"""FastAPI application: ``/health``, ``/players``, ``/tournaments``, ``/match``.

Run it with::

    uvicorn api.main:app --reload    # from src/, or with src/ on PYTHONPATH

The app is built by :func:`create_app` so the startup context can be injected.
Importing this module is cheap and side-effect-free, the pipeline load (~1.8 s;
see :func:`api.deps.build_api_context`) happens in the **lifespan**, i.e. when
the app actually starts, so a test may import ``api.main`` (or generate its
OpenAPI schema) without touching the vendored dataset.

Conventions:
  * Startup state is loaded once into ``app.state`` by the lifespan handler and
    read by every handler through a dependency (:func:`get_context`,
    :func:`get_registry`). Handlers never load anything themselves.
  * All responses are ``api.schemas`` models (``response_model=`` on every
    route); no handler returns a raw dict.
  * Anything configurable (CORS origins, titles, the draws directory) is a
    ``config`` constant, overridable per app instance for tests.

**``/tournaments/{id}/bracket`` versus simulability.** ``/bracket`` calls
``load_draw`` and nothing else, so ``simulate_bracket``'s placeholder-entrant
refusal never fires there. A draw containing ``Qualifier`` slots loads fine
(placeholders take the default skill profile, as everywhere else), so such a
draw is legitimately listed and its bracket legitimately served. The
simulability question is answered at ``/simulate`` instead, see
:func:`tournament_simulation` for the decision and its reasoning.

**What this module may and may not simulate.** ``/simulate`` reads a file that
``scripts/precompute_sim.py`` wrote offline and simulates nothing, Phase 3's
global rule ("never run a full 5,000-run MC synchronously inside a request
handler") is a rule, not a performance preference, so this module holds no
reference to ``monte_carlo`` or ``simulate_bracket`` at all. ``/storybook`` is
the deliberate exception the same rule allows: **one** bracket, not thousands,
run live per request via ``sim.tournament.storybook_run``. The line is the cost
of a single run (0.4–1.4 s for a 128 draw; measured below), and
``tests/test_api_simulate.py`` pins it, the Monte Carlo entry points must stay
un-imported here even though the storybook one is now imported.
"""

from __future__ import annotations

import importlib
import json
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Path as PathParam, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import StringConstraints, ValidationError

import config
from api.deps import ApiContext, ContextFactory, build_api_context, log_context
from api.registry import TournamentEntry, TournamentRegistry, build_registry, cache_path_for
from api.schemas import (
    CLASSIFIER_LIMITATION,
    CLASSIFIER_LIMITATION_DETAIL,
    CONTENT_NOTE_CURATED,
    CONTENT_NOTE_UPLOAD,
    CONTENT_SOURCE_CURATED,
    CONTENT_SOURCE_UPLOAD,
    FORECAST_CLASSIFIER_LIMITATION,
    FORECAST_CLASSIFIER_LIMITATION_DETAIL,
    IS_FORECAST,
    MATCH_CLASSIFIER_LIMITATION,
    MATCH_CLASSIFIER_LIMITATION_DETAIL,
    BracketResponse,
    BracketSlot,
    HealthResponse,
    InvalidDraw,
    MatchMetadata,
    MatchSimulateRequest,
    MatchSimulateResponse,
    PlayerSearchResponse,
    PlayerSummary,
    RankingsResponse,
    SetScore,
    SimulationResponse,
    StorybookChampion,
    StorybookMatch,
    StorybookMetadata,
    StorybookPlayerRun,
    StorybookResponse,
    StorybookRound,
    SurfaceSkill,
    TotalGamesSummary,
    TournamentListResponse,
    TournamentSummary,
    UploadResponse,
)
from api.uploads import UploadedDraw, UploadStore
from common.names import NameIndex, NameMatch, resolve_name
from features.elo import elo_cache_path
from features.serve import SkillTable
from sim.draw import Draw, DrawValidationError, parse_draw
from sim.reconcile import ClassifierProb
from sim.single_match import SingleMatchResult, simulate_single_match
from sim.tournament import StorybookResult, storybook_run

logger = logging.getLogger(__name__)

# Deterministic surface ordering in every response body (VALID_SURFACES is a set).
SURFACE_ORDER = sorted(config.VALID_SURFACES)

# The ``/players`` query type. ``strip_whitespace`` runs **before** ``min_length``
# in Pydantic v2, which is the whole point: a bare ``min_length=1`` accepts " "
# and "\t", and ``common.names.resolve_name`` then strips it to "", at which
# point the substring strategy's ``"" in name`` matches *every* player and the
# endpoint returns the entire universe (1,408 players, ~470 KB) for what is
# semantically an empty query. Stripping first makes whitespace-only input take
# the same 422 path as "" and a missing query, and the handler searches (and
# echoes) the stripped string.
PlayerQuery = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


# --------------------------------------------------------------------------- #
# Security headers (security hardening)
# --------------------------------------------------------------------------- #
# Sent on every response by the middleware below. Scoped to what this service
# actually is, a JSON API whose only HTML is the auto-generated docs UI:
#
#   * Content-Security-Policy: JSON responses render no markup, so the strict
#     `default-src 'none'` policy costs nothing and is pure defence-in-depth
#     (a browser coerced into treating a response as a document can load
#     nothing). `frame-ancestors 'none'` + `base-uri 'none'` round it out. The
#     Swagger/ReDoc HTML at /docs and /redoc is the one exception (see
#     DOCS_CSP): it deliberately pulls its bundle from the jsDelivr CDN and runs
#     an inline bootstrap, so the strict policy would blank it out.
#   * X-Content-Type-Options: nosniff, stop a browser MIME-sniffing a JSON body
#     into something executable; the one header that most matters for an API.
#   * X-Frame-Options: DENY (paired with frame-ancestors 'none' for old
#     browsers that predate CSP), nothing here is meant to be iframed.
#   * Referrer-Policy: no-referrer, a JSON API and a query-string SPA have no
#     use for the Referer header, so send the least. (Stricter than the
#     strict-origin-when-cross-origin default; nothing here needs referrer info.)
#
# HSTS is deliberately NOT set here. Render terminates TLS at its edge and owns
# the certificate and the redirect; forcing Strict-Transport-Security from the
# app behind that proxy is the platform's call, not the app's, and a wrong
# max-age set here could outlive the deployment. See the security report.
API_CSP = "default-src 'none'; frame-ancestors 'none'; base-uri 'none'"

# Relaxed policy for the Swagger UI / ReDoc HTML only. FastAPI serves those
# pages with a script + stylesheet from cdn.jsdelivr.net and an inline
# initialiser, and Swagger uses data: images; this is exactly what they need and
# nothing more. Applied only to /docs, /redoc and /docs/oauth2-redirect.
DOCS_CSP = (
    "default-src 'none'; "
    "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
    "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
    "img-src 'self' data: https://fastapi.tiangolo.com; "
    "font-src 'self' https://cdn.jsdelivr.net; "
    "frame-ancestors 'none'; base-uri 'none'"
)
DOCS_PATHS = frozenset({"/docs", "/redoc", "/docs/oauth2-redirect"})

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


def _apply_security_headers(response, *, csp: str = API_CSP) -> None:
    """Set the standard security headers + a CSP on ``response`` if unset."""
    for header, value in SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    response.headers.setdefault("Content-Security-Policy", csp)


def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all 500 handler that still carries the security headers.

    An unhandled exception is turned into a 500 by Starlette's
    ``ServerErrorMiddleware``, which sits **outside** the ``BaseHTTPMiddleware``
    that adds :data:`SECURITY_HEADERS`, so without this, a raw 500 goes out
    bare (verified: no CSP, no nosniff). Registering a handler for ``Exception``
    routes those 500s here instead, where the headers are set explicitly.

    The body is a fixed, generic message: ``exc`` is never echoed, so an internal
    error (a stack detail, a path, a value) cannot leak to the client. More
    specific handlers (``HTTPException``, ``RequestValidationError``) still win by
    MRO, so this only ever fires for genuinely unexpected errors, the 4xx paths
    are untouched.
    """
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    response = JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
    _apply_security_headers(response)
    return response


# --------------------------------------------------------------------------- #
# Rate limiting for /storybook (security hardening)
# --------------------------------------------------------------------------- #
# A deliberately tiny in-memory, per-IP sliding-window limiter, built here rather
# than pulled in as a dependency: the one endpoint that needs it does not justify
# a rate-limit library (and slowapi's decorator is incompatible with this
# module's `from __future__ import annotations` + FastAPI forward-ref
# signatures). Applied as a route *dependency*, so it reads the client IP without
# touching the handler's own signature.
#
# HONEST SCOPE, repeated verbatim from config.API_STORYBOOK_RATE_LIMIT
# because it is the whole point: the window counts live in this process's memory
# only. Render's free tier cold-starts and restarts routinely and every restart
# wipes the counters; a multi-replica deployment gives each replica its own. So
# this is resource-exhaustion hygiene against one careless/looping client, NOT
# abuse prevention. It gates no authorization (there is none). The client key
# below prefers Cloudflare's non-spoofable CF-Connecting-IP (Render fronts
# onrender.com with Cloudflare) and only falls back to the spoofable
# X-Forwarded-For off-platform.
_PERIOD_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}


def _parse_rate(spec: str) -> tuple[int, int]:
    """``"12/minute"`` → ``(12, 60)``, (max requests, window seconds)."""
    count_str, _, period = spec.partition("/")
    period = period.strip().rstrip("s")  # tolerate "minute"/"minutes"
    if not count_str.strip().isdigit() or period not in _PERIOD_SECONDS:
        raise ValueError(
            f"rate limit {spec!r} must be '<int>/<second|minute|hour|day>'"
        )
    return int(count_str), _PERIOD_SECONDS[period]


def _client_key(request: Request) -> str:
    """Per-client key for the rate limit.

    Behind Render's proxy ``request.client.host`` is the *proxy's* address, which
    would lump every visitor into one bucket, so a forwarded header is preferred
    and the socket peer is the fallback for direct (local, un-proxied) requests.
    The order matters:

      1. ``CF-Connecting-IP``, Render fronts ``*.onrender.com`` with Cloudflare,
         which **overwrites** this header with the real connecting IP on every
         request (a client-sent value is discarded, not appended), so it is not
         client-spoofable in the deployed topology. Preferred when present.
      2. First hop of ``X-Forwarded-For``, the fallback for a deployment without
         Cloudflare in front. This one *is* client-supplied and spoofable: an
         attacker can rotate it to dodge the limit. Acceptable because the limit
         is hygiene, not a security control, and gates no authorization decision.
      3. The socket peer, direct local dev, no proxy at all.
    """
    cf_ip = request.headers.get("cf-connecting-ip")
    if cf_ip:
        return cf_ip.strip() or "unknown"
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    client = request.client
    return client.host if client else "unknown"


class SlidingWindowRateLimiter:
    """Thread-safe, in-memory, per-key sliding-window limiter.

    Sync FastAPI dependencies run in a threadpool, so access is guarded by a
    lock. Memory is bounded by the number of *active* client keys in the window
    (each key holds at most ``max_requests`` timestamps, pruned on every touch).
    """

    def __init__(self, spec: str) -> None:
        self.max_requests, self.window = _parse_rate(spec)
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> float | None:
        """Register a hit for ``key``; return ``None`` if allowed, else the
        ``Retry-After`` seconds until the window frees a slot."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            cutoff = now - self.window
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.max_requests:
                retry_after = self.window - (now - hits[0])
                return max(1, int(retry_after) + 1)
            hits.append(now)
            return None


class LiveComputeGate:
    """A shared, in-memory cap on *concurrent* live simulations (security-review-2).

    The per-IP rate limiters bound each client's request *rate*; nothing bounds
    how many live simulations execute *at the same time*. ``/storybook`` and
    ``/match/simulate`` are sync handlers, so FastAPI dispatches them into the
    anyio threadpool (default 40 workers), a single client firing concurrent
    requests under its own rate limit, or a handful of clients each under theirs,
    can pile a threadpool's worth of 128-match brackets onto one 0.1-CPU instance
    at once. Summed at their per-IP limits the three live-compute endpoints
    oversubscribe that CPU budget several times over, and queued CPU-bound
    handlers turn into ballooning latency and edge timeouts.

    This bounds the number in flight to ``max_concurrency`` and **sheds** the
    excess immediately (:meth:`acquire` returns ``False``) rather than blocking a
    threadpool thread waiting for a slot, blocking would re-create the pile-up it
    exists to prevent. Same honest scope as the rate limiters: per-process, resets
    on restart, independent per replica; it is peak-load hygiene, not a
    substitute for the platform's volumetric defense.
    """

    def __init__(self, max_concurrency: int) -> None:
        if max_concurrency < 1:
            raise ValueError(
                f"max_concurrency must be >= 1, got {max_concurrency}"
            )
        self.max_concurrency = max_concurrency
        self._sem = threading.BoundedSemaphore(max_concurrency)

    def acquire(self) -> bool:
        """Take a slot without blocking; ``True`` if one was free, else ``False``."""
        return self._sem.acquire(blocking=False)

    def release(self) -> None:
        self._sem.release()


# A factory with ``cli.simulate_match.make_classifier_prob``'s signature:
# ``(estimator, data, surface_history, h2h_history) -> ClassifierProb``. Called
# once at startup with the pieces :class:`~api.deps.ApiContext` already carries.
ClassifierFactory = Callable[[Any, Any, dict, dict], ClassifierProb]


@dataclass(frozen=True)
class ClassifierAdapter:
    """The startup-built ``ClassifierProb`` and the name it is disclosed under.

    The two travel together because they must agree: ``name`` is what every
    ``metadata.adapter`` field reports, and it is *derived from* the factory that
    actually ran rather than written down beside it, so a response cannot name an
    adapter other than the one that produced its numbers.
    """

    call: ClassifierProb
    name: str


def resolve_classifier_factory(spec: str = config.API_CLASSIFIER_ADAPTER):
    """Import the ``ClassifierProb`` factory named by ``spec`` (``"module:attr"``).

    **Why an import by name instead of an import statement.** The default
    adapter, ``common.classifier_adapter:make_classifier_prob``, is a UI-free
    module ``api/`` could import outright, so no ``cli/`` module is loaded by the
    running server at all (``tests/test_api_storybook.py`` proves it in a clean
    interpreter). Late-binding is kept because its purpose stands on its own:
    "which model does this server publish" is deployment configuration, and
    swapping the adapter, for a differently-calibrated one, or back to an older
    one for comparison, is one config line rather than a code change. Whatever
    runs is reported: every response names :attr:`ClassifierAdapter.name`,
    derived from the factory that actually ran.

    Args:
        spec: ``"<module>:<attribute>"``, e.g.
            ``"common.classifier_adapter:make_classifier_prob"``.

    Returns:
        The factory callable.

    Raises:
        ValueError: If ``spec`` is not ``module:attribute``.
        ImportError: If the module or attribute does not exist, raised at
            **startup**, so a misconfigured server fails immediately rather than
            on the first ``/storybook`` request.
    """
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(
            f"classifier adapter {spec!r} must be written '<module>:<attribute>', "
            f"e.g. 'cli.simulate_match:make_classifier_prob'."
        )
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise ImportError(
            f"module {module_name!r} has no attribute {attribute!r} "
            f"(from config.API_CLASSIFIER_ADAPTER = {spec!r})."
        ) from exc


def adapter_name(factory: ClassifierFactory) -> str:
    """Dotted name of ``factory``, for ``metadata.adapter``.

    Derived, never hardcoded, see :class:`ClassifierAdapter`. For the shipped
    default this reads ``"common.classifier_adapter.make_classifier_prob"``, the
    same string ``scripts/precompute_sim.py`` writes into its cache files
    (``tests/test_api_storybook.py`` pins the two together).
    """
    module = getattr(factory, "__module__", None) or "?"
    qualname = getattr(factory, "__qualname__", None) or type(factory).__name__
    return f"{module}.{qualname}"


def build_classifier(
    context: ApiContext, factory: ClassifierFactory
) -> ClassifierAdapter:
    """Build the adapter once, at startup, from the loaded context.

    Cheap: the adapter defers both its as-of-now rolling-form table (0.08 s
    over the full frame) and every ``predict_proba`` to first use, so this adds
    nothing to the ~2 s startup and 0.08 s to the first simulated request. Building it
    *here* rather than per request is what keeps ``/storybook`` to one bracket's
    worth of work, and, because the adapter's memo table is shared by every
    request, repeat matchups across seeds cost one ``predict_proba`` in total
    rather than one per request.
    """
    call = factory(
        context.estimator,
        context.data,
        context.surface_history,
        context.h2h_history,
    )
    # Warm the one piece of state the adapter otherwise defers to the first
    # simulate request: the as-of-now rolling-form table (a ~0.08 s scan of the
    # full frame, built once and kept). Doing it here folds that cost into the
    # ~2 s startup, the same eager-init discipline the rest of this module
    # follows, so the first /match or /storybook request is not the one that
    # pays for it.
    #
    # Best-effort by design. The `form_table` property is read only if the
    # adapter exposes one (the factory is pluggable via
    # config.API_CLASSIFIER_ADAPTER), and building it is wrapped so that any
    # failure, chiefly a minimal test-fixture frame that lacks the rolling
    # columns, falls back to the original lazy behaviour instead of aborting
    # startup. Warming can only make the first request faster; it must never be
    # the thing that stops the app from starting.
    _warm_form_table(call)
    return ClassifierAdapter(call=call, name=adapter_name(factory))


def _warm_form_table(call: ClassifierProb) -> None:
    """Trigger the adapter's deferred rolling-form build at startup, if it has one."""
    # Check the *class*, not the instance: `hasattr(call, "form_table")` would
    # invoke the property getter and let its exception escape this guard. The
    # class carries the property descriptor whether or not it can be evaluated.
    if not hasattr(type(call), "form_table"):
        return
    try:
        form_table = call.form_table
    except Exception as exc:  # noqa: BLE001, warming must never break startup
        logger.warning(
            "Skipped rolling-form warm-up (%s); it will build on first request.",
            exc,
        )
        return
    logger.info(
        "Warmed rolling-form table at startup (%d players).", len(form_table.form)
    )


def get_context(request: Request) -> ApiContext:
    """FastAPI dependency: the startup-loaded context for this app instance.

    Reads ``app.state.context``, which the lifespan handler populated once. It is
    a dependency rather than a module global so that two apps (e.g. the real one
    and a test one) can coexist in a process with independent state.
    """
    return request.app.state.context


def get_registry(request: Request) -> TournamentRegistry:
    """FastAPI dependency: the startup-scanned draws catalogue for this app.

    Same pattern (and same reason) as :func:`get_context`, read once from
    ``app.state``, never rebuilt per request.
    """
    return request.app.state.registry


def get_classifier(request: Request) -> ClassifierAdapter:
    """FastAPI dependency: the startup-built ``ClassifierProb`` adapter.

    Same ``app.state`` pattern as :func:`get_context`. Built once, a per-request
    adapter would throw away the memo table that makes a live storybook viable.
    """
    return request.app.state.classifier


def get_cache_dir(request: Request) -> Path:
    """FastAPI dependency: where this app reads precomputed simulations.

    Stored on ``app.state`` like the context and the registry, so a test app can
    point at a fixture cache directory without touching ``data/cache/``.
    """
    return request.app.state.cache_dir


def get_upload_store(request: Request) -> UploadStore:
    """FastAPI dependency: this app's in-memory store of uploaded draws.

    One store per app instance, created at startup and never persisted, the
    ephemeral contract in :mod:`api.uploads`. A fresh app (a restart, or a test
    app) starts with an empty store, which is exactly why an upload does not
    survive a restart.
    """
    return request.app.state.upload_store


def _player_summary(
    name: str, index: NameIndex, skill_table: SkillTable
) -> PlayerSummary:
    """Assemble one player's wire model from the skill table.

    A player with no aggregated data on a surface gets ``SkillTable.default`` for
    it (``n_serve_pts == 0`` flags that to the client) rather than being omitted,
    so the ``skills`` map has the same keys for everyone.
    """
    player_id = index.id_for(name)
    skills: dict[str, SurfaceSkill] = {}
    for surface in SURFACE_ORDER:
        skill = skill_table.get(player_id, surface)
        skills[surface] = SurfaceSkill(
            spw=skill.spw,
            rpw=skill.rpw,
            n_serve_pts=skill.n_serve_pts,
            n_return_pts=skill.n_return_pts,
        )
    return PlayerSummary(player_id=player_id, name=name, skills=skills)


def _bracket_response(draw: Draw, tournament_id: str | None = None) -> BracketResponse:
    """Assemble a resolved draw's wire model.

    Every slot is included, placeholders are flagged via ``is_placeholder``,
    never filtered out, because a bracket with holes in it is not a bracket.

    ``tournament_id`` overrides the draw's own id in the response. Curated draws
    pass ``None`` and report their real id; an uploaded draw reports the
    generated upload id it is addressed by, so the id a client fetched with is
    the id it gets back (the draw's descriptive ``tournament_id`` never becomes
    an addressable handle, see ``api.uploads``).
    """
    return BracketResponse(
        tournament_id=tournament_id if tournament_id is not None else draw.tournament_id,
        name=draw.name,
        surface=draw.surface,
        best_of=draw.best_of,
        final_set_tiebreak=draw.final_set_tiebreak,
        draw_size=draw.draw_size,
        slots=[
            BracketSlot(
                position=slot.position,
                player=slot.player,
                is_placeholder=slot.is_placeholder,
                player_id=slot.player_id,
                seed=draw.seed_of(slot),
            )
            for slot in draw.bracket
        ],
    )


def _unknown_tournament(tournament_id: str, registry: TournamentRegistry) -> HTTPException:
    """404 for ``/simulate``, grouping ids by what can actually be simulated.

    ``/bracket``'s 404 lists every registered id in one breath, correctly: a draw
    file that failed validation *is* fetchable there, as a 422 carrying its
    problem list. Simulation is different, and in two ways, an unloadable file
    can never be simulated, and neither can a perfectly loadable draw that still
    has ``Qualifier`` slots. One flat "Known ids" list would imply a capability
    two of those groups do not have, so the same courtesy is extended split by
    what the caller can do with each id.
    """
    groups = [
        ("Simulatable ids", [e.tournament_id for e in registry.valid if e.is_simulatable]),
        (
            "Draws awaiting entrants (listed, but not simulatable until their "
            "placeholder slots are filled)",
            [e.tournament_id for e in registry.valid if not e.is_simulatable],
        ),
        (
            "Draw files that failed validation (listed, but not simulatable)",
            [e.tournament_id for e in registry.invalid],
        ),
    ]
    parts = [f"No tournament with id {tournament_id!r}."]
    for label, ids in groups:
        if ids:
            parts.append(f"{label}: " + ", ".join(repr(i) for i in ids) + ".")
    if len(parts) == 1:
        parts.append("No draws are registered.")
    return HTTPException(status_code=404, detail=" ".join(parts))


def _invalid_draw_file(entry: TournamentEntry) -> HTTPException:
    """422 for a draw file that failed validation."""
    return HTTPException(
        status_code=422,
        detail={
            "message": (
                f"Draw file {entry.source!r} failed validation with "
                f"{len(entry.problems)} problem(s)."
            ),
            "tournament_id": entry.tournament_id,
            "source": entry.source,
            "problems": list(entry.problems),
        },
    )


def _not_simulatable(entry: TournamentEntry) -> HTTPException:
    """409 for a draw holding placeholder entrants.

    Mirrors, over HTTP, the refusal ``simulate_bracket`` already makes offline:
    a placeholder has no ``player_id`` and no classifier history, so no match-win
    probability exists for it. Every offending slot is named rather than a
    probability invented for it.
    """
    placeholders = entry.placeholder_slots
    return HTTPException(
        status_code=409,
        detail={
            "reason": "draw_not_simulatable",
            "message": (
                f"Draw {entry.draw.tournament_id!r} has "
                f"{len(placeholders)} placeholder entrant(s) and cannot "
                f"be simulated: a placeholder has no player id and no "
                f"classifier history, so no match-win probability can be "
                f"computed for it. Fill these slots with real entrants in "
                f"{entry.source!r}, then precompute."
            ),
            "tournament_id": entry.draw.tournament_id,
            "source": entry.source,
            "placeholder_slots": [
                {"position": slot.position, "player": slot.player}
                for slot in placeholders
            ],
        },
    )


def _simulatable_entry(
    tournament_id: str, registry: TournamentRegistry
) -> TournamentEntry:
    """Look an id up and refuse it unless it can actually be simulated.

    The gate **both** simulation endpoints go through, so ``/simulate`` (cached)
    and ``/storybook`` (live) cannot disagree about which draws are runnable or
    about how they say no: unknown id → 404 grouped by simulability, a file that
    failed validation → 422 with the validator's problems, a draw still holding
    ``Qualifier`` slots → 409.

    The verdict itself is :attr:`~api.registry.TournamentEntry.is_simulatable`,
    the one definition of the rule, living in the registry next to the flag it
    reads, not a placeholder comprehension repeated per endpoint.

    Returns:
        The entry, with a non-``None`` ``draw`` that ``simulate_bracket``/
        ``storybook_run`` will accept.
    """
    entry = registry.get(tournament_id)
    if entry is None:
        raise _unknown_tournament(tournament_id, registry)
    if entry.draw is None:
        raise _invalid_draw_file(entry)
    if not entry.is_simulatable:
        raise _not_simulatable(entry)
    return entry


def _unknown_upload(tournament_id: str) -> HTTPException:
    """404 for an upload id that is not (or no longer) in the store.

    The message names the ephemeral contract rather than pretending the id was
    never valid: an id from before a restart, or one evicted by newer uploads,
    is genuinely gone, and the fix is to upload again.
    """
    return HTTPException(
        status_code=404,
        detail={
            "reason": "upload_not_found",
            "message": (
                f"No uploaded draw with id {tournament_id!r}. Uploaded draws are "
                f"held in memory only and are cleared when the server restarts "
                f"or when older uploads are evicted, so a shared link can stop "
                f"working. Upload the draw again to get a new id."
            ),
            "tournament_id": tournament_id,
        },
    )


def _upload_not_simulatable(uploaded: UploadedDraw) -> HTTPException:
    """409 for an uploaded draw holding placeholder entrants.

    The same refusal, and the same body shape, as :func:`_not_simulatable`, an
    uploaded ``Qualifier`` slot is no more simulatable than a curated one.
    """
    placeholders = [slot for slot in uploaded.draw.bracket if slot.is_placeholder]
    return HTTPException(
        status_code=409,
        detail={
            "reason": "draw_not_simulatable",
            "message": (
                f"Uploaded draw {uploaded.upload_id!r} has {len(placeholders)} "
                f"placeholder entrant(s) and cannot be simulated: a placeholder "
                f"has no player id and no classifier history, so no match-win "
                f"probability can be computed for it. Replace them with real "
                f"entrants and upload again."
            ),
            "tournament_id": uploaded.upload_id,
            "source": uploaded.upload_id,
            "placeholder_slots": [
                {"position": slot.position, "player": slot.player}
                for slot in placeholders
            ],
        },
    )


def _extract_event_date(data: Any) -> tuple[date | None, bool]:
    """Read an uploaded draw's optional ``event_date`` and derive is_forecast.

    Returns ``(event_date, is_forecast)``. A draw with no ``event_date`` is
    treated as historical (``(None, False)``), the safe default, since claiming
    a forecast we cannot support would be the dishonest direction. A future date
    makes ``is_forecast`` true.

    The field is read from the raw JSON here, not through ``parse_draw`` (which
    ignores it as an unknown key), so adding it required no change to the draw
    schema in ``sim/``.

    Raises:
        HTTPException: 422 if ``event_date`` is present but not a ``YYYY-MM-DD``
            string, a malformed optional field is a client error worth naming,
            not something to silently ignore.
    """
    if not isinstance(data, dict):
        return None, False
    raw = data.get(config.UPLOAD_EVENT_DATE_FIELD)
    if raw is None:
        return None, False
    if not isinstance(raw, str):
        raise HTTPException(
            status_code=422,
            detail={
                "reason": "invalid_event_date",
                "message": (
                    f"{config.UPLOAD_EVENT_DATE_FIELD!r} must be a "
                    f"'YYYY-MM-DD' string, got {type(raw).__name__}."
                ),
            },
        )
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "reason": "invalid_event_date",
                "message": (
                    f"{config.UPLOAD_EVENT_DATE_FIELD!r} must be a valid "
                    f"'YYYY-MM-DD' date: {exc}."
                ),
            },
        ) from exc
    return parsed, parsed > date.today()


@dataclass(frozen=True)
class DrawDisclosure:
    """Everything a storybook response discloses about the *draw* it ran.

    Bundled so the one response builder serves both a curated draw and an
    uploaded one without a pile of keyword arguments, and so the two concerns
    stay distinct: ``is_forecast`` / ``classifier_limitation`` are about the
    *model*'s knowledge (do the inputs postdate the event?), while
    ``content_source`` / ``content_note`` are about the *draw*'s provenance (is
    it a verified example or an unverified upload?). See ``api.schemas`` for the
    canonical text of each.
    """

    source: str
    is_forecast: bool
    classifier_limitation: str
    classifier_limitation_detail: str
    content_source: str
    content_note: str


def _curated_disclosure(entry: TournamentEntry) -> DrawDisclosure:
    """Disclosure for a git-committed example draw (historical, verified)."""
    return DrawDisclosure(
        source=entry.source,
        is_forecast=IS_FORECAST,
        classifier_limitation=CLASSIFIER_LIMITATION,
        classifier_limitation_detail=CLASSIFIER_LIMITATION_DETAIL,
        content_source=CONTENT_SOURCE_CURATED,
        content_note=CONTENT_NOTE_CURATED,
    )


def _upload_disclosure(uploaded: UploadedDraw) -> DrawDisclosure:
    """Disclosure for a user-submitted draw.

    Two things differ from a curated draw, and they are independent. The
    *model* limitation flips only when the draw is future-dated: a not-yet-played
    event is a genuine forecast (``is_forecast=True``), an already-played one is
    the same retrospective a curated draw would be. The *content* note is always
    the upload note, unverified, and held in memory only.
    """
    if uploaded.is_forecast:
        limitation = FORECAST_CLASSIFIER_LIMITATION
        detail = FORECAST_CLASSIFIER_LIMITATION_DETAIL
    else:
        limitation = CLASSIFIER_LIMITATION
        detail = CLASSIFIER_LIMITATION_DETAIL
    return DrawDisclosure(
        # The upload id is the addressable handle and the honest "source": there
        # is no file. It names exactly the resource the numbers came from.
        source=uploaded.upload_id,
        is_forecast=uploaded.is_forecast,
        classifier_limitation=limitation,
        classifier_limitation_detail=detail,
        content_source=CONTENT_SOURCE_UPLOAD,
        content_note=CONTENT_NOTE_UPLOAD,
    )


def _storybook_response(
    result: StorybookResult,
    *,
    disclosure: DrawDisclosure,
    context: ApiContext,
    adapter: str,
    tournament_id: str | None = None,
) -> StorybookResponse:
    """Assemble the wire model from a finished :class:`StorybookResult`.

    **Presentation over already-built data**, exactly as
    ``scripts/precompute_sim.build_payload`` is for a Monte Carlo result: every
    field is read off the storybook run's own structures, ``rounds``/``matches`` (including
    the ``scoreline`` and the rendered ``line`` they already carry),
    ``champion``/``champion_seed``, and the ``runs`` summaries, and nothing is
    re-derived from the underlying ``BracketResult`` a second time. The CLI's
    text renderer (``render_storybook``) is deliberately **not** called: it is a
    different presentation of the same structure, and this endpoint returns JSON.

    ``disclosure`` carries the draw-specific provenance (curated vs upload,
    historical vs forecast); ``tournament_id`` overrides the response id for an
    uploaded draw, exactly as in :func:`_bracket_response`.
    """
    return StorybookResponse(
        tournament_id=tournament_id if tournament_id is not None else result.tournament_id,
        name=result.name,
        surface=result.surface,
        best_of=result.best_of,
        final_set_tiebreak=result.final_set_tiebreak,
        draw_size=result.draw_size,
        match_count=len(result.matches),
        rounds=[
            StorybookRound(
                index=storybook_round.index,
                label=storybook_round.label,
                matches=[
                    StorybookMatch(
                        round_index=played.round_index,
                        round_label=played.round_label,
                        match_index=played.match_index,
                        winner=played.winner,
                        loser=played.loser,
                        winner_seed=played.winner_seed,
                        loser_seed=played.loser_seed,
                        scoreline=played.scoreline,
                        line=played.line,
                    )
                    for played in storybook_round.matches
                ],
            )
            for storybook_round in result.rounds
        ],
        champion=StorybookChampion(
            position=result.champion.position,
            player=result.champion.player,
            player_id=result.champion.player_id,
            seed=result.champion_seed,
        ),
        runs=[
            StorybookPlayerRun(
                position=run.position,
                player=run.player,
                player_id=run.player_id,
                seed=run.seed,
                is_champion=run.is_champion,
                furthest_round=run.furthest_round,
                matches_won=run.matches_won,
                beat=list(run.beat),
                eliminated_by=run.eliminated_by,
            )
            for run in result.runs
        ],
        metadata=StorybookMetadata(
            seed=result.seed,
            mode=result.mode,
            w=result.w,
            data_through_year=context.data_through_year,
            estimator_class=context.estimator_class,
            adapter=adapter,
            is_forecast=disclosure.is_forecast,
            classifier_limitation=disclosure.classifier_limitation,
            classifier_limitation_detail=disclosure.classifier_limitation_detail,
            source=disclosure.source,
            content_source=disclosure.content_source,
            content_note=disclosure.content_note,
        ),
    )


def _resolve_match_player(query: str, index: NameIndex) -> str:
    """Resolve one single-match player query to a canonical display name.

    Uses the same shared resolver as ``/players`` (``common.names.resolve_name``),
    then reduces its three outcomes to the one a *simulation* needs, a single,
    unambiguous name, rather than the list a search returns. The canonical name
    (not the raw query) is what the classifier's name-keyed feature lookup needs,
    so this is where a fragment like ``"alcaraz"`` becomes ``"Carlos Alcaraz"``.

    Raises:
        HTTPException: 422 ``player_not_found`` when nothing resolves, or 422
            ``player_ambiguous`` (carrying the candidate list) when several do.
            The model refuses to score a player it has no history for, so an
            unresolved name is a request error, not a silent default.
    """
    match: NameMatch | None = resolve_name(query, index)
    if match is None:
        raise HTTPException(
            status_code=422,
            detail={
                "reason": "player_not_found",
                "message": (
                    f"No player matches {query!r}. Pick a player from the "
                    f"suggestions, or try a fuller name."
                ),
                "query": query,
            },
        )
    if match.is_ambiguous:
        raise HTTPException(
            status_code=422,
            detail={
                "reason": "player_ambiguous",
                "message": (
                    f"{query!r} matches {len(match.candidates)} players. Pick one."
                ),
                "query": query,
                "candidates": list(match.candidates),
            },
        )
    return match.name


def _match_response(
    result: SingleMatchResult, context: ApiContext, adapter: str
) -> MatchSimulateResponse:
    """Assemble the wire model from a finished :class:`SingleMatchResult`.

    Presentation over already-built data, the same pattern as
    :func:`_storybook_response`: every number is read off the aggregate the
    simulator returned, and the required :class:`ModelDisclosure` fields are
    filled with the single-match wording (a hypothetical matchup is neither a
    played draw nor a scheduled event, so ``is_forecast`` is ``false`` and the
    limitation copy is the match-specific one).
    """
    return MatchSimulateResponse(
        player_a=result.player_a,
        player_b=result.player_b,
        surface=result.surface,
        best_of=result.best_of,
        final_set_rule=result.final_set_rule,
        win_prob_a=result.win_prob_a,
        win_prob_b=result.win_prob_b,
        p_clf=result.p_clf,
        analytic_win_prob_a=result.analytic_win_prob_a,
        set_scores=[
            SetScore(
                sets_a=outcome.sets_a,
                sets_b=outcome.sets_b,
                count=outcome.count,
                prob=outcome.prob,
            )
            for outcome in result.set_scores
        ],
        total_games=TotalGamesSummary(
            mean=result.total_games.mean,
            std=result.total_games.std,
            minimum=result.total_games.minimum,
            maximum=result.total_games.maximum,
            distribution=result.total_games.distribution,
        ),
        metadata=MatchMetadata(
            mode=result.mode,
            w=result.w,
            data_through_year=context.data_through_year,
            estimator_class=context.estimator_class,
            adapter=adapter,
            is_forecast=False,
            classifier_limitation=MATCH_CLASSIFIER_LIMITATION,
            classifier_limitation_detail=MATCH_CLASSIFIER_LIMITATION_DETAIL,
            # No draw file backs a single match; name the run, honestly, rather
            # than borrow a filename it never had.
            source="single-match simulation",
            runs=result.n_runs,
            seed=result.seed,
        ),
    )


def create_app(
    context_factory: ContextFactory = build_api_context,
    draws_dir: Path | str = config.DRAWS_DIR,
    cache_dir: Path | str = config.CACHE_DIR,
    classifier_factory: ClassifierFactory | None = None,
    upload_store: UploadStore | None = None,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        context_factory: Called **exactly once**, on startup, to produce the
            :class:`~api.deps.ApiContext`. Defaults to the real pipeline loader;
            tests inject a fixture factory so they never load vendored data.
        draws_dir: Directory scanned once at startup for draw files. Defaults to
            ``config.DRAWS_DIR``; tests point it at a fixture directory so the
            suite never depends on the shipped draws.
        cache_dir: Directory of precomputed simulation results read by
            ``/simulate``. Defaults to ``config.CACHE_DIR``; the same
            injection seam, for the same reason. It is **not** scanned at
            startup, a cache file appearing while the server runs is picked up
            on the next request, which is what makes the precompute script
            usable against a live server.
        classifier_factory: Builds the ``ClassifierProb`` adapter ``/storybook``
            simulates with, called **exactly once** at startup with the
            context's estimator and histories. Defaults to ``None``, meaning
            "resolve :data:`config.API_CLASSIFIER_ADAPTER`", see
            :func:`resolve_classifier_factory` for why the default is a name
            rather than an import. Tests inject a deterministic stub so the
            suite never needs the vendored data or the pinned model to exercise
            an endpoint; ``tests/test_classifier_adapter.py`` drives the *real*
            resolved factory end-to-end over a small engineered frame, so the
            stub does not stand in for the shipped adapter anywhere it matters.
        upload_store: In-memory store backing ``POST /tournaments/upload`` and
            the serving of uploaded draws. Defaults to ``None``, meaning "build a
            fresh :class:`~api.uploads.UploadStore` capped at
            ``config.API_UPLOAD_MAX_DRAWS``". Tests inject one with a small cap to
            exercise eviction. Never persisted: a new app starts with an empty
            store, which *is* the ephemeral behaviour uploaded draws promise.

    Returns:
        The configured app. Nothing is loaded, scanned or resolved until it
        starts.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Lifespan, not the deprecated @app.on_event("startup"). Runs once per
        # app instance; the factory result is the single shared context.
        context = context_factory()
        app.state.context = context
        log_context(context)
        # Scanned once here, not per request. Draw files are hand-entered and
        # near-static, resolving them needs the skill table that was just built,
        # and the whole scan costs ~4 ms for the shipped 8- and 128-slot draws
        # (measured 2026-08-03), repeating it per request would buy nothing but
        # a slower endpoint. This also matches the pattern the rest of startup
        # follows: the app's expensive, immutable state is assembled at startup
        # and only read afterwards. The trade-off is explicit: a draw file added or edited
        # while the server is running is not picked up until it restarts
        # (`--reload` covers the dev loop).
        app.state.registry = build_registry(context.skill_table, draws_dir)
        # A path, not a scan: see the `cache_dir` argument note.
        app.state.cache_dir = Path(cache_dir)
        # Built once, here, for the same reason as everything above it, and
        # resolved here rather than at import time so a bad
        # config.API_CLASSIFIER_ADAPTER fails at startup, not mid-request.
        factory = classifier_factory or resolve_classifier_factory()
        app.state.classifier = build_classifier(context, factory)
        logger.info(
            "Live simulation adapter: %s (reconciliation %s)",
            app.state.classifier.name,
            config.SIM_CLI_RECONCILE_MODE,
        )
        # In-memory store for uploaded draws (never persisted). A fresh one per
        # app instance is what makes uploads ephemeral by construction: a restart
        # is a new process, a new store, and every prior upload is forgotten.
        # `is not None`, not `or`: an empty store is falsy (its __len__ is 0), so
        # `upload_store or UploadStore()` would silently discard an injected
        # empty store.
        app.state.upload_store = (
            upload_store if upload_store is not None else UploadStore()
        )
        logger.info(
            "Upload store: capacity %d draw(s), in memory only (cleared on restart)",
            app.state.upload_store.max_uploads,
        )
        yield
        app.state.context = None
        app.state.registry = None
        app.state.cache_dir = None
        app.state.classifier = None
        app.state.upload_store = None

    app = FastAPI(
        title=config.API_TITLE,
        version=config.API_VERSION,
        lifespan=lifespan,
    )

    # Per-IP rate limiter for /storybook only (attached as that route's
    # dependency below). Built per app instance, a fresh window store for every
    # test app. The honest limits of the in-memory store are documented on
    # config.API_STORYBOOK_RATE_LIMIT.
    storybook_limiter = SlidingWindowRateLimiter(config.API_STORYBOOK_RATE_LIMIT)

    def storybook_rate_limit(request: Request) -> None:
        """Dependency: 429 when a client exceeds the /storybook window.

        Kept a dependency (not a param on the handler) so the compute-heavy route
        is guarded before its body runs without altering its signature.
        """
        retry_after = storybook_limiter.check(_client_key(request))
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail={
                    "reason": "rate_limited",
                    "message": (
                        "Too many storybook simulations from this client. This "
                        "endpoint runs a live bracket per request and is rate "
                        f"limited to {config.API_STORYBOOK_RATE_LIMIT} per IP. "
                        "Retry after the window, or precompute via /simulate."
                    ),
                },
                headers={"Retry-After": str(retry_after)},
            )

    # Per-IP limiter for POST /tournaments/upload, same in-memory limiter and
    # same honest scope as /storybook's. Uploads validate and name-resolve up to
    # 128 entrants, so they get a tighter window than a GET.
    upload_limiter = SlidingWindowRateLimiter(config.API_UPLOAD_RATE_LIMIT)

    def upload_rate_limit(request: Request) -> None:
        """Dependency: 429 when a client exceeds the /upload window."""
        retry_after = upload_limiter.check(_client_key(request))
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail={
                    "reason": "rate_limited",
                    "message": (
                        "Too many uploads from this client. Uploading is rate "
                        f"limited to {config.API_UPLOAD_RATE_LIMIT} per IP. Retry "
                        "after the window."
                    ),
                },
                headers={"Retry-After": str(retry_after)},
            )

    # Per-IP limiter for POST /match/simulate. Same in-memory limiter and same
    # honest scope as /storybook's, this is the second endpoint that runs real
    # per-request compute (a live single-match Monte Carlo). More generous than
    # /storybook because a single match is ~5x cheaper and is the featured path a
    # user re-runs on every tweak. See config.API_MATCH_RATE_LIMIT.
    match_limiter = SlidingWindowRateLimiter(config.API_MATCH_RATE_LIMIT)

    def match_rate_limit(request: Request) -> None:
        """Dependency: 429 when a client exceeds the /match/simulate window."""
        retry_after = match_limiter.check(_client_key(request))
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail={
                    "reason": "rate_limited",
                    "message": (
                        "Too many match simulations from this client. This "
                        "endpoint runs a live Monte Carlo per request and is rate "
                        f"limited to {config.API_MATCH_RATE_LIMIT} per IP. Retry "
                        "after the window."
                    ),
                },
                headers={"Retry-After": str(retry_after)},
            )

    # Shared concurrency cap across BOTH live-compute routes (/storybook and
    # /match/simulate). One gate per app instance so the peak simultaneous load,
    # summed across every IP and every live endpoint, is a small constant. See
    # config.API_LIVE_COMPUTE_MAX_CONCURRENCY for why the per-IP rate limits above
    # do not cover this. Excess is shed with 503, not queued.
    live_compute_gate = LiveComputeGate(config.API_LIVE_COMPUTE_MAX_CONCURRENCY)

    def live_compute_slot(request: Request):
        """Dependency: hold one live-compute slot for the request, or 503.

        A generator dependency so FastAPI runs the teardown (slot release) after
        the response is sent, whether the handler returned or raised. Acquisition
        is non-blocking: over the cap, the request is rejected immediately with a
        503 + Retry-After rather than parked on a threadpool thread.
        """
        if not live_compute_gate.acquire():
            raise HTTPException(
                status_code=503,
                detail={
                    "reason": "server_busy",
                    "message": (
                        "The server is running the maximum number of live "
                        "simulations at once "
                        f"({config.API_LIVE_COMPUTE_MAX_CONCURRENCY}). This is a "
                        "small demo instance; retry in a moment."
                    ),
                },
                headers={"Retry-After": "2"},
            )
        try:
            yield
        finally:
            live_compute_gate.release()

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Attach the standard security headers to every response.

        The docs UI (/docs, /redoc) gets a looser CSP so Swagger/ReDoc can pull
        their bundle from the CDN and run their inline bootstrap; everything else
        every JSON body, gets the strict ``default-src 'none'`` policy.
        """
        response = await call_next(request)
        # Read the raw ASGI routed path, NOT request.url.path. request.url is
        # reconstructed from the Host header and is spoofable
        # (PYSEC-2026-161/248: a crafted Host or leading-slash-less path can make
        # request.url.path differ from the path actually routed), which for a
        # path-keyed security decision like "docs get a looser CSP" is exactly
        # the sink those advisories warn about. scope["path"] is the path the
        # router dispatched on and cannot be moved by a header.
        path = request.scope.get("path", "")
        csp = DOCS_CSP if path in DOCS_PATHS else API_CSP
        _apply_security_headers(response, csp=csp)
        return response

    # Explicit allow-list from config, never "*". See config.API_ALLOWED_ORIGINS.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.API_ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Catch-all so an unhandled exception (which bypasses the header middleware
    # above, ServerErrorMiddleware sits outside it) still returns the security
    # headers and a generic, non-leaking body. See unexpected_error_handler.
    app.add_exception_handler(Exception, unexpected_error_handler)

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(context: ApiContext = Depends(get_context)) -> HealthResponse:
        """Liveness + the provenance of the loaded dataset.

        ``data_through_year`` is the **observed** maximum season in the loaded
        frame, not ``config.END_YEAR``: the configured range is an intent, and
        the two diverge whenever a season fails to vendor or has not started. A
        client deciding whether a probability is current needs what is actually
        there.
        """
        return HealthResponse(
            status="ok",
            data_through_year=context.data_through_year,
            n_players=context.n_players,
        )

    @app.get("/players", response_model=PlayerSearchResponse, tags=["players"])
    def search_players(
        query: Annotated[
            PlayerQuery,
            Query(
                description="Player name or fragment, e.g. 'alcaraz' or "
                "'C Alcaraz'. Whitespace-only is rejected, not treated as a "
                "match-everything wildcard.",
            ),
        ],
        context: ApiContext = Depends(get_context),
    ) -> PlayerSearchResponse:
        """Fuzzy-search the skill table; return every match with its skills.

        Matching is the shared resolver (``common.names.resolve_name``),
        exact → initials → substring → fuzzy, not a second search
        implementation.

        **Ambiguity is a result, not an error** (a deliberate divergence from the
        CLI). ``cli/interactive`` resolves to exactly one player because it must
        then predict a specific matchup, so it treats several candidates as a
        failure and prints a "did you mean" list. This endpoint is a *search*:
        the conventional HTTP shape for one is a list, possibly of length 0 or n,
        with the caller choosing. So all three resolver outcomes collapse to one
        200 response:

          * unique match   → one entry
          * ambiguous      → **every** candidate, in resolver order
          * nothing at all → ``players: []``, ``count: 0``, ``strategy: null``

        No 404: a search that legitimately found nothing succeeded. ``strategy``
        is echoed so a client can weight a ``fuzzy`` hit differently from an
        ``exact`` one. A missing, empty or whitespace-only ``query`` is the one
        failure, a 422 from validation (see :data:`PlayerQuery`), since it is a
        malformed request rather than a search.

        **Not paginated**, deliberately: the whole result set is returned however
        large. This is a documented call for the current scale, not an oversight
        the worst realistic query (``?query=a``) is 1,220 of 1,408 players and
        ~407 KB, assembled in ~12 ms, so the server-side cost is negligible. The
        cost that *does* matter is bandwidth to a browser type-ahead, which is
        the frontend's to manage (debounce, or ask for a limit here then).
        """
        index = context.skill_table.name_index
        match = resolve_name(query, index)

        if match is None:
            return PlayerSearchResponse(query=query, count=0, strategy=None, players=[])

        names = match.candidates if match.is_ambiguous else [match.name]
        players = [
            _player_summary(name, index, context.skill_table) for name in names
        ]
        return PlayerSearchResponse(
            query=query,
            count=len(players),
            strategy=match.strategy.value,
            players=players,
        )

    @app.get(
        "/tournaments", response_model=TournamentListResponse, tags=["tournaments"]
    )
    def list_tournaments(
        registry: TournamentRegistry = Depends(get_registry),
    ) -> TournamentListResponse:
        """Every draw file in the draws directory, split by whether it loaded.

        **A malformed file is listed, not skipped, and never 500s the listing.**
        Draw files are hand-entered JSON, so a typo is the expected failure mode,
        and the three plausible responses to one are not equal: propagating it
        would take the whole catalogue down over an unrelated file; silently
        skipping it would make the event vanish with the reason visible only in a
        server log, which is exactly when an operator goes looking for it. So
        each file is loaded independently and a failure becomes a row in
        ``invalid`` carrying the draw validator's full problem list. ``count``
        and ``tournaments`` cover the **valid** draws only, so a client rendering
        an event picker can ignore ``invalid`` entirely.

        Note that "valid" here means *loadable*, not *simulatable*: a draw
        containing ``Qualifier`` placeholders loads fine and is listed. See the
        module docstring.
        """
        return TournamentListResponse(
            count=len(registry.valid),
            tournaments=[
                TournamentSummary(
                    tournament_id=entry.draw.tournament_id,
                    name=entry.draw.name,
                    surface=entry.draw.surface,
                    best_of=entry.draw.best_of,
                    draw_size=entry.draw.draw_size,
                )
                for entry in registry.valid
            ],
            invalid=[
                InvalidDraw(
                    tournament_id=entry.tournament_id,
                    source=entry.source,
                    problems=list(entry.problems),
                )
                for entry in registry.invalid
            ],
        )

    @app.get("/rankings", response_model=RankingsResponse, tags=["rankings"])
    def rankings(cache_dir: Path = Depends(get_cache_dir)) -> RankingsResponse:
        """Precomputed Elo leaderboards: overall plus one per surface.

        **This handler never computes anything.** Like ``/simulate``, it reads a
        JSON file produced offline (``scripts/precompute_elo.py``) on the weekly
        refresh cadence, so ratings regenerate with the rest of the data rather
        than on the request path.

        The ratings are a **display feature only**, Ace's own Elo, not the
        official ATP ranking, and deliberately walled off from the model
        (``features/elo.py``). Each player carries an ``is_active`` flag so a
        client can separate current form from a long-retired player's frozen
        peak; the ordering itself is by rating so the stale entries are still
        visible, not silently dropped.

        Failures:

          * **No cache file → 425** (``reason: rankings_missing``) with the exact
            command that produces one, the same shape ``/simulate`` uses for a
            missing Monte Carlo cache, and for the same reason: rating is never
            run inside a request.
          * **An unreadable/invalid cache file → 422** (``reason:
            rankings_unreadable``), the server's artefact is bad, so a 500 would
            wrongly imply a transient fault; regenerate it.
        """
        path = elo_cache_path(cache_dir)
        if not path.is_file():
            raise HTTPException(
                status_code=425,
                detail={
                    "reason": "rankings_missing",
                    "message": (
                        "No precomputed Elo rankings. They are generated offline "
                        "on the refresh cadence, never inside a request, so run "
                        "the precompute and retry."
                    ),
                    "command": "python scripts/precompute_elo.py",
                },
            )
        try:
            return RankingsResponse.model_validate_json(path.read_text(encoding="utf-8"))
        except (ValidationError, ValueError, OSError) as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "reason": "rankings_unreadable",
                    "message": (
                        f"The cached Elo rankings could not be read as a "
                        f"{RankingsResponse.__name__}: {exc}. Re-run "
                        f"scripts/precompute_elo.py to regenerate them."
                    ),
                },
            ) from exc

    @app.post(
        "/tournaments/upload",
        response_model=UploadResponse,
        status_code=201,
        tags=["tournaments"],
        dependencies=[Depends(upload_rate_limit)],
    )
    async def tournament_upload(
        request: Request,
        context: ApiContext = Depends(get_context),
        store: UploadStore = Depends(get_upload_store),
    ) -> UploadResponse:
        """Accept a draw JSON, validate it, and hold it in memory for simulation.

        **Ephemeral by design.** The draw is stored in this process's memory only
        (:mod:`api.uploads`), never on disk, because the deploy target has no
        persistent disk, so it does not survive a restart, a cold-start wake or
        a redeploy, and the response says as much (``ephemeral: true``,
        ``content_note``). It is assigned a generated ``upload-…`` id in a
        namespace separate from the curated draws, so it can never collide with
        or shadow the git-committed, verified example draws.

        **Validation is the draw schema's, not a second parser.** The body goes straight to
        :func:`sim.draw.parse_draw`, so an uploaded draw is held to exactly the
        rules a curated one is, and a malformed one comes back as the same
        accumulated problem list ``/bracket`` and ``/simulate`` already return,
        every problem at once, so a hand-entered 128-draw is fixable in one pass.

        Guards, in order:

          * **Non-JSON content type → 415.** Only ``application/json`` is
            accepted; a form post or a binary blob is refused before anything is
            read.
          * **Body over ``config.API_UPLOAD_MAX_BYTES`` → 413.** Enforced while
            streaming, so an oversized upload is rejected without buffering the
            whole thing, a 128-slot draw is tens of KB, the cap is 1 MiB.
          * **Unparseable JSON → 422** (``reason: invalid_json``).
          * **A bad ``event_date`` → 422** (``reason: invalid_event_date``).
          * **A draw that fails validation → 422** (``reason: draw_invalid``)
            with the validator's full ``problems`` list.

        On success the draw is stored (evicting the oldest if the store is at
        capacity) and a 201 acknowledges it with the id to simulate it by. Note
        the aggregate Monte Carlo board (``/simulate``) is **not** available for
        uploads, see :func:`tournament_simulation`; the live storybook is.
        """
        # 1. Content type: refuse anything that is not JSON up front.
        media_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if media_type != "application/json":
            raise HTTPException(
                status_code=415,
                detail={
                    "reason": "unsupported_media_type",
                    "message": (
                        "Upload a draw as application/json. Got "
                        f"{media_type or 'no'} content type."
                    ),
                },
            )

        # 2. Size cap, enforced while streaming so an oversized body is rejected
        #    without ever being fully buffered (Content-Length can lie or be
        #    absent, so the stream length is what is trusted).
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > config.API_UPLOAD_MAX_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail={
                        "reason": "payload_too_large",
                        "message": (
                            "Uploaded draw exceeds the "
                            f"{config.API_UPLOAD_MAX_BYTES}-byte limit. A draw is "
                            "tens of KB; this is far larger."
                        ),
                    },
                )

        # 3. Parse JSON. RecursionError is caught alongside the ordinary decode
        #    errors: deeply-nested JSON (e.g. thousands of open brackets, well
        #    under the size cap) overflows the decoder's recursion and would
        #    otherwise escape this handler as an uncaught 500 with a full stack
        #    trace logged on every request, a cheap error/log-spam vector on a
        #    public endpoint. It is malformed input like any other, so it earns
        #    the same 422.
        try:
            data = json.loads(bytes(raw))
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "reason": "invalid_json",
                    "message": f"Request body is not valid JSON: {exc}.",
                },
            ) from exc

        # 4. Optional event_date → is_forecast (422 on a malformed value).
        event_date, is_forecast = _extract_event_date(data)

        # 5. Validate the draw itself, the draw schema's rules and problem list.
        try:
            draw = parse_draw(data, context.skill_table, source="<uploaded draw>")
        except DrawValidationError as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "reason": "draw_invalid",
                    "message": (
                        f"Uploaded draw failed validation with "
                        f"{len(exc.problems)} problem(s)."
                    ),
                    "source": exc.source,
                    "problems": list(exc.problems),
                },
            ) from exc

        # 6. Store (memory only) and acknowledge.
        record = store.add(draw, is_forecast=is_forecast, event_date=event_date)
        placeholder_count = sum(1 for slot in draw.bracket if slot.is_placeholder)
        logger.info(
            "Stored uploaded draw %s (%d slots, forecast=%s); store now holds %d",
            record.upload_id,
            draw.draw_size,
            is_forecast,
            len(store),
        )
        return UploadResponse(
            tournament_id=record.upload_id,
            name=draw.name,
            surface=draw.surface,
            best_of=draw.best_of,
            final_set_tiebreak=draw.final_set_tiebreak,
            draw_size=draw.draw_size,
            is_simulatable=placeholder_count == 0,
            placeholder_count=placeholder_count,
            is_forecast=is_forecast,
            ephemeral=True,
            content_note=CONTENT_NOTE_UPLOAD,
        )

    @app.get(
        "/tournaments/{tournament_id}/bracket",
        response_model=BracketResponse,
        tags=["tournaments"],
    )
    def tournament_bracket(
        tournament_id: Annotated[
            str,
            PathParam(description="Id from ``GET /tournaments`` or ``POST /tournaments/upload``."),
        ],
        registry: TournamentRegistry = Depends(get_registry),
        store: UploadStore = Depends(get_upload_store),
    ) -> BracketResponse:
        """The resolved draw: positions, entrants, ids and seeds.

        The draw comes from :func:`sim.draw.load_draw` via the startup
        registry, this handler re-parses nothing and re-validates nothing, so
        there is one draw parser in the codebase and one set of rules. An
        **uploaded** draw (an ``upload-…`` id) is served the same way from the
        in-memory store, so a client renders a user draw's bracket with no
        special case.

        Failures:

          * **Unknown upload id → 404** (``reason: upload_not_found``), the id
            is in the upload namespace but no longer held (evicted, or the server
            restarted). The message says so.
          * **Unknown registered id → 404**, naming the ids that *are* registered
            (a draws directory holds a handful of files, so listing them beats a
            bare "not found", the same courtesy ``cli/simulate_tournament``
            extends).
          * **A file that failed validation → 422**, with the
            :class:`~sim.draw.DrawValidationError` ``problems`` list verbatim
            rather than a generic message, so a hand-entered draw is fixable in
            one pass. 422 is the ticket's choice and is a deliberate stretch of
            the status code: the unprocessable entity is the server's draw file,
            not the client's request, but nothing about the request can fix it
            and a 500 would suggest a transient fault rather than a bad file.
        """
        if store.is_upload_id(tournament_id):
            uploaded = store.get(tournament_id)
            if uploaded is None:
                raise _unknown_upload(tournament_id)
            return _bracket_response(uploaded.draw, tournament_id=uploaded.upload_id)

        entry = registry.get(tournament_id)
        if entry is None:
            known = ", ".join(repr(known_id) for known_id in registry.ids())
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No tournament with id {tournament_id!r}. "
                    + (f"Known ids: {known}." if known else "No draws are registered.")
                ),
            )
        if entry.draw is None:
            raise _invalid_draw_file(entry)
        return _bracket_response(entry.draw)

    @app.get(
        "/tournaments/{tournament_id}/simulate",
        response_model=SimulationResponse,
        tags=["tournaments"],
    )
    def tournament_simulation(
        tournament_id: Annotated[
            str,
            PathParam(description="Id from ``GET /tournaments``."),
        ],
        top: Annotated[
            int | None,
            Query(
                ge=1,
                description="Return only the ``top`` most likely champions. "
                "Omit for the whole field.",
            ),
        ] = None,
        registry: TournamentRegistry = Depends(get_registry),
        cache_dir: Path = Depends(get_cache_dir),
        store: UploadStore = Depends(get_upload_store),
    ) -> SimulationResponse:
        """Precomputed Monte Carlo title and round-survival probabilities.

        **This handler never simulates anything.** It reads the JSON file
        ``scripts/precompute_sim.py`` wrote offline, validates it and returns it.
        Phase 3's global rules forbid running a 5,000-run Monte Carlo inside a
        request handler, a 128-draw job is ~40 s, so the endpoint is a cache
        reader by design, not as an optimisation. The proof is structural: this
        module imports no simulation entry point at all
        (``tests/test_api_simulate.py`` pins it with raising traps on
        ``monte_carlo`` and ``simulate_bracket``).

        **Read ``metadata`` before rendering the numbers.** Every response
        carries the reconciliation ``mode``/``w``, an ``is_forecast`` flag and a
        plain-language ``classifier_limitation``. The classifier's feature row is
        fully real (``ace-04-current-state.md §7`` seam 7, closed), so what the
        disclosure states is narrow: the model is an as-of-now snapshot, which
        makes a simulation of an already-played draw retrospective rather than a
        forecast. They remain required schema fields, so a cache file cannot
        publish bare probabilities.

        Four failure modes, each distinguishable without parsing prose:

          * **Unknown id → 404**, listing simulatable ids separately from draw
            files that failed validation (see :func:`_unknown_tournament`).
          * **A draw file that failed validation → 422** with the validator's
            problem list, the same body ``/bracket`` returns for the same file.
          * **A draw containing placeholder entrants → 409**
            (``reason: "draw_not_simulatable"``). The registry keeps listing such
            draws and ``/bracket`` keeps serving them, and simulability is
            answered *here*, at the endpoint that needs it, rather than by a
            ``simulatable`` flag on ``/tournaments`` or by filtering the
            catalogue. Reasons: a flag would be a second mechanism for a
            distinction the error already makes, and would have to be kept in
            sync with cache presence to mean anything useful to a client;
            filtering would hide a real event by silently skipping it. A 409 also
            matches the standard error set, an informative structured error, not
            a crash, and mirrors, over HTTP, the refusal ``simulate_bracket``
            already makes offline, naming every offending slot rather than
            inventing a probability for it.
          * **No cache file → 425** (``reason: "cache_missing"``) with the exact
            command that produces one. Deliberately *not* the same code as the
            placeholder case: 409 means "this draw can never be simulated as
            written, edit the file", 425 means "not computed yet, run the
            script". Both return immediately; neither blocks on a Monte Carlo
            run, and no background job is kicked off, because a request that
            silently triggers a 40 s job is the thing the Phase 3 rule exists to
            prevent.

        The first three come from :func:`_simulatable_entry`, shared with
        ``/storybook`` so the two simulation endpoints cannot disagree about
        which draws are runnable. It runs **before** the cache lookup, so the
        answer for a given draw is a property of the draw rather than of whatever
        happens to be on disk.

        A cache file that fails to parse, or that describes a different draw than
        the registry holds (a stale file left after the draw was edited), is also
        a 422, same reasoning as an invalid draw file: it is the server's
        artefact, a 500 would imply a transient fault, and the fix is named in
        the message.

        **Uploaded draws are not served here → 409.** The Monte Carlo board is
        cache-only by architectural rule (never run 5,000 runs in a request), and
        an uploaded draw has no cache: there is no persistent disk to precompute
        one onto, so there is no honest way to produce this board for it. Uploaded
        draws support the live storybook instead (``/storybook``), and the error
        says so rather than 404-ing with a list of curated ids the caller cannot
        use.
        """
        if store.is_upload_id(tournament_id):
            raise HTTPException(
                status_code=409,
                detail={
                    "reason": "monte_carlo_unavailable_for_upload",
                    "message": (
                        "Uploaded draws do not support the precomputed Monte "
                        "Carlo board: it is cache-only and there is no persistent "
                        "disk to precompute an upload onto. Use "
                        f"/tournaments/{tournament_id}/storybook for a live run "
                        "instead."
                    ),
                    "tournament_id": tournament_id,
                },
            )

        entry = _simulatable_entry(tournament_id, registry)

        path = cache_path_for(tournament_id, cache_dir)
        if path is None or not path.is_file():
            raise HTTPException(
                status_code=425,
                detail={
                    "reason": "cache_missing",
                    "message": (
                        f"No precomputed simulation for {tournament_id!r}. Monte "
                        f"Carlo is never run inside a request, so generate the "
                        f"cache offline and retry."
                    ),
                    "tournament_id": tournament_id,
                    "command": (
                        f"python scripts/precompute_sim.py --draw {tournament_id} "
                        f"--runs {config.MC_RUNS} --seed {config.MC_SEED}"
                    ),
                },
            )

        try:
            cached = SimulationResponse.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (ValidationError, ValueError, OSError) as exc:
            raise HTTPException(
                status_code=422,
                detail={
                    "reason": "cache_unreadable",
                    "message": (
                        f"The cached simulation for {tournament_id!r} could not "
                        f"be read as a {SimulationResponse.__name__}: {exc}. "
                        f"Re-run the precompute script to regenerate it."
                    ),
                    "tournament_id": tournament_id,
                },
            ) from exc

        if (
            cached.tournament_id != entry.draw.tournament_id
            or cached.draw_size != entry.draw.draw_size
        ):
            raise HTTPException(
                status_code=422,
                detail={
                    "reason": "cache_stale",
                    "message": (
                        f"The cached simulation for {tournament_id!r} describes "
                        f"{cached.tournament_id!r} with a {cached.draw_size}-slot "
                        f"draw, but the registered draw is "
                        f"{entry.draw.tournament_id!r} with {entry.draw.draw_size} "
                        f"slots. Re-run the precompute script."
                    ),
                    "tournament_id": tournament_id,
                },
            )

        if top is not None and top < cached.count:
            # players are already sorted by title probability, descending.
            return cached.model_copy(
                update={"players": cached.players[:top], "count": top}
            )
        return cached

    @app.get(
        "/tournaments/{tournament_id}/storybook",
        response_model=StorybookResponse,
        tags=["tournaments"],
        dependencies=[Depends(storybook_rate_limit), Depends(live_compute_slot)],
    )
    def tournament_storybook(
        tournament_id: Annotated[
            str,
            PathParam(description="Id from ``GET /tournaments``."),
        ],
        seed: Annotated[
            int | None,
            Query(
                ge=0,
                le=config.API_STORYBOOK_SEED_MAX,
                description="RNG seed. The same seed always replays the same "
                f"tournament. Omit for the server default "
                f"({config.API_STORYBOOK_SEED}); the seed used is echoed in "
                f"``metadata.seed`` either way.",
            ),
        ] = None,
        registry: TournamentRegistry = Depends(get_registry),
        context: ApiContext = Depends(get_context),
        classifier: ClassifierAdapter = Depends(get_classifier),
        store: UploadStore = Depends(get_upload_store),
    ) -> StorybookResponse:
        """One tournament played out point by point, for one seed, live.

        **This one does simulate, and that is within the Phase 3 rule, not an
        exception to it.** The rule forbids a full Monte Carlo (thousands of
        brackets) in a request handler; a storybook is *one* bracket, 127
        matches for a Slam, which the ticket budgets at "well under a couple
        seconds". Measured on the shipped 128 draw ``usopen_2024_atp_full``
        (2026-08-03, dev machine):
        **1.54 s for the first request** on a cold server, **0.80 s
        for a subsequent request with a fresh seed**, and **0.45 s to replay a
        seed already served**. The spread is almost entirely the adapter's
        ``predict_proba``: 127 matches need 127 distinct matchups priced, and the
        memo table lives on the startup-built adapter, so it is shared across
        requests and warms as the server runs; the cold request additionally pays
        the one-off 0.08 s rolling-form table build. 0.45 s is the floor, the
        point-by-point simulation of 127 matches itself.
        ``storybook_run`` is called **exactly once** per request; the rounds,
        scorelines and champion in the response are read off the single
        :class:`~sim.tournament.StorybookResult` it returns.

        **Determinism is the contract, not a nicety.** Same id + same seed → a
        byte-identical body: the response carries no timestamp and every field is
        a function of the draw, the seed and the startup state. That is what makes
        the URL shareable, and it is why an omitted ``?seed=`` falls back to a
        *fixed* default (:data:`config.API_STORYBOOK_SEED`) rather than a
        time-derived one, a bare ``/storybook`` that returned a different story
        on refresh would be unreproducible and uncacheable. ``metadata.seed``
        always echoes what ran, so a client can turn any response into an explicit,
        shareable URL.

        **Reconciliation mode is ``config.SIM_CLI_RECONCILE_MODE`` (``blend``),
        not ``config.RECONCILE_MODE``**, the same value the cached ``/simulate``
        board was produced under, so the two endpoints publish one model.
        Blending lets the point model contribute half the match-win probability
        (see :data:`config.SIM_CLI_RECONCILE_MODE`).
        ``metadata`` carries ``mode``/``w``, ``is_forecast`` and
        ``classifier_limitation``, the identical
        :class:`~api.schemas.ModelDisclosure` block ``/simulate`` serves,
        inherited rather than restated.

        Failures are ``/simulate``'s, from the same
        :func:`_simulatable_entry` gate: unknown id → 404, unloadable draw file →
        422, placeholder entrants → 409. There is no 425 here, nothing is
        cached, so there is nothing to be missing.

        **Uploaded draws run here too.** An ``upload-…`` id is served from the
        in-memory store with the same live run and the same placeholder refusal;
        the only differences are in ``metadata``, an uploaded draw discloses
        that it is user-submitted and unverified (``content_source``,
        ``content_note``), and a future-dated one reads ``is_forecast: true``,
        the one case in this project where that is honest. An upload id that is
        no longer held (evicted, or the server restarted) is a 404
        (``reason: upload_not_found``).

        **Rate limited** (``config.API_STORYBOOK_RATE_LIMIT``, per IP): this is
        the only endpoint that runs real per-request compute, so a client over
        the limit gets a 429. The limiter's store is in-memory and resets on
        restart, see the config constant for the honest scope of that.
        """
        if seed is None:
            seed = config.API_STORYBOOK_SEED

        if store.is_upload_id(tournament_id):
            uploaded = store.get(tournament_id)
            if uploaded is None:
                raise _unknown_upload(tournament_id)
            if any(slot.is_placeholder for slot in uploaded.draw.bracket):
                raise _upload_not_simulatable(uploaded)
            result = storybook_run(
                uploaded.draw,
                context.skill_table,
                classifier.call,
                seed=seed,
                mode=config.SIM_CLI_RECONCILE_MODE,
                w=config.RECONCILE_BLEND_WEIGHT,
            )
            return _storybook_response(
                result,
                disclosure=_upload_disclosure(uploaded),
                context=context,
                adapter=classifier.name,
                tournament_id=uploaded.upload_id,
            )

        entry = _simulatable_entry(tournament_id, registry)
        result = storybook_run(
            entry.draw,
            context.skill_table,
            classifier.call,
            seed=seed,
            # See the docstring: the same mode the cached board was produced under.
            mode=config.SIM_CLI_RECONCILE_MODE,
            w=config.RECONCILE_BLEND_WEIGHT,
        )
        return _storybook_response(
            result,
            disclosure=_curated_disclosure(entry),
            context=context,
            adapter=classifier.name,
        )

    @app.post(
        "/match/simulate",
        response_model=MatchSimulateResponse,
        tags=["match"],
        dependencies=[Depends(match_rate_limit), Depends(live_compute_slot)],
    )
    def match_simulate(
        body: MatchSimulateRequest,
        context: ApiContext = Depends(get_context),
        classifier: ClassifierAdapter = Depends(get_classifier),
    ) -> MatchSimulateResponse:
        """Simulate one matchup live and report the betting-relevant outputs.

        **This one runs a live Monte Carlo, and that is a deliberate, bounded
        exception to the cache-only rule, not a violation of it.** The rule
        forbids a live *aggregate* because a 128-draw board is 5,000 brackets x
        127 matches. A single match is a different cost class: there is one matchup,
        so the classifier is priced once and the serve shift solved once, and
        each of ``config.API_MATCH_SIM_RUNS`` runs is only a point-by-point
        scoreline draw. Measured ~0.24 s for 3,000 runs, cheaper than one live
        ``/storybook`` bracket. The reasoning is recorded in
        ``ace-04-current-state.md``'s seam tracking so "why does a single match
        run live when ``/simulate`` does not" has a durable answer.

        **What it returns**, all from one seeded pass over the matchup
        (``sim.single_match``): the match win probability for each player, the
        distribution over final set scores (e.g. how often 3-0 / 3-1 / 3-2 in a
        best-of-5), and a total-games (over/under) summary. Hold/break rates are
        deliberately **not** here, a simulated set records only game counts, not
        which games were holds versus breaks, so surfacing them would need new
        instrumentation in ``sim/match.py``; that is a follow-up, not this
        endpoint's scope.

        **Determinism is the contract.** Same players, surface, format and seed
        give a byte-identical body, so a shared single-match link is
        reproducible. ``metadata.seed`` echoes whatever ran (the caller's, or the
        server default), so any response can be turned into an explicit URL.

        **Players must be known to the model.** Each name is resolved through the
        skill table's name index (the resolver ``/players`` uses). An unknown or
        ambiguous name is a 422 (``player_not_found`` / ``player_ambiguous``),
        never a silent default, the model refuses to score a player it has no
        history for, which is why the UI restricts selection to players it knows.

        **Reconciliation mode is ``config.SIM_CLI_RECONCILE_MODE`` (``blend``)**,
        the same value ``/simulate`` and ``/storybook`` publish under, so all
        three endpoints report one model. ``metadata`` carries the shared
        :class:`~api.schemas.ModelDisclosure` block plus this run's ``runs`` and
        ``seed``; ``is_forecast`` is ``false`` and the limitation text is the
        single-match wording (this is a model estimate of a hypothetical matchup,
        not a betting recommendation).

        **Rate limited** (``config.API_MATCH_RATE_LIMIT``, per IP): a client over
        the window gets a 429. The limiter's store is in-memory and resets on
        restart, see the config constant for the honest scope of that.
        """
        seed = body.seed if body.seed is not None else config.MC_SEED
        if seed > config.API_MATCH_SEED_MAX:
            raise HTTPException(
                status_code=422,
                detail={
                    "reason": "seed_out_of_range",
                    "message": (
                        f"seed must be between 0 and {config.API_MATCH_SEED_MAX}."
                    ),
                },
            )
        final_set_rule = body.final_set_rule or config.SIM_CLI_FINAL_SET_RULE

        index = context.skill_table.name_index
        player_a = _resolve_match_player(body.player_a, index)
        player_b = _resolve_match_player(body.player_b, index)

        try:
            result = simulate_single_match(
                player_a,
                player_b,
                body.surface,
                context.skill_table,
                classifier.call,
                best_of=body.best_of,
                final_set_rule=final_set_rule,
                n_runs=config.API_MATCH_SIM_RUNS,
                seed=seed,
                mode=config.SIM_CLI_RECONCILE_MODE,
                w=config.RECONCILE_BLEND_WEIGHT,
            )
        except ValueError as exc:
            # Reaches here only if a name resolved in the skill table but has no
            # classifier-visible history (skill table and match frame disagreeing)
            # a rare wiring case, surfaced honestly rather than defaulted.
            raise HTTPException(
                status_code=422,
                detail={"reason": "player_not_simulatable", "message": str(exc)},
            ) from exc

        return _match_response(result, context, classifier.name)

    return app


# Module-level app for `uvicorn api.main:app`. Construction is cheap; the
# pipeline load waits for startup.
app = create_app()
