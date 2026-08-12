"""FastAPI application: ``/health``, ``/players`` (T3.1), ``/tournaments`` (T3.2–T3.4).

Run it with::

    uvicorn api.main:app --reload    # from src/, or with src/ on PYTHONPATH

The app is built by :func:`create_app` so the startup context can be injected.
Importing this module is cheap and side-effect-free — the pipeline load (~1.8 s;
see :func:`api.deps.build_api_context`) happens in the **lifespan**, i.e. when
the app actually starts, so a test may import ``api.main`` (or generate its
OpenAPI schema) without touching the vendored dataset.

Conventions this ticket sets for T3.2–T3.4:
  * Startup state is loaded once into ``app.state`` by the lifespan handler and
    read by every handler through a dependency (:func:`get_context`,
    :func:`get_registry`). Handlers never load anything themselves.
  * All responses are ``api.schemas`` models (``response_model=`` on every
    route); no handler returns a raw dict.
  * Anything configurable (CORS origins, titles, the draws directory) is a
    ``config`` constant, overridable per app instance for tests.

**T3.2 note, answered by T3.3.** ``/tournaments/{id}/bracket`` calls
``load_draw`` and nothing else, so T2.2's placeholder-entrant refusal — which
lives in ``simulate_bracket`` — never fires there. A draw containing
``Qualifier`` slots loads fine (placeholders take the default skill profile, as
everywhere else), so ``data/draws/example_usopen_2026.json`` is legitimately
listed and its bracket legitimately served. T3.3 keeps that unchanged and answers
the simulability question at ``/simulate`` instead — see
:func:`tournament_simulation` for the decision and its reasoning.

**T3.3/T3.4: what this module may and may not simulate.** ``/simulate`` reads a
file that ``scripts/precompute_sim.py`` wrote offline and simulates nothing —
Phase 3's global rule ("never run a full 5,000-run MC synchronously inside a
request handler") is a rule, not a performance preference, so this module holds
no reference to ``monte_carlo`` or ``simulate_bracket`` at all. ``/storybook``
(T3.4) is the deliberate exception the same rule allows: **one** bracket, not
thousands, run live per request via ``sim.tournament.storybook_run``. The line
is the cost of a single run (0.4–1.4 s for a 128 draw; measured below), and
``tests/test_api_simulate.py`` pins it — the Monte Carlo entry points must stay
un-imported here even though the storybook one is now imported.
"""

from __future__ import annotations

import importlib
import logging
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Path as PathParam, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import StringConstraints, ValidationError

import config
from api.deps import ApiContext, ContextFactory, build_api_context, log_context
from api.registry import TournamentEntry, TournamentRegistry, build_registry, cache_path_for
from api.schemas import (
    CLASSIFIER_LIMITATION,
    CLASSIFIER_LIMITATION_DETAIL,
    IS_FORECAST,
    BracketResponse,
    BracketSlot,
    HealthResponse,
    InvalidDraw,
    PlayerSearchResponse,
    PlayerSummary,
    SimulationResponse,
    StorybookChampion,
    StorybookMatch,
    StorybookMetadata,
    StorybookPlayerRun,
    StorybookResponse,
    StorybookRound,
    SurfaceSkill,
    TournamentListResponse,
    TournamentSummary,
)
from common.names import NameIndex, resolve_name
from features.serve import SkillTable
from sim.draw import Draw
from sim.reconcile import ClassifierProb
from sim.tournament import StorybookResult, storybook_run

logger = logging.getLogger(__name__)

# Deterministic surface ordering in every response body (VALID_SURFACES is a set).
SURFACE_ORDER = sorted(config.VALID_SURFACES)

# The ``/players`` query type. ``strip_whitespace`` runs **before** ``min_length``
# in Pydantic v2, which is the whole point: a bare ``min_length=1`` accepts " "
# and "\t", and ``common.names.resolve_name`` then strips it to "" — at which
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
# actually is — a JSON API whose only HTML is the auto-generated docs UI:
#
#   * Content-Security-Policy: JSON responses render no markup, so the strict
#     `default-src 'none'` policy costs nothing and is pure defence-in-depth
#     (a browser coerced into treating a response as a document can load
#     nothing). `frame-ancestors 'none'` + `base-uri 'none'` round it out. The
#     Swagger/ReDoc HTML at /docs and /redoc is the one exception (see
#     DOCS_CSP): it deliberately pulls its bundle from the jsDelivr CDN and runs
#     an inline bootstrap, so the strict policy would blank it out.
#   * X-Content-Type-Options: nosniff — stop a browser MIME-sniffing a JSON body
#     into something executable; the one header that most matters for an API.
#   * X-Frame-Options: DENY (paired with frame-ancestors 'none' for old
#     browsers that predate CSP) — nothing here is meant to be iframed.
#   * Referrer-Policy: no-referrer — a JSON API and a query-string SPA have no
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
# ⚠️ HONEST SCOPE — repeated verbatim from config.API_STORYBOOK_RATE_LIMIT
# because it is the whole point: the window counts live in this process's memory
# only. Render's free tier cold-starts and restarts routinely and every restart
# wipes the counters; a multi-replica deployment gives each replica its own. So
# this is resource-exhaustion hygiene against one careless/looping client, NOT
# abuse prevention. It gates no authorization (there is none), so the spoofable
# client key below is acceptable.
_PERIOD_SECONDS = {"second": 1, "minute": 60, "hour": 3600, "day": 86400}


def _parse_rate(spec: str) -> tuple[int, int]:
    """``"12/minute"`` → ``(12, 60)`` — (max requests, window seconds)."""
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
    would lump every visitor into one bucket, so prefer the first hop of
    ``X-Forwarded-For`` when present and fall back to the socket peer for direct
    (local, un-proxied) requests.

    ⚠️ ``X-Forwarded-For`` is client-supplied and therefore spoofable: an
    attacker can rotate it to dodge the limit. Acceptable because this limit is
    hygiene, not a security control, and gates no authorization decision — a
    forged value costs nothing but the attacker's own evasion.
    """
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

    **Why an import by name instead of an import statement.** T3.4 needed an
    adapter to simulate live and the only one that existed was
    ``cli.simulate_match.make_classifier_prob``, which ``api/`` may not import;
    late-binding kept ``api/`` free of any ``cli/`` symbol while disclosing the
    real runtime dependency in every response. **T3.5 removed that dependency**:
    the default is now ``common.classifier_adapter:make_classifier_prob``, a
    UI-free module ``api/`` could import outright, so no ``cli/`` module is
    loaded by the running server at all (``tests/test_api_storybook.py`` proves
    it in a clean interpreter).

    The indirection is kept because its *other* purpose stands on its own:
    "which model does this server publish" is deployment configuration, and
    swapping the adapter — for a differently-calibrated one, or back to an older
    one for comparison — is one config line rather than a code change. Whatever
    runs is reported: every response names :attr:`ClassifierAdapter.name`,
    derived from the factory that actually ran.

    Args:
        spec: ``"<module>:<attribute>"``, e.g.
            ``"common.classifier_adapter:make_classifier_prob"``.

    Returns:
        The factory callable.

    Raises:
        ValueError: If ``spec`` is not ``module:attribute``.
        ImportError: If the module or attribute does not exist — raised at
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

    Derived, never hardcoded — see :class:`ClassifierAdapter`. For the shipped
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

    Cheap: the T3.5 adapter defers both its as-of-now rolling-form table (0.08 s
    over the full frame) and every ``predict_proba`` to first use, so this adds
    nothing to the ~2 s startup and 0.08 s to the first simulated request. Building it
    *here* rather than per request is what keeps ``/storybook`` to one bracket's
    worth of work — and, because the adapter's memo table is shared by every
    request, repeat matchups across seeds cost one ``predict_proba`` in total
    rather than one per request.
    """
    return ClassifierAdapter(
        call=factory(
            context.estimator,
            context.data,
            context.surface_history,
            context.h2h_history,
        ),
        name=adapter_name(factory),
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

    Same pattern (and same reason) as :func:`get_context` — read once from
    ``app.state``, never rebuilt per request.
    """
    return request.app.state.registry


def get_classifier(request: Request) -> ClassifierAdapter:
    """FastAPI dependency: the startup-built ``ClassifierProb`` adapter (T3.4).

    Same ``app.state`` pattern as :func:`get_context`. Built once — a per-request
    adapter would throw away the memo table that makes a live storybook viable.
    """
    return request.app.state.classifier


def get_cache_dir(request: Request) -> Path:
    """FastAPI dependency: where this app reads precomputed simulations (T3.3).

    Stored on ``app.state`` like the context and the registry, so a test app can
    point at a fixture cache directory without touching ``data/cache/``.
    """
    return request.app.state.cache_dir


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


def _bracket_response(draw: Draw) -> BracketResponse:
    """Assemble a resolved draw's wire model.

    Every slot is included — placeholders are flagged via ``is_placeholder``,
    never filtered out, because a bracket with holes in it is not a bracket.
    """
    return BracketResponse(
        tournament_id=draw.tournament_id,
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
    problem list. Simulation is different, and in two ways — an unloadable file
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
    """422 for a draw file that failed validation — T3.2's shape, reused."""
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
    """409 for a draw holding placeholder entrants — T3.3's shape, reused.

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

    The verdict itself is :attr:`~api.registry.TournamentEntry.is_simulatable` —
    T3.3's one definition of the rule, living in the registry next to the flag it
    reads — not a placeholder comprehension repeated per endpoint.

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


def _storybook_response(
    result: StorybookResult,
    *,
    entry: TournamentEntry,
    context: ApiContext,
    adapter: str,
) -> StorybookResponse:
    """Assemble the wire model from a finished :class:`StorybookResult`.

    **Presentation over already-built data**, exactly as
    ``scripts/precompute_sim.build_payload`` is for a Monte Carlo result: every
    field is read off T2.4's own structures — ``rounds``/``matches`` (including
    the ``scoreline`` and the rendered ``line`` they already carry),
    ``champion``/``champion_seed``, and the ``runs`` summaries — and nothing is
    re-derived from the underlying ``BracketResult`` a second time. The CLI's
    text renderer (``render_storybook``) is deliberately **not** called: it is a
    different presentation of the same structure, and this endpoint returns JSON.
    """
    return StorybookResponse(
        tournament_id=result.tournament_id,
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
            is_forecast=IS_FORECAST,
            classifier_limitation=CLASSIFIER_LIMITATION,
            classifier_limitation_detail=CLASSIFIER_LIMITATION_DETAIL,
            source=entry.source,
        ),
    )


def create_app(
    context_factory: ContextFactory = build_api_context,
    draws_dir: Path | str = config.DRAWS_DIR,
    cache_dir: Path | str = config.CACHE_DIR,
    classifier_factory: ClassifierFactory | None = None,
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
            ``/simulate`` (T3.3). Defaults to ``config.CACHE_DIR``; the same
            injection seam, for the same reason. It is **not** scanned at
            startup — a cache file appearing while the server runs is picked up
            on the next request, which is what makes the precompute script
            usable against a live server.
        classifier_factory: Builds the ``ClassifierProb`` adapter ``/storybook``
            simulates with (T3.4), called **exactly once** at startup with the
            context's estimator and histories. Defaults to ``None``, meaning
            "resolve :data:`config.API_CLASSIFIER_ADAPTER`" — see
            :func:`resolve_classifier_factory` for why the default is a name
            rather than an import. Tests inject a deterministic stub so the
            suite never needs the vendored data or the pinned model to exercise
            an endpoint; ``tests/test_classifier_adapter.py`` drives the *real*
            resolved factory end-to-end over a small engineered frame, so the
            stub does not stand in for the shipped adapter anywhere it matters.

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
        # (measured 2026-08-03) — repeating it per request would buy nothing but
        # a slower endpoint. This also matches the pattern T3.1 set: the
        # app's expensive, immutable state is assembled at startup and only read
        # afterwards. The trade-off is explicit: a draw file added or edited
        # while the server is running is not picked up until it restarts
        # (`--reload` covers the dev loop).
        app.state.registry = build_registry(context.skill_table, draws_dir)
        # A path, not a scan: see the `cache_dir` argument note.
        app.state.cache_dir = Path(cache_dir)
        # Built once, here, for the same reason as everything above it — and
        # resolved here rather than at import time so a bad
        # config.API_CLASSIFIER_ADAPTER fails at startup, not mid-request.
        factory = classifier_factory or resolve_classifier_factory()
        app.state.classifier = build_classifier(context, factory)
        logger.info(
            "Live simulation adapter: %s (reconciliation %s)",
            app.state.classifier.name,
            config.SIM_CLI_RECONCILE_MODE,
        )
        yield
        app.state.context = None
        app.state.registry = None
        app.state.cache_dir = None
        app.state.classifier = None

    app = FastAPI(
        title=config.API_TITLE,
        version=config.API_VERSION,
        lifespan=lifespan,
    )

    # Per-IP rate limiter for /storybook only (attached as that route's
    # dependency below). Built per app instance — a fresh window store for every
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

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        """Attach the standard security headers to every response.

        The docs UI (/docs, /redoc) gets a looser CSP so Swagger/ReDoc can pull
        their bundle from the CDN and run their inline bootstrap; everything else
        — every JSON body — gets the strict ``default-src 'none'`` policy.
        """
        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        csp = DOCS_CSP if request.url.path in DOCS_PATHS else API_CSP
        response.headers.setdefault("Content-Security-Policy", csp)
        return response

    # Explicit allow-list from config — never "*". See config.API_ALLOWED_ORIGINS.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.API_ALLOWED_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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

        Matching is the shared T0.6 resolver (``common.names.resolve_name``) —
        exact → initials → substring → fuzzy — not a second search
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
        failure — a 422 from validation (see :data:`PlayerQuery`), since it is a
        malformed request rather than a search.

        **Not paginated**, deliberately: the whole result set is returned however
        large. This is a documented call for the current scale, not an oversight
        — the worst realistic query (``?query=a``) is 1,220 of 1,408 players and
        ~407 KB, assembled in ~12 ms, so the server-side cost is negligible. The
        cost that *does* matter is bandwidth to a browser type-ahead, which is
        T4.1's to manage (debounce, or ask for a limit here then).
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
        ``invalid`` carrying the T2.1 validator's full problem list. ``count``
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

    @app.get(
        "/tournaments/{tournament_id}/bracket",
        response_model=BracketResponse,
        tags=["tournaments"],
    )
    def tournament_bracket(
        tournament_id: Annotated[
            str,
            PathParam(description="Id from ``GET /tournaments``."),
        ],
        registry: TournamentRegistry = Depends(get_registry),
    ) -> BracketResponse:
        """The resolved draw: positions, entrants, ids and seeds.

        The draw comes from :func:`sim.draw.load_draw` (T2.1) via the startup
        registry — this handler re-parses nothing and re-validates nothing, so
        there is one draw parser in the codebase and one set of rules.

        Two failures, both deliberate:

          * **Unknown id → 404**, naming the ids that *are* registered (a draws
            directory holds a handful of files, so listing them beats a bare
            "not found" — the same courtesy ``cli/simulate_tournament`` extends).
          * **A file that failed validation → 422**, with the
            :class:`~sim.draw.DrawValidationError` ``problems`` list verbatim
            rather than a generic message, so a hand-entered draw is fixable in
            one pass. 422 is the ticket's choice and is a deliberate stretch of
            the status code: the unprocessable entity is the server's draw file,
            not the client's request, but nothing about the request can fix it
            and a 500 would suggest a transient fault rather than a bad file.
        """
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
    ) -> SimulationResponse:
        """Precomputed Monte Carlo title and round-survival probabilities.

        **This handler never simulates anything.** It reads the JSON file
        ``scripts/precompute_sim.py`` wrote offline, validates it and returns it.
        Phase 3's global rules forbid running a 5,000-run Monte Carlo inside a
        request handler — a 128-draw job is ~40 s — so the endpoint is a cache
        reader by design, not as an optimisation. The proof is structural: this
        module imports no simulation entry point at all
        (``tests/test_api_simulate.py`` pins it with raising traps on
        ``monte_carlo`` and ``simulate_bracket``).

        **Read ``metadata`` before rendering the numbers.** Every response
        carries the reconciliation ``mode``/``w``, an ``is_forecast`` flag and a
        plain-language ``classifier_limitation``. Since T3.5 the classifier's
        feature row is fully real (``ace-04-current-state.md §7`` seam 7,
        closed), so what the disclosure now states is narrower: the model is an
        as-of-now snapshot, which makes a simulation of an already-played draw
        retrospective rather than a forecast. They remain required schema fields,
        so a cache file cannot publish bare probabilities.

        Four failure modes, each distinguishable without parsing prose:

          * **Unknown id → 404**, listing simulatable ids separately from draw
            files that failed validation (see :func:`_unknown_tournament`).
          * **A draw file that failed validation → 422** with the validator's
            problem list — the same body ``/bracket`` returns for the same file.
          * **A draw containing placeholder entrants → 409**
            (``reason: "draw_not_simulatable"``). This is T3.3's answer to the
            open question T3.2 left: the registry keeps listing such draws and
            ``/bracket`` keeps serving them, and simulability is answered *here*,
            at the endpoint that needs it, rather than by a ``simulatable`` flag
            on ``/tournaments`` or by filtering the catalogue. Reasons: a flag
            would be a second mechanism for a distinction the error already
            makes, and would have to be kept in sync with cache presence to mean
            anything useful to a client; filtering would hide a real event
            (T3.2's explicit rejection of silent skipping, and its pinned
            behaviour). A 409 also matches the standard T3.2 set — an informative
            structured error, not a crash — and mirrors, over HTTP, the refusal
            ``simulate_bracket`` already makes offline, naming every offending
            slot rather than inventing a probability for it.
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
        a 422 — same reasoning as an invalid draw file: it is the server's
        artefact, a 500 would imply a transient fault, and the fix is named in
        the message.
        """
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
        dependencies=[Depends(storybook_rate_limit)],
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
    ) -> StorybookResponse:
        """One tournament played out point by point, for one seed — live.

        **This one does simulate, and that is within the Phase 3 rule, not an
        exception to it.** The rule forbids a full Monte Carlo (thousands of
        brackets) in a request handler; a storybook is *one* bracket — 127
        matches for a Slam — which the ticket budgets at "well under a couple
        seconds". Measured on the shipped 128 draw ``usopen_2024_atp_full``
        (2026-08-03, dev machine, re-measured on T3.5's real-feature adapter):
        **1.54 s for the first request** on a cold server (was 1.42 s), **0.80 s
        for a subsequent request with a fresh seed**, and **0.45 s to replay a
        seed already served**. The spread is almost entirely the adapter's
        ``predict_proba``: 127 matches need 127 distinct matchups priced, and the
        memo table lives on the startup-built adapter, so it is shared across
        requests and warms as the server runs; the cold request additionally pays
        the one-off 0.08 s rolling-form table build. 0.45 s is the floor — the
        point-by-point simulation of 127 matches itself.
        ``storybook_run`` is called **exactly once** per request; the rounds,
        scorelines and champion in the response are read off the single
        :class:`~sim.tournament.StorybookResult` it returns.

        **Determinism is the contract, not a nicety.** Same id + same seed → a
        byte-identical body: the response carries no timestamp and every field is
        a function of the draw, the seed and the startup state. That is what makes
        the URL shareable, and it is why an omitted ``?seed=`` falls back to a
        *fixed* default (:data:`config.API_STORYBOOK_SEED`) rather than a
        time-derived one — a bare ``/storybook`` that returned a different story
        on refresh would be unreproducible and uncacheable. ``metadata.seed``
        always echoes what ran, so a client can turn any response into an explicit,
        shareable URL.

        **Reconciliation mode is ``config.SIM_CLI_RECONCILE_MODE`` (``blend``),
        not ``config.RECONCILE_MODE``** — the same value T3.3 chose and the same
        one the cached ``/simulate`` board was produced under, so the two
        endpoints publish one model. It began as the seam-7 mitigation and is
        kept, post-T3.5, because blending lets the point model contribute half
        the match-win probability (see :data:`config.SIM_CLI_RECONCILE_MODE`).
        ``metadata`` carries ``mode``/``w``, ``is_forecast`` and
        ``classifier_limitation`` — the identical
        :class:`~api.schemas.ModelDisclosure` block ``/simulate`` serves,
        inherited rather than restated.

        Failures are ``/simulate``'s, from the same
        :func:`_simulatable_entry` gate: unknown id → 404, unloadable draw file →
        422, placeholder entrants → 409. There is no 425 here — nothing is
        cached, so there is nothing to be missing.

        **Rate limited** (``config.API_STORYBOOK_RATE_LIMIT``, per IP): this is
        the only endpoint that runs real per-request compute, so a client over
        the limit gets a 429. The limiter's store is in-memory and resets on
        restart — see the config constant for the honest scope of that.
        """
        entry = _simulatable_entry(tournament_id, registry)
        if seed is None:
            seed = config.API_STORYBOOK_SEED

        result = storybook_run(
            entry.draw,
            context.skill_table,
            classifier.call,
            seed=seed,
            # See the docstring: T3.3's choice, inherited deliberately.
            mode=config.SIM_CLI_RECONCILE_MODE,
            w=config.RECONCILE_BLEND_WEIGHT,
        )
        return _storybook_response(
            result, entry=entry, context=context, adapter=classifier.name
        )

    return app


# Module-level app for `uvicorn api.main:app`. Construction is cheap; the
# pipeline load waits for startup.
app = create_app()
