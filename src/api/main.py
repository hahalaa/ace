"""FastAPI application: ``/health``, ``/players`` (T3.1), ``/tournaments`` (T3.2/T3.3).

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

**T3.3: nothing in this module simulates anything.** ``/simulate`` reads a file
that ``scripts/precompute_sim.py`` wrote offline. That is a Phase 3 global rule
("never run a full 5,000-run MC synchronously inside a request handler"), not a
performance preference, and it is why this module imports neither
``sim.tournament`` nor any simulation entry point at all.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path as PathParam, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import StringConstraints, ValidationError

import config
from api.deps import ApiContext, ContextFactory, build_api_context, log_context
from api.registry import TournamentEntry, TournamentRegistry, build_registry, cache_path_for
from api.schemas import (
    BracketResponse,
    BracketSlot,
    HealthResponse,
    InvalidDraw,
    PlayerSearchResponse,
    PlayerSummary,
    SimulationResponse,
    SurfaceSkill,
    TournamentListResponse,
    TournamentSummary,
)
from common.names import NameIndex, resolve_name
from features.serve import SkillTable
from sim.draw import Draw

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


def create_app(
    context_factory: ContextFactory = build_api_context,
    draws_dir: Path | str = config.DRAWS_DIR,
    cache_dir: Path | str = config.CACHE_DIR,
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

    Returns:
        The configured app. Nothing is loaded or scanned until it starts.
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
        yield
        app.state.context = None
        app.state.registry = None
        app.state.cache_dir = None

    app = FastAPI(
        title=config.API_TITLE,
        version=config.API_VERSION,
        lifespan=lifespan,
    )

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
        plain-language ``classifier_limitation``, because the model behind these
        probabilities has a known defect (``ace-04-current-state.md §7`` seam 7:
        20 of the classifier's 27 features are synthetic constants, flattening it
        toward 0.5). They are required schema fields, so a cache file cannot
        publish bare probabilities.

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

        The placeholder check runs **before** the cache lookup, so the answer for
        a given draw is a property of the draw rather than of whatever happens to
        be on disk.

        A cache file that fails to parse, or that describes a different draw than
        the registry holds (a stale file left after the draw was edited), is also
        a 422 — same reasoning as an invalid draw file: it is the server's
        artefact, a 500 would imply a transient fault, and the fix is named in
        the message.
        """
        entry = registry.get(tournament_id)
        if entry is None:
            raise _unknown_tournament(tournament_id, registry)
        if entry.draw is None:
            raise _invalid_draw_file(entry)

        placeholders = entry.placeholder_slots
        if placeholders:
            raise HTTPException(
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

        path = cache_path_for(tournament_id, cache_dir)
        if path is None or not path.is_file():
            raise HTTPException(
                status_code=425,
                detail={
                    "reason": "cache_missing",
                    "message": (
                        f"No precomputed simulation for {tournament_id!r}. Monte "
                        f"Carlo is never run inside a request — generate the "
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

    return app


# Module-level app for `uvicorn api.main:app`. Construction is cheap; the
# pipeline load waits for startup.
app = create_app()
