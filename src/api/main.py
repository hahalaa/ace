"""FastAPI application: ``/health``, ``/players`` (T3.1), ``/tournaments`` (T3.2).

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

**T3.2 note for T3.3.** ``/tournaments/{id}/bracket`` calls ``load_draw`` and
nothing else, so T2.2's placeholder-entrant refusal — which lives in
``simulate_bracket`` — never fires here. A draw containing ``Qualifier`` slots
loads fine (placeholders take the default skill profile, as everywhere else),
so ``data/draws/example_usopen_2026.json`` is legitimately listed and its
bracket legitimately served. The registry deliberately does **not** try to
guess simulability; deciding what ``/simulate`` does with such a draw is T3.3's
open question (see its ticket text).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Path as PathParam, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import StringConstraints

import config
from api.deps import ApiContext, ContextFactory, build_api_context, log_context
from api.registry import TournamentRegistry, build_registry
from api.schemas import (
    BracketResponse,
    BracketSlot,
    HealthResponse,
    InvalidDraw,
    PlayerSearchResponse,
    PlayerSummary,
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


def create_app(
    context_factory: ContextFactory = build_api_context,
    draws_dir: Path | str = config.DRAWS_DIR,
) -> FastAPI:
    """Build the FastAPI app.

    Args:
        context_factory: Called **exactly once**, on startup, to produce the
            :class:`~api.deps.ApiContext`. Defaults to the real pipeline loader;
            tests inject a fixture factory so they never load vendored data.
        draws_dir: Directory scanned once at startup for draw files. Defaults to
            ``config.DRAWS_DIR``; tests point it at a fixture directory so the
            suite never depends on the shipped draws.

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
        yield
        app.state.context = None
        app.state.registry = None

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
            raise HTTPException(
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
        return _bracket_response(entry.draw)

    return app


# Module-level app for `uvicorn api.main:app`. Construction is cheap; the
# pipeline load waits for startup.
app = create_app()
