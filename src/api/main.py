"""FastAPI application: ``/health`` and ``/players`` (T3.1).

Run it with::

    uvicorn api.main:app --reload    # from src/, or with src/ on PYTHONPATH

The app is built by :func:`create_app` so the startup context can be injected.
Importing this module is cheap and side-effect-free — the pipeline load (~1.8 s;
see :func:`api.deps.build_api_context`) happens in the **lifespan**, i.e. when
the app actually starts, so a test may import ``api.main`` (or generate its
OpenAPI schema) without touching the vendored dataset.

Conventions this ticket sets for T3.2–T3.4:
  * Startup state is loaded once into ``app.state.context`` by the lifespan
    handler and read by every handler through the :func:`get_context` dependency.
    Handlers never load anything themselves.
  * All responses are ``api.schemas`` models (``response_model=`` on every
    route); no handler returns a raw dict.
  * Anything configurable (CORS origins, titles) is a ``config`` constant.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import StringConstraints

import config
from api.deps import ApiContext, ContextFactory, build_api_context, log_context
from api.schemas import (
    HealthResponse,
    PlayerSearchResponse,
    PlayerSummary,
    SurfaceSkill,
)
from common.names import NameIndex, resolve_name
from features.serve import SkillTable

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


def create_app(context_factory: ContextFactory = build_api_context) -> FastAPI:
    """Build the FastAPI app.

    Args:
        context_factory: Called **exactly once**, on startup, to produce the
            :class:`~api.deps.ApiContext`. Defaults to the real pipeline loader;
            tests inject a fixture factory so they never load vendored data.

    Returns:
        The configured app. Nothing is loaded until it starts.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Lifespan, not the deprecated @app.on_event("startup"). Runs once per
        # app instance; the factory result is the single shared context.
        context = context_factory()
        app.state.context = context
        log_context(context)
        yield
        app.state.context = None

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

    return app


# Module-level app for `uvicorn api.main:app`. Construction is cheap; the
# pipeline load waits for startup.
app = create_app()
