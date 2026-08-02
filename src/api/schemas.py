"""Pydantic response models for the API (T3.1).

Every endpoint returns one of these — no handler returns a raw dict. Three
conventions established here for the rest of Phase 3 to follow:

  * **The wire model is not the domain model.** ``SurfaceSkill`` mirrors
    ``features.serve.PlayerSkill`` rather than serialising it directly, so the
    internal dataclass can change without silently reshaping the HTTP contract.
  * **Search responses echo their query.** A client that fired several requests
    can attribute a response without tracking state.
  * **Every model forbids extra fields.** ``extra="forbid"`` is what makes the
    "no raw dicts" rule *enforceable* rather than aspirational: Pydantic v2
    ignores undeclared keys by default, so a test that round-trips a response
    body through ``model_validate`` would happily accept a handler that had
    leaked an extra field onto the wire. With ``forbid`` that round-trip fails,
    and ``tests/test_api.py`` pins each body's exact key set on top of it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Liveness plus the provenance of the data behind every other answer."""

    model_config = ConfigDict(extra="forbid")

    status: str = Field(description="``\"ok\"`` when the app has loaded its state.")
    data_through_year: int = Field(
        description="Latest season actually present in the loaded data — the "
        "observed maximum, not the configured ``END_YEAR``."
    )
    n_players: int = Field(
        description="Number of players searchable via ``/players``."
    )


class SurfaceSkill(BaseModel):
    """One player's serve/return profile on one surface (T1.1).

    ``n_serve_pts``/``n_return_pts`` are the raw sample sizes behind the shrunk
    rates: **0 means this is the surface-baseline default profile**, not a
    measured one, so a client can tell an unknown player from a measured average
    one.
    """

    model_config = ConfigDict(extra="forbid")

    spw: float = Field(description="Serve-points-won rate, in ``[0, 1]``.")
    rpw: float = Field(description="Return-points-won rate, in ``[0, 1]``.")
    n_serve_pts: float = Field(description="Raw serve points in the sample.")
    n_return_pts: float = Field(description="Raw return points in the sample.")


class PlayerSummary(BaseModel):
    """A resolved player: identity plus per-surface skills."""

    model_config = ConfigDict(extra="forbid")

    player_id: str | None = Field(
        description="Canonical join key for ``sim/``/``api/``. ``None`` only if "
        "the name carries no id in the skill table."
    )
    name: str = Field(description="Canonical display name.")
    skills: dict[str, SurfaceSkill] = Field(
        description="Per-surface skills, keyed by surface name "
        "(``Hard``/``Clay``/``Grass``)."
    )


class PlayerSearchResponse(BaseModel):
    """Result of a ``/players`` search.

    A **list**, deliberately — including when the query is ambiguous or matches
    nothing. See the endpoint docstring in ``api/main.py`` for why this diverges
    from the CLI's error-on-ambiguity behaviour.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        description="The query as searched, echoed back (leading/trailing "
        "whitespace stripped by validation)."
    )
    count: int = Field(description="Number of players returned.")
    strategy: str | None = Field(
        default=None,
        description="Which resolver strategy matched (``exact``/``initials``/"
        "``substring``/``fuzzy``); ``None`` when nothing matched.",
    )
    players: list[PlayerSummary] = Field(
        default_factory=list,
        description="Matching players; empty when the query resolves to nothing.",
    )
