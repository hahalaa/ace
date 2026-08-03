"""Pydantic response models for the API (T3.1, T3.2).

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


# --------------------------------------------------------------------------- #
# Tournaments (T3.2)
# --------------------------------------------------------------------------- #
class TournamentSummary(BaseModel):
    """One listable event: exactly the five fields the ticket specifies.

    Every field is non-null, which is the point of keeping failed draw files in
    a *separate* list (:class:`InvalidDraw`) rather than folding them in here
    with nulled-out metadata: a client rendering the tournament picker never has
    to null-check, and never has to filter out an event it cannot open.
    """

    model_config = ConfigDict(extra="forbid")

    tournament_id: str = Field(
        description="Lookup key for ``/tournaments/{id}/bracket``; the draw "
        "file's own ``tournament_id``."
    )
    name: str = Field(description="Human-readable event name.")
    surface: str = Field(description="``Hard``, ``Clay`` or ``Grass``.")
    best_of: int = Field(description="Sets to win the match format over: 3 or 5.")
    draw_size: int = Field(description="Number of slots in the bracket.")


class InvalidDraw(BaseModel):
    """A draw file in the directory that did not load.

    Listed rather than silently skipped. Draw files are hand-entered, so a
    typo is the expected failure; dropping the file from the listing would make
    the event simply vanish, with the reason visible only in a server log. The
    ``tournament_id`` here is the file's stem (a failed file has no trustworthy
    id of its own) and it is addressable: fetching its bracket returns the same
    ``problems`` with a 422.
    """

    model_config = ConfigDict(extra="forbid")

    tournament_id: str = Field(
        description="Id this file is addressable by — its filename stem, since "
        "the draw's own ``tournament_id`` could not be trusted."
    )
    source: str = Field(description="Draw file basename, so the file can be fixed.")
    problems: list[str] = Field(
        description="Every problem the T2.1 validator found, in its own order — "
        "the whole list, so a hand-entered draw is fixable in one pass."
    )


class TournamentListResponse(BaseModel):
    """Everything in the draws directory, split by whether it loaded."""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(description="Number of **valid** tournaments returned.")
    tournaments: list[TournamentSummary] = Field(
        default_factory=list,
        description="Valid, openable events, in draw-filename order.",
    )
    invalid: list[InvalidDraw] = Field(
        default_factory=list,
        description="Draw files that failed to load. Normally empty; a bad file "
        "shows up here instead of breaking the listing.",
    )


class BracketSlot(BaseModel):
    """One entrant position in a resolved bracket.

    Mirrors :class:`sim.draw.DrawSlot` plus the slot's seed. Seeds are attached
    **per slot** rather than repeated as a name → seed map: the validator
    guarantees every seeded name is present in the bracket, so nothing is lost,
    and a bracket renderer wants the seed next to the name it draws.
    """

    model_config = ConfigDict(extra="forbid")

    position: int = Field(
        description="1-based slot number. Slots ``2k-1`` and ``2k`` meet in round 1."
    )
    player: str = Field(
        description="Entrant string exactly as written in the draw file."
    )
    is_placeholder: bool = Field(
        description="True when the entrant names an unfilled slot (``Qualifier``, "
        "``Bye``, …) rather than a person. Such a slot resolves to the "
        "surface-baseline skill profile, but cannot be simulated (T2.2)."
    )
    player_id: str | None = Field(
        description="Skill-table id — the canonical join key. ``None`` for "
        "placeholders."
    )
    seed: int | None = Field(
        default=None, description="Seed number, or ``None`` if unseeded."
    )


class BracketResponse(BaseModel):
    """A resolved draw: its match format and every slot in position order."""

    model_config = ConfigDict(extra="forbid")

    tournament_id: str = Field(description="The draw's own id.")
    name: str = Field(description="Human-readable event name.")
    surface: str = Field(description="``Hard``, ``Clay`` or ``Grass``.")
    best_of: int = Field(description="3 or 5.")
    final_set_tiebreak: str = Field(
        description="Deciding-set rule: ``7pt_at_6_6``, ``10pt_at_6_6`` or "
        "``advantage``."
    )
    draw_size: int = Field(description="Number of slots; equals ``len(slots)``.")
    slots: list[BracketSlot] = Field(
        default_factory=list,
        description="Every slot, ordered by ``position`` — placeholders "
        "included and flagged, never dropped.",
    )
