"""Pydantic response models for the API (T3.1, T3.2, T3.3, T3.4).

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

from datetime import datetime

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


# --------------------------------------------------------------------------- #
# The model disclosure (T3.3), shared by every endpoint that publishes a
# simulated probability (T3.4). Rewritten by T3.5, which fixed what the original
# text described.
# --------------------------------------------------------------------------- #
# The canonical disclosure text. It lives here, next to the schema that requires
# it, because two producers need the identical wording: ``scripts/precompute_sim.py``
# writes it into every cache file, and ``api/main.py`` attaches it to every live
# storybook. A second copy of this paragraph is exactly the "second disclosure
# format" the requirement exists to prevent.
#
# **What changed in T3.5.** Until then this text disclosed seam 7: the
# classifier was fed a row whose 20 recent-form features were synthetic
# constants, flattening its probability toward 0.5. That is fixed —
# ``common.classifier_adapter`` builds all 27 features from the pipeline's real
# leakage-safe state, and the adapter's held-out Brier score now equals the
# training pipeline's own (0.2188). The field is kept, and still required,
# because a real limitation remains: every input is an **as-of-now snapshot** of
# the loaded data, so simulating a draw that has already been played uses form
# and skills its participants did not have at the time.
#
# **These strings must stay self-contained.** They are published on every
# /simulate and /storybook response and rendered in the web app's disclosure
# panel, so their audience is a reader of the deployed site — someone with no
# access to this repository's internal design notes (which are not distributed)
# and no idea what a ticket number refers to. The text carried a
# "(ace-04-current-state.md §7 seam 7, closed in T3.5)" citation until the
# 2026-08-09 audit; the claim it supported is true and stays, the pointer to a
# file no reader has does not. Describe the limitation, never cite where it is
# discussed.
#
# **Two fields, one disclosure.** CLASSIFIER_LIMITATION is the one-line summary
# a reader sees first; CLASSIFIER_LIMITATION_DETAIL is the full technical
# account behind it, unchanged in substance from the single paragraph this used
# to be. Both are required response fields — the split is a presentation choice
# (the web app shows the summary and tucks the detail behind a collapsed
# toggle), so the API body stays fully self-describing: a non-browser consumer
# still receives the complete disclosure, not just the teaser.
CLASSIFIER_LIMITATION = (
    "Not a forecast. This draw already happened, so the model is scoring a "
    "result it could have seen."
)

CLASSIFIER_LIMITATION_DETAIL = (
    "The classifier is fed a fully populated feature row: all 27 features, "
    "including recent form, computed from the pipeline's own leakage-safe match "
    "history, where every value is derived only from matches played before the "
    "one being predicted. The remaining caveat is timing: rankings, surface "
    "records, head-to-heads and recent form are an as-of-now snapshot of the "
    "loaded data, so simulating a draw that has already been played gives every "
    "entrant knowledge from after the event. Read a past event's numbers as a "
    "retrospective, not as a forecast; a draw that has not yet been played "
    "carries no such caveat."
)

# Whether output produced under the adapter above may be presented as a
# prediction. ``False``: the draws shipped today are historical, and the
# as-of-now snapshot above makes their numbers retrospective. One flag, so the
# two producers cannot disagree about it. (Before T3.5 this flag stood for the
# seam-7 defect; it now stands for the snapshot caveat, which is why it did not
# flip when the defect was fixed.)
IS_FORECAST = False

# --------------------------------------------------------------------------- #
# Disclosure for a genuinely future-dated draw (draw upload feature).
# --------------------------------------------------------------------------- #
# The as-of-now-snapshot caveat above exists because every shipped draw already
# happened. An *uploaded* draw whose declared event date is in the future is the
# one case where that does not apply: none of its matches have been played, so
# the snapshot is simply the latest available form rather than after-the-fact
# knowledge. That draw's disclosure reads is_forecast=true and swaps in the
# wording below. It is still not a confident prediction — the model's ceiling is
# unchanged — but it is an honest forward projection, not a retrospective, and it
# must not carry the "this already happened" sentence.
FORECAST_CLASSIFIER_LIMITATION = (
    "A model projection for an event that has not been played. Treat it as an "
    "estimate, not a confident prediction."
)

FORECAST_CLASSIFIER_LIMITATION_DETAIL = (
    "The classifier is fed a fully populated feature row: all 27 features, "
    "including recent form, computed from the pipeline's own leakage-safe match "
    "history. Because this draw's event date is in the future, none of its "
    "matches have happened yet, so those inputs — rankings, surface records, "
    "head-to-heads and recent form — are the latest available form rather than "
    "knowledge from after the event. This is a genuine forward projection. It is "
    "still only as good as the model behind it: match outcomes are noisy, an "
    "as-of-now snapshot cannot see an injury or a late withdrawal, and the "
    "numbers should be read as an estimate rather than a settled forecast."
)

# --------------------------------------------------------------------------- #
# Content-authenticity note (draw upload feature).
# --------------------------------------------------------------------------- #
# A DISTINCT concern from is_forecast/classifier_limitation, deliberately not
# folded into them: those describe what the *model* knows; this describes where
# the *draw* came from. A curated draw in data/draws/ is git-committed and
# checked against real historical records; an uploaded draw is neither, and it
# lives in this process's memory only, so it can vanish on a restart. Both facts
# belong in the disclosure a reader sees, and neither is about the classifier.
CONTENT_SOURCE_CURATED = "curated"
CONTENT_SOURCE_UPLOAD = "user_upload"

CONTENT_NOTE_CURATED = (
    "Curated example draw, verified against the real event's records."
)

CONTENT_NOTE_UPLOAD = (
    "User-submitted draw. It has not been checked against any official record, "
    "and it is held in memory only, so it may be cleared when the server "
    "restarts. Save anything you want to keep."
)


class ModelDisclosure(BaseModel):
    """What produced a published probability, and what its limits are.

    The base of both metadata blocks, so the disclosure is **inherited rather
    than retyped**: ``/simulate`` (cached Monte Carlo) and ``/storybook`` (live
    single run) run the same reconciliation over the same adapter, so a consumer
    that learned to read one has learned to read both, and neither can drift
    into publishing a subset of the fields.

    Every field is **required** — that is what makes disclosure structural. A
    cache file or a handler that omits any of it fails validation instead of
    serving bare probabilities.

    What is *not* here is as deliberate as what is. Run-shape provenance
    (``runs``/``workers``) belongs to a Monte Carlo aggregate and is filler for a
    single live run; ``generated_at`` describes a cache *file* and would make an
    otherwise byte-identical storybook differ between two calls with the same
    seed. Subclasses add what actually applies to them.
    """

    model_config = ConfigDict(extra="forbid")

    mode: str = Field(
        description="Reconciliation mode the probabilities were produced under: "
        "``blend`` (classifier diluted by the point model) or "
        "``classifier_anchor`` (the classifier *is* the target). Read together "
        "with ``classifier_limitation``."
    )
    w: float = Field(
        description="Weight on the classifier's probability in ``blend`` mode; "
        "unused by ``classifier_anchor``."
    )
    data_through_year: int = Field(
        description="Latest season present in the data the skill table and "
        "classifier were built from."
    )
    estimator_class: str = Field(
        description="Class name of the pinned classifier used. The training "
        "pipeline persists whichever of four candidate models (logistic "
        "regression, decision tree, random forest, gradient-boosted trees) "
        "scores best on the held-out season, so this can differ between "
        "deployments built in different environments — which is why it is "
        "reported rather than assumed."
    )
    adapter: str = Field(
        description="Which ``ClassifierProb`` adapter produced ``P_clf`` — the "
        "subject of ``classifier_limitation``."
    )
    is_forecast: bool = Field(
        description="**Machine-readable honesty flag.** ``false`` means these "
        "numbers demonstrate the simulation machinery and must not be presented "
        "as a prediction of the event; see ``classifier_limitation``."
    )
    classifier_limitation: str = Field(
        description="One-line, plain-language summary of the known limitation of "
        "the model behind these probabilities. Required, so numbers cannot be "
        "published without it; the full account is in "
        "``classifier_limitation_detail``."
    )
    classifier_limitation_detail: str = Field(
        description="The complete technical statement behind "
        "``classifier_limitation``. Required too, so a non-browser consumer "
        "receives the whole disclosure in the response body, not just the "
        "summary; the web app renders it behind a collapsed toggle."
    )
    source: str = Field(
        description="Draw file basename the simulation was run against."
    )


# --------------------------------------------------------------------------- #
# Cached Monte Carlo simulation (T3.3)
# --------------------------------------------------------------------------- #
# These three models are **also the cache file format**. ``scripts/precompute_sim.py``
# assembles a SimulationResponse, validates it and writes its JSON; the endpoint
# reads that file straight back through the same model. So there is one
# definition of the format rather than a writer and a reader that must be kept in
# step, and ``extra="forbid"`` plus the absence of defaults on the metadata
# fields means a cache file written by an older schema fails loudly on read
# instead of being served with fields silently missing.
class SimulationMetadata(ModelDisclosure):
    """Provenance of a cached Monte Carlo run — what produced these numbers.

    :class:`ModelDisclosure` plus the three facts that only a cached aggregate
    has: how many runs are behind the counts, the base seed that reproduces
    them, and when the file was written.
    """

    runs: int = Field(description="Number of bracket simulations behind the counts.")
    seed: int = Field(
        description="Base RNG seed. Re-running the precompute script with this "
        "seed, runs and mode reproduces the file exactly. (Not to be confused "
        "with a player's tournament seed in ``SimulationPlayer.seed``.)"
    )
    workers: int = Field(
        description="Processes used. Provenance only — the aggregate is "
        "worker-count-independent by construction (T2.3)."
    )
    generated_at: datetime = Field(
        description="When the cache file was written (UTC, ISO-8601)."
    )


class SimulationPlayer(BaseModel):
    """One entrant's simulated outcome — mirrors ``sim.tournament.PlayerOutcome``.

    Both the raw **counts** and the derived **probabilities** are carried: counts
    are what T2.3 actually stores (integers make two runs comparable for exact
    equality), and probabilities are what a client renders. Nothing here is
    computed that ``PlayerOutcome`` does not already expose.
    """

    model_config = ConfigDict(extra="forbid")

    position: int = Field(description="1-based bracket slot.")
    player: str = Field(description="Entrant display name.")
    player_id: str | None = Field(description="Skill-table id; the join key.")
    seed: int | None = Field(
        description="The player's **tournament seed**, or ``None`` if unseeded. "
        "Unrelated to ``SimulationMetadata.seed``, which is the RNG seed."
    )
    titles: int = Field(description="Runs this entrant won.")
    matches_won: int = Field(description="Matches won across every run.")
    p_title: float = Field(description="``titles / runs`` — P(wins the event).")
    expected_rounds_won: float = Field(
        description="``matches_won / runs`` — mean matches won per run."
    )
    reached: dict[str, int] = Field(
        description="Round label → runs in which the entrant contested it."
    )
    p_reach: dict[str, float] = Field(
        description="Round label → P(contests that round). Keys are the draw's "
        "own labels (``QF``/``SF``/``F`` for an 8-draw, ``R128 … F`` for 128), "
        "never a hardcoded set."
    )


class SimulationResponse(BaseModel):
    """A precomputed Monte Carlo result, as cached and as served.

    **Read ``metadata`` before rendering the numbers.** T3.3 published these
    probabilities under a classifier whose feature row was 20/27 synthetic
    constants (``ace-04-current-state.md §7`` seam 7); T3.5 closed that seam, so
    the row is now built entirely from the pipeline's own leakage-safe state. The
    limitation that remains is narrower and still travels *with* the numbers:
    every input is an **as-of-now snapshot** of the loaded data, which makes a
    simulation of an already-played draw retrospective rather than a forecast.
    ``metadata.classifier_limitation`` (prose), ``metadata.is_forecast``
    (machine-readable) and ``metadata.mode``/``w`` (how ``P_clf`` and the point
    model are fused) are required fields of every response, not a footnote in the
    docs.
    """

    model_config = ConfigDict(extra="forbid")

    tournament_id: str = Field(description="The draw's own id.")
    name: str = Field(description="Human-readable event name.")
    surface: str = Field(description="``Hard``, ``Clay`` or ``Grass``.")
    best_of: int = Field(description="3 or 5.")
    final_set_tiebreak: str = Field(description="Deciding-set rule.")
    draw_size: int = Field(
        description="Entrants in the draw — the **full** field size, which "
        "exceeds ``count`` when ``?top=`` truncated the rows."
    )
    round_labels: list[str] = Field(
        default_factory=list,
        description="The draw's round labels, first round first.",
    )
    count: int = Field(
        description="Players in ``players`` — the whole field unless ``?top=`` "
        "was given."
    )
    players: list[SimulationPlayer] = Field(
        default_factory=list,
        description="Entrants sorted by title probability, descending (ties "
        "broken by bracket position, so the order is deterministic).",
    )
    metadata: SimulationMetadata = Field(
        description="Provenance: runs, seed, mode/weight, data year, and the "
        "model's known limitation. Required."
    )


# --------------------------------------------------------------------------- #
# Live storybook run (T3.4)
# --------------------------------------------------------------------------- #
# Wire models over ``sim.tournament``'s T2.4 storybook types. Presentation only,
# in the same sense ``scripts/precompute_sim.build_payload`` is: every field is
# read off ``StorybookResult``/``StorybookRound``/``StorybookMatch``/``PlayerRun``,
# and nothing is re-derived from the underlying ``BracketResult`` a second time.
class StorybookMetadata(ModelDisclosure):
    """Provenance of one live storybook run.

    :class:`ModelDisclosure` — the same seam-7 disclosure ``/simulate`` carries,
    inherited rather than restated — plus the one fact that reproduces this
    particular story: its ``seed``.

    Note what is deliberately absent: no ``runs``/``workers`` (this *is* one run,
    in-process) and no ``generated_at``. Everything here is a function of the
    draw, the seed and the startup state, so two calls with the same seed return
    byte-identical bodies — which is what makes the URL shareable.
    """

    seed: int = Field(
        description="RNG seed this story was played out under — the one the "
        "caller passed, or the server default when they passed none. Echoed so "
        "``?seed=`` on the same URL replays this exact tournament. (Not to be "
        "confused with a player's tournament seed in ``StorybookPlayerRun.seed``.)"
    )
    # ------------------------------------------------------------------ #
    # Where the DRAW came from — added by the draw upload feature.
    # ------------------------------------------------------------------ #
    # A concern separate from the inherited is_forecast/classifier_limitation
    # (which are about the model's knowledge). These say whether the draw itself
    # is a curated, verified example or user-submitted and unverified. Required
    # here (not on the shared base) because ``/storybook`` is always built live
    # by the handler, so every response can set them — unlike the cached
    # ``/simulate`` format, whose on-disk files predate these fields and must not
    # gain a required key. ``/simulate`` never serves an uploaded draw anyway.
    content_source: str = Field(
        description="``curated`` for a git-committed example draw verified "
        "against the real event, or ``user_upload`` for a draw a visitor "
        "submitted. Distinct from ``is_forecast``, which is about the model."
    )
    content_note: str = Field(
        description="Plain-language note on the draw's provenance — for an "
        "upload, that it is unverified and held in memory only. Render it "
        "alongside the model disclosure, not in place of it."
    )


class StorybookMatch(BaseModel):
    """One played-out match — mirrors ``sim.tournament.StorybookMatch``.

    Both ``scoreline`` and the rendered ``line`` are carried: a bracket renderer
    wants the score next to two names it positions itself, and a feed wants the
    one-liner. ``line`` is the T2.4 dataclass's own field, not a second
    formatting rule invented here.
    """

    model_config = ConfigDict(extra="forbid")

    round_index: int = Field(description="0-based round, first round first.")
    round_label: str = Field(description="``R128`` … ``QF``/``SF``/``F``.")
    match_index: int = Field(
        description="0-based position of the match within its round, top of the "
        "draw first."
    )
    winner: str = Field(description="Winner's display name.")
    loser: str = Field(description="Loser's display name.")
    winner_seed: int | None = Field(
        description="Winner's tournament seed, ``None`` if unseeded."
    )
    loser_seed: int | None = Field(
        description="Loser's tournament seed, ``None`` if unseeded."
    )
    scoreline: str = Field(
        description="Winner-first set-by-set score, e.g. ``6-4 3-6 7-6(5) 6-2``."
    )
    line: str = Field(description="``\"<winner> def. <loser> <scoreline>\"``.")


class StorybookRound(BaseModel):
    """One round of the story, its matches in draw order."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(description="0-based round number.")
    label: str = Field(description="The draw's own label for this round.")
    matches: list[StorybookMatch] = Field(
        default_factory=list, description="Every match in the round."
    )


class StorybookPlayerRun(BaseModel):
    """One entrant's tournament, summarised — mirrors ``sim.tournament.PlayerRun``.

    Included so a client never has to walk ``rounds`` to answer "how far did this
    player get, and who beat them" — T2.4 already computed it from the bracket,
    and re-deriving it here would be a second answer to the same question.
    """

    model_config = ConfigDict(extra="forbid")

    position: int = Field(description="1-based bracket slot.")
    player: str = Field(description="Entrant display name.")
    player_id: str | None = Field(description="Skill-table id; the join key.")
    seed: int | None = Field(
        description="The player's **tournament seed**, or ``None`` if unseeded."
    )
    is_champion: bool = Field(description="True for the one entrant who won.")
    furthest_round: str = Field(
        description="Label of the last round contested — so a beaten finalist "
        "and the champion both read ``F``."
    )
    matches_won: int = Field(description="Matches won (``0`` for a first-round exit).")
    beat: list[str] = Field(
        default_factory=list, description="Opponents beaten, in round order."
    )
    eliminated_by: str | None = Field(
        description="Who knocked them out; ``None`` **iff** ``is_champion``."
    )


class StorybookChampion(BaseModel):
    """The title winner — identity only; their matches are in ``rounds``."""

    model_config = ConfigDict(extra="forbid")

    position: int = Field(description="1-based bracket slot they entered from.")
    player: str = Field(description="Display name.")
    player_id: str | None = Field(description="Skill-table id; the join key.")
    seed: int | None = Field(description="Tournament seed, ``None`` if unseeded.")


class StorybookResponse(BaseModel):
    """One tournament played out point by point, for one seed.

    **Read ``metadata`` before rendering the scorelines**, for the same reason
    :class:`SimulationResponse` says so: the same reconciliation over the same
    adapter produced them, and ``ace-04-current-state.md §7`` seam 7 applies
    identically to a live run. The disclosure is the inherited
    :class:`ModelDisclosure` block, not a second format.

    Deterministic in full: every field is a function of the draw, ``metadata.seed``
    and the server's startup state, so the same URL returns the same body.
    """

    model_config = ConfigDict(extra="forbid")

    tournament_id: str = Field(description="The draw's own id.")
    name: str = Field(description="Human-readable event name.")
    surface: str = Field(description="``Hard``, ``Clay`` or ``Grass``.")
    best_of: int = Field(description="3 or 5.")
    final_set_tiebreak: str = Field(description="Deciding-set rule.")
    draw_size: int = Field(description="Entrants in the draw.")
    match_count: int = Field(
        description="Matches played — ``draw_size - 1`` for a full bracket."
    )
    rounds: list[StorybookRound] = Field(
        default_factory=list, description="Every round, first round first."
    )
    champion: StorybookChampion = Field(description="Who won the title.")
    runs: list[StorybookPlayerRun] = Field(
        default_factory=list,
        description="Every entrant's run, best finish first (ties broken by "
        "bracket position, so the order is deterministic).",
    )
    metadata: StorybookMetadata = Field(
        description="Provenance: the seed that replays this story, the "
        "reconciliation mode/weight, and the model's known limitation. Required."
    )


# --------------------------------------------------------------------------- #
# Draw upload (POST /tournaments/upload)
# --------------------------------------------------------------------------- #
class UploadResponse(BaseModel):
    """Acknowledgement that an uploaded draw was accepted and stored.

    Returned by ``POST /tournaments/upload`` on success. It carries the
    generated, namespaced ``tournament_id`` the caller then uses to fetch the
    bracket (``/tournaments/{id}/bracket``) or run a live storybook
    (``/tournaments/{id}/storybook``) — **not** the draw's own ``tournament_id``,
    which stays purely descriptive so uploads can never collide with the curated
    draws.

    It is deliberately honest about two things the caller must know up front: the
    draw is **ephemeral** (memory only, cleared on restart — ``ephemeral`` is
    always true and ``content_note`` says so in words), and whether it can be
    simulated at all (a draw with ``Qualifier`` slots loads and lists but cannot
    be run, exactly as a curated one cannot).
    """

    model_config = ConfigDict(extra="forbid")

    tournament_id: str = Field(
        description="The generated upload id to address this draw by "
        "(``upload-…``). Use it for ``/bracket`` and ``/storybook``."
    )
    name: str = Field(description="The draw's human-readable event name.")
    surface: str = Field(description="``Hard``, ``Clay`` or ``Grass``.")
    best_of: int = Field(description="3 or 5.")
    final_set_tiebreak: str = Field(description="Deciding-set rule.")
    draw_size: int = Field(description="Number of slots in the bracket.")
    is_simulatable: bool = Field(
        description="False when the draw still holds placeholder entrants "
        "(``Qualifier``/``Bye``); such a draw lists and shows its bracket but "
        "cannot be simulated."
    )
    placeholder_count: int = Field(
        description="How many slots are placeholders — ``0`` for a simulatable "
        "draw. Non-zero explains why ``is_simulatable`` is false."
    )
    is_forecast: bool = Field(
        description="True when the draw's declared ``event_date`` is in the "
        "future, so a simulation is a genuine forward projection."
    )
    ephemeral: bool = Field(
        description="Always true: uploaded draws are held in memory only and are "
        "lost on a server restart. Surfaced so the client can warn the user.",
    )
    content_note: str = Field(
        description="Plain-language note that this draw is user-submitted, "
        "unverified and temporary. Show it near the upload result."
    )
