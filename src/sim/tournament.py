"""Bracket simulation (T2.2), the Monte Carlo runner (T2.3), storybook (T2.4).

Plays one full tournament: pair the current round's entrants, simulate every
match, advance the winners, repeat until one player is left. The entry point is
:func:`simulate_bracket`, which serves two callers with very different speed
needs through one flag:

* ``outcome_only=False`` (default — storybook T2.4, one-off/manual runs): every
  match is a real point-by-point simulation via
  :func:`~sim.reconcile.simulate_reconciled_match`, so each result carries a
  full scoreline (``6-4 3-6 7-6(5) 6-2``).
* ``outcome_only=True`` (Monte Carlo T2.3): every match is **one Bernoulli
  draw** against the reconciled analytic match-win probability. No point, game,
  set or tiebreak is simulated and ``scoreline`` is ``None``.

**What "reconciled" means here, and why it is the same in both modes (T2.2
decision).** The ticket calls the ``outcome_only=True`` probability "the
reconciled analytic match-win probability
(``sim/reconcile.py``'s ``match_win_prob_point``)", but those are two different
things: ``match_win_prob_point`` alone is the raw ``§2``–``§4`` composition of
the point model, with no classifier input at all. The *reconciled* probability
is what :func:`~sim.reconcile.simulate_reconciled_match` computes internally —
``P_clf`` from the classifier, fused with the point model per
``config.RECONCILE_MODE``, then realised through the serve shift ``δ``. This
module implements the **reconciled** reading:

    ``p_clf = classifier(a, b, surface)`` → ``reconciled_prob`` → ``solve_delta``
    → ``match_win_prob_point(pA + δ, pB − δ, …)``

so that a Monte Carlo pass and a storybook run of the same draw are driven by
the *same* probability model rather than two models of different quality. The
final ``match_win_prob_point`` call (rather than the ``target`` handed to
``solve_delta``) is deliberate: when a target is outside δ's reachable window
``solve_delta`` saturates, and the post-δ value is what
``outcome_only=False`` would actually simulate. The one remaining difference is
that the analytic composition averages over who serves first while the
point-by-point path pins it (see the serve-first convention below); that is the
same fraction-of-a-percent effect ``match_win_prob_point`` already documents.

**Performance — why ``prob_cache`` exists (measured, not assumed).** The
expensive part of a reconciled probability is not the classifier, it is the
bisection in :func:`~sim.reconcile.solve_delta`: ~3.2 ms versus ~0.2 ms for a
single :func:`~sim.reconcile.match_win_prob_point`, on this dev machine. A
128-draw × 5,000-run Monte Carlo pass is ~635,000 match evaluations ≈ **40
minutes** if each one re-solves δ — 40–80× over T2.3's budget — no matter how
well the caller memoises the classifier. But a reconciled probability is a
*pure function* of ``(entrant A, entrant B, surface, format, mode, w)``, and a
128-draw bracket admits at most 8,128 distinct matchups, so caching collapses
that to **~35–42 s** for the whole job: the per-run cost decays as the cache
fills (~52 ms/run over the first 100 runs, ~5 ms/run by run 1,200) until almost
every lookup is a dictionary hit. ``prob_cache`` is the caller-owned dict that
makes this possible; see :func:`simulate_bracket` for the contract T2.3 must
honour, which covers both this cache and the classifier adapter's own. These
figures match the ones recorded in ``ace-04-current-state.md`` and the ticket
index — keep the three in step if they are ever re-measured.

**Serve-first convention (deterministic, not drawn per match).** In every
match, player **A is the entrant sitting in the lower-numbered bracket
position** — the first-listed of the pairing — and **A serves the first game**.
Both halves matter:

* It fixes match orientation, so a given pairing is always presented to the
  classifier as ``(A, B, surface)`` in one stable order. Cache keys — the
  caller's and this module's — therefore never split across two orientations.
* It replaces T1.8's per-match coin flip for the opening server, which is why
  :func:`~sim.reconcile.simulate_reconciled_match` grew an optional
  ``first_server`` argument (``None`` still means "draw it", so no existing
  caller changed). The effect on a result is tiny; the point is that it is a
  documented convention rather than RNG state.

The bracket order is preserved round to round, so the "lower position" rule
holds automatically in every round: winners stay in bracket order, and each
match pairs consecutive winners out of disjoint, ascending position blocks.

**Placeholder entrants are rejected (T2.2 decision, a known gap).** T2.1 lets a
draw carry ``"Qualifier"``/``"Bye"``/… slots, which get the default surface
profile. Those cannot be *simulated* here: a placeholder has no ``player_id``
for the reconciliation join and no classifier-visible history at all, so no
``P_clf`` exists for it. Rather than invent one (a modelling decision this
ticket does not own — silently substituting 0.5, or bypassing the classifier
for part of the bracket, would make some matches quietly lower-quality than
others), :func:`simulate_bracket` refuses upfront and names every offending
slot, in T2.1's fail-loudly-but-complete style. Consequence to be aware of: the
shipped ``data/draws/example_usopen_2026.json`` contains placeholders and so
cannot be simulated as-is. Deciding how an unfilled slot should be modelled is
left to the project owner.

**Classifier quality caveat (``ace-04-current-state.md §7`` seam 7).** This
module takes its classifier as an injected callable (T1.8's dependency
inversion, exercised by T1.10) and never constructs one. The only adapter that
exists today, ``cli/simulate_match.make_classifier_prob``, builds a feature row
whose 20 recent-form columns are synthetic constants, which flattens ``P_clf``
toward 0.5. A bracket exercises that adapter far harder than T1.10 did — every
distinct matchup in the draw, feeding every Monte Carlo run — so a caller
should pick its reconciliation ``mode`` with that in mind (T1.10 defaults to
``"blend"`` for exactly this reason) or build an adapter with real engineered
rows. Nothing here fixes it; it is inherited, not resolved.

**Monte Carlo (T2.3).** :func:`monte_carlo` runs :func:`simulate_bracket` many
times — always with ``outcome_only=True``, never a scoreline — and aggregates
per-entrant title and round-survival probabilities into a
:class:`MonteCarloResult`. Three properties are load-bearing and are described
in full on :func:`monte_carlo` itself: exactly one ``prob_cache`` spans a whole
single-threaded job (and one *fresh* cache lives inside each worker process when
``workers > 1``); every run's RNG is spawned from one
``np.random.SeedSequence(seed)``, so the aggregate is bit-identical however the
runs are split across workers; and the reconciliation ``mode`` is a pass-through
parameter, because this module does not build the classifier and therefore does
not own the choice (see the classifier caveat below).

**Storybook (T2.4).** :func:`storybook_run` is the opposite end of the same
machinery: **one** ``simulate_bracket(..., outcome_only=False)`` call, every
match played out point by point, wrapped in a presentation-friendly structure
for the shareable "watch the tournament play out" view.
:func:`render_storybook` turns that structure into text. The two are kept
strictly apart — :class:`StorybookResult` is data the API/frontend can consume
directly, and the renderer only ever composes its public fields, never
computes a fact of its own.

The **run summary** shape T2.5 (and later the API) can rely on is
:class:`PlayerRun`, one per entrant: identity (``position``, ``player``,
``player_id``, ``seed``), how far they got (``furthest_round`` — the label of
the last round they *contested* — plus ``matches_won`` and ``is_champion``),
who they beat on the way (``beat``, opponent names in round order), and how it
ended (``eliminated_by``, ``None`` iff champion, and ``last_match``, the
:class:`StorybookMatch` in which they went out — the final, won, for the
champion). ``StorybookResult.runs`` is sorted by finishing position: champion,
runner-up, then losing semi-finalists and so on, ties broken by bracket
position.

Like every other ``sim/`` module this one never imports from ``cli/``: its
project imports are ``config``, ``features.serve`` (typing), ``sim.draw``,
``sim.match`` (result types only) and ``sim.reconcile``.
"""

from __future__ import annotations

import pickle
from collections.abc import Mapping, MutableMapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import config
from sim.draw import Draw, DrawSlot
from sim.match import MatchResult
from sim.points import matchup_point_probs
from sim.reconcile import (
    ClassifierProb,
    match_win_prob_point,
    reconciled_prob,
    simulate_reconciled_match,
    solve_delta,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from features.serve import SkillTable

# Index of player A in every bracket match (the lower bracket position), and the
# player who serves first — see the serve-first convention in the module
# docstring. Named rather than inlined so the convention is greppable.
PLAYER_A = 0
BRACKET_FIRST_SERVER = PLAYER_A

# Rounds are labelled by how many players *start* the round; the last three get
# their traditional names instead of R8/R4/R2. Domain vocabulary rather than a
# tunable, so it lives here and not in config.
_NAMED_ROUNDS = {8: "QF", 4: "SF", 2: "F"}


# ---------------------------------------------------------------------------
# Round labelling (pure).
# ---------------------------------------------------------------------------


def round_label(players_remaining: int) -> str:
    """Label for the round that ``players_remaining`` players start.

    ``128 → "R128"``, ``16 → "R16"``, ``8 → "QF"``, ``4 → "SF"``, ``2 → "F"``.

    Args:
        players_remaining: Players contesting the round; a power of two ≥ 2.

    Returns:
        The round's label.

    Raises:
        ValueError: If ``players_remaining`` is not a power of two ≥ 2 (a
            bracket that cannot halve cleanly has no meaningful round label).
    """
    if players_remaining < 2 or players_remaining & (players_remaining - 1):
        raise ValueError(
            f"players_remaining must be a power of two >= 2, "
            f"got {players_remaining!r}"
        )
    return _NAMED_ROUNDS.get(players_remaining, f"R{players_remaining}")


def round_labels(draw_size: int) -> tuple[str, ...]:
    """Every round label for a ``draw_size`` bracket, first round first.

    An 8-draw gives ``("QF", "SF", "F")``; a 128-draw gives
    ``("R128", "R64", "R32", "R16", "QF", "SF", "F")`` — ``log2(draw_size)``
    rounds, halving the field each time.

    Raises:
        ValueError: If ``draw_size`` is not a power of two ≥ 2.
    """
    labels = []
    remaining = draw_size
    while remaining >= 2:
        labels.append(round_label(remaining))
        remaining //= 2
    if not labels:
        raise ValueError(f"draw_size must be a power of two >= 2, got {draw_size!r}")
    return tuple(labels)


# ---------------------------------------------------------------------------
# Scoreline rendering (pure).
# ---------------------------------------------------------------------------


def format_scoreline(result: MatchResult) -> str:
    """Render a match scoreline **winner-first**, e.g. ``6-4 3-6 7-6(5) 6-2``.

    Every set is written from the match winner's perspective (the tennis
    convention), so a set the winner lost reads ``3-6``. A tiebreak set carries
    the loser-of-that-set's point total in parentheses — ``7-6(5)`` won,
    ``6-7(5)`` lost.

    Note this is deliberately *not*
    ``cli/simulate_match.format_scoreline``, which renders from player A's
    perspective: the CLI names its two players explicitly, whereas a bracket
    line reads "winner def. loser". The layering rule also forbids ``sim/``
    importing the CLI's copy.
    """
    parts = []
    for s in result.sets:
        if result.winner == PLAYER_A:
            games_w, games_l = s.games_a, s.games_b
        else:
            games_w, games_l = s.games_b, s.games_a
        line = f"{games_w}-{games_l}"
        if s.tb_score is not None:
            line += f"({min(s.tb_score)})"  # the set loser's tiebreak points
        parts.append(line)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Result types (frozen dataclasses, matching the rest of sim/).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BracketMatch:
    """One simulated match in the bracket.

    Attributes:
        round_index: 0-based round number (0 = first round).
        round_label: The round's label (``"R32"``, ``"QF"``, …).
        match_index: 0-based position of the match within its round, top of the
            draw first.
        slot_a: The entrant in the lower bracket position — player ``0``, and
            the first server (see the module docstring).
        slot_b: The other entrant — player ``1``.
        winner: Whichever of ``slot_a``/``slot_b`` won.
        scoreline: The rendered scoreline, or ``None`` when the match was
            decided by a single Bernoulli draw (``outcome_only=True``).
        result: The underlying :class:`~sim.match.MatchResult`, or ``None`` in
            ``outcome_only`` mode — nothing was simulated to record.
    """

    round_index: int
    round_label: str
    match_index: int
    slot_a: DrawSlot
    slot_b: DrawSlot
    winner: DrawSlot
    scoreline: str | None = None
    result: MatchResult | None = None

    @property
    def loser(self) -> DrawSlot:
        """The beaten entrant."""
        return (
            self.slot_b
            if self.winner.position == self.slot_a.position
            else self.slot_a
        )

    def involves(self, position: int) -> bool:
        """True if the entrant from bracket ``position`` played this match."""
        return position in (self.slot_a.position, self.slot_b.position)


@dataclass(frozen=True)
class BracketRound:
    """One round of the bracket.

    Attributes:
        index: 0-based round number.
        label: The round's label (``"R32"``, ``"QF"``, …).
        matches: The round's matches, top of the draw first.
    """

    index: int
    label: str
    matches: tuple[BracketMatch, ...]


@dataclass(frozen=True)
class BracketResult:
    """One simulated tournament: every round, every match, and the champion.

    Carries the draw's identity and format alongside the results so a stored
    result is self-describing, and enough per-match detail to reconstruct any
    entrant's path (:meth:`path_of`).

    Attributes:
        tournament_id, name, surface, best_of, final_set_tiebreak, draw_size:
            Copied from the :class:`~sim.draw.Draw` that was simulated.
        rounds: Every round in order.
        champion: The entrant left standing.
        outcome_only: Which mode produced this result — ``True`` means no
            scorelines were generated (see :func:`simulate_bracket`).
    """

    tournament_id: str
    name: str
    surface: str
    best_of: int
    final_set_tiebreak: str
    draw_size: int
    rounds: tuple[BracketRound, ...]
    champion: DrawSlot
    outcome_only: bool

    @property
    def matches(self) -> tuple[BracketMatch, ...]:
        """Every match played, in round then draw order."""
        return tuple(m for r in self.rounds for m in r.matches)

    def path_of(self, position: int) -> tuple[BracketMatch, ...]:
        """Every match the entrant from bracket ``position`` played, in order.

        The run ends with a defeat, or with the final for the champion — so
        ``len(path_of(p))`` is the number of rounds that entrant contested and
        the last match says how their tournament finished.
        """
        return tuple(m for m in self.matches if m.involves(position))


# ---------------------------------------------------------------------------
# Simulation.
# ---------------------------------------------------------------------------


def _reject_placeholders(draw: Draw) -> None:
    """Fail loudly if the draw still has unfilled slots — all of them at once.

    See the module docstring: a placeholder has neither a ``player_id`` nor any
    classifier-visible history, so no reconciled match-win probability exists
    for it.
    """
    placeholders = [slot for slot in draw.bracket if slot.is_placeholder]
    if not placeholders:
        return
    listed = ", ".join(f"{slot.position}:{slot.player!r}" for slot in placeholders)
    raise ValueError(
        f"{draw.tournament_id}: cannot simulate a bracket with "
        f"{len(placeholders)} placeholder entrant(s) — a placeholder has no "
        f"player id and no classifier history, so no reconciled match-win "
        f"probability can be computed for it. Fill these slots with real "
        f"entrants first: {listed}"
    )


def reconciled_win_prob(
    draw: Draw,
    slot_a: DrawSlot,
    slot_b: DrawSlot,
    skill_table: "SkillTable",
    classifier: ClassifierProb,
    mode: str = config.RECONCILE_MODE,
    w: float = config.RECONCILE_BLEND_WEIGHT,
    prob_cache: MutableMapping[tuple, float] | None = None,
) -> float:
    """P(``slot_a`` beats ``slot_b``) under reconciliation — deterministic, no RNG.

    The analytic twin of :func:`~sim.reconcile.simulate_reconciled_match`: the
    same resolve → point model → ``P_clf`` → reconcile → solve ``δ`` chain, but
    it returns the resulting probability instead of simulating a scoreline. It
    composes ``sim/reconcile.py``'s public pieces (``reconciled_prob``,
    ``solve_delta``, ``match_win_prob_point``) rather than reimplementing any of
    the maths.

    Point-win probabilities come from ``Draw.skill_for`` on the ids T2.1 already
    resolved, which agrees with what ``simulate_reconciled_match`` resolves for
    itself — both go through ``SkillTable.resolve_name``.

    Args:
        draw: The draw being simulated (supplies surface and match format).
        slot_a, slot_b: The two entrants; ``slot_a`` is player A.
        skill_table: The id-keyed T1.1 skill table.
        classifier: The injected ``(a, b, surface) -> P_clf`` adapter.
        mode, w: Reconciliation mode / blend weight.
        prob_cache: Optional caller-owned cache — see :func:`simulate_bracket`.

    Returns:
        P(A beats B) in ``[0, 1]``.
    """
    key = (
        slot_a.player,
        slot_b.player,
        draw.surface,
        draw.best_of,
        draw.final_set_tiebreak,
        mode,
        w,
    )
    if prob_cache is not None:
        cached = prob_cache.get(key)
        if cached is not None:
            return cached

    skill_a = draw.skill_for(slot_a, skill_table)
    skill_b = draw.skill_for(slot_b, skill_table)
    pA, pB = matchup_point_probs(skill_a, skill_b, config.SURFACE_MU[draw.surface])

    p_clf = float(classifier(slot_a.player, slot_b.player, draw.surface))
    p_point = (
        match_win_prob_point(pA, pB, draw.best_of, draw.final_set_tiebreak)
        if mode == "blend"
        else 0.0  # unused by classifier_anchor
    )
    target = reconciled_prob(p_clf, p_point, mode, w)
    delta = solve_delta(target, pA, pB, draw.best_of, draw.final_set_tiebreak)

    # solve_delta keeps these inside [P_MIN, P_MAX]; clamp defensively regardless
    # — the same belt-and-braces simulate_reconciled_match applies.
    pA_adj = min(max(pA + delta, config.P_MIN), config.P_MAX)
    pB_adj = min(max(pB - delta, config.P_MIN), config.P_MAX)

    # Re-read the post-δ probability rather than trusting `target`: when the
    # target is beyond δ's reachable window solve_delta saturates, and this is
    # the probability the outcome_only=False path would actually simulate.
    prob = match_win_prob_point(
        pA_adj, pB_adj, draw.best_of, draw.final_set_tiebreak
    )
    if prob_cache is not None:
        prob_cache[key] = prob
    return prob


def _simulate_match(
    draw: Draw,
    slot_a: DrawSlot,
    slot_b: DrawSlot,
    skill_table: "SkillTable",
    classifier: ClassifierProb,
    rng: np.random.Generator,
    outcome_only: bool,
    mode: str,
    w: float,
    prob_cache: MutableMapping[tuple, float] | None,
) -> tuple[DrawSlot, str | None, MatchResult | None]:
    """Play one bracket match; return ``(winner, scoreline, result)``.

    The single place the two modes diverge. ``outcome_only=True`` consumes
    exactly one ``rng`` draw and touches nothing in ``sim/match.py``;
    ``outcome_only=False`` hands the whole match to
    :func:`~sim.reconcile.simulate_reconciled_match` (which does the resolve →
    reconcile → simulate sequence itself — it is not re-assembled here).
    """
    if outcome_only:
        p_a = reconciled_win_prob(
            draw, slot_a, slot_b, skill_table, classifier, mode, w, prob_cache
        )
        a_won = bool(rng.random() < p_a)
        return (slot_a if a_won else slot_b), None, None

    result = simulate_reconciled_match(
        slot_a.player,
        slot_b.player,
        draw.surface,
        draw.best_of,
        draw.final_set_tiebreak,
        skill_table,
        classifier,
        rng,
        mode=mode,
        w=w,
        first_server=BRACKET_FIRST_SERVER,
    )
    winner = slot_a if result.winner == PLAYER_A else slot_b
    return winner, format_scoreline(result), result


def simulate_bracket(
    draw: Draw,
    skill_table: "SkillTable",
    classifier: ClassifierProb,
    rng: np.random.Generator,
    outcome_only: bool = False,
    mode: str = config.RECONCILE_MODE,
    w: float = config.RECONCILE_BLEND_WEIGHT,
    prob_cache: MutableMapping[tuple, float] | None = None,
) -> BracketResult:
    """Simulate one full bracket, round by round, to a single champion.

    A ``draw_size = N`` draw plays ``N − 1`` matches across ``log2(N)`` rounds.
    Winners keep their bracket order into the next round, so the top half of the
    draw meets the bottom half only in the final.

    Args:
        draw: A validated :class:`~sim.draw.Draw` (T2.1). Its ``surface``,
            ``best_of`` and ``final_set_tiebreak`` govern every match.
        skill_table: The id-keyed T1.1 skill table.
        classifier: The injected ``(player_a, player_b, surface) -> P_clf``
            adapter (:class:`~sim.reconcile.ClassifierProb`). This module never
            builds one — see the caching contract below.
        rng: A ``numpy`` ``Generator`` (``numpy.random.default_rng(seed)``,
            never the global ``np.random``). **One** generator is threaded
            through every round and every match, so a given seed replays the
            whole bracket in either mode.
        outcome_only: ``False`` (default) simulates each match point-by-point
            and records a real scoreline. ``True`` decides each match with a
            single Bernoulli draw against the reconciled analytic probability —
            no point/game/set simulation happens at all, and every
            ``scoreline``/``result`` is ``None``.
        mode: Reconciliation mode (``"classifier_anchor"``/``"blend"``);
            defaults to ``config.RECONCILE_MODE``. Read the classifier caveat in
            the module docstring before accepting the default.
        w: Blend weight on ``P_clf`` in ``"blend"`` mode.
        prob_cache: A caller-owned mutable mapping used to memoise reconciled
            match-win probabilities across calls (``outcome_only=True`` only —
            the point-by-point path re-solves δ inside
            ``simulate_reconciled_match``). Optional and ``None`` by default,
            which is correct but uncached.

    Returns:
        A :class:`BracketResult`.

    Raises:
        ValueError: If the draw contains placeholder entrants (see the module
            docstring), or the draw size is not a power of two ≥ 2.

    **Caching contract for the Monte Carlo caller (T2.3).** A Monte Carlo pass
    runs this function thousands of times, and *both* expensive inputs to a
    match are pure functions of ``(entrant A, entrant B, surface)``, so both
    must be memoised **for the whole job**, not per bracket:

    1. **The δ solve** — pass one ``prob_cache`` dict into every
       ``simulate_bracket`` call of the job. Without it a 128 × 5,000 pass
       re-solves ~635,000 bisections (~40 min measured); with it, at most 8,128
       distinct matchups are ever solved. A fresh dict per bracket caches
       nothing, because a bracket meets each matchup once.
    2. **The classifier** — the adapter itself must memoise on
       ``(player_a, player_b, surface)`` and be built **once**, outside the run
       loop (``cli/simulate_match.make_classifier_prob`` shows the pattern).
       Rebuilding it per bracket discards the cache exactly as often as it would
       be used.

    Nothing inside this function defeats either cache: it never rebuilds
    per-match state, and it always presents a pairing in one stable orientation
    (lower bracket position first), so a matchup maps to exactly one cache key
    however many times it recurs.
    """
    _reject_placeholders(draw)

    labels = round_labels(draw.draw_size)
    survivors: list[DrawSlot] = list(draw.bracket)  # position-ordered by T2.1
    rounds: list[BracketRound] = []

    for round_index, label in enumerate(labels):
        matches: list[BracketMatch] = []
        winners: list[DrawSlot] = []
        # Pair survivors 2k-1 vs 2k in bracket order (T2.1's first-round rule,
        # which the preserved ordering extends to every later round). slot_a is
        # therefore always the lower bracket position — the serve-first and
        # cache-orientation convention in the module docstring.
        for match_index in range(len(survivors) // 2):
            slot_a = survivors[2 * match_index]
            slot_b = survivors[2 * match_index + 1]
            winner, scoreline, result = _simulate_match(
                draw,
                slot_a,
                slot_b,
                skill_table,
                classifier,
                rng,
                outcome_only,
                mode,
                w,
                prob_cache,
            )
            matches.append(
                BracketMatch(
                    round_index=round_index,
                    round_label=label,
                    match_index=match_index,
                    slot_a=slot_a,
                    slot_b=slot_b,
                    winner=winner,
                    scoreline=scoreline,
                    result=result,
                )
            )
            winners.append(winner)
        rounds.append(
            BracketRound(index=round_index, label=label, matches=tuple(matches))
        )
        survivors = winners

    return BracketResult(
        tournament_id=draw.tournament_id,
        name=draw.name,
        surface=draw.surface,
        best_of=draw.best_of,
        final_set_tiebreak=draw.final_set_tiebreak,
        draw_size=draw.draw_size,
        rounds=tuple(rounds),
        champion=survivors[0],
        outcome_only=outcome_only,
    )


# ---------------------------------------------------------------------------
# Monte Carlo aggregation (T2.3).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerOutcome:
    """One entrant's aggregated Monte Carlo record.

    Counts are stored, not probabilities: integers make two runs comparable for
    *exact* equality (the T2.3 reproducibility and single-vs-multi-worker
    criteria), and the probabilities are one division away.

    Attributes:
        position: The entrant's 1-based bracket slot.
        player, player_id, seed: Copied from the :class:`~sim.draw.DrawSlot` and
            the draw's seed list (``seed`` is ``None`` for an unseeded entrant).
        n_runs: Simulations behind these counts.
        titles: Runs this entrant won.
        reached: Round label → number of runs in which the entrant *contested*
            that round. Keys are the draw's own labels (see
            :func:`round_labels`), so an 8-draw carries ``QF/SF/F`` and a
            128-draw ``R128 … F`` — nothing assumes a particular draw size.
        matches_won: Total matches won across every run; ``/ n_runs`` gives the
            expected number of rounds won.
    """

    position: int
    player: str
    player_id: str | None
    seed: int | None
    n_runs: int
    titles: int
    reached: Mapping[str, int]
    matches_won: int

    @property
    def p_title(self) -> float:
        """P(this entrant wins the tournament)."""
        return self.titles / self.n_runs

    @property
    def expected_rounds_won(self) -> float:
        """Mean number of matches won — 0 for a first-round exit every time."""
        return self.matches_won / self.n_runs

    def p_reach(self, label: str) -> float:
        """P(this entrant reaches — i.e. plays in — the round called ``label``).

        Unknown labels give ``0.0`` rather than raising, so a caller can ask for
        ``"QF"`` of a draw too small to have one.
        """
        return self.reached.get(label, 0) / self.n_runs

    def to_record(self) -> dict:
        """A flat, tidy dict — one row of the output table.

        Carries a ``p_reach_<label>`` key per round the draw actually has, so
        the record's shape follows the draw rather than a hardcoded QF/SF/F.
        """
        record = {
            "position": self.position,
            "player": self.player,
            "player_id": self.player_id,
            "seed": self.seed,
            "n_runs": self.n_runs,
            "titles": self.titles,
            "p_title": self.p_title,
            "expected_rounds_won": self.expected_rounds_won,
        }
        record.update(
            {f"p_reach_{label}": count / self.n_runs
             for label, count in self.reached.items()}
        )
        return record


@dataclass(frozen=True)
class MonteCarloResult:
    """The aggregate of a whole Monte Carlo job.

    Self-describing in the same way :class:`BracketResult` is — the draw's
    identity and format, plus every knob that could change the numbers (runs,
    seed, reconciliation mode/weight) — so a stored result can be reproduced
    from what it carries. ``workers`` is recorded for provenance only: it
    provably does not affect the counts (see :func:`monte_carlo`).

    Attributes:
        tournament_id, name, surface, best_of, final_set_tiebreak, draw_size:
            Copied from the :class:`~sim.draw.Draw`.
        n_runs, seed, workers, mode, w: The job's parameters.
        round_labels: The draw's round labels, first round first.
        players: Every entrant's :class:`PlayerOutcome`, **sorted by title
            probability descending** (ties broken by bracket position, so the
            order is deterministic).
    """

    tournament_id: str
    name: str
    surface: str
    best_of: int
    final_set_tiebreak: str
    draw_size: int
    n_runs: int
    seed: int
    workers: int
    mode: str
    w: float
    round_labels: tuple[str, ...]
    players: tuple[PlayerOutcome, ...]

    def to_records(self) -> list[dict]:
        """The tidy output: one dict per entrant, title probability descending.

        Ready to hand to a CLI table or ``pandas.DataFrame(...)`` — this module
        stays pandas-free on purpose (it is on the hot path).
        """
        return [player.to_record() for player in self.players]

    def by_position(self) -> dict[int, PlayerOutcome]:
        """The same outcomes keyed by bracket position."""
        return {player.position: player for player in self.players}

    @property
    def champion_probabilities(self) -> dict[str, float]:
        """Entrant name → P(title), in the same sorted order."""
        return {player.player: player.p_title for player in self.players}


# One chunk of a Monte Carlo job: the counts three parallel accumulators build.
# Plain lists of ints rather than arrays — the inner loop does ~2·(N−1) scalar
# increments per run, where a Python list beats numpy element assignment.
_ChunkCounts = tuple[list[int], list[list[int]], list[int]]


def _accumulate(
    result: BracketResult,
    titles: list[int],
    reached: list[list[int]],
    matches_won: list[int],
) -> None:
    """Fold one bracket's outcome into the running counts (in place).

    ``reached[r][p]`` counts entrant at position ``p + 1`` contesting round
    ``r``; every player in a match reached that round, and the winner banks a
    match won.
    """
    for bracket_round in result.rounds:
        row = reached[bracket_round.index]
        for bracket_match in bracket_round.matches:
            row[bracket_match.slot_a.position - 1] += 1
            row[bracket_match.slot_b.position - 1] += 1
            matches_won[bracket_match.winner.position - 1] += 1
    titles[result.champion.position - 1] += 1


def _run_chunk(payload: tuple) -> _ChunkCounts:
    """Simulate one contiguous chunk of runs and return its counts.

    Module-level (not a closure) and single-argument so it can be the target of
    a :class:`~concurrent.futures.ProcessPoolExecutor` ``map``.

    **This is where the cache lives, and the reason it is here.** ``prob_cache``
    is created once per *chunk*:

    * ``workers = 1`` → exactly one chunk → **one dict for the entire job**,
      which is :func:`simulate_bracket`'s documented contract (a per-bracket
      dict would cache nothing, since a bracket meets each matchup once).
    * ``workers > 1`` → one chunk per worker process → each worker builds its
      **own** cache, in its own address space. No attempt is made to share one
      across processes: a plain dict cannot be, and the alternatives (a manager
      proxy) cost more per lookup than the δ solve they would save.
    """
    draw, skill_table, classifier, seed_seqs, mode, w = payload

    n_slots = draw.draw_size
    n_rounds = len(round_labels(n_slots))
    titles = [0] * n_slots
    reached = [[0] * n_slots for _ in range(n_rounds)]
    matches_won = [0] * n_slots

    prob_cache: dict[tuple, float] = {}
    for seed_seq in seed_seqs:
        result = simulate_bracket(
            draw,
            skill_table,
            classifier,
            np.random.default_rng(seed_seq),
            outcome_only=True,  # T2.3: never a scoreline, at any run count.
            mode=mode,
            w=w,
            prob_cache=prob_cache,
        )
        _accumulate(result, titles, reached, matches_won)
    return titles, reached, matches_won


def _split_evenly(items: Sequence, parts: int) -> list[Sequence]:
    """Split ``items`` into at most ``parts`` contiguous, near-equal chunks.

    Contiguity is cosmetic — each run carries its own spawned seed, so the split
    cannot change any result — but it keeps a worker's slice easy to reason
    about. Empty chunks are dropped, so ``parts > len(items)`` simply yields
    fewer chunks.
    """
    n = len(items)
    parts = max(1, min(parts, n))
    size, remainder = divmod(n, parts)
    chunks: list[Sequence] = []
    start = 0
    for index in range(parts):
        stop = start + size + (1 if index < remainder else 0)
        if stop > start:
            chunks.append(items[start:stop])
        start = stop
    return chunks


def _require_picklable(classifier: ClassifierProb) -> None:
    """Fail early and legibly if the classifier cannot cross a process boundary.

    ``workers > 1`` sends the adapter to each worker by pickle. The obvious
    adapter shape — a closure over the estimator and the pipeline's histories,
    which is exactly what ``cli/simulate_match.make_classifier_prob`` returns —
    is *not* picklable, and the raw ``PicklingError`` from deep inside the pool
    says nothing useful about how to fix it.

    **This is a scope boundary, not an inherent limitation.** Nothing about a
    classifier adapter *requires* a closure: wrapping the same estimator and
    histories in a module-level class with a ``__call__`` makes it picklable and
    the multi-worker path works unchanged. T2.3's ticket scopes its diff to this
    module and ``config.py``, so rewriting ``cli/simulate_match.py`` was out of
    bounds; building that adapter belongs to **T2.5** (the tournament CLI, the
    first caller likely to want ``--workers``). Until then this check turns the
    gap into a fast, legible refusal rather than a crash inside a worker.
    """
    try:
        pickle.dumps(classifier)
    except Exception as exc:
        raise ValueError(
            "workers > 1 requires a picklable classifier adapter, and this one "
            f"is not ({type(exc).__name__}: {exc}). A closure (e.g. the callable "
            "returned by cli/simulate_match.make_classifier_prob) cannot cross a "
            "process boundary — wrap the estimator in a module-level class with "
            "a __call__, or run with workers=1."
        ) from exc


def monte_carlo(
    draw: Draw,
    skill_table: "SkillTable",
    classifier: ClassifierProb,
    n_runs: int = config.MC_RUNS,
    seed: int = config.MC_SEED,
    workers: int = 1,
    mode: str = config.RECONCILE_MODE,
    w: float = config.RECONCILE_BLEND_WEIGHT,
) -> MonteCarloResult:
    """Run the bracket ``n_runs`` times and aggregate per-entrant probabilities.

    Every run goes through ``simulate_bracket(..., outcome_only=True)`` — one
    Bernoulli draw per match against the reconciled analytic probability. No run
    generates a scoreline: at 128 × 5,000 that would be 635,000 point-by-point
    matches for information nothing downstream reads. Storybook scorelines are a
    separate, single-run mode (T2.4).

    Args:
        draw: A validated, placeholder-free :class:`~sim.draw.Draw`.
        skill_table: The id-keyed T1.1 skill table.
        classifier: The injected ``(a, b, surface) -> P_clf`` adapter, built
            **once** by the caller (this module never constructs one). With
            ``workers > 1`` it must also be picklable — see below.
        n_runs: Number of bracket simulations (default ``config.MC_RUNS``).
        seed: Base seed; every run's RNG is spawned from it.
        workers: ``1`` (default) runs in this process. ``> 1`` splits the runs
            across that many processes.
        mode: Reconciliation mode — **a pass-through, defaulting to
            ``config.RECONCILE_MODE``**. See the note below before accepting it.
        w: Blend weight on ``P_clf`` in ``"blend"`` mode.

    Returns:
        A :class:`MonteCarloResult` whose ``players`` are sorted by title
        probability, descending.

    Raises:
        ValueError: If ``n_runs``/``workers`` is below 1, the draw contains
            placeholder entrants (rejected up front, before any process is
            spawned), or ``workers > 1`` with an unpicklable classifier.

    **Seeding — reproducible, and independent of ``workers``.**
    ``np.random.SeedSequence(seed).spawn(n_runs)`` gives run *i* its own child
    sequence, so run *i* is the same simulation whichever worker executes it.
    The aggregate is a sum of integer counts, which is associative and
    commutative, so splitting the runs cannot perturb it: ``workers=4`` returns
    counts *identical* to ``workers=1``, not merely close.

    **The ``prob_cache``, and where it lives.** :func:`simulate_bracket`'s
    contract is that one cache must span the whole job, because the δ bisection
    (~3.2 ms) dominates and a 128-draw admits at most 8,128 distinct matchups
    against ~635,000 evaluations. That dict is created inside
    :func:`_run_chunk`, once per chunk: single-threaded there is exactly one
    chunk and therefore exactly one dict for all ``n_runs``; with
    ``workers > 1`` each worker process gets its own fresh dict and no attempt
    is made to share one across processes. The trade is deliberate — W workers
    each pay their own cache-fill, so the speedup is sub-linear, but wall-clock
    still improves once ``n_runs`` per worker is well past the fill. Measured on
    the 128 × 5,000 job: a quarter-size chunk still visits 2,397 of the 3,452
    distinct matchups the whole job visits (69%, not 25%), which is why four
    workers buy far less than 4×.

    **``workers > 1`` needs a picklable classifier, and — on macOS — a
    ``__main__`` guard.** The adapter is sent to each worker by pickle;
    :func:`_require_picklable` refuses up front and explains the fix if it
    cannot be. Separately, ``ProcessPoolExecutor``'s default start method is
    *spawn* on macOS (and Windows), so each worker re-imports the parent's
    ``__main__`` module: **any script or CLI that calls this with
    ``workers > 1`` must guard its entry point with
    ``if __name__ == "__main__":``**, or the re-import will re-run the job in
    every child. Flagged here for T2.5, which is the first caller likely to
    expose a ``--workers`` flag.

    **Reconciliation mode is the caller's decision, not this function's (T2.3
    decision).** ``mode`` defaults to ``config.RECONCILE_MODE``
    (``"classifier_anchor"``), the same default :func:`simulate_bracket` uses,
    and this function deliberately does **not** substitute a "safer" one of its
    own. ``ace-04-current-state.md §7`` seam 7 records that the only adapter
    built so far (``cli/simulate_match.make_classifier_prob``) emits a
    form-blind ``P_clf`` flattened toward 0.5, and that
    ``config.SIM_CLI_RECONCILE_MODE = "blend"`` is a **CLI-scoped mitigation for
    that one adapter, explicitly not a project-wide recommendation**. The defect
    belongs to the adapter, not to the reconciliation default: a caller feeding
    real engineered feature rows should get ``"classifier_anchor"``, and a
    caller reusing the flattened CLI adapter should pass ``mode="blend"`` (and
    say so in whatever it publishes). Baking the workaround in here would push
    a degraded adapter's problem into the core and silently change the model for
    callers that do not have it.
    """
    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1, got {n_runs!r}")
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers!r}")
    # Refuse an unsimulatable draw here rather than inside every worker.
    _reject_placeholders(draw)

    labels = round_labels(draw.draw_size)
    seed_seqs = np.random.SeedSequence(seed).spawn(n_runs)
    chunks = _split_evenly(seed_seqs, workers)

    if len(chunks) == 1:
        counts = [_run_chunk((draw, skill_table, classifier, chunks[0], mode, w))]
    else:
        _require_picklable(classifier)
        payloads = [
            (draw, skill_table, classifier, chunk, mode, w) for chunk in chunks
        ]
        with ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            counts = list(pool.map(_run_chunk, payloads))

    n_slots = draw.draw_size
    titles = [0] * n_slots
    reached = [[0] * n_slots for _ in labels]
    matches_won = [0] * n_slots
    for chunk_titles, chunk_reached, chunk_matches_won in counts:
        for position in range(n_slots):
            titles[position] += chunk_titles[position]
            matches_won[position] += chunk_matches_won[position]
        for round_index, row in enumerate(chunk_reached):
            target = reached[round_index]
            for position in range(n_slots):
                target[position] += row[position]

    players = [
        PlayerOutcome(
            position=slot.position,
            player=slot.player,
            player_id=slot.player_id,
            seed=draw.seed_of(slot),
            n_runs=n_runs,
            titles=titles[slot.position - 1],
            reached={
                label: reached[round_index][slot.position - 1]
                for round_index, label in enumerate(labels)
            },
            matches_won=matches_won[slot.position - 1],
        )
        for slot in draw.bracket
    ]
    players.sort(key=lambda outcome: (-outcome.titles, outcome.position))

    return MonteCarloResult(
        tournament_id=draw.tournament_id,
        name=draw.name,
        surface=draw.surface,
        best_of=draw.best_of,
        final_set_tiebreak=draw.final_set_tiebreak,
        draw_size=draw.draw_size,
        n_runs=n_runs,
        seed=seed,
        workers=workers,
        mode=mode,
        w=w,
        round_labels=labels,
        players=tuple(players),
    )


# ---------------------------------------------------------------------------
# Storybook single run (T2.4).
# ---------------------------------------------------------------------------


def format_match_line(winner: str, loser: str, scoreline: str) -> str:
    """Render one bracket match as ``"A def. B 6-4 3-6 7-6(5) 6-2"``.

    The scoreline is passed in already rendered — :func:`format_scoreline` (used
    by :func:`_simulate_match` and stored on every :class:`BracketMatch`) is the
    single scoreline formatter in this module, and this only wraps names around
    its output.
    """
    return f"{winner} def. {loser} {scoreline}"


@dataclass(frozen=True)
class StorybookMatch:
    """One played-out match, ready to display.

    Attributes:
        round_index, round_label, match_index: Where the match sits in the
            bracket — the same coordinates as :class:`BracketMatch`.
        winner, loser: Display names.
        winner_seed, loser_seed: Their tournament seeds, ``None`` if unseeded.
        scoreline: Winner-first scoreline (``6-4 3-6 7-6(5) 6-2``).
        line: The rendered match line — ``"<winner> def. <loser> <scoreline>"``.
        match: The underlying :class:`BracketMatch`, whose ``result`` carries
            the full set-by-set :class:`~sim.match.MatchResult` for any caller
            that wants more than the line.
    """

    round_index: int
    round_label: str
    match_index: int
    winner: str
    loser: str
    winner_seed: int | None
    loser_seed: int | None
    scoreline: str
    line: str
    match: BracketMatch


@dataclass(frozen=True)
class StorybookRound:
    """One round of a storybook run, top of the draw first."""

    index: int
    label: str
    matches: tuple[StorybookMatch, ...]


@dataclass(frozen=True)
class PlayerRun:
    """One entrant's tournament, summarised — the "run summary" T2.5 renders.

    Attributes:
        position: 1-based bracket slot.
        player, player_id, seed: Identity, copied from the
            :class:`~sim.draw.DrawSlot` and the draw's seed list.
        is_champion: True for the one entrant who won the title.
        furthest_round: Label of the last round this entrant *contested* — so a
            beaten finalist and the champion both read ``"F"``, and a
            first-round loser reads the first round's label.
        matches_won: How many matches they won (``0`` for a first-round exit).
        beat: The opponents they beat, in round order.
        eliminated_by: Who knocked them out; ``None`` **iff** ``is_champion``.
        last_match: The match their run ended on — their defeat, or the final
            for the champion.
    """

    position: int
    player: str
    player_id: str | None
    seed: int | None
    is_champion: bool
    furthest_round: str
    matches_won: int
    beat: tuple[str, ...]
    eliminated_by: str | None
    last_match: StorybookMatch


@dataclass(frozen=True)
class StorybookResult:
    """One fully played-out tournament, in presentation form.

    Self-describing in the same way :class:`MonteCarloResult` is: the draw's
    identity and format plus every knob that could change the story (``seed``,
    ``mode``, ``w``), so a shared result records exactly how to reproduce it.

    Attributes:
        tournament_id, name, surface, best_of, final_set_tiebreak, draw_size:
            Copied from the :class:`~sim.draw.Draw`.
        seed, mode, w: The run's parameters.
        rounds: Every round, first round first.
        champion: The winning :class:`~sim.draw.DrawSlot`.
        champion_seed: The champion's seed, ``None`` if unseeded.
        runs: Every entrant's :class:`PlayerRun`, best finish first.
        bracket: The underlying :class:`BracketResult`, unmodified — nothing
            the simulation produced is discarded by this wrapper.
    """

    tournament_id: str
    name: str
    surface: str
    best_of: int
    final_set_tiebreak: str
    draw_size: int
    seed: int
    mode: str
    w: float
    rounds: tuple[StorybookRound, ...]
    champion: DrawSlot
    champion_seed: int | None
    runs: tuple[PlayerRun, ...]
    bracket: BracketResult

    @property
    def matches(self) -> tuple[StorybookMatch, ...]:
        """Every match, in round then draw order."""
        return tuple(m for r in self.rounds for m in r.matches)

    @property
    def final(self) -> StorybookMatch:
        """The title match."""
        return self.rounds[-1].matches[0]

    def run_of(self, position: int) -> PlayerRun:
        """The :class:`PlayerRun` of the entrant from bracket ``position``."""
        for run in self.runs:
            if run.position == position:
                return run
        raise KeyError(f"no entrant at bracket position {position!r}")


def _storybook_match(draw: Draw, bracket_match: BracketMatch) -> StorybookMatch:
    """Wrap one :class:`BracketMatch` for display.

    Raises:
        ValueError: If the match has no scoreline — i.e. it came from an
            ``outcome_only=True`` bracket, which has no story to tell.
    """
    if bracket_match.scoreline is None:
        raise ValueError(
            f"{bracket_match.round_label} match {bracket_match.match_index} has "
            f"no scoreline: a storybook needs a bracket simulated with "
            f"outcome_only=False"
        )
    winner, loser = bracket_match.winner, bracket_match.loser
    return StorybookMatch(
        round_index=bracket_match.round_index,
        round_label=bracket_match.round_label,
        match_index=bracket_match.match_index,
        winner=winner.player,
        loser=loser.player,
        winner_seed=draw.seed_of(winner),
        loser_seed=draw.seed_of(loser),
        scoreline=bracket_match.scoreline,
        line=format_match_line(winner.player, loser.player, bracket_match.scoreline),
        match=bracket_match,
    )


def _player_run(
    draw: Draw,
    result: BracketResult,
    slot: DrawSlot,
    by_coords: Mapping[tuple[int, int], StorybookMatch],
) -> PlayerRun:
    """Summarise one entrant's tournament from the bracket they played in."""
    path = result.path_of(slot.position)  # ends in a defeat, or the final
    last = path[-1]
    # A run ends the first time the entrant loses, so a run whose last match is
    # a win can only be the champion's.
    is_champion = last.winner.position == slot.position
    beat = tuple(
        m.loser.player for m in path if m.winner.position == slot.position
    )
    eliminated_by = None if is_champion else last.winner.player
    return PlayerRun(
        position=slot.position,
        player=slot.player,
        player_id=slot.player_id,
        seed=draw.seed_of(slot),
        is_champion=is_champion,
        furthest_round=last.round_label,
        matches_won=len(beat),
        beat=beat,
        eliminated_by=eliminated_by,
        last_match=by_coords[(last.round_index, last.match_index)],
    )


def storybook_run(
    draw: Draw,
    skill_table: "SkillTable",
    classifier: ClassifierProb,
    seed: int,
    mode: str = config.RECONCILE_MODE,
    w: float = config.RECONCILE_BLEND_WEIGHT,
) -> StorybookResult:
    """Play one full tournament, every match point by point, and wrap it up.

    Exactly **one** :func:`simulate_bracket` call with ``outcome_only=False``:
    a ``draw_size = N`` draw is ``N − 1`` real matches (127 for a Slam), which
    is cheap in absolute terms — the Monte Carlo budget only bites because it
    multiplies that by thousands of runs.

    Args:
        draw: A validated, placeholder-free :class:`~sim.draw.Draw`.
        skill_table: The id-keyed T1.1 skill table.
        classifier: The injected ``(a, b, surface) -> P_clf`` adapter; this
            module never builds one.
        seed: Base seed. One ``numpy.random.default_rng(seed)`` is built here
            and threaded through the whole bracket by :func:`simulate_bracket`,
            so the same seed replays the same tournament — scoreline for
            scoreline — which is what makes a shared storybook link meaningful.
        mode: Reconciliation mode — a pass-through, see the note below.
        w: Blend weight on ``P_clf`` in ``"blend"`` mode.

    Returns:
        A :class:`StorybookResult`. Render it with :func:`render_storybook`, or
        consume the structure directly (API/frontend).

    Raises:
        ValueError: If the draw contains placeholder entrants (raised by
            :func:`simulate_bracket`).

    **Reconciliation mode is still the caller's decision (T2.4 decision).**
    ``mode`` defaults to ``config.RECONCILE_MODE``, exactly as
    :func:`simulate_bracket` and :func:`monte_carlo` do, and this function does
    **not** substitute ``"blend"`` of its own accord — even though a storybook
    is the first *shareable* output in the project and its readability depends
    directly on the reconciled probabilities not collapsing toward a coin flip.
    The reasoning is the same as T2.3's and it does not weaken here: the flat,
    form-blind ``P_clf`` is a defect of one adapter
    (``cli/simulate_match.make_classifier_prob``, ``ace-04-current-state.md``
    §7 seam 7), not of the reconciliation default, and a caller that feeds real
    engineered feature rows wants ``"classifier_anchor"`` — silently diluting
    its classifier by half to protect a different caller's bad rows would be
    the core making a modelling decision on evidence it does not have. What
    changes for a *showcase* run is only the cost of getting it wrong, and that
    is addressed where the knowledge lives: the layer that builds the adapter
    picks the mode (T2.5 should pass ``config.SIM_CLI_RECONCILE_MODE`` while it
    reuses the CLI adapter), and :class:`StorybookResult` records ``mode``/``w``
    so any published story states which model produced it.
    """
    rng = np.random.default_rng(seed)
    bracket = simulate_bracket(
        draw,
        skill_table,
        classifier,
        rng,
        outcome_only=False,  # T2.4: the whole point — every match gets a scoreline.
        mode=mode,
        w=w,
    )

    rounds = tuple(
        StorybookRound(
            index=bracket_round.index,
            label=bracket_round.label,
            matches=tuple(
                _storybook_match(draw, m) for m in bracket_round.matches
            ),
        )
        for bracket_round in bracket.rounds
    )
    by_coords = {
        (m.round_index, m.match_index): m for r in rounds for m in r.matches
    }

    runs = [_player_run(draw, bracket, slot, by_coords) for slot in draw.bracket]
    # Best finish first: matches won orders the field by exit round exactly
    # (the champion wins one more than the finalist, and so on down), with
    # bracket position breaking ties so the order is deterministic.
    runs.sort(key=lambda run: (-run.matches_won, run.position))

    return StorybookResult(
        tournament_id=draw.tournament_id,
        name=draw.name,
        surface=draw.surface,
        best_of=draw.best_of,
        final_set_tiebreak=draw.final_set_tiebreak,
        draw_size=draw.draw_size,
        seed=seed,
        mode=mode,
        w=w,
        rounds=rounds,
        champion=bracket.champion,
        champion_seed=draw.seed_of(bracket.champion),
        runs=tuple(runs),
        bracket=bracket,
    )


def render_storybook(result: StorybookResult) -> str:
    """Render a :class:`StorybookResult` as a round-by-round text summary.

    A header naming the tournament and the run's parameters, then one block per
    round with a line per match, ending with the champion. Presentation only —
    every fact comes from :class:`StorybookResult`'s public fields, so the API
    and frontend can consume that structure and lay it out differently without
    reimplementing anything this function knows.

    Args:
        result: A completed storybook run.

    Returns:
        The rendered story, newline-separated and without a trailing newline.
    """
    reconciliation = result.mode
    if result.mode == "blend":
        reconciliation += f" (w={result.w:g})"
    lines = [
        f"{result.name} — {result.surface}, best of {result.best_of}, "
        f"{result.draw_size} entrants",
        f"seed {result.seed} · reconciliation {reconciliation}",
    ]

    for storybook_round in result.rounds:
        lines.append("")
        lines.append(storybook_round.label)
        lines.extend(f"  {m.line}" for m in storybook_round.matches)

    champion_seed = (
        f" [{result.champion_seed}]" if result.champion_seed is not None else ""
    )
    lines.append("")
    lines.append(f"Champion: {result.champion.player}{champion_seed}")
    return "\n".join(lines)
