"""Single bracket simulation (T2.2).

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

Like every other ``sim/`` module this one never imports from ``cli/``: its
project imports are ``config``, ``features.serve`` (typing), ``sim.draw``,
``sim.match`` (result types only) and ``sim.reconcile``.
"""

from __future__ import annotations

from collections.abc import MutableMapping
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
