"""Aggregate one matchup over many live simulations (single-match view).

The tournament path answers "who wins the draw"; this module answers the
betting-relevant questions a single matchup raises — how often each player wins,
what the set score tends to be, and how many games get played — by simulating
**one** matchup thousands of times and tallying the point-by-point results.

**Why this can run live when a Monte Carlo draw cannot.** ``/simulate`` is
cache-only (``ace-04-current-state.md``, T3.3) because a 128-draw aggregate is
5,000 brackets × 127 matches ≈ 635k match evaluations. A single match is a
different cost class entirely: there is exactly **one** matchup, so the
classifier is priced once, the serve shift ``δ`` is solved **once**
(:func:`~sim.reconcile.solve_reconciled_serve_probs`), and each of the ``n_runs``
draws is only a point-by-point scoreline over already-adjusted probabilities.
Measured at ~0.08 ms per run — 3,000 runs in ~0.24 s, cheaper than one live
``/storybook`` bracket. That is what makes a live single-match endpoint a
deliberate, bounded exception to T3.3's rule rather than a violation of it.

**What falls out for free, and what does not.** :class:`~sim.match.MatchResult`
carries every :class:`~sim.match.SetResult` (its winner and game score), so the
set-score distribution and the total-games distribution are pure tallies of what
the existing simulator already produces. **Hold/break rates do not** — a set
records only game *counts*, not which games were service holds versus breaks, so
reporting them would need new instrumentation inside ``sim/match.py``. That is
left as a follow-up rather than built here.

This module belongs to the ``sim/`` core: it composes the public pieces of
``sim.reconcile`` and ``sim.match`` and imports nothing from ``cli/`` or ``api/``.
Determinism is by ``seed`` — the same seed replays the same aggregate exactly, so
a shared single-match URL reproduces byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

import config
from sim.match import simulate_match_bo3, simulate_match_bo5
from sim.reconcile import (
    ClassifierProb,
    match_win_prob_point,
    solve_reconciled_serve_probs,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from features.serve import SkillTable


@dataclass(frozen=True)
class SetScoreOutcome:
    """One observed final set tally and how often it happened.

    Oriented to A/B (not winner/loser), so ``(3, 1)`` and ``(1, 3)`` are distinct
    entries — a client that wants the by-winner grouping can fold them, but the
    joint is the honest thing to report.

    Attributes:
        sets_a: Sets player A won in the final scoreline.
        sets_b: Sets player B won.
        count: Simulations that ended on this set tally.
        prob: ``count / n_runs``.
    """

    sets_a: int
    sets_b: int
    count: int
    prob: float


@dataclass(frozen=True)
class TotalGames:
    """The total-games (over/under) summary across every simulated match.

    Attributes:
        mean: Mean total games (both players, all sets).
        std: Population standard deviation of the total.
        minimum: Fewest total games seen.
        maximum: Most total games seen.
        distribution: ``(total_games, count)`` pairs, ascending — the full
            histogram, so a client can draw the spread rather than only quote the
            mean.
    """

    mean: float
    std: float
    minimum: int
    maximum: int
    distribution: list[tuple[int, int]]


@dataclass(frozen=True)
class SingleMatchResult:
    """Everything the single-match view reports for one matchup.

    Identity + format, the reconciled model numbers behind the estimate, and the
    three free distributions (win probability, set score, total games). ``seed``
    and ``n_runs`` are carried so the result is reproducible and self-describing.
    """

    player_a: str
    player_b: str
    surface: str
    best_of: int
    final_set_rule: str
    n_runs: int
    seed: int
    mode: str
    w: float
    # Model numbers (deterministic, not sampled).
    p_clf: float
    p_point: float
    target: float
    analytic_win_prob_a: float
    # Sampled outcomes.
    wins_a: int
    wins_b: int
    win_prob_a: float
    win_prob_b: float
    set_scores: list[SetScoreOutcome]
    total_games: TotalGames


def simulate_single_match(
    player_a: str,
    player_b: str,
    surface: str,
    skill_table: "SkillTable",
    classifier: ClassifierProb,
    *,
    best_of: int = config.SIM_CLI_BEST_OF,
    final_set_rule: str = config.SIM_CLI_FINAL_SET_RULE,
    n_runs: int = config.API_MATCH_SIM_RUNS,
    seed: int = config.MC_SEED,
    mode: str = config.SIM_CLI_RECONCILE_MODE,
    w: float = config.RECONCILE_BLEND_WEIGHT,
) -> SingleMatchResult:
    """Simulate one matchup ``n_runs`` times and aggregate the outcomes.

    Solves the reconciliation once (:func:`~sim.reconcile.solve_reconciled_serve_probs`),
    then draws ``n_runs`` point-by-point scorelines with the adjusted serve
    probabilities, threading a single seeded ``Generator`` through every draw so
    the whole aggregate is reproducible by ``seed``. The opening server of each
    draw is a fresh coin flip from that generator, mirroring
    :func:`~sim.reconcile.simulate_reconciled_match`.

    Args:
        player_a, player_b: Canonical display names (already resolved). Player A is
            index ``0``; ``win_prob_a`` is P(A wins).
        surface: ``"Hard"``/``"Clay"``/``"Grass"``.
        skill_table: The id-keyed serve/return skill table.
        classifier: The :class:`~sim.reconcile.ClassifierProb` adapter.
        best_of: ``3`` or ``5``.
        final_set_rule: ``"7pt_at_6_6"``/``"10pt_at_6_6"``/``"advantage"``.
        n_runs: Simulations behind the distributions.
        seed: Seed for ``numpy.random.default_rng`` — the same seed replays the
            same aggregate exactly.
        mode, w: Reconciliation mode / blend weight.

    Returns:
        A :class:`SingleMatchResult`.

    Raises:
        ValueError: If either name fails to resolve, the surface is unknown, or
            ``best_of`` is not 3/5.
    """
    if n_runs <= 0:
        raise ValueError(f"n_runs must be positive, got {n_runs!r}")

    reconciled = solve_reconciled_serve_probs(
        player_a, player_b, surface, best_of, final_set_rule, skill_table, classifier,
        mode, w,
    )
    pA_adj, pB_adj = reconciled.pA_adj, reconciled.pB_adj

    if best_of == 3:
        simulate = simulate_match_bo3
    elif best_of == 5:
        simulate = simulate_match_bo5
    else:
        raise ValueError(f"best_of must be 3 or 5, got {best_of!r}")

    rng = np.random.default_rng(seed)
    wins_a = 0
    set_score_counts: dict[tuple[int, int], int] = {}
    total_games_counts: dict[int, int] = {}
    for _ in range(n_runs):
        first_server = int(rng.integers(2))
        result = simulate(pA_adj, pB_adj, first_server, final_set_rule, rng)
        if result.winner == 0:
            wins_a += 1
        sets_a = sum(1 for s in result.sets if s.winner == 0)
        sets_b = len(result.sets) - sets_a
        set_score_counts[(sets_a, sets_b)] = set_score_counts.get((sets_a, sets_b), 0) + 1
        total = sum(s.games_a + s.games_b for s in result.sets)
        total_games_counts[total] = total_games_counts.get(total, 0) + 1

    wins_b = n_runs - wins_a
    set_scores = [
        SetScoreOutcome(sets_a=a, sets_b=b, count=count, prob=count / n_runs)
        # Sort by the winner's margin then orientation, so 3-0, 3-1, 3-2, 0-3, …
        # read in a natural order for a table.
        for (a, b), count in sorted(
            set_score_counts.items(), key=lambda kv: (-kv[0][0], kv[0][1])
        )
    ]

    totals = np.array(
        [t for t, c in total_games_counts.items() for _ in range(c)], dtype=float
    )
    total_games = TotalGames(
        mean=float(totals.mean()),
        std=float(totals.std()),  # population sd (ddof=0), matching np default
        minimum=int(totals.min()),
        maximum=int(totals.max()),
        distribution=sorted(total_games_counts.items()),
    )

    return SingleMatchResult(
        player_a=player_a,
        player_b=player_b,
        surface=surface,
        best_of=best_of,
        final_set_rule=final_set_rule,
        n_runs=n_runs,
        seed=seed,
        mode=mode,
        w=w,
        p_clf=reconciled.p_clf,
        p_point=reconciled.p_point,
        target=reconciled.target,
        analytic_win_prob_a=match_win_prob_point(
            pA_adj, pB_adj, best_of, final_set_rule
        ),
        wins_a=wins_a,
        wins_b=wins_b,
        win_prob_a=wins_a / n_runs,
        win_prob_b=wins_b / n_runs,
        set_scores=set_scores,
        total_games=total_games,
    )
