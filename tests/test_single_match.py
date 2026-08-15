"""Tests for sim/single_match.py — the single-match live aggregate.

Reuses the tiny 2-player skill table + constant-classifier stubs from the
reconciliation tests, so these exercise the aggregation logic (counting,
determinism, distribution shape) without loading the vendored data.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

import config
import sim.single_match as SM
from features.serve import PlayerSkill, SkillTable
from sim.reconcile import solve_reconciled_serve_probs
from sim.single_match import (
    SingleMatchResult,
    SetScoreOutcome,
    TotalGames,
    simulate_single_match,
)


def make_skill_table() -> SkillTable:
    """A tiny 2-player id-keyed skill table (A slightly stronger than B)."""
    mu = dict(config.SURFACE_MU)
    skills = {
        ("A", "Hard"): PlayerSkill(spw=0.66, rpw=0.40, n_serve_pts=5000, n_return_pts=5000),
        ("B", "Hard"): PlayerSkill(spw=0.64, rpw=0.38, n_serve_pts=5000, n_return_pts=5000),
    }
    name_to_id = {"Player A": "A", "Player B": "B"}
    return SkillTable(skills, name_to_id, mu, cutoff=None)


def const_classifier(p: float):
    """A ClassifierProb stub that always returns P(A beats B) = p."""

    def _clf(player_a: str, player_b: str, surface: str) -> float:
        return p

    return _clf


def _run(seed=7, n_runs=400, best_of=5, mode="blend", p_clf=0.62, **kw):
    return simulate_single_match(
        "Player A",
        "Player B",
        "Hard",
        make_skill_table(),
        const_classifier(p_clf),
        best_of=best_of,
        final_set_rule="10pt_at_6_6" if best_of == 5 else "7pt_at_6_6",
        n_runs=n_runs,
        seed=seed,
        mode=mode,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Shape and internal consistency
# --------------------------------------------------------------------------- #
def test_returns_single_match_result():
    result = _run()
    assert isinstance(result, SingleMatchResult)
    assert result.player_a == "Player A"
    assert result.player_b == "Player B"
    assert result.best_of == 5


def test_win_counts_partition_the_runs():
    result = _run(n_runs=500)
    assert result.wins_a + result.wins_b == 500
    assert result.win_prob_a == pytest.approx(result.wins_a / 500)
    assert result.win_prob_b == pytest.approx(1.0 - result.win_prob_a)


def test_set_score_counts_sum_to_runs_and_are_oriented():
    result = _run(n_runs=500, best_of=5)
    assert sum(o.count for o in result.set_scores) == 500
    assert all(isinstance(o, SetScoreOutcome) for o in result.set_scores)
    for o in result.set_scores:
        # Exactly one player reaches 3 sets in a best-of-5.
        assert max(o.sets_a, o.sets_b) == 3
        assert min(o.sets_a, o.sets_b) in (0, 1, 2)
        assert o.prob == pytest.approx(o.count / 500)


def test_set_scores_sorted_most_decisive_first():
    result = _run(n_runs=500, best_of=5)
    # A-wins tallies (sets_a == 3) come before B-wins, and within a winner the
    # margin narrows (3-0 before 3-1 before 3-2).
    keys = [(o.sets_a, o.sets_b) for o in result.set_scores]
    assert keys == sorted(keys, key=lambda k: (-k[0], k[1]))


def test_total_games_summary_is_consistent():
    result = _run(n_runs=500)
    tg = result.total_games
    assert isinstance(tg, TotalGames)
    assert sum(count for _, count in tg.distribution) == 500
    assert tg.minimum == min(t for t, _ in tg.distribution)
    assert tg.maximum == max(t for t, _ in tg.distribution)
    # Mean lies inside the observed range.
    assert tg.minimum <= tg.mean <= tg.maximum
    # A best-of-5 cannot have fewer than 3 sets worth of games.
    assert tg.minimum >= 3 * 6


# --------------------------------------------------------------------------- #
# Determinism — a shared seed must replay identically
# --------------------------------------------------------------------------- #
def test_same_seed_is_identical():
    a = _run(seed=11)
    b = _run(seed=11)
    assert a == b


def test_different_seeds_differ():
    a = _run(seed=1)
    b = _run(seed=2)
    assert (a.wins_a, [o.count for o in a.set_scores]) != (
        b.wins_a,
        [o.count for o in b.set_scores],
    )


# --------------------------------------------------------------------------- #
# The sampled win rate agrees with the exact analytic probability
# --------------------------------------------------------------------------- #
def test_sampled_win_prob_matches_analytic():
    # Many runs so the binomial noise is small; the analytic figure is the exact
    # composed probability the sampling estimates.
    result = _run(seed=3, n_runs=4000, best_of=5, p_clf=0.7)
    assert result.win_prob_a == pytest.approx(result.analytic_win_prob_a, abs=0.03)
    # In blend mode the target is a mix of p_clf and the point model.
    assert result.p_clf == pytest.approx(0.7)


def test_classifier_anchor_target_is_p_clf():
    result = _run(mode="classifier_anchor", p_clf=0.66, n_runs=2000)
    # The reconciled target is p_clf exactly, and the analytic win prob solved to
    # reproduce it should be very close.
    assert result.target == pytest.approx(0.66)
    assert result.analytic_win_prob_a == pytest.approx(0.66, abs=0.02)


# --------------------------------------------------------------------------- #
# Failure modes — fail loudly, never a silent default
# --------------------------------------------------------------------------- #
def test_unknown_player_raises():
    with pytest.raises(ValueError):
        simulate_single_match(
            "Nobody Here",
            "Player B",
            "Hard",
            make_skill_table(),
            const_classifier(0.5),
            best_of=5,
            n_runs=10,
            seed=0,
        )


def test_bad_best_of_raises():
    with pytest.raises(ValueError):
        simulate_single_match(
            "Player A",
            "Player B",
            "Hard",
            make_skill_table(),
            const_classifier(0.5),
            best_of=4,
            n_runs=10,
            seed=0,
        )


def test_non_positive_runs_raises():
    with pytest.raises(ValueError):
        _run(n_runs=0)


# --------------------------------------------------------------------------- #
# The extracted reconciliation helper
# --------------------------------------------------------------------------- #
def test_solve_reconciled_serve_probs_stays_in_bounds():
    reconciled = solve_reconciled_serve_probs(
        "Player A", "Player B", "Hard", 5, "10pt_at_6_6",
        make_skill_table(), const_classifier(0.9), mode="classifier_anchor",
    )
    assert config.P_MIN <= reconciled.pA_adj <= config.P_MAX
    assert config.P_MIN <= reconciled.pB_adj <= config.P_MAX
    assert reconciled.target == pytest.approx(0.9)


def test_solves_reconciliation_once_per_aggregate_not_per_run():
    # The whole point of the solve_reconciled_serve_probs extraction: δ is a ~3 ms
    # bisection, so simulate_single_match must solve it ONCE and loop only the
    # scoreline draw — not re-solve on every simulated match. A per-run solve is
    # correctness-preserving (identical results, just ~40x slower), so only a
    # call-count assertion catches that regression. Same spy pattern as
    # test_reconcile.py::test_simulate_reconciled_match_resolves_each_name_once.
    with mock.patch.object(
        SM, "solve_reconciled_serve_probs", wraps=solve_reconciled_serve_probs
    ) as spy:
        _run(n_runs=50)
    # Once for the aggregate, regardless of n_runs — not once per match.
    assert spy.call_count == 1


def test_solve_reconciled_serve_probs_makes_no_rng_draws(monkeypatch):
    # Pure model composition — it must not touch the global RNG.
    def _boom(*args, **kwargs):
        raise AssertionError("solve_reconciled_serve_probs must not use np.random")

    monkeypatch.setattr(np.random, "default_rng", _boom)
    reconciled = solve_reconciled_serve_probs(
        "Player A", "Player B", "Hard", 3, "7pt_at_6_6",
        make_skill_table(), const_classifier(0.6), mode="blend",
    )
    assert 0.0 <= reconciled.target <= 1.0
