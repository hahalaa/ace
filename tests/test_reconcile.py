"""Tests for sim/reconcile.py, the point-model/classifier reconciliation.

Covers the ticket's acceptance + testing criteria:
  * match_win_prob_point is deterministic and makes NO RNG draws (the property
    the simulator's outcome-only performance budget depends on).
  * The analytic composition agrees with a point-by-point MC estimate.
  * Both reconciliation modes are implemented and config-selectable.
  * The δ-solver is numerically stable, monotone, saturates gracefully, and keeps
    the adjusted probabilities inside [P_MIN, P_MAX].
  * classifier_anchor reproduces the target P as the *simulated* match-win rate.
  * simulate_reconciled_match resolves names→ids exactly once and its winner
    distribution matches the reconciled probability.
  * A name that does not resolve raises loudly.
"""

from __future__ import annotations

import inspect
from unittest import mock

import numpy as np
import pytest

import config
from features.serve import PlayerSkill, SkillTable
from sim.match import MatchResult
import sim.reconcile as R


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
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


def _simulated_win_rate(pA, pB, best_of, rule, n, seed) -> float:
    """Fraction of A wins over n seeded point-by-point matches."""
    rng = np.random.default_rng(seed)
    sim = R.simulate_match_bo3 if best_of == 3 else R.simulate_match_bo5
    wins = 0
    for _ in range(n):
        first_server = int(rng.integers(2))
        if sim(pA, pB, first_server, rule, rng).winner == 0:
            wins += 1
    return wins / n


# --------------------------------------------------------------------------- #
# match_win_prob_point, deterministic, no RNG (the hot-path property)
# --------------------------------------------------------------------------- #
def test_match_win_prob_point_has_no_rng_parameter():
    # The signature itself must not accept an rng, it cannot draw randomness.
    params = inspect.signature(R.match_win_prob_point).parameters
    assert "rng" not in params
    assert "seed" not in params


def test_match_win_prob_point_is_deterministic():
    a = R.match_win_prob_point(0.64, 0.60, 5, "10pt_at_6_6")
    b = R.match_win_prob_point(0.64, 0.60, 5, "10pt_at_6_6")
    assert a == b


def test_match_win_prob_point_makes_no_global_rng_draws(monkeypatch):
    # Poison every global-np.random entry point; if the analytic path touched the
    # global RNG at all, this would raise. It must not.
    def _boom(*args, **kwargs):
        raise AssertionError("match_win_prob_point must not use the global np.random")

    monkeypatch.setattr(np.random, "default_rng", _boom)
    monkeypatch.setattr(np.random, "random", _boom)
    monkeypatch.setattr(np.random, "rand", _boom)
    val = R.match_win_prob_point(0.66, 0.58, 3, "7pt_at_6_6")
    assert 0.0 <= val <= 1.0


def test_match_win_prob_point_symmetry():
    # Equal players → 0.5 regardless of format/rule.
    for best_of, rule in [(3, "7pt_at_6_6"), (5, "10pt_at_6_6"), (5, "advantage")]:
        assert R.match_win_prob_point(0.62, 0.62, best_of, rule) == pytest.approx(0.5, abs=1e-9)


def test_match_win_prob_point_validates_inputs():
    with pytest.raises(ValueError):
        R.match_win_prob_point(0.6, 0.6, 4, "7pt_at_6_6")  # bad best_of
    with pytest.raises(ValueError):
        R.match_win_prob_point(0.6, 0.6, 3, "nonsense")  # bad rule
    with pytest.raises(ValueError):
        R.match_win_prob_point(1.0, 0.6, 3, "7pt_at_6_6")  # p out of (0,1) via hold_prob


@pytest.mark.parametrize(
    "pA,pB,best_of,rule",
    [
        (0.64, 0.60, 5, "10pt_at_6_6"),
        (0.66, 0.58, 3, "7pt_at_6_6"),
        (0.62, 0.62, 5, "advantage"),
        (0.70, 0.56, 5, "10pt_at_6_6"),
    ],
)
def test_analytic_matches_mc(pA, pB, best_of, rule):
    # Cross-validate the closed-form composition against the point-by-point sim.
    analytic = R.match_win_prob_point(pA, pB, best_of, rule)
    rng = np.random.default_rng(0)
    mc = R.match_win_prob_point_mc(pA, pB, best_of, rule, 30000, rng)
    assert analytic == pytest.approx(mc, abs=0.02)


def test_match_win_prob_point_monotone_in_delta():
    pA, pB = 0.6377, 0.5977
    prev = -1.0
    for delta in np.linspace(-0.05, 0.05, 11):
        p = R.match_win_prob_point(pA + delta, pB - delta, 3, "7pt_at_6_6")
        assert p > prev  # strictly increasing in δ
        prev = p


# --------------------------------------------------------------------------- #
# δ-solver
# --------------------------------------------------------------------------- #
def test_solve_delta_reproduces_target_analytically():
    pA, pB = 0.6377, 0.5977
    for target in (0.55, 0.65, 0.75, 0.40):
        delta = R.solve_delta(target, pA, pB, 3, "7pt_at_6_6")
        got = R.match_win_prob_point(pA + delta, pB - delta, 3, "7pt_at_6_6")
        assert got == pytest.approx(target, abs=2 * config.RECONCILE_DELTA_TOL)


def test_solve_delta_keeps_probs_in_bounds():
    pA, pB = 0.6377, 0.5977
    for target in (0.001, 0.3, 0.7, 0.999):
        delta = R.solve_delta(target, pA, pB, 5, "10pt_at_6_6")
        assert config.P_MIN <= pA + delta <= config.P_MAX
        assert config.P_MIN <= pB - delta <= config.P_MAX


def test_solve_delta_saturates_beyond_reach():
    pA, pB = 0.6377, 0.5977
    d_hi = min(config.P_MAX - pA, pB - config.P_MIN)
    d_lo = max(config.P_MIN - pA, pB - config.P_MAX)
    # An unreachable-high target saturates at the top of the window...
    assert R.solve_delta(0.999, pA, pB, 3, "7pt_at_6_6") == pytest.approx(d_hi)
    # ...and an unreachable-low target at the bottom.
    assert R.solve_delta(0.001, pA, pB, 3, "7pt_at_6_6") == pytest.approx(d_lo)


def test_solve_delta_degenerate_window_returns_zero():
    # Both probs pinned to P_MAX → no symmetric room to move → δ = 0.
    delta = R.solve_delta(0.9, config.P_MAX, config.P_MAX, 3, "7pt_at_6_6")
    assert delta == 0.0


# --------------------------------------------------------------------------- #
# reconciliation modes
# --------------------------------------------------------------------------- #
def test_reconciled_prob_classifier_anchor():
    assert R.reconciled_prob(0.72, 0.60, mode="classifier_anchor") == 0.72


def test_reconciled_prob_blend():
    assert R.reconciled_prob(0.72, 0.60, mode="blend", w=0.5) == pytest.approx(0.66)
    assert R.reconciled_prob(0.72, 0.60, mode="blend", w=0.25) == pytest.approx(0.63)


def test_reconciled_prob_rejects_unknown_mode():
    with pytest.raises(ValueError):
        R.reconciled_prob(0.6, 0.6, mode="average")


# --------------------------------------------------------------------------- #
# simulate_reconciled_match
# --------------------------------------------------------------------------- #
def test_simulate_reconciled_match_returns_matchresult():
    table = make_skill_table()
    rng = np.random.default_rng(3)
    res = R.simulate_reconciled_match(
        "Player A", "Player B", "Hard", 5, "10pt_at_6_6",
        table, const_classifier(0.7), rng,
    )
    assert isinstance(res, MatchResult)
    assert res.best_of == 5
    assert res.winner in (0, 1)
    assert len(res.sets) >= 3


def test_classifier_anchor_reproduces_target_as_simulated_rate():
    # THE headline acceptance criterion: solve δ against P_clf, then the *simulated*
    # match-win rate must reproduce P_clf (loose tolerance / seeded).
    table = make_skill_table()
    target = 0.70
    rng = np.random.default_rng(7)
    n = 8000
    wins = 0
    for _ in range(n):
        res = R.simulate_reconciled_match(
            "Player A", "Player B", "Hard", 3, "7pt_at_6_6",
            table, const_classifier(target), rng, mode="classifier_anchor",
        )
        if res.winner == 0:
            wins += 1
    assert wins / n == pytest.approx(target, abs=0.03)


def test_blend_mode_reproduces_blended_rate():
    # blend mode is config-selectable and its winner distribution matches the
    # blended probability w·P_clf + (1−w)·P_point.
    table = make_skill_table()
    p_clf = 0.85
    # derive the expected blended target the same way the function does
    skill_a = table.get("A", "Hard")
    skill_b = table.get("B", "Hard")
    from sim.points import matchup_point_probs

    pA, pB = matchup_point_probs(skill_a, skill_b, config.SURFACE_MU["Hard"])
    p_point = R.match_win_prob_point(pA, pB, 3, "7pt_at_6_6")
    expected = 0.5 * p_clf + 0.5 * p_point

    rng = np.random.default_rng(11)
    n = 8000
    wins = 0
    for _ in range(n):
        res = R.simulate_reconciled_match(
            "Player A", "Player B", "Hard", 3, "7pt_at_6_6",
            table, const_classifier(p_clf), rng, mode="blend", w=0.5,
        )
        if res.winner == 0:
            wins += 1
    assert wins / n == pytest.approx(expected, abs=0.03)


def test_simulate_reconciled_match_resolves_each_name_once():
    table = make_skill_table()
    rng = np.random.default_rng(1)
    with mock.patch.object(table, "resolve_name", wraps=table.resolve_name) as spy:
        R.simulate_reconciled_match(
            "Player A", "Player B", "Hard", 3, "7pt_at_6_6",
            table, const_classifier(0.6), rng,
        )
    # Exactly two resolutions total: one per player, once each (not per set/point).
    assert spy.call_count == 2
    assert {c.args[0] for c in spy.call_args_list} == {"Player A", "Player B"}


def test_unresolved_player_a_raises():
    table = make_skill_table()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Nobody"):
        R.simulate_reconciled_match(
            "Nobody", "Player B", "Hard", 3, "7pt_at_6_6",
            table, const_classifier(0.6), rng,
        )


def test_unresolved_player_b_raises():
    table = make_skill_table()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Ghost"):
        R.simulate_reconciled_match(
            "Player A", "Ghost", "Hard", 3, "7pt_at_6_6",
            table, const_classifier(0.6), rng,
        )


def test_unknown_surface_raises():
    table = make_skill_table()
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="surface"):
        R.simulate_reconciled_match(
            "Player A", "Player B", "Carpet", 3, "7pt_at_6_6",
            table, const_classifier(0.6), rng,
        )


# --------------------------------------------------------------------------- #
# first_server (a backwards-compatible addition to this reconciliation entry point)
# --------------------------------------------------------------------------- #
# `first_server=None` must behave exactly as the earlier code did, same result
# *and* same RNG stream position, because every existing caller relies on it.
# The pinned branch must consume no draw at all, which is what makes that true.
# These tests exist to pin that guarantee: without them, removing the rng draw,
# adding one to the pinned branch, or ignoring the caller's value all pass.


class _IntegersSpy:
    """Wraps a Generator, recording only the ``integers`` calls made through it."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng
        self.calls: list[tuple] = []

    def integers(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._rng.integers(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._rng, name)


def test_first_server_none_draws_exactly_one_integer_from_rng():
    """The default branch draws the opening server from the threaded RNG."""
    table = make_skill_table()
    spy = _IntegersSpy(np.random.default_rng(7))

    R.simulate_reconciled_match(
        "Player A", "Player B", "Hard", 5, "10pt_at_6_6",
        table, const_classifier(0.6), spy,
    )

    assert spy.calls == [((2,), {})]


def test_pinned_first_server_consumes_no_integers_draw():
    """The new branch must NOT draw, that is what keeps reconciliation's callers intact."""
    table = make_skill_table()
    spy = _IntegersSpy(np.random.default_rng(7))

    R.simulate_reconciled_match(
        "Player A", "Player B", "Hard", 5, "10pt_at_6_6",
        table, const_classifier(0.6), spy, first_server=0,
    )

    assert spy.calls == []


def test_first_server_none_reproduces_the_pre_t2_2_rng_stream():
    """Byte-for-byte equivalence with the reconciliation code path.

    Earlier the opening server was *always* ``int(rng.integers(2))``. Drawing
    that value by hand and pinning it must give the same match and leave the
    generator in the same state as letting the default branch draw it.
    """
    table = make_skill_table()
    r_default = np.random.default_rng(11)
    r_manual = np.random.default_rng(11)
    expected_server = int(r_manual.integers(2))

    got = R.simulate_reconciled_match(
        "Player A", "Player B", "Hard", 5, "advantage",
        table, const_classifier(0.6), r_default,
    )
    want = R.simulate_reconciled_match(
        "Player A", "Player B", "Hard", 5, "advantage",
        table, const_classifier(0.6), r_manual, first_server=expected_server,
    )

    assert got == want
    assert r_default.bit_generator.state == r_manual.bit_generator.state


@pytest.mark.parametrize(
    ("best_of", "rule", "sim_name"),
    [(3, "7pt_at_6_6", "simulate_match_bo3"), (5, "10pt_at_6_6", "simulate_match_bo5")],
)
@pytest.mark.parametrize("first_server", [0, 1])
def test_pinned_first_server_reaches_the_match_simulator(
    best_of, rule, sim_name, first_server, monkeypatch
):
    """The caller's value is *honoured*, not merely accepted and discarded."""
    table = make_skill_table()
    seen = []
    real = getattr(R, sim_name)

    def spy(pA, pB, fs, final_set_rule, rng):
        seen.append(fs)
        return real(pA, pB, fs, final_set_rule, rng)

    monkeypatch.setattr(R, sim_name, spy)
    R.simulate_reconciled_match(
        "Player A", "Player B", "Hard", best_of, rule,
        table, const_classifier(0.6), np.random.default_rng(3),
        first_server=first_server,
    )

    assert seen == [first_server]


@pytest.mark.parametrize("bad", [2, -1, 99, "0"])
def test_invalid_first_server_raises(bad):
    table = make_skill_table()
    with pytest.raises(ValueError, match="first_server must be 0, 1 or None"):
        R.simulate_reconciled_match(
            "Player A", "Player B", "Hard", 5, "10pt_at_6_6",
            table, const_classifier(0.6), np.random.default_rng(1),
            first_server=bad,
        )


# --------------------------------------------------------------------------- #
# model pinning
# --------------------------------------------------------------------------- #
def test_load_pinned_classifier_records_identity():
    if not config.MODEL_PATH.exists():
        pytest.skip("persisted classifier not present (run predictor.py)")
    pinned = R.load_pinned_classifier()
    assert isinstance(pinned.estimator_class, str) and pinned.estimator_class
    assert hasattr(pinned.estimator, "predict_proba")
    assert pinned.path == str(config.MODEL_PATH)
