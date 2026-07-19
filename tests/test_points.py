"""Tests for src/sim/points.py — the T1.2 point-win probability model.

Covers ace-03-tennis-math.md §1: P = spw_server − rpw_returner + (1 − μ),
clamped to [P_MIN, P_MAX]. Every expected value here is hand-computed.
"""
import pytest

import config
from features.serve import PlayerSkill
from sim.points import matchup_point_probs, point_win_prob


def test_average_server_vs_average_returner_returns_mu():
    """§1 sanity check: an average server (spw = μ) against an average returner
    (rpw = 1 − μ) must return exactly μ — not merely a spot-check near it."""
    for mu in (0.6196, 0.6423, 0.6606, 0.55, 0.72):
        p = point_win_prob(
            server_spw=mu,
            returner_rpw=1.0 - mu,
            mu_surface=mu,
            p_min=0.0,
            p_max=1.0,
        )
        assert p == pytest.approx(mu, abs=1e-12)


def test_formula_matches_hand_computed_unclamped():
    """P = spw − rpw + (1 − μ), inside the bounds so no clamp fires."""
    # 0.66 − 0.30 + (1 − 0.64) = 0.72
    p = point_win_prob(0.66, 0.30, 0.64, p_min=0.0, p_max=1.0)
    assert p == pytest.approx(0.72, abs=1e-12)


def test_strong_server_weak_returner_is_high():
    """A big server facing a poor returner should clear the surface baseline."""
    mu = config.SURFACE_MU["Grass"]
    p = point_win_prob(0.72, 1.0 - mu - 0.05, mu, p_min=0.0, p_max=1.0)
    # 0.72 − (0.3394 − 0.05) + (1 − 0.6606) = 0.72 − 0.2894 + 0.3394 = 0.77
    assert p == pytest.approx(0.72 - (1.0 - mu - 0.05) + (1.0 - mu), abs=1e-12)
    assert p > mu


def test_clamps_to_upper_bound():
    """A degenerate high raw value is pulled down to P_MAX."""
    # 0.99 − 0.01 + (1 − 0.62) = 1.36 → clamp to P_MAX.
    p = point_win_prob(0.99, 0.01, 0.62)
    assert p == config.P_MAX


def test_clamps_to_lower_bound():
    """A degenerate low raw value is lifted to P_MIN."""
    # 0.30 − 0.80 + (1 − 0.62) = -0.12 → clamp to P_MIN.
    p = point_win_prob(0.30, 0.80, 0.62)
    assert p == config.P_MIN


def test_default_clamp_bounds_come_from_config():
    """Omitting p_min/p_max uses config.P_MIN / config.P_MAX."""
    assert point_win_prob(0.0, 1.0, 0.62) == config.P_MIN
    assert point_win_prob(1.0, 0.0, 0.62) == config.P_MAX


def test_matchup_returns_both_directions():
    """Wrapper computes §1 for each player serving; spw/rpw differ per player,
    so the two directions are genuinely distinct."""
    mu = config.SURFACE_MU["Hard"]
    skill_a = PlayerSkill(spw=0.68, rpw=0.38, n_serve_pts=500, n_return_pts=500)
    skill_b = PlayerSkill(spw=0.63, rpw=0.34, n_serve_pts=500, n_return_pts=500)

    p_a, p_b = matchup_point_probs(skill_a, skill_b, mu, p_min=0.0, p_max=1.0)

    expected_a = 0.68 - 0.34 + (1.0 - mu)  # A serves: A's spw vs B's rpw
    expected_b = 0.63 - 0.38 + (1.0 - mu)  # B serves: B's spw vs A's rpw
    assert p_a == pytest.approx(expected_a, abs=1e-12)
    assert p_b == pytest.approx(expected_b, abs=1e-12)
    assert p_a != p_b


def test_matchup_of_two_average_players_is_mu_both_ways():
    """Two average players (from SkillTable.default) → each serves at μ."""
    mu = config.SURFACE_MU["Clay"]
    avg = PlayerSkill(spw=mu, rpw=1.0 - mu, n_serve_pts=0, n_return_pts=0)
    p_a, p_b = matchup_point_probs(avg, avg, mu, p_min=0.0, p_max=1.0)
    assert p_a == pytest.approx(mu, abs=1e-12)
    assert p_b == pytest.approx(mu, abs=1e-12)
