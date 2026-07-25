"""Tests for src/sim/points.py — the T1.2 point-win probability model.

Covers ace-03-tennis-math.md §1 with the T1.9b skill-gap amplification:
    base = spw_server − rpw_returner + (1 − μ)      # §1
    P    = μ + γ·(base − μ)                          # T1.9b (γ = config.POINT_GAP_GAMMA)
then clamped to [P_MIN, P_MAX]. Every expected value here is hand-computed. The
§1-identity, clamp-guard and default-bound tests are γ-invariant (the identity
holds for any γ, the clamp fires regardless of γ) and are unchanged from T1.2;
the value-pinning tests were recomputed under the amplified formula for T1.9b.
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
    """Amplified §1 with an explicit γ, inside the bounds so no clamp fires.

    Recomputed under the T1.9b amplified formula (was pinned to 0.72 at γ=1):
      base = 0.66 − 0.30 + (1 − 0.64) = 0.72
      P    = μ + γ·(base − μ) = 0.64 + 1.9·(0.72 − 0.64) = 0.792
    γ is passed explicitly so this value is config-independent.
    """
    p = point_win_prob(0.66, 0.30, 0.64, p_min=0.0, p_max=1.0, gamma=1.9)
    assert p == pytest.approx(0.792, abs=1e-12)


def test_gamma_one_recovers_pure_additive_formula():
    """γ = 1 must reproduce the un-amplified §1 value exactly (backward-compat).

    Anchors the amplification semantics: γ=1 ⇒ P = base = spw − rpw + (1 − μ),
    the pre-T1.9b formula. 0.66 − 0.30 + (1 − 0.64) = 0.72.
    """
    p = point_win_prob(0.66, 0.30, 0.64, p_min=0.0, p_max=1.0, gamma=1.0)
    assert p == pytest.approx(0.72, abs=1e-12)


def test_strong_server_weak_returner_is_high():
    """A big server facing a poor returner should clear the surface baseline.

    Recomputed under the T1.9b amplified formula with an explicit γ.
    """
    mu = config.SURFACE_MU["Grass"]
    gamma = 1.9
    returner_rpw = 1.0 - mu - 0.05
    p = point_win_prob(0.72, returner_rpw, mu, p_min=0.0, p_max=1.0, gamma=gamma)
    # base = 0.72 − (0.3394 − 0.05) + (1 − 0.6606) = 0.77; P = μ + γ·(0.77 − μ).
    base = 0.72 - returner_rpw + (1.0 - mu)
    assert p == pytest.approx(mu + gamma * (base - mu), abs=1e-12)
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
    """Wrapper computes amplified §1 for each player serving; spw/rpw differ per
    player, so the two directions are genuinely distinct.

    matchup_point_probs does not take a γ argument — it applies the shipped
    config.POINT_GAP_GAMMA — so the expected values reference that constant
    (recomputed under the amplified formula; was the pure base at γ=1).
    """
    mu = config.SURFACE_MU["Hard"]
    gamma = config.POINT_GAP_GAMMA
    skill_a = PlayerSkill(spw=0.68, rpw=0.38, n_serve_pts=500, n_return_pts=500)
    skill_b = PlayerSkill(spw=0.63, rpw=0.34, n_serve_pts=500, n_return_pts=500)

    p_a, p_b = matchup_point_probs(skill_a, skill_b, mu, p_min=0.0, p_max=1.0)

    base_a = 0.68 - 0.34 + (1.0 - mu)  # A serves: A's spw vs B's rpw
    base_b = 0.63 - 0.38 + (1.0 - mu)  # B serves: B's spw vs A's rpw
    assert p_a == pytest.approx(mu + gamma * (base_a - mu), abs=1e-12)
    assert p_b == pytest.approx(mu + gamma * (base_b - mu), abs=1e-12)
    assert p_a != p_b


def test_matchup_of_two_average_players_is_mu_both_ways():
    """Two average players (from SkillTable.default) → each serves at μ."""
    mu = config.SURFACE_MU["Clay"]
    avg = PlayerSkill(spw=mu, rpw=1.0 - mu, n_serve_pts=0, n_return_pts=0)
    p_a, p_b = matchup_point_probs(avg, avg, mu, p_min=0.0, p_max=1.0)
    assert p_a == pytest.approx(mu, abs=1e-12)
    assert p_b == pytest.approx(mu, abs=1e-12)


def test_point_gap_gamma_is_pinned_to_calibrated_value():
    """Regression guard: the T1.9b amplification factor must not drift silently.

    γ=1.9 was calibrated against scripts/validate_sim.py's hard set-count gate,
    which is run manually (not in CI). Without this pin an accidental edit to
    config.POINT_GAP_GAMMA would reintroduce the skill-gap compression bug undetected
    by the test suite. A deliberate recalibration should update this value consciously.
    """
    assert config.POINT_GAP_GAMMA == 1.9
