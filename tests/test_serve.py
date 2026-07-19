"""Tests for src/features/serve.py — the T1.1 serve/return skill table.

The skill table keys on ``player_id`` and aggregates per-match serve stats into
recency- and volume-weighted, empirical-Bayes-shrunk rates per surface. These
tests build a small preprocessed-style p1/p2 frame directly (the columns
``build_skill_table`` consumes) so every expected value is hand-computable.
"""
import numpy as np
import pandas as pd
import pytest

import config
import features.serve as serve
from features.serve import PlayerSkill


def _row(
    date,
    surface,
    p1_id,
    p2_id,
    p1_won,
    p1_svpt,
    p2_won,
    p2_svpt,
    has=True,
    score="6-4 6-4",
    p1_name=None,
    p2_name=None,
):
    """One preprocessed match row. Serve points won are stashed entirely in the
    1stWon column (serve.py sums 1stWon + 2ndWon), keeping the arithmetic clean."""
    return {
        "tourney_date": pd.Timestamp(date),
        "surface": surface,
        "score": score,
        "has_serve_stats": has,
        "p1_id": p1_id,
        "p2_id": p2_id,
        "p1_name": p1_name if p1_name is not None else f"Player {p1_id}",
        "p2_name": p2_name if p2_name is not None else f"Player {p2_id}",
        "p1_1stWon": p1_won,
        "p1_2ndWon": 0,
        "p1_svpt": p1_svpt,
        "p2_1stWon": p2_won,
        "p2_2ndWon": 0,
        "p2_svpt": p2_svpt,
    }


def _frame(rows):
    return pd.DataFrame(rows)


def _shrink(rate_raw, n, prior, k=None):
    """Empirical-Bayes shrinkage the table applies: (n·rate + k·μ)/(n + k)."""
    if k is None:
        k = config.SERVE_SHRINKAGE_K
    return (n * rate_raw + k * prior) / (n + k)


# --------------------------------------------------------------------------
# Core aggregation: hand-calculated weighted + shrunk rates
# --------------------------------------------------------------------------

def test_spw_and_rpw_match_hand_calculated_weighted_shrunk_value():
    """Two Hard matches, same date (recency weight = 1) → pooled rate, shrunk."""
    mu = config.SURFACE_MU["Hard"]
    rows = [
        # X serves 100 pts winning 60; opponent serves 100 pts winning 70.
        _row("2024-05-01", "Hard", "X", "O1", p1_won=60, p1_svpt=100, p2_won=70, p2_svpt=100),
        # X serves 120 pts winning 80; opponent serves 110 pts winning 60.
        _row("2024-05-01", "Hard", "X", "O2", p1_won=80, p1_svpt=120, p2_won=60, p2_svpt=110),
    ]
    st = serve.build_skill_table(_frame(rows))
    sk = st.get("X", "Hard")

    spw_raw = (60 + 80) / (100 + 120)
    n_s = 100 + 120
    assert sk.spw == pytest.approx(_shrink(spw_raw, n_s, mu))
    assert sk.n_serve_pts == pytest.approx(n_s)

    # Return points won = opp_svpt - opp_won, over opp_svpt played.
    rpw_raw = ((100 - 70) + (110 - 60)) / (100 + 110)
    n_r = 100 + 110
    assert sk.rpw == pytest.approx(_shrink(rpw_raw, n_r, 1 - mu))
    assert sk.n_return_pts == pytest.approx(n_r)


def test_recency_weighting_pulls_rate_toward_the_recent_match():
    """Same player, two Hard matches a half-life apart, different serve rates.

    The older match carries weight 0.5 (one half-life), the recent one 1.0, so
    the weighted raw rate sits nearer the recent match's 0.70 than a flat mean."""
    half_life = int(config.SERVE_RECENCY_HALFLIFE_DAYS)
    recent = pd.Timestamp("2024-06-01")
    old = recent - pd.Timedelta(days=half_life)
    rows = [
        _row(recent, "Hard", "X", "O1", p1_won=70, p1_svpt=100, p2_won=64, p2_svpt=100),
        _row(old, "Hard", "X", "O2", p1_won=50, p1_svpt=100, p2_won=64, p2_svpt=100),
    ]
    st = serve.build_skill_table(_frame(rows))
    sk = st.get("X", "Hard")

    mu = config.SURFACE_MU["Hard"]
    spw_raw = (1.0 * 70 + 0.5 * 50) / (1.0 * 100 + 0.5 * 100)  # 0.6333...
    n_s = 200  # raw point total is unweighted
    assert sk.spw == pytest.approx(_shrink(spw_raw, n_s, mu))
    # Sanity: nearer the recent 0.70 than a naive average would be.
    flat_raw = (70 + 50) / 200
    assert spw_raw > flat_raw


# --------------------------------------------------------------------------
# Shrinkage toward the surface baseline
# --------------------------------------------------------------------------

def test_low_sample_player_is_shrunk_toward_baseline():
    """A single low-volume match should be pulled most of the way to μ."""
    mu = config.SURFACE_MU["Hard"]
    # A wildly high raw serve rate on tiny volume.
    rows = [_row("2024-05-01", "Hard", "X", "O1", p1_won=40, p1_svpt=40, p2_won=64, p2_svpt=100)]
    st = serve.build_skill_table(_frame(rows))
    sk = st.get("X", "Hard")

    # Raw rate is 1.0; shrunk with n=40 against k=200 lands close to μ.
    assert sk.spw == pytest.approx(_shrink(1.0, 40, mu))
    assert abs(sk.spw - mu) < abs(1.0 - mu)  # moved toward baseline
    assert sk.spw < 0.75  # nowhere near the raw 1.0


def test_high_sample_player_barely_shrinks():
    """Large volume overwhelms the k-point prior — rate stays near raw."""
    mu = config.SURFACE_MU["Hard"]
    rows = [_row("2024-05-01", "Hard", "X", "O1", p1_won=7000, p1_svpt=10000, p2_won=64, p2_svpt=100)]
    st = serve.build_skill_table(_frame(rows))
    sk = st.get("X", "Hard")
    assert sk.spw == pytest.approx(_shrink(0.70, 10000, mu))
    assert abs(sk.spw - 0.70) < 0.01


# --------------------------------------------------------------------------
# Exclusions: missing stats, Carpet/Unknown, retirements
# --------------------------------------------------------------------------

def test_missing_serve_stat_rows_are_excluded():
    good = _row("2024-05-01", "Hard", "X", "O1", p1_won=60, p1_svpt=100, p2_won=64, p2_svpt=100)
    # A row flagged has_serve_stats=False with values that would skew the rate.
    bad = _row("2024-05-02", "Hard", "X", "O2", p1_won=100, p1_svpt=100, p2_won=0, p2_svpt=100, has=False)
    st = serve.build_skill_table(_frame([good, bad]))
    sk = st.get("X", "Hard")
    assert sk.n_serve_pts == pytest.approx(100)  # only the good row counted


def test_carpet_and_unknown_surface_rows_are_excluded():
    good = _row("2024-05-01", "Hard", "X", "O1", p1_won=60, p1_svpt=100, p2_won=64, p2_svpt=100)
    carpet = _row("2024-05-02", "Carpet", "X", "O2", p1_won=99, p1_svpt=100, p2_won=0, p2_svpt=100)
    unknown = _row("2024-05-03", np.nan, "X", "O3", p1_won=99, p1_svpt=100, p2_won=0, p2_svpt=100)
    st = serve.build_skill_table(_frame([good, carpet, unknown]))

    assert st.get("X", "Hard").n_serve_pts == pytest.approx(100)
    # No Carpet/Unknown key was created (get on a non-baseline surface raises).
    assert ("X", "Carpet") not in st._skills
    with pytest.raises(ValueError):
        st.get("X", "Carpet")


def test_retirement_and_walkover_rows_are_excluded():
    good = _row("2024-05-01", "Hard", "X", "O1", p1_won=60, p1_svpt=100, p2_won=64, p2_svpt=100)
    ret = _row("2024-05-02", "Hard", "X", "O2", p1_won=99, p1_svpt=100, p2_won=0, p2_svpt=100, score="6-4 2-0 RET")
    wo = _row("2024-05-03", "Hard", "X", "O3", p1_won=99, p1_svpt=100, p2_won=0, p2_svpt=100, score="W/O")
    st = serve.build_skill_table(_frame([good, ret, wo]))
    assert st.get("X", "Hard").n_serve_pts == pytest.approx(100)


def test_null_id_rows_are_skipped():
    """A row with a missing player id can't be keyed — drop it (both sides)."""
    rows = [_row("2024-05-01", "Hard", None, "O1", p1_won=60, p1_svpt=100, p2_won=64, p2_svpt=100)]
    st = serve.build_skill_table(_frame(rows))
    # Neither the null-id player nor the opponent (whose only match this was) is aggregated.
    assert st._skills == {}
    assert st.get("O1", "Hard") == st.default("Hard")


# --------------------------------------------------------------------------
# Keying on id, name carry-through, defaults, resolution
# --------------------------------------------------------------------------

def test_table_keys_on_id_not_name_and_name_collisions_stay_separate():
    """Two different players sharing a display name keep separate id-keyed rows."""
    rows = [
        _row("2024-05-01", "Hard", "J1", "O1", p1_won=70, p1_svpt=100, p2_won=64, p2_svpt=100, p1_name="John Doe"),
        _row("2024-05-02", "Hard", "J1", "O2", p1_won=70, p1_svpt=100, p2_won=64, p2_svpt=100, p1_name="John Doe"),
        _row("2024-05-03", "Hard", "J2", "O3", p1_won=50, p1_svpt=100, p2_won=64, p2_svpt=100, p1_name="John Doe"),
    ]
    st = serve.build_skill_table(_frame(rows))
    assert ("J1", "Hard") in st._skills
    assert ("J2", "Hard") in st._skills
    assert st.get("J1", "Hard").spw != st.get("J2", "Hard").spw
    # Name→id resolution picks the more frequent id (J1 appears twice).
    assert st.resolve_name("John Doe") == "J1"


def test_default_profile_is_the_surface_baseline():
    st = serve.build_skill_table(_frame([_row("2024-05-01", "Hard", "X", "O1", 60, 100, 64, 100)]))
    for surface in ("Hard", "Clay", "Grass"):
        mu = config.SURFACE_MU[surface]
        d = st.default(surface)
        assert d == PlayerSkill(spw=mu, rpw=1 - mu, n_serve_pts=0.0, n_return_pts=0.0)


def test_get_unknown_id_returns_default():
    st = serve.build_skill_table(_frame([_row("2024-05-01", "Hard", "X", "O1", 60, 100, 64, 100)]))
    assert st.get("NOPE", "Hard") == st.default("Hard")
    assert st.get(None, "Hard") == st.default("Hard")


def test_resolve_name_delegates_to_common_names():
    rows = [
        _row("2024-05-01", "Hard", "A0E2", "O1", 60, 100, 64, 100, p1_name="Carlos Alcaraz"),
        _row("2024-05-02", "Hard", "S0AG", "O2", 60, 100, 64, 100, p1_name="Jannik Sinner"),
    ]
    st = serve.build_skill_table(_frame(rows))
    assert st.resolve_name("Carlos Alcaraz") == "A0E2"      # exact
    assert st.resolve_name("C Alcaraz") == "A0E2"           # initials strategy
    assert st.resolve_name("Nobody Here") is None           # no match


def test_get_and_default_reject_unknown_surface():
    st = serve.build_skill_table(_frame([_row("2024-05-01", "Hard", "X", "O1", 60, 100, 64, 100)]))
    with pytest.raises(ValueError):
        st.default("Carpet")
    with pytest.raises(ValueError):
        st.get("X", "Unknown")


# --------------------------------------------------------------------------
# Leakage-safety: snapshot cutoff and as-of variant
# --------------------------------------------------------------------------

def test_snapshot_cutoff_is_the_latest_included_match():
    rows = [
        _row("2024-01-01", "Hard", "X", "O1", 60, 100, 64, 100),
        _row("2024-06-01", "Hard", "X", "O2", 60, 100, 64, 100),
    ]
    st = serve.build_skill_table(_frame(rows))
    assert st.cutoff == pd.Timestamp("2024-06-01")


def test_as_of_uses_only_matches_strictly_before_the_cutoff():
    rows = [
        _row("2024-01-01", "Hard", "X", "O1", p1_won=60, p1_svpt=100, p2_won=64, p2_svpt=100),
        _row("2024-06-01", "Hard", "X", "O2", p1_won=99, p1_svpt=200, p2_won=64, p2_svpt=100),
    ]
    st = serve.build_skill_table(_frame(rows), as_of="2024-03-01")
    # Only the January match is in scope.
    assert st.get("X", "Hard").n_serve_pts == pytest.approx(100)
    assert st.cutoff == pd.Timestamp("2024-01-01")


def test_as_of_is_exclusive_on_the_boundary_date():
    rows = [_row("2024-03-01", "Hard", "X", "O1", 60, 100, 64, 100)]
    st = serve.build_skill_table(_frame(rows), as_of="2024-03-01")
    assert st._skills == {}  # the boundary match is excluded


# --------------------------------------------------------------------------
# Surface baseline μ
# --------------------------------------------------------------------------

def test_compute_surface_mu_pools_both_players_and_is_in_band():
    rows = [
        _row("2024-05-01", "Hard", "X", "O1", p1_won=64, p1_svpt=100, p2_won=66, p2_svpt=100),
        _row("2024-05-02", "Clay", "X", "O2", p1_won=60, p1_svpt=100, p2_won=64, p2_svpt=100),
    ]
    mu = serve.compute_surface_mu(_frame(rows))
    assert mu["Hard"] == pytest.approx((64 + 66) / 200)
    assert mu["Clay"] == pytest.approx((60 + 64) / 200)


def test_config_surface_mu_is_in_sane_band():
    """The vendored, data-derived baselines must sit in the documented band."""
    for surface, mu in config.SURFACE_MU.items():
        assert 0.60 <= mu <= 0.68, f"{surface} μ={mu} outside sane band"


def test_empty_frame_yields_empty_table_with_defaults():
    empty = _frame([]).reindex(
        columns=[
            "tourney_date", "surface", "score", "has_serve_stats",
            "p1_id", "p2_id", "p1_name", "p2_name",
            "p1_1stWon", "p1_2ndWon", "p1_svpt", "p2_1stWon", "p2_2ndWon", "p2_svpt",
        ]
    )
    st = serve.build_skill_table(empty)
    assert st._skills == {}
    assert st.cutoff is None
    assert st.get("anyone", "Hard") == st.default("Hard")
