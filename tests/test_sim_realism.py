"""Tests for scripts/validate_sim.py — the T1.9 simulation-realism harness.

A smaller, **seeded** version of the validation the script runs directly. Its job
is to catch *gross* breakage (a simulator that stops producing plausible match
shapes at all, or that loses determinism), **not** to enforce tight statistical
agreement with historical data — that rigour lives in ``validate_sim.py`` itself,
run manually. So the bands here are deliberately generous: each brackets *both*
the current simulated value and the historical value, so the test stays green
now and would remain green if the point model later improves toward history (the
known skill-gap-compression finding documented in ``validate_sim.py``).

Speed/stability: a narrow year slice is loaded once (module fixture) and only a
few hundred matches are simulated from a fixed seed, so the metrics are exact and
reproducible.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# scripts/ is not on pyproject's ``pythonpath = ["src"]``; add it (mirrors
# tests/test_refresh_data.py). validate_sim itself adds src/ on import.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))

import validate_sim as V  # noqa: E402
from sim.match import simulate_match_bo3, simulate_match_bo5  # noqa: E402

# A narrow, recent slice — enough real matchups to sample, cheap to load.
_START, _END = 2024, 2026
_SEED = 4242
_N = 600


@pytest.fixture(scope="module")
def loaded():
    """Load + preprocess a small slice once and build the snapshot skill table."""
    from data.loader import load_atp_data
    from data.preprocess import preprocess_data
    from features.serve import build_skill_table

    raw = load_atp_data(_START, _END)
    processed = preprocess_data(raw)
    processed["tourney_date"] = pd.to_datetime(processed["tourney_date"])
    table = build_skill_table(processed)
    return raw, processed, table


def _sim(loaded, best_of, match_fn, rule, slam_only, seed=_SEED, n=_N):
    _, processed, table = loaded
    pool = V.matchup_pool(processed, best_of=best_of, slam_only=slam_only)
    rng = np.random.default_rng(seed)
    return V.simulated_shape_metrics(pool, table, match_fn, rule, n, rng)


# --------------------------------------------------------------------------- #
# Determinism (acceptance: "CI test version is seeded and stable")
# --------------------------------------------------------------------------- #
def test_simulation_is_deterministic_bo5(loaded):
    a = _sim(loaded, 5, simulate_match_bo5, V.BO5_FINAL_SET_RULE, slam_only=True)
    b = _sim(loaded, 5, simulate_match_bo5, V.BO5_FINAL_SET_RULE, slam_only=True)
    assert a == b  # same seed -> identical metrics, exactly


def test_simulation_is_deterministic_bo3(loaded):
    a = _sim(loaded, 3, simulate_match_bo3, V.BO3_FINAL_SET_RULE, slam_only=False)
    b = _sim(loaded, 3, simulate_match_bo3, V.BO3_FINAL_SET_RULE, slam_only=False)
    assert a == b


def test_different_seed_changes_outcome(loaded):
    a = _sim(loaded, 5, simulate_match_bo5, V.BO5_FINAL_SET_RULE, True, seed=1)
    b = _sim(loaded, 5, simulate_match_bo5, V.BO5_FINAL_SET_RULE, True, seed=2)
    assert a.setcount != b.setcount  # not frozen to one RNG stream


# --------------------------------------------------------------------------- #
# Simulated shapes fall in generous plausibility bands (catch gross breakage)
# --------------------------------------------------------------------------- #
def test_simulated_bo5_shapes_plausible(loaded):
    m = _sim(loaded, 5, simulate_match_bo5, V.BO5_FINAL_SET_RULE, slam_only=True)
    assert set(m.setcount) <= {3, 4, 5}
    assert m.setcount.get(3, 0.0) == pytest.approx(1 - m.setcount.get(4, 0) - m.setcount.get(5, 0), abs=1e-9)
    # Bands bracket both current sim (~0.39/0.29) and historical (~0.44/0.22).
    assert 0.25 <= m.setcount.get(3, 0.0) <= 0.60   # straight-set share
    assert 0.10 <= m.setcount.get(5, 0.0) <= 0.40   # five-set share
    assert 8.5 <= m.games_per_set <= 12.0
    assert 0.08 <= m.tb_freq <= 0.38
    assert 0.10 <= m.break_rate <= 0.30


def test_simulated_bo3_shapes_plausible(loaded):
    m = _sim(loaded, 3, simulate_match_bo3, V.BO3_FINAL_SET_RULE, slam_only=False)
    assert set(m.setcount) <= {2, 3}
    assert sum(m.setcount.values()) == pytest.approx(1.0, abs=1e-9)
    assert 0.40 <= m.setcount.get(2, 0.0) <= 0.75   # straight-set share
    assert 0.25 <= m.setcount.get(3, 0.0) <= 0.60
    assert 8.5 <= m.games_per_set <= 12.0
    assert 0.08 <= m.tb_freq <= 0.38
    assert 0.10 <= m.break_rate <= 0.30


# --------------------------------------------------------------------------- #
# Historical side computes and is itself plausible
# --------------------------------------------------------------------------- #
def test_historical_bo5_distribution(loaded):
    raw, _, _ = loaded
    slam_mask = raw["tourney_level"] == "G"
    h = V.historical_shape_metrics(raw, slam_mask, best_of=5)
    assert h.n_matches > 0
    assert set(h.setcount) <= {3, 4, 5}
    assert sum(h.setcount.values()) == pytest.approx(1.0, abs=1e-9)
    # Real Slams: straight-sets are the plurality, five-setters the minority.
    assert h.setcount.get(3, 0.0) > h.setcount.get(5, 0.0)
    assert 8.5 <= h.games_per_set <= 11.0

    rate, n_used = V.historical_break_rate(raw, slam_mask & (raw["best_of"] == 5))
    assert n_used > 0
    assert 0.10 <= rate <= 0.30


def test_historical_bo3_distribution(loaded):
    raw, _, _ = loaded
    mask = raw["tourney_level"] != "D"
    h = V.historical_shape_metrics(raw, mask, best_of=3)
    assert h.n_matches > 0
    assert set(h.setcount) <= {2, 3}
    assert sum(h.setcount.values()) == pytest.approx(1.0, abs=1e-9)
    assert h.setcount.get(2, 0.0) > h.setcount.get(3, 0.0)


# --------------------------------------------------------------------------- #
# Score parser unit tests (no data needed)
# --------------------------------------------------------------------------- #
def test_parse_completed_sets_basic():
    assert V.parse_completed_sets("6-4 3-6 6-2") == [(6, 4, False), (3, 6, False), (6, 2, False)]


def test_parse_completed_sets_detects_tiebreaks():
    parsed = V.parse_completed_sets("7-6(5) 6-7(10-8) 6-3")
    assert parsed == [(7, 6, True), (6, 7, True), (6, 3, False)]


def test_parse_completed_sets_rejects_garbage():
    assert V.parse_completed_sets("") is None
    assert V.parse_completed_sets(float("nan")) is None
    assert V.parse_completed_sets("W/O") is None
    assert V.parse_completed_sets("6-4 UNK") is None


def test_is_incomplete_markers():
    assert V._is_incomplete("6-4 2-0 RET")
    assert V._is_incomplete("6-4 W/O")
    assert not V._is_incomplete("6-4 6-3")
    assert not V._is_incomplete(float("nan"))


# --------------------------------------------------------------------------- #
# Confound control: recent-only pool reproduces the compressed gap (§7)
# --------------------------------------------------------------------------- #
def test_matchup_pool_since_filters(loaded):
    """The ``since`` filter yields a strictly smaller, non-empty recent pool."""
    _, processed, _ = loaded
    full = V.matchup_pool(processed, best_of=5, slam_only=True)
    recent = V.matchup_pool(
        processed, best_of=5, slam_only=True, since=pd.Timestamp("2025-01-01")
    )
    assert 0 < len(recent) < len(full)  # 2025+ is a proper subset of 2024-2026


def test_confound_gap_stable_on_recent_pool(loaded):
    """The mean |pA-pB| gap barely moves on a ~contemporaneous recent pool.

    This is the harness's own robustness argument: the compression is not an
    artifact of scoring old matchups with latest-known snapshot skills.
    """
    _, processed, table = loaded
    full = V.matchup_pool(processed, best_of=5, slam_only=True)
    recent = V.matchup_pool(
        processed, best_of=5, slam_only=True, since=pd.Timestamp("2025-01-01")
    )
    full_gap, full_n = V.mean_point_gap(full, table)
    recent_gap, recent_n = V.mean_point_gap(recent, table)
    assert full_n > 0 and recent_n > 0
    # Both compressed (well below the ~0.10+ gaps that reproduce history) and
    # near-equal — the stale-snapshot confound does not explain the compression.
    assert full_gap < 0.10 and recent_gap < 0.10
    assert abs(full_gap - recent_gap) < 0.03


def test_mean_point_gap_empty_pool_is_nan(loaded):
    """An empty pool yields a NaN gap and n=0 rather than dividing by zero."""
    _, _, table = loaded
    empty = pd.DataFrame({"p1_id": [], "p2_id": [], "surface": []})
    gap, n = V.mean_point_gap(empty, table)
    assert n == 0 and gap != gap  # NaN


# --------------------------------------------------------------------------- #
# End-to-end: run_validation wires hist + sim + the gate together
# --------------------------------------------------------------------------- #
def test_run_validation_smoke():
    checks, metrics = V.run_validation(_START, _END, n_matches=200, seed=_SEED)
    assert {"bo5_hist", "bo5_sim", "bo3_hist", "bo3_sim"} <= metrics.keys()
    # The set-count and games/set checks are the hard gate; tb/break are soft.
    hard_names = {c.name for c in checks if c.hard}
    assert "bo5 3-0 sets" in hard_names and "bo5 games/set" in hard_names
    assert any(not c.hard for c in checks)  # soft checks exist
    # Every check has a boolean verdict; the report can always render.
    assert all(isinstance(c.passed, bool) for c in checks)


def test_run_validation_emits_confound():
    """run_validation wires the stale-snapshot control into its output."""
    _, metrics = V.run_validation(_START, _END, n_matches=200, seed=_SEED)
    assert "confound" in metrics
    c = metrics["confound"]
    assert isinstance(c, V.ConfoundResult)
    assert c.since_year == V.CONFOUND_SINCE_YEAR
    assert c.full_n > 0 and c.recent_n > 0
    # Gaps populated and near-equal (robustness holds on the small CI slice too).
    assert abs(c.full_gap - c.recent_gap) < 0.03
    assert set(c.recent_setcount) <= {3, 4, 5}
