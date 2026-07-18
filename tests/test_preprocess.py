"""Tests for src/data/preprocess.py — serve stats / ids carried to p1/p2 (T0.4).

The p1/p2 swap is driven by a hardcoded ``default_rng(seed=42)`` inside
``preprocess_data``. These tests deliberately do **not** recompute that mask:
they assert the *invariant* it exists to produce (every p1 field comes from the
same original player, consistent with ``target``), so a mis-mapped column can't
pass by reimplementing the bug.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make src/ importable (mirrors tests/test_loader.py; the pyproject pythonpath
# config lands in T0.5).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import data.preprocess as preprocess  # noqa: E402

# Serve-stat values are seeded per row as winner = 100 + i, loser = 200 + i, so a
# column sourced from the wrong player is unambiguous in an assertion failure.
N_ROWS = 8


def _raw_matches(n=N_ROWS, **overrides):
    """A raw winner/loser frame using the real TML-Database column names."""
    rows = {
        "tourney_date": pd.to_datetime(["2024-01-01"] * n),
        "surface": ["Hard"] * n,
        "tourney_level": ["G"] * n,
        "round": ["R32"] * n,
        "best_of": [5] * n,
        "winner_id": [f"W{i}" for i in range(n)],
        "loser_id": [f"L{i}" for i in range(n)],
        "winner_name": [f"Winner {i}" for i in range(n)],
        "loser_name": [f"Loser {i}" for i in range(n)],
        "winner_rank": [1.0] * n,
        "loser_rank": [2.0] * n,
        "winner_age": [25.0] * n,
        "loser_age": [26.0] * n,
        "score": ["6-4 6-4"] * n,
    }
    for stat in preprocess.SERVE_STAT_COLUMNS:
        rows[f"w_{stat}"] = [float(100 + i) for i in range(n)]
        rows[f"l_{stat}"] = [float(200 + i) for i in range(n)]
    rows.update(overrides)
    return pd.DataFrame(rows)


@pytest.fixture
def processed():
    return preprocess.preprocess_data(_raw_matches())


def test_fixture_exercises_both_swapped_and_unswapped_rows(processed):
    """Guard against the mapping tests below going vacuous if the mask changes."""
    assert set(processed["target"]) == {0, 1}


def test_new_columns_are_present(processed):
    for stat in preprocess.SERVE_STAT_COLUMNS:
        assert f"p1_{stat}" in processed.columns
        assert f"p2_{stat}" in processed.columns
    for col in ["p1_id", "p2_id", "best_of", "tourney_level", "round", "has_serve_stats"]:
        assert col in processed.columns


def test_serve_stats_and_ids_follow_the_same_swap_as_the_target(processed):
    """p1/p2 serve columns and ids map to the correct original player either way.

    target == 1 means p1 is the match winner; target == 0 means p1 is the loser.
    """
    for i, row in processed.iterrows():
        p1_is_winner = row["target"] == 1
        # The row index survives preprocessing, so the seeded values are recoverable.
        winner_val, loser_val = float(100 + i), float(200 + i)

        expected_p1_id = f"W{i}" if p1_is_winner else f"L{i}"
        expected_p2_id = f"L{i}" if p1_is_winner else f"W{i}"
        assert row["p1_id"] == expected_p1_id
        assert row["p2_id"] == expected_p2_id
        # Name must agree with id — they are swapped by the same mask.
        assert row["p1_name"] == (f"Winner {i}" if p1_is_winner else f"Loser {i}")

        for stat in preprocess.SERVE_STAT_COLUMNS:
            assert row[f"p1_{stat}"] == (winner_val if p1_is_winner else loser_val)
            assert row[f"p2_{stat}"] == (loser_val if p1_is_winner else winner_val)


def test_player_ids_stay_strings_even_when_digit_only():
    """Real ids are alphanumeric ("D875"), but some are digit-only ("104631").

    A file whose ids all happen to be digit-only reads back as an integer column,
    which would silently make p1_id an int and break the join with the string-keyed
    skill table (T1.1). Ints in, strings out.
    """
    raw = _raw_matches(
        winner_id=[104631] * N_ROWS,
        loser_id=[200000 + i for i in range(N_ROWS)],
    )
    assert pd.api.types.is_integer_dtype(raw["winner_id"]), "fixture must model an int column"

    out = preprocess.preprocess_data(raw)

    assert out["p1_id"].dtype == "string"
    assert out["p2_id"].dtype == "string"
    assert all(isinstance(v, str) for v in out["p1_id"])
    assert all(isinstance(v, str) for v in out["p2_id"])
    assert "104631" in set(out["p1_id"]) | set(out["p2_id"])


def test_missing_player_id_stays_missing():
    """The real 2014-2026 data has exactly one row with a null loser_id.

    It must stay missing rather than becoming the literal string "nan", which is
    what a naive str cast produces and which would join as a real player id.
    """
    raw = _raw_matches()
    raw["loser_id"] = raw["loser_id"].astype("object")
    raw.loc[0, "loser_id"] = np.nan

    out = preprocess.preprocess_data(raw)

    side = "p2" if out.loc[0, "target"] == 1 else "p1"
    assert pd.isna(out.loc[0, f"{side}_id"])


def test_serve_stat_nans_are_preserved_not_imputed():
    raw = _raw_matches()
    raw.loc[0, "w_svpt"] = np.nan
    raw.loc[0, "w_ace"] = np.nan

    out = preprocess.preprocess_data(raw)

    # Whichever side row 0's winner landed on, the NaN survives on exactly that side.
    side = "p1" if out.loc[0, "target"] == 1 else "p2"
    assert pd.isna(out.loc[0, f"{side}_svpt"])
    assert pd.isna(out.loc[0, f"{side}_ace"])


def test_has_serve_stats_is_false_when_svpt_is_nan():
    raw = _raw_matches()
    raw.loc[0, "w_svpt"] = np.nan

    out = preprocess.preprocess_data(raw)

    assert not out.loc[0, "has_serve_stats"]
    assert out.loc[1:, "has_serve_stats"].all()


def test_has_serve_stats_is_false_when_svpt_is_zero():
    raw = _raw_matches()
    raw.loc[0, "l_svpt"] = 0.0

    out = preprocess.preprocess_data(raw)

    assert not out.loc[0, "has_serve_stats"]
    assert out.loc[1:, "has_serve_stats"].all()


def test_has_serve_stats_is_false_when_a_non_svpt_column_is_missing():
    """The flag means the whole stat line is usable, not just svpt."""
    raw = _raw_matches()
    raw.loc[0, "l_bpFaced"] = np.nan

    out = preprocess.preprocess_data(raw)

    assert not out.loc[0, "has_serve_stats"]


def test_target_distribution_is_unchanged_by_t0_4():
    """Regression: golden captured from the pre-T0.4 code on this exact fixture.

    Pins the swap mask — an extra rng draw added anywhere before it would shift
    every downstream p1/p2 assignment and silently change the training labels.
    """
    out = preprocess.preprocess_data(_raw_matches(n=8))

    assert out["target"].tolist() == [0, 1, 0, 0, 1, 0, 0, 0]


def test_existing_classifier_columns_are_unchanged(processed):
    """The baseline columns preprocess emitted before T0.4 still come through."""
    for col in [
        "tourney_date", "surface", "tourney_level", "target",
        "p1_name", "p1_rank", "p1_age", "p2_name", "p2_rank", "p2_age",
        "p1_games_won", "p1_games_lost", "p1_sets_won", "p1_sets_lost",
        "p2_games_won", "p2_games_lost", "p2_sets_won", "p2_sets_lost",
    ]:
        assert col in processed.columns

    # Scores parse from the winner's perspective, then swap with the same mask.
    for _, row in processed.iterrows():
        if row["target"] == 1:
            assert row["p1_games_won"] == 12 and row["p1_sets_won"] == 2
        else:
            assert row["p1_games_won"] == 8 and row["p1_sets_won"] == 0
