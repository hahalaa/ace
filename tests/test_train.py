"""Tests for the train/test year split in src/model/train.py (T0.3).

The split is now Train: year < TEST_YEAR, Test: year == TEST_YEAR, decoupling
the held-out season from END_YEAR so the partial 2026 season isn't the test set.
"""
import os
import sys

import pandas as pd

# Make src/ importable (mirrors tests/test_loader.py's src-on-path pattern; the
# pyproject pythonpath config lands in T0.5).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import config  # noqa: E402


def _year_masks(df):
    """Reproduce train.py's split so the invariant is asserted directly."""
    train_mask = df["tourney_date"].dt.year < config.TEST_YEAR
    test_mask = df["tourney_date"].dt.year == config.TEST_YEAR
    return train_mask, test_mask


def test_config_split_is_sane():
    assert config.TEST_YEAR < config.END_YEAR
    assert config.TEST_YEAR > config.START_YEAR


def test_masks_non_empty_and_non_overlapping():
    # A fixture spanning several years, including TEST_YEAR and END_YEAR.
    dates = pd.to_datetime(
        [
            f"{config.START_YEAR}-05-01",
            f"{config.TEST_YEAR - 1}-06-01",
            f"{config.TEST_YEAR}-07-01",
            f"{config.TEST_YEAR}-09-01",
            f"{config.END_YEAR}-01-15",
        ]
    )
    df = pd.DataFrame({"tourney_date": dates})

    train_mask, test_mask = _year_masks(df)

    # Non-empty
    assert train_mask.sum() > 0
    assert test_mask.sum() > 0

    # Non-overlapping
    assert not (train_mask & test_mask).any()

    # Test set is exactly the TEST_YEAR rows
    assert (df.loc[test_mask, "tourney_date"].dt.year == config.TEST_YEAR).all()

    # The partial END_YEAR season is neither train nor test
    end_year_rows = df["tourney_date"].dt.year == config.END_YEAR
    assert not (train_mask & end_year_rows).any()
    assert not (test_mask & end_year_rows).any()
