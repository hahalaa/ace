"""Tests for the train/test year split in src/model/train.py (T0.3).

The split is Train: year < TEST_YEAR, Test: year == TEST_YEAR, decoupling the
held-out season from END_YEAR so the partial 2026 season isn't the test set.

These tests exercise ``model.train``'s own mask logic. They must never
reimplement the split locally: a copied mask passes even when train.py's real
split regresses.
"""
import os
import sys

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless: train_and_evaluate writes the accuracy plot

# Make src/ importable (mirrors tests/test_loader.py's src-on-path pattern; the
# pyproject pythonpath config lands in T0.5).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import config  # noqa: E402
import model.train as train  # noqa: E402


def _multi_year_frame():
    """A fixture spanning several seasons, including TEST_YEAR and END_YEAR."""
    dates = pd.to_datetime(
        [
            f"{config.START_YEAR}-05-01",
            f"{config.TEST_YEAR - 1}-06-01",
            f"{config.TEST_YEAR}-07-01",
            f"{config.TEST_YEAR}-09-01",
            f"{config.END_YEAR}-01-15",
        ]
    )
    return pd.DataFrame({"tourney_date": dates})


def test_config_split_is_sane():
    assert config.START_YEAR < config.TEST_YEAR < config.END_YEAR


def test_masks_non_empty_and_non_overlapping():
    df = _multi_year_frame()

    train_mask, test_mask = train.year_split_masks(df)

    assert train_mask.sum() > 0
    assert test_mask.sum() > 0
    assert not (train_mask & test_mask).any()


def test_test_mask_is_exactly_the_test_year():
    df = _multi_year_frame()

    _, test_mask = train.year_split_masks(df)

    assert (df.loc[test_mask, "tourney_date"].dt.year == config.TEST_YEAR).all()
    assert test_mask.sum() == (df["tourney_date"].dt.year == config.TEST_YEAR).sum()


def test_train_mask_is_strictly_before_the_test_year():
    df = _multi_year_frame()

    train_mask, _ = train.year_split_masks(df)

    assert (df.loc[train_mask, "tourney_date"].dt.year < config.TEST_YEAR).all()


def test_partial_end_year_season_is_neither_train_nor_test():
    df = _multi_year_frame()

    train_mask, test_mask = train.year_split_masks(df)

    end_year_rows = df["tourney_date"].dt.year == config.END_YEAR
    assert end_year_rows.any(), "fixture must contain an END_YEAR row to be meaningful"
    assert not (train_mask & end_year_rows).any()
    assert not (test_mask & end_year_rows).any()


def _trainable_frame():
    """Minimal frame train_and_evaluate can actually fit: every MODEL_FEATURE,
    a two-class target, and rows either side of the split."""
    years = [config.TEST_YEAR - 2, config.TEST_YEAR - 1, config.TEST_YEAR, config.END_YEAR]
    dates, targets = [], []
    for year in years:
        for i in range(10):
            dates.append(pd.Timestamp(f"{year}-06-01"))
            targets.append(i % 2)  # both classes present in every season

    df = pd.DataFrame({"tourney_date": dates, "target": targets})
    # Features correlate with target so the fits are well-posed; values are
    # irrelevant to the split, which is what these tests are about.
    for feature in config.MODEL_FEATURES:
        df[feature] = df["target"] * 1.0 + range(len(df))
    return df


def test_train_and_evaluate_derives_its_split_from_year_split_masks(monkeypatch, tmp_path):
    """Guard the seam the helper exists for: train_and_evaluate must derive its
    split from year_split_masks rather than an inline copy, or the mask tests
    above would be green while the real split regressed.
    """
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(config, "ACCURACY_PLOT", tmp_path / "accuracy_comparison.png")

    calls = []
    real_masks = train.year_split_masks

    def spy(df):
        train_mask, test_mask = real_masks(df)
        calls.append((train_mask, test_mask))
        return train_mask, test_mask

    monkeypatch.setattr(train, "year_split_masks", spy)

    df = _trainable_frame()
    model = train.train_and_evaluate(df)

    assert model is not None
    assert len(calls) == 1, "train_and_evaluate must use year_split_masks exactly once"

    train_mask, test_mask = calls[0]
    years = df["tourney_date"].dt.year
    assert (years[train_mask] < config.TEST_YEAR).all()
    assert (years[test_mask] == config.TEST_YEAR).all()
