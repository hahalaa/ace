"""Tests for model/calibrate.py (T1.8) — the pure calibration helpers.

The full run_calibration pipeline (load → engineer → predict → plot) is exercised
manually / by the script entry point; here we pin the pure, fixture-sized helpers
the ticket calls out: the Brier score and the decile binning.
"""

from __future__ import annotations

import numpy as np
import pytest

import config
from model.calibrate import brier_score, compute_calibration


def test_brier_score_tiny_fixture():
    # mean((0.9-1)^2, (0.1-0)^2) = mean(0.01, 0.01) = 0.01
    assert brier_score([0.9, 0.1], [1, 0]) == pytest.approx(0.01)


def test_brier_score_perfect_and_worst():
    assert brier_score([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert brier_score([0.0, 1.0], [1, 0]) == pytest.approx(1.0)


def test_brier_score_validates():
    with pytest.raises(ValueError):
        brier_score([0.5, 0.5], [1])  # length mismatch
    with pytest.raises(ValueError):
        brier_score([], [])  # empty


def test_compute_calibration_tiny_fixture():
    probs = [0.1, 0.2, 0.9, 0.8]
    outcomes = [0, 1, 1, 1]
    res = compute_calibration(probs, outcomes, n_bins=2)
    # bin 0 = [0,0.5): preds {0.1,0.2}, outcomes {0,1}
    # bin 1 = [0.5,1]: preds {0.9,0.8}, outcomes {1,1}
    assert list(res.bin_count) == [2, 2]
    assert res.bin_pred[0] == pytest.approx(0.15)
    assert res.bin_emp[0] == pytest.approx(0.5)
    assert res.bin_pred[1] == pytest.approx(0.85)
    assert res.bin_emp[1] == pytest.approx(1.0)


def test_compute_calibration_p_equals_one_in_last_bin():
    res = compute_calibration([1.0], [1], n_bins=10)
    assert list(res.bin_count) == [1]
    assert res.bin_pred[0] == pytest.approx(1.0)


def test_compute_calibration_drops_empty_bins():
    # All predictions in a single decile → exactly one populated bin.
    res = compute_calibration([0.31, 0.34, 0.39], [0, 1, 0], n_bins=10)
    assert len(res.bin_count) == 1
    assert int(res.bin_count[0]) == 3
    assert res.bin_emp[0] == pytest.approx(1 / 3)


def test_compute_calibration_validates():
    with pytest.raises(ValueError):
        compute_calibration([0.5], [0, 1])  # length mismatch
    with pytest.raises(ValueError):
        compute_calibration([], [])  # empty
