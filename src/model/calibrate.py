"""Calibration / reliability check on the held-out season (T1.8, ``§6``).

The headline "is the model honest?" artefact. Bins the classifier's predicted
``P(P1 wins)`` on ``config.TEST_YEAR`` (2025 — the decoupled held-out season, **not**
``END_YEAR``) into deciles, compares predicted vs realised win rate, computes the
Brier score, and saves the reliability curve to ``config.CALIBRATION_PLOT``
(``outputs/calibration.png``).

Landed as its **own file** (``model/calibrate.py``), not folded into
``model/viz.py``: ``viz`` is a thin one-function feature-importance plotter with no
data-loading responsibilities, whereas calibration runs the whole
load→preprocess→engineer pipeline, pins and scores the classifier, and computes a
metric. Keeping that off ``viz`` preserves ``viz``'s single-purpose shape.

Scope note: this calibrates the classifier's own predicted probability, which is
exactly the authoritative match-win probability under the default
``RECONCILE_MODE = "classifier_anchor"`` (there the reconciled prob *equals*
``P_clf``). Under ``"blend"`` the authoritative prob shifts by ``(1−w)·(P_point −
P_clf)`` per match; recalibrating that blended curve is out of this ticket's tight
scope (it needs the skill table + per-row name→id resolution — a T1.9-scale job).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
import data.loader as loader
import data.preprocess as preprocess
import features.engineering as features
import model.train as train
import sim.reconcile as reconcile


@dataclass(frozen=True)
class CalibrationResult:
    """Per-bin reliability data (empty bins dropped).

    Attributes:
        bin_pred: Mean predicted probability within each populated bin.
        bin_emp: Empirical win rate (mean outcome) within each bin.
        bin_count: Number of samples in each bin.
        n_bins: The number of equal-width bins the ``[0, 1]`` range was split into.
    """

    bin_pred: np.ndarray
    bin_emp: np.ndarray
    bin_count: np.ndarray
    n_bins: int


def brier_score(probs, outcomes) -> float:
    """Mean squared error between predicted probabilities and 0/1 outcomes.

    ``Brier = mean((p − y)²)`` — lower is better; 0 is perfect. Pure and array-
    based so it is trivially testable on a fixture.

    Args:
        probs: Predicted probabilities of the positive class.
        outcomes: Realised binary outcomes (0/1), aligned with ``probs``.

    Returns:
        The Brier score.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if probs.shape != outcomes.shape:
        raise ValueError(
            f"probs and outcomes must have the same shape, got {probs.shape} "
            f"vs {outcomes.shape}"
        )
    if probs.size == 0:
        raise ValueError("brier_score requires at least one sample")
    return float(np.mean((probs - outcomes) ** 2))


def compute_calibration(probs, outcomes, n_bins: int = config.CALIBRATION_BINS) -> CalibrationResult:
    """Bin predictions into equal-width bins and compute per-bin reliability.

    Splits ``[0, 1]`` into ``n_bins`` equal-width bins, assigns each prediction to a
    bin (the top edge, ``p == 1.0``, falls in the last bin), and returns the mean
    predicted probability and empirical win rate per **populated** bin. Empty bins
    are dropped so the reliability curve has no spurious points.

    Args:
        probs: Predicted probabilities in ``[0, 1]``.
        outcomes: Realised binary outcomes (0/1).
        n_bins: Number of equal-width bins.

    Returns:
        A :class:`CalibrationResult`.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if probs.shape != outcomes.shape:
        raise ValueError(
            f"probs and outcomes must have the same shape, got {probs.shape} "
            f"vs {outcomes.shape}"
        )
    if probs.size == 0:
        raise ValueError("compute_calibration requires at least one sample")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin index in [0, n_bins-1]; clip so p == 1.0 lands in the last bin.
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)

    bin_pred, bin_emp, bin_count = [], [], []
    for b in range(n_bins):
        mask = idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        bin_pred.append(float(probs[mask].mean()))
        bin_emp.append(float(outcomes[mask].mean()))
        bin_count.append(count)

    return CalibrationResult(
        bin_pred=np.array(bin_pred),
        bin_emp=np.array(bin_emp),
        bin_count=np.array(bin_count),
        n_bins=n_bins,
    )


def plot_calibration(result: CalibrationResult, brier: float, save_path=config.CALIBRATION_PLOT) -> None:
    """Save a reliability curve (predicted vs empirical) with the Brier score.

    Point size scales with per-bin sample count; the dashed diagonal is perfect
    calibration. Imported lazily so importing this module stays cheap and
    headless-safe.
    """
    import os

    import matplotlib

    matplotlib.use("Agg")  # headless — no display needed
    import matplotlib.pyplot as plt

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect calibration")
    sizes = 20.0 + 200.0 * result.bin_count / max(result.bin_count.max(), 1)
    ax.scatter(
        result.bin_pred,
        result.bin_emp,
        s=sizes,
        c="tab:blue",
        alpha=0.8,
        edgecolors="k",
        zorder=3,
        label="Model (bin size ∝ n)",
    )
    ax.plot(result.bin_pred, result.bin_emp, color="tab:blue", alpha=0.4, zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(P1 wins)")
    ax.set_ylabel("Empirical win rate")
    ax.set_title(
        f"Reliability curve — TEST_YEAR {config.TEST_YEAR}\nBrier score = {brier:.4f}"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _load_test_year_frame():
    """Run the pipeline and return the engineered ``TEST_YEAR`` rows.

    Mirrors ``predictor.main``'s load→preprocess→engineer path, then selects the
    held-out season via ``train.year_split_masks`` (so the split stays defined in
    exactly one place).
    """
    data = loader.load_atp_data(config.START_YEAR, config.END_YEAR)
    processed = preprocess.preprocess_data(data)
    final_df, *_ = features.add_features(processed)
    _, test_mask = train.year_split_masks(final_df)
    return final_df.loc[test_mask]


def run_calibration(classifier=None, save_path=config.CALIBRATION_PLOT):
    """Compute + plot the calibration of the classifier on the held-out season.

    Loads the pinned classifier (via :func:`~sim.reconcile.load_pinned_classifier`)
    when none is passed, predicts ``P(P1 wins)`` on the ``TEST_YEAR`` rows, computes
    the Brier score and the decile reliability curve, and saves the plot.

    Args:
        classifier: An estimator exposing ``predict_proba``; if ``None``, the pinned
            persisted model is loaded and its identity reported.
        save_path: Where to write the reliability PNG.

    Returns:
        ``(brier, CalibrationResult)``.
    """
    if classifier is None:
        pinned = reconcile.load_pinned_classifier()
        print(
            f"📌 Pinned classifier: {pinned.estimator_class} "
            f"({pinned.n_features_in} features) from {pinned.path}"
        )
        classifier = pinned.estimator

    test = _load_test_year_frame()
    X = test[config.MODEL_FEATURES]
    y = test["target"].to_numpy()
    probs = classifier.predict_proba(X)[:, 1]  # P(P1 wins)

    brier = brier_score(probs, y)
    result = compute_calibration(probs, y, config.CALIBRATION_BINS)
    plot_calibration(result, brier, save_path)

    print(f"🎯 Brier score on TEST_YEAR {config.TEST_YEAR}: {brier:.4f} "
          f"(n={y.size}, {len(result.bin_count)} populated bins)")
    print(f"   [Saved {save_path}]")
    return brier, result


if __name__ == "__main__":
    run_calibration()
