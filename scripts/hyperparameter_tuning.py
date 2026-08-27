"""RandomForest hyperparameter search on the confirmed 27-feature set.

Phase 4 of the model-accuracy lineage (after the H2H experiment, the feature-
FAMILY ablation, and the individual-feature LOO ablation). Those three varied the
FEATURES with the estimator fixed. This one inverts that: FEATURES are frozen at
all 27 MODEL_FEATURES, and only the pinned estimator's (RandomForest's) three
capacity/regularisation knobs move.

Three-tier validation, deliberately not the prior experiments' single 2024 split,
because scoring ~27 configs against one arbitrary held-out year is the optimizer's
curse:

  TIER 1 (SEARCH): forward-chaining CV entirely WITHIN the training years
    (2014-2023). 2024 and 2025 are untouched here. Five expanding-window folds,
    each validation block a full season:
        train 2014-2018 -> val 2019
        train 2014-2019 -> val 2020
        train 2014-2020 -> val 2021
        train 2014-2021 -> val 2022
        train 2014-2022 -> val 2023
    Time-aware, not random: the deployment regime is strictly train-past ->
    predict-future, and player skill is non-stationary, so a random fold that
    trains on 2022 to predict 2015 would both leak temporal structure and
    mis-estimate forward error. The feature pipeline is already leakage-safe at
    row grain (rolling stats .shift(1); H2H/surface built in date order), so the
    only leakage the fold design must still prevent is this cross-time one.

  TIER 2 (CONFIRMATION): the SINGLE CV winner (only that one, not a shortlist) is
    trained on 2014-2023 and scored once on the untouched 2024 season, with a
    paired bootstrap CI of the Brier difference vs the baseline config -- the same
    check every prior experiment used.

  TIER 3 (FINAL): only if tier 2 confirms. Not run here -- a separate step retrains
    on <TEST_YEAR and touches 2025 + the market benchmark exactly once.

Primary metric is BRIER (a proper scoring rule, and what the whole downstream --
reconciliation, calibration, the market benchmark -- is judged on, and what
all three prior experiments decided on). Accuracy is reported alongside because
train.py's best-of-four actually SELECTS on accuracy; the tension is noted in the
write-up.

Evaluation only. Writes nothing to config, MODEL_FEATURES, or any artefact.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

# Bootstrap: run-as-script puts scripts/ on sys.path[0]; add src/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import config
import data.loader as loader
import data.preprocess as preprocess
import features.engineering as features

# --- Pre-committed, bounded grid (report before running) -------------------
# Baseline = (100, 10, 1) is deliberately a grid point, evaluated identically.
GRID_N_ESTIMATORS = [100, 300, 500]
GRID_MAX_DEPTH = [10, 16, None]        # None = unlimited depth
GRID_MIN_SAMPLES_LEAF = [1, 5, 20]
BASELINE = dict(n_estimators=100, max_depth=10, min_samples_leaf=1)
FIXED = dict(random_state=42, n_jobs=-1)  # max_features='sqrt' default; RF is
# deterministic under fixed random_state regardless of n_jobs, so parallelism is
# a pure speedup and does not affect any reported number.

VAL_YEAR = 2024                        # tier-2 confirmation season (untouched in tier 1)
CV_START = 2014                        # first training year
CV_FOLDS = [                           # expanding-window forward-chaining folds
    (2014, 2018, 2019),
    (2014, 2019, 2020),
    (2014, 2020, 2021),
    (2014, 2021, 2022),
    (2014, 2022, 2023),
]
N_BOOT = 20_000
BOOT_SEED = 42


def build_frame() -> pd.DataFrame:
    """Load -> preprocess -> engineer, exactly as predictor.main / calibrate do."""
    data = loader.load_atp_data(config.START_YEAR, config.END_YEAR)
    processed = preprocess.preprocess_data(data)
    final_df, *_ = features.add_features(processed)
    return final_df


def combos() -> list[dict]:
    """The full grid as a list of RF kwargs dicts."""
    out = []
    for n, d, leaf in itertools.product(
        GRID_N_ESTIMATORS, GRID_MAX_DEPTH, GRID_MIN_SAMPLES_LEAF
    ):
        out.append(dict(n_estimators=n, max_depth=d, min_samples_leaf=leaf))
    return out


def label(params: dict) -> str:
    d = params["max_depth"]
    return f"n={params['n_estimators']:<3} depth={str(d):<4} leaf={params['min_samples_leaf']:<2}"


def fit_predict(params: dict, train: pd.DataFrame, val: pd.DataFrame) -> np.ndarray:
    """Fit RF(params) on train, return per-row P(win) on val."""
    model = RandomForestClassifier(**params, **FIXED)
    model.fit(train[config.MODEL_FEATURES], train["target"])
    return model.predict_proba(val[config.MODEL_FEATURES])[:, 1]


def sq_err(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p - y) ** 2


def cv_evaluate(params: dict, df: pd.DataFrame) -> dict:
    """Forward-chaining CV for one config. Returns pooled + per-fold Brier/acc."""
    years = df["tourney_date"].dt.year
    pooled_se: list[np.ndarray] = []
    pooled_correct: list[np.ndarray] = []
    per_fold_brier: list[float] = []
    for tr_start, tr_end, vyear in CV_FOLDS:
        train = df.loc[(years >= tr_start) & (years <= tr_end)]
        val = df.loc[years == vyear]
        p = fit_predict(params, train, val)
        y = val["target"].to_numpy(dtype=float)
        se = sq_err(p, y)
        pooled_se.append(se)
        pooled_correct.append(((p >= 0.5).astype(float) == y).astype(float))
        per_fold_brier.append(float(se.mean()))
    all_se = np.concatenate(pooled_se)
    all_correct = np.concatenate(pooled_correct)
    return dict(
        pooled_brier=float(all_se.mean()),
        mean_fold_brier=float(np.mean(per_fold_brier)),
        std_fold_brier=float(np.std(per_fold_brier)),
        pooled_acc=float(all_correct.mean()),
        per_fold_brier=per_fold_brier,
        pooled_se=all_se,
    )


def boot_ci(delta_se: np.ndarray, boot_idx: np.ndarray) -> tuple[float, float, float]:
    """Paired bootstrap 95% CI of mean per-row Brier difference (cand - base)."""
    point = float(delta_se.mean())
    boot_means = delta_se[boot_idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return point, float(lo), float(hi)


def verdict(lo: float, hi: float) -> str:
    if hi < 0:
        return "BETTER"
    if lo > 0:
        return "worse"
    return "noise"


def main() -> None:
    print("=" * 78)
    print("TIER 1 - forward-chaining CV search within 2014-2023 (2024/2025 untouched)")
    print("=" * 78)
    grid = combos()
    print(f"Grid: n_estimators {GRID_N_ESTIMATORS} x max_depth {GRID_MAX_DEPTH} "
          f"x min_samples_leaf {GRID_MIN_SAMPLES_LEAF} = {len(grid)} configs")
    print(f"Folds ({len(CV_FOLDS)}): " +
          "; ".join(f"tr{a}-{b}->val{v}" for a, b, v in CV_FOLDS))
    print(f"Fixed: {FIXED}, max_features=sqrt (default). Metric: pooled Brier.\n")

    df = build_frame()
    fold_sizes = {v: int((df['tourney_date'].dt.year == v).sum())
                  for _, _, v in CV_FOLDS}
    print(f"val-block match counts: {fold_sizes}\n")

    results = []
    for params in grid:
        r = cv_evaluate(params, df)
        r["params"] = params
        r["is_baseline"] = (
            params["n_estimators"] == BASELINE["n_estimators"]
            and params["max_depth"] == BASELINE["max_depth"]
            and params["min_samples_leaf"] == BASELINE["min_samples_leaf"]
        )
        results.append(r)
        tag = "  <-- BASELINE" if r["is_baseline"] else ""
        print(f"{label(params)}  pooled_Brier={r['pooled_brier']:.6f}  "
              f"mean_fold={r['mean_fold_brier']:.6f}(sd {r['std_fold_brier']:.6f})  "
              f"acc={r['pooled_acc']:.4f}{tag}")

    baseline = next(r for r in results if r["is_baseline"])
    ranked = sorted(results, key=lambda r: r["pooled_brier"])
    winner = ranked[0]

    print("\n--- CV ranking (best pooled Brier first) ---")
    print("| rank | config | pooled Brier | Δ vs base | mean-fold(sd) | acc |")
    print("|---|---|---|---|---|---|")
    for i, r in enumerate(ranked, 1):
        d = r["pooled_brier"] - baseline["pooled_brier"]
        tag = " **(baseline)**" if r["is_baseline"] else ""
        print(f"| {i} | {label(r['params'])}{tag} | {r['pooled_brier']:.6f} | "
              f"{d:+.6f} | {r['mean_fold_brier']:.6f}({r['std_fold_brier']:.6f}) | "
              f"{r['pooled_acc']:.4f} |")

    print(f"\nBaseline pooled Brier: {baseline['pooled_brier']:.6f}")
    print(f"CV winner:  {label(winner['params'])}  "
          f"pooled Brier={winner['pooled_brier']:.6f}  "
          f"(Δ {winner['pooled_brier'] - baseline['pooled_brier']:+.6f})")

    # Paired bootstrap of the CV winner vs baseline on the POOLED cv rows, to see
    # whether the CV margin itself is inside noise before spending the 2024 bite.
    if not winner["is_baseline"]:
        rng = np.random.default_rng(BOOT_SEED)
        n = len(baseline["pooled_se"])
        bidx = rng.integers(0, n, size=(N_BOOT, n))
        delta = winner["pooled_se"] - baseline["pooled_se"]
        pt, lo, hi = boot_ci(delta, bidx)
        print(f"CV winner vs baseline, paired bootstrap on pooled CV rows: "
              f"Δ={pt:+.6f} 95%CI[{lo:+.6f},{hi:+.6f}] -> {verdict(lo, hi)}")
    else:
        print("CV winner IS the baseline config -> tier 2/3 unnecessary; stop.")
        return

    # --- TIER 2: single CV winner, one shot on untouched 2024 ---------------
    print("\n" + "=" * 78)
    print(f"TIER 2 - CV winner confirmed once on untouched {VAL_YEAR}")
    print("=" * 78)
    years = df["tourney_date"].dt.year
    train_23 = df.loc[(years >= CV_START) & (years < VAL_YEAR)]
    val_24 = df.loc[years == VAL_YEAR]
    assert train_23["tourney_date"].dt.year.max() < VAL_YEAR
    assert config.TEST_YEAR not in train_23["tourney_date"].dt.year.unique()
    print(f"train {CV_START}-{VAL_YEAR-1} (n={len(train_23)}), "
          f"val [{VAL_YEAR}] (n={len(val_24)})")

    y24 = val_24["target"].to_numpy(dtype=float)
    p_base = fit_predict(baseline["params"], train_23, val_24)
    p_win = fit_predict(winner["params"], train_23, val_24)
    se_base, se_win = sq_err(p_base, y24), sq_err(p_win, y24)
    b_base, b_win = float(se_base.mean()), float(se_win.mean())
    acc_base = float(((p_base >= 0.5) == y24).mean())
    acc_win = float(((p_win >= 0.5) == y24).mean())

    rng = np.random.default_rng(BOOT_SEED)
    bidx = rng.integers(0, len(val_24), size=(N_BOOT, len(val_24)))
    pt, lo, hi = boot_ci(se_win - se_base, bidx)

    print(f"\nbaseline {label(baseline['params'])}: Brier={b_base:.6f} acc={acc_base:.4f}")
    print(f"winner   {label(winner['params'])}: Brier={b_win:.6f} acc={acc_win:.4f}")
    print(f"\nΔBrier (winner - baseline) on {VAL_YEAR}: {pt:+.6f}  "
          f"95%CI[{lo:+.6f},{hi:+.6f}]  -> {verdict(lo, hi)}")
    print("\nDECISION RULE: tier-2 confirms only if the 2024 CI is entirely below 0")
    print("(BETTER). 'noise' or 'worse' => stop, keep baseline hyperparameters.")


if __name__ == "__main__":
    main()
