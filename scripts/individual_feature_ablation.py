"""Individual-feature leave-one-out ablation across all 27 MODEL_FEATURES.

Phase 3 of the model-accuracy lineage (after the §9 H2H experiment and the §10
feature-FAMILY ablation). This asks, at single-feature grain: can any one of the
27 features be dropped without cost, and, the key cross-check against §10, are
the individually-neutral features collectively load-bearing?

Discipline (identical to §10, tightened for 27 comparisons):
  * Validation season is 2024. Train strictly on years < 2024. TEST_YEAR (2025)
    and the market benchmark are NOT touched here, only the final confirmatory
    step (a separate script/run) may touch them, and only if a candidate clears
    the corrected bar.
  * Estimator FIXED to RandomForestClassifier(n_estimators=100, max_depth=10,
    random_state=42) for every ablation run, isolating the feature variable.
  * Paired bootstrap of the Brier difference vs baseline. Report BOTH the raw
    95% CI and the Bonferroni-corrected CI (alpha = 0.05/27), the corrected
    verdict being the decision rule.

Evaluation only. Writes nothing to config, MODEL_FEATURES, or any artefact.
"""

from __future__ import annotations

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

VAL_YEAR = 2024
RF_KWARGS = dict(n_estimators=100, max_depth=10, random_state=42)
N_BOOT = 20_000          # tail resolution for a 99.8% CI needs more than §10's 5,000
BOOT_SEED = 42
ALPHA = 0.05
N_COMPARISONS = len(config.MODEL_FEATURES)  # 27
ALPHA_CORR = ALPHA / N_COMPARISONS


def build_frame() -> pd.DataFrame:
    """Load -> preprocess -> engineer, exactly as predictor.main / calibrate do."""
    data = loader.load_atp_data(config.START_YEAR, config.END_YEAR)
    processed = preprocess.preprocess_data(data)
    final_df, *_ = features.add_features(processed)
    return final_df


def split(df: pd.DataFrame):
    """Train = years < VAL_YEAR; validation = exactly [VAL_YEAR]. Assert both."""
    years = df["tourney_date"].dt.year
    train = df.loc[years < VAL_YEAR]
    val = df.loc[years == VAL_YEAR]
    train_years = sorted(train["tourney_date"].dt.year.unique().tolist())
    val_years = sorted(val["tourney_date"].dt.year.unique().tolist())
    assert max(train_years) < VAL_YEAR, f"train leaks >= {VAL_YEAR}: {train_years}"
    assert val_years == [VAL_YEAR], f"val must be exactly [{VAL_YEAR}], got {val_years}"
    assert VAL_YEAR not in train_years
    assert config.TEST_YEAR not in train_years and config.TEST_YEAR not in val_years
    print(f"train years {train_years[0]}-{train_years[-1]} (n={len(train)}), "
          f"val [{VAL_YEAR}] (n={len(val)})")
    return train, val


def sq_err(feature_list, train, val) -> np.ndarray:
    """Fit RF on feature_list, return per-row squared error on the val set."""
    model = RandomForestClassifier(**RF_KWARGS)
    model.fit(train[feature_list], train["target"])
    p = model.predict_proba(val[feature_list])[:, 1]
    y = val["target"].to_numpy(dtype=float)
    return (p - y) ** 2


def boot_ci(delta_se: np.ndarray, boot_idx: np.ndarray):
    """Paired bootstrap CIs of mean per-row Brier difference (candidate - base).

    delta_se: per-row (candidate_sq_err - baseline_sq_err), length n_val.
    boot_idx: (N_BOOT, n_val) shared resample indices.
    Returns (point_delta, raw95_lo, raw95_hi, corr_lo, corr_hi).
    """
    point = float(delta_se.mean())
    boot_means = delta_se[boot_idx].mean(axis=1)
    raw_lo, raw_hi = np.percentile(boot_means, [2.5, 97.5])
    corr_lo, corr_hi = np.percentile(
        boot_means, [100 * ALPHA_CORR / 2, 100 * (1 - ALPHA_CORR / 2)]
    )
    return point, float(raw_lo), float(raw_hi), float(corr_lo), float(corr_hi)


def verdict(lo: float, hi: float) -> str:
    """A CI entirely below 0 => better; entirely above 0 => worse; else noise."""
    if hi < 0:
        return "BETTER"
    if lo > 0:
        return "worse"
    return "noise"


def main() -> None:
    print(f"Bonferroni: {N_COMPARISONS} comparisons, alpha_corr = {ALPHA_CORR:.5f} "
          f"(~{100*(1-ALPHA_CORR):.3f}% CI)\n")
    df = build_frame()
    train, val = split(df)

    base_se = sq_err(config.MODEL_FEATURES, train, val)
    base_brier = float(base_se.mean())
    print(f"\nBASELINE (all {len(config.MODEL_FEATURES)}): Brier = {base_brier:.6f} "
          f"(n={len(val)})")
    print(f"  sanity vs §10 family baseline (0.217470): "
          f"delta = {base_brier - 0.217470:+.6f}\n")

    rng = np.random.default_rng(BOOT_SEED)
    boot_idx = rng.integers(0, len(val), size=(N_BOOT, len(val)))

    rows = []
    for feat in config.MODEL_FEATURES:
        reduced = [f for f in config.MODEL_FEATURES if f != feat]
        cand_se = sq_err(reduced, train, val)
        delta = cand_se - base_se
        pt, rlo, rhi, clo, chi = boot_ci(delta, boot_idx)
        raw_v = verdict(rlo, rhi)
        corr_v = verdict(clo, chi)
        rows.append(dict(feature=feat, brier=float(cand_se.mean()), delta=pt,
                         raw_lo=rlo, raw_hi=rhi, raw=raw_v,
                         corr_lo=clo, corr_hi=chi, corr=corr_v))
        print(f"drop {feat:<32} Brier={cand_se.mean():.6f} d={pt:+.6f}  "
              f"raw95[{rlo:+.5f},{rhi:+.5f}]={raw_v:<6} "
              f"corr[{clo:+.5f},{chi:+.5f}]={corr_v}")

    # Compounding-effect candidate: drop every feature that is neutral-or-better
    # under the CORRECTED bar, all at once.
    neutral_or_better = [r["feature"] for r in rows if r["corr"] in ("noise", "BETTER")]
    print(f"\nCompounding candidate: drop all {len(neutral_or_better)} "
          f"corrected-neutral-or-better features at once")
    print(f"  = {neutral_or_better}")
    kept = [f for f in config.MODEL_FEATURES if f not in neutral_or_better]
    print(f"  keeping {len(kept)}: {kept}")
    if kept:
        comb_se = sq_err(kept, train, val)
        delta = comb_se - base_se
        pt, rlo, rhi, clo, chi = boot_ci(delta, boot_idx)
        print(f"COMBINED  Brier={comb_se.mean():.6f} d={pt:+.6f}  "
              f"raw95[{rlo:+.5f},{rhi:+.5f}]={verdict(rlo,rhi):<6} "
              f"corr[{clo:+.5f},{chi:+.5f}]={verdict(clo,chi)}")
    else:
        print("COMBINED skipped: nothing left to keep.")

    winners = [r["feature"] for r in rows if r["corr"] == "BETTER"]
    print(f"\n=== corrected-bar improvements among LOO runs: {winners or 'NONE'} ===")

    # Emit a markdown table for the write-up.
    print("\n\n| drop | Brier | Δ | raw 95% CI | raw | corrected CI | corrected |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['feature']} | {r['brier']:.6f} | {r['delta']:+.6f} | "
              f"[{r['raw_lo']:+.5f}, {r['raw_hi']:+.5f}] | {r['raw']} | "
              f"[{r['corr_lo']:+.5f}, {r['corr_hi']:+.5f}] | {r['corr']} |")


if __name__ == "__main__":
    main()
