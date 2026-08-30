"""Benchmark the model's held-out-season predictions against de-vigged market odds, evaluation only.

Joins the model's own ``TEST_YEAR`` rows to a static, hand-committed bookmaker-price snapshot and
reports Brier + a reliability curve for both sides over the identical matched set. The dependency
is strictly one-way (enforced by ``tests/test_benchmark_vs_market.py``); the odds snapshot is a
manual download no automated job may fetch. The probability scored is the classifier's, which
equals the reconciled one only under the default ``classifier_anchor``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Running as a script puts scripts/ on sys.path[0], not src/; add src/.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import model.calibrate as calibrate  # noqa: E402
import sim.reconcile as reconcile  # noqa: E402
from common.names import NameIndex, resolve_name  # noqa: E402

# Bookmaker prefix -> (winner-price column, loser-price column), a property of the vendor's layout.
BOOK_COLUMNS: dict[str, tuple[str, str]] = {
    "B365": ("B365W", "B365L"),
    "PS": ("PSW", "PSL"),
    "Max": ("MaxW", "MaxL"),
    "Avg": ("AvgW", "AvgL"),
    "BFE": ("BFEW", "BFEL"),
}

# Columns the join needs; a snapshot missing any is rejected up front.
REQUIRED_COLUMNS = ("Date", "Winner", "Loser", "Comment")


def devig(odds_winner: float, odds_loser: float) -> float:
    """Proportional de-vig of a two-way decimal-odds market: ``(1/oW) / (1/oW + 1/oL)``; raises ValueError on a price <= 1.0."""
    for label, value in (("odds_winner", odds_winner), ("odds_loser", odds_loser)):
        v = float(value)
        if not np.isfinite(v):
            raise ValueError(f"{label} must be finite, got {value!r}")
        if v <= 1.0:
            raise ValueError(f"{label} must be > 1.0 (decimal odds), got {value!r}")

    inv_w = 1.0 / float(odds_winner)
    inv_l = 1.0 / float(odds_loser)
    return inv_w / (inv_w + inv_l)


def overround(odds_winner: float, odds_loser: float) -> float:
    """The book's margin: ``1/oW + 1/oL − 1``. Reported, never subtracted twice."""
    return 1.0 / float(odds_winner) + 1.0 / float(odds_loser) - 1.0


@dataclass(frozen=True)
class NameResolution:
    """Outcome of resolving one vendor-format name; ``reason`` is None, ``"no_match"`` or ``"ambiguous"``."""

    raw: str
    name: str | None
    reason: str | None = None
    candidates: tuple[str, ...] = ()


def market_name_query_forms(raw: str) -> list[str]:
    """Rewrite a vendor ``"Surname F."`` name into query forms ``common.names.resolve_name`` handles, most specific first (formatting, not matching)."""
    text = str(raw).strip()
    forms: list[str] = []

    parts = text.split()
    if len(parts) >= 2 and parts[-1].endswith("."):
        initials = parts[-1].rstrip(".")
        surname = " ".join(parts[:-1])
        first_initial = initials.split(".")[0]
        if len(first_initial) == 1:
            forms.append(f"{first_initial} {surname}")
        forms.append(surname)

    forms.append(text)

    seen: list[str] = []
    for form in forms:
        if form and form not in seen:
            seen.append(form)
    return seen


def resolve_market_name(
    raw: str, index: NameIndex, cache: dict[str, NameResolution] | None = None
) -> NameResolution:
    """Resolve one vendor name to a canonical model name via the first unambiguous query form, else the most informative failure (ambiguity beats no-match). Never raises or prints."""
    if cache is not None and raw in cache:
        return cache[raw]

    failure = NameResolution(raw=raw, name=None, reason="no_match")
    for query in market_name_query_forms(raw):
        match = resolve_name(query, index)
        if match is None:
            continue
        if match.is_ambiguous:
            if failure.reason != "ambiguous":
                failure = NameResolution(
                    raw=raw,
                    name=None,
                    reason="ambiguous",
                    candidates=tuple(match.candidates),
                )
            continue
        result = NameResolution(raw=raw, name=match.name, reason=None)
        break
    else:
        result = failure

    if cache is not None:
        cache[raw] = result
    return result


@dataclass
class JoinReport:
    """Everything the join produced; ``skipped`` carries every unusable market row with its date, players and reason."""

    matched: pd.DataFrame = field(default_factory=pd.DataFrame)
    skipped: list[dict] = field(default_factory=list)
    n_market_rows: int = 0
    n_model_rows: int = 0
    unresolved_names: dict[str, NameResolution] = field(default_factory=dict)

    @property
    def n_matched(self) -> int:
        return len(self.matched)

    @property
    def resolution_rate(self) -> float:
        """Fraction of market rows joined to a model row. 0.0 when the file is empty."""
        return self.n_matched / self.n_market_rows if self.n_market_rows else 0.0

    def skipped_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.skipped:
            counts[row["reason"]] = counts.get(row["reason"], 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def load_market_snapshot(path=config.BENCHMARK_ODDS_SNAPSHOT) -> pd.DataFrame:
    """Read the committed odds snapshot offline, ``Date`` parsed to datetime; raises if missing or short a required column."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No market-odds snapshot at {path}. This file is a one-time MANUAL "
            "download and no script fetches it. See this module's docstring for "
            "how to reproduce it."
        )

    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Market snapshot {path} is missing required column(s): {missing}. "
            f"Found: {list(df.columns)}"
        )
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def market_prob_for_p1(row: pd.Series, p1_name: str, p1_is_winner: bool, book: str) -> float | None:
    """De-vigged ``P(p1 wins)``, flipped onto p1 when p1 lost, or ``None`` if either price is absent or unusable."""
    col_w, col_l = BOOK_COLUMNS[book]
    odds_w, odds_l = row.get(col_w), row.get(col_l)
    if pd.isna(odds_w) or pd.isna(odds_l):
        return None
    try:
        p_winner = devig(float(odds_w), float(odds_l))
    except ValueError:
        return None
    return p_winner if p1_is_winner else 1.0 - p_winner


def join_market_to_model(
    market: pd.DataFrame,
    model_rows: pd.DataFrame,
    book: str = config.BENCHMARK_DEFAULT_BOOK,
    min_days: int = config.BENCHMARK_JOIN_MIN_DAYS,
    max_days: int = config.BENCHMARK_JOIN_MAX_DAYS,
) -> JoinReport:
    """Join market rows onto the model's held-out-season rows, keyed on the unordered name pair plus a date window (each model row consumed once, nearest date first)."""
    if book not in BOOK_COLUMNS:
        raise KeyError(f"Unknown book {book!r}; known: {sorted(BOOK_COLUMNS)}")
    for col in BOOK_COLUMNS[book]:
        if col not in market.columns:
            raise KeyError(f"Market snapshot has no column {col!r} for book {book!r}")

    report = JoinReport(n_market_rows=len(market), n_model_rows=len(model_rows))

    names = sorted(set(model_rows["p1_name"]) | set(model_rows["p2_name"]))
    index = NameIndex.from_names(names)
    cache: dict[str, NameResolution] = {}

    # Candidate model rows per unordered name pair, so the scan is a dict lookup.
    dates = pd.to_datetime(model_rows["tourney_date"])
    pairs: dict[frozenset, list[tuple[pd.Timestamp, object]]] = {}
    for idx, p1, p2, when in zip(
        model_rows.index, model_rows["p1_name"], model_rows["p2_name"], dates
    ):
        pairs.setdefault(frozenset((p1, p2)), []).append((when, idx))

    used: set[object] = set()
    matched_records: list[dict] = []

    for _, row in market.iterrows():
        def skip(reason: str, **extra) -> None:
            report.skipped.append(
                {
                    "reason": reason,
                    "date": row["Date"].date().isoformat(),
                    "winner": row["Winner"],
                    "loser": row["Loser"],
                    **extra,
                }
            )

        res_w = resolve_market_name(row["Winner"], index, cache)
        res_l = resolve_market_name(row["Loser"], index, cache)
        for res in (res_w, res_l):
            if res.name is None:
                report.unresolved_names.setdefault(res.raw, res)
        if res_w.name is None or res_l.name is None:
            unresolved = [r.raw for r in (res_w, res_l) if r.name is None]
            reasons = [r.reason for r in (res_w, res_l) if r.name is None]
            skip("unresolved_name", players=unresolved, detail=reasons)
            continue

        candidates = pairs.get(frozenset((res_w.name, res_l.name)))
        if not candidates:
            skip("no_model_match", players=[res_w.name, res_l.name])
            continue

        in_window = [
            (abs((row["Date"] - when).days), when, idx)
            for when, idx in candidates
            if idx not in used and min_days <= (row["Date"] - when).days <= max_days
        ]
        if not in_window:
            skip("no_model_match_in_window", players=[res_w.name, res_l.name])
            continue

        _, _, idx = min(in_window, key=lambda c: c[0])

        p1_name = model_rows.at[idx, "p1_name"]
        p1_is_winner = p1_name == res_w.name
        p_market = market_prob_for_p1(row, p1_name, p1_is_winner, book)
        if p_market is None:
            skip("missing_odds", players=[res_w.name, res_l.name], detail=book)
            continue

        used.add(idx)
        matched_records.append(
            {
                "model_index": idx,
                "date": row["Date"],
                "p1_name": p1_name,
                "p2_name": model_rows.at[idx, "p2_name"],
                "tournament": row.get("Tournament"),
                "comment": row.get("Comment"),
                "p_market": p_market,
                "outcome": int(model_rows.at[idx, "target"]),
                "overround": overround(
                    float(row[BOOK_COLUMNS[book][0]]), float(row[BOOK_COLUMNS[book][1]])
                ),
            }
        )

    report.matched = pd.DataFrame(matched_records)
    return report


def plot_comparison(
    model_result: calibrate.CalibrationResult,
    market_result: calibrate.CalibrationResult,
    model_brier: float,
    market_brier: float,
    n: int,
    book: str,
    save_path=config.BENCHMARK_PLOT,
) -> None:
    """Save one reliability chart carrying both curves (a second plot, not an edit of outputs/calibration.png)."""
    import os

    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect calibration")

    for result, colour, label, brier in (
        (model_result, "tab:blue", "Reconciled model", model_brier),
        (market_result, "tab:orange", f"Market ({book}, de-vigged)", market_brier),
    ):
        sizes = 20.0 + 200.0 * result.bin_count / max(result.bin_count.max(), 1)
        ax.scatter(
            result.bin_pred,
            result.bin_emp,
            s=sizes,
            c=colour,
            alpha=0.8,
            edgecolors="k",
            zorder=3,
            label=f"{label}: Brier {brier:.4f}",
        )
        ax.plot(result.bin_pred, result.bin_emp, color=colour, alpha=0.45, zorder=2)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted P(P1 wins)")
    ax.set_ylabel("Empirical win rate")
    ax.set_title(
        f"Model vs market, TEST_YEAR {config.TEST_YEAR}\n"
        f"same {n} matched matches (bin size ∝ n)"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def format_report(
    join: JoinReport,
    model_brier: float,
    market_brier: float,
    model_result: calibrate.CalibrationResult,
    market_result: calibrate.CalibrationResult,
    book: str,
    top_unresolved: int,
) -> str:
    """Render the side-by-side text report (returned, not printed)."""
    lines: list[str] = []
    add = lines.append

    add(f"Market-odds benchmark, TEST_YEAR {config.TEST_YEAR}")
    add("=" * 72)
    add(f"Snapshot          : {config.BENCHMARK_ODDS_SNAPSHOT} (static, manual download)")
    add(f"Bookmaker         : {book} ({'/'.join(BOOK_COLUMNS[book])}), proportional de-vig")
    add(f"Model probability : pinned classifier, RECONCILE_MODE={config.RECONCILE_MODE!r}")
    add("")
    add("JOIN")
    add("-" * 72)
    add(f"Market rows                : {join.n_market_rows}")
    add(f"Model {config.TEST_YEAR} rows            : {join.n_model_rows}")
    add(
        f"Matched (both scored on)   : {join.n_matched} "
        f"({join.resolution_rate:.1%} of market rows)"
    )
    for reason, count in join.skipped_by_reason().items():
        add(f"  skipped: {reason:<24} {count}")
    add(f"Unresolved distinct names  : {len(join.unresolved_names)}")
    for res in list(join.unresolved_names.values())[:top_unresolved]:
        detail = f" candidates={list(res.candidates)}" if res.candidates else ""
        add(f"  {res.raw!r}: {res.reason}{detail}")
    if len(join.unresolved_names) > top_unresolved:
        add(f"  … {len(join.unresolved_names) - top_unresolved} more (--top-unresolved)")
    add("")
    add("BRIER SCORE (lower is better; identical match set)")
    add("-" * 72)
    add(f"  Reconciled model : {model_brier:.4f}")
    add(f"  Market ({book:<4})     : {market_brier:.4f}")
    delta = model_brier - market_brier
    verdict = "market is better" if delta > 0 else "model is better"
    add(f"  Difference       : {delta:+.4f}  ({verdict})")
    if not join.matched.empty:
        add(f"  Mean overround   : {join.matched['overround'].mean():.4f}")
    add("")
    add("  READ THE GAP AS A LOWER BOUND: the model's side is the flattered one.")
    add(f"      model/train.py picks the best of four estimators by accuracy on")
    add(f"      TEST_YEAR ({config.TEST_YEAR}) itself, the very season scored above. The model's")
    add("      number therefore carries a model-selection advantage on this data that")
    add("      the market's carries no equivalent of: the bookmaker priced each match")
    add("      before it was played and was never tuned against this sample. A clean")
    add("      out-of-sample comparison would, if anything, widen the gap rather than")
    add("      close it, so do not cite this as the model's best case against the market.")
    add("")
    add("CALIBRATION (equal-width bins, empty bins dropped)")
    add("-" * 72)
    add(f"{'bin pred':>10} {'model emp':>10} {'n':>6}   |{'bin pred':>10} {'mkt emp':>10} {'n':>6}")
    for i in range(max(len(model_result.bin_count), len(market_result.bin_count))):
        left = (
            f"{model_result.bin_pred[i]:>10.3f} {model_result.bin_emp[i]:>10.3f} "
            f"{model_result.bin_count[i]:>6d}"
            if i < len(model_result.bin_count)
            else " " * 28
        )
        right = (
            f"{market_result.bin_pred[i]:>10.3f} {market_result.bin_emp[i]:>10.3f} "
            f"{market_result.bin_count[i]:>6d}"
            if i < len(market_result.bin_count)
            else ""
        )
        add(f"{left}   |{right}")
    add("")
    add(f"Plot   : {config.BENCHMARK_PLOT}")
    add(f"Report : {config.BENCHMARK_REPORT}")
    return "\n".join(lines)


def run_benchmark(
    snapshot_path=config.BENCHMARK_ODDS_SNAPSHOT,
    book: str = config.BENCHMARK_DEFAULT_BOOK,
    classifier=None,
    model_rows: pd.DataFrame | None = None,
    top_unresolved: int = 20,
    save_plot: bool = True,
):
    """Run the whole comparison, write the plot + report, and return ``(join_report, model_brier, market_brier, report_text)``."""
    market = load_market_snapshot(snapshot_path)

    if model_rows is None:
        # Reuse the model's own held-out-season selection, so this scores the same rows as calibration.png.
        model_rows = calibrate._load_test_year_frame()

    join = join_market_to_model(market, model_rows, book=book)
    if join.matched.empty:
        raise RuntimeError(
            "No market rows joined to the model's held-out season. Refusing to "
            "report a Brier comparison over an empty set. "
            f"Skips: {join.skipped_by_reason()}"
        )

    if classifier is None:
        pinned = reconcile.load_pinned_classifier()
        print(
            f"Pinned classifier: {pinned.estimator_class} "
            f"({pinned.n_features_in} features) from {pinned.path}"
        )
        classifier = pinned.estimator

    subset = model_rows.loc[join.matched["model_index"]]
    p_model = classifier.predict_proba(subset[config.MODEL_FEATURES])[:, 1]
    p_market = join.matched["p_market"].to_numpy(dtype=float)
    outcomes = join.matched["outcome"].to_numpy(dtype=float)

    # Both probabilities and the outcomes must describe one identical match set.
    if not (len(p_model) == len(p_market) == len(outcomes) == join.n_matched):
        raise RuntimeError(
            "Refusing to report: the model and market probabilities are not over "
            f"the same match set: model={len(p_model)}, market={len(p_market)}, "
            f"outcomes={len(outcomes)}, matched={join.n_matched}."
        )

    model_brier = calibrate.brier_score(p_model, outcomes)
    market_brier = calibrate.brier_score(p_market, outcomes)
    model_result = calibrate.compute_calibration(p_model, outcomes, config.CALIBRATION_BINS)
    market_result = calibrate.compute_calibration(p_market, outcomes, config.CALIBRATION_BINS)

    if save_plot:
        plot_comparison(
            model_result, market_result, model_brier, market_brier,
            n=len(outcomes), book=book,
        )

    text = format_report(
        join, model_brier, market_brier, model_result, market_result, book, top_unresolved
    )

    if save_plot:
        config.BENCHMARK_REPORT.parent.mkdir(parents=True, exist_ok=True)
        config.BENCHMARK_REPORT.write_text(text + "\n")

    return join, model_brier, market_brier, text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the reconciled model's held-out-season calibration against "
            "de-vigged bookmaker odds (evaluation only)."
        )
    )
    parser.add_argument(
        "--snapshot", default=config.BENCHMARK_ODDS_SNAPSHOT,
        help="Static odds CSV (default: the committed snapshot).",
    )
    parser.add_argument(
        "--book", default=config.BENCHMARK_DEFAULT_BOOK, choices=sorted(BOOK_COLUMNS),
        help="Which bookmaker's two price columns to de-vig.",
    )
    parser.add_argument(
        "--top-unresolved", type=int, default=20,
        help="How many unresolved names to list in the report.",
    )
    args = parser.parse_args(argv)

    try:
        _, _, _, text = run_benchmark(
            snapshot_path=args.snapshot, book=args.book, top_unresolved=args.top_unresolved
        )
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"{exc}")
        return 1

    print()
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
