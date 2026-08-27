"""Benchmark the reconciled model against de-vigged market odds.

Phase 6 is **not required for v1**. This is a one-off, read-only "how good are we,
really" check: it takes the model's own held-out-season predictions (produced by
``model/calibrate.py``), joins them to a static snapshot of
bookmaker prices for the same season, and reports Brier score + a reliability
curve for **both, over the identical matched set** (evaluation only, never
folded back into training)::

    python scripts/benchmark_vs_market.py
    python scripts/benchmark_vs_market.py --book PS --top-unresolved 40

Outputs ``config.BENCHMARK_PLOT`` (``outputs/calibration_vs_market.png``) next to
the model's own ``outputs/calibration.png``, plus a text report at
``config.BENCHMARK_REPORT``.

--------------------------------------------------------------------------------
The hard boundary: evaluation only, permanently
--------------------------------------------------------------------------------
``tennis-data.co.uk`` data must never reach ``data/preprocess.py``, ``features/``,
``sim/``, or any training/fitting code. ``ace-02-data-schema.md`` records why the
source was rejected for the primary pipeline (no serve/return columns at all, no
TLS, framed around betting-system development), and folding
market odds into the model would be self-defeating: any accuracy gain would just
be the model re-deriving the market's answer rather than an independent one.

The dependency is therefore strictly one-way. This script imports the pipeline;
**nothing in the pipeline imports this script**, and the market frame never
leaves this module, it is read here, joined here, scored here, and the only
things that escape are a PNG, a text report and a printed summary.
``tests/test_benchmark_vs_market.py`` proves that by AST-walking every module
under ``src/`` and ``scripts/`` rather than taking it on trust.

--------------------------------------------------------------------------------
Provenance of the snapshot, a one-time MANUAL download (2026-08-09)
--------------------------------------------------------------------------------
``data/benchmarks/tennis_data_atp_2025.csv`` is committed, static, and fetched by
no automated script. Do **not** wire it into ``scripts/refresh_data.py`` or
``.github/workflows/refresh-and-simulate.yml``; the whole point of the refresh workflow's gates is
that unattended jobs only touch sources that can be verified in transit.

How it was obtained, and the HTTP/HTTPS decision:

1. The origin serves **no TLS at all**, ``https://www.tennis-data.co.uk/…`` and
   ``https://tennis-data.co.uk/…`` both fail the handshake outright
   (``tlsv1 alert internal error``), so there is no secure origin to prefer.
2. A secure mirror *does* exist: the Wayback Machine serves the same object over
   HTTPS. The capture at
   ``https://web.archive.org/web/20260307200645id_/http://www.tennis-data.co.uk/2025/2025.xlsx``
   was downloaded over TLS (427,318 bytes, sha256
   ``8acfcf78d18e0024a145009ff382e940d765d6fab6087d696fae2bea2dd217e2``).
3. The live origin file was **also** fetched, over plain HTTP
   (``http://www.tennis-data.co.uk/2025/2025.xlsx``, 426,707 bytes, sha256
   ``941aaa1abc49131f51e1f7f6eee93dac829dd72578ca10b341dfc4c9d41ba013``), and the
   two were compared cell by cell. They agree on **2,642 of 2,644 rows**. The two
   that differ are corrections the vendor made after the March 2026 capture:
   Rome R2 Moutet–Humbert and Roland Garros R1 Quinn–Dimitrov. Both are
   retirements *in the corrected data*, and in both the older capture has
   ``Winner``/``Loser``, and their prices, ranks, sets and game scores, the
   wrong way round. (The stale copy is not merely inverted on the Roland Garros
   row: it also labels that match ``Completed`` rather than ``Retired``.)

   Which copy is right was settled against a source independent of both: this
   repo's own vendored ``data/raw/atp_matches_2025.csv`` records
   ``Corentin Moutet d. Ugo Humbert 6-3 4-0 RET`` and
   ``Ethan Quinn d. Grigor Dimitrov 2-6 3-6 6-2 RET``, the live-origin values.
4. **The committed file is the corrected origin copy**, converted ``.xlsx`` →
   ``.csv``. The archived HTTPS copy is retained as the *integrity control*
   rather than as the data: agreeing with an independently-delivered,
   TLS-authenticated archive on 2,642/2,644 rows is a far stronger statement
   about the HTTP bytes than the transport alone would have been, and choosing
   the archive instead would have meant knowingly benchmarking against two
   inverted results. Serving the stale copy was the only thing TLS-purity would
   have bought here.

Why plain HTTP is acceptable for this step and forbidden in the scheduled
refresh: this is a
one-time fetch of a static file that a human downloaded, diffed against a second
independent source, inspected, and committed, so any tampering had to survive
review and is frozen in git history. An automated, unattended refresh has none of
those properties, nobody looks at the bytes, and a bad fetch would silently
retrain the model.

The ``.xlsx`` → ``.csv`` conversion is deliberate too: reading the vendor's
workbook needs ``openpyxl``, and Phase 6 is optional, so the conversion was done
once by hand rather than adding a dependency to ``requirements.txt`` that CI and
both Docker images would then install forever. The CSV also matches how every
other vendored table in this repo is stored (``data/raw/*.csv``).

--------------------------------------------------------------------------------
Columns actually present in the file (verified, not assumed)
--------------------------------------------------------------------------------
The ticket called ``B365W``/``B365L`` illustrative. They are in fact real. The
snapshot's 38 columns are::

    ATP Location Tournament Date Series Court Surface Round "Best of"
    Winner Loser WRank LRank WPts LPts W1 L1 W2 L2 W3 L3 W4 L4 W5 L5
    Wsets Lsets Comment B365W B365L PSW PSL MaxW MaxL AvgW AvgL BFEW BFEL

Odds are **decimal**, and the columns are keyed by *result* (``…W`` = the price on
the player who went on to win), not by home/away, so a row leaks its own outcome
and must be re-oriented onto the model's p1/p2 convention before scoring, which
:func:`market_prob_for_p1` does. Coverage over the 2,644 rows: ``B365`` 2,632,
``PS`` 2,534, ``Max``/``Avg`` 2,644, ``BFE`` 651.

--------------------------------------------------------------------------------
What "the model's prediction" means here
--------------------------------------------------------------------------------
The same thing it means in ``model/calibrate.py``, and for the same reason: under
the default ``config.RECONCILE_MODE = "classifier_anchor"`` the reconciled
match-win probability *equals* ``P_clf``, so scoring the pinned classifier's
``predict_proba`` on the held-out season is scoring the reconciled model. Under
``"blend"`` the authoritative probability shifts by ``(1−w)·(P_point − P_clf)``
per match and would need the skill table plus per-row name→id resolution to
reproduce, out of scope here exactly as it was there. The report states which
mode it ran under so the number is never read as more than it is.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Running this as a script puts scripts/ on sys.path[0], not src/, the same
# bootstrap cli/simulate_match.py and scripts/validate_sim.py carry.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import model.calibrate as calibrate  # noqa: E402
import sim.reconcile as reconcile  # noqa: E402
from common.names import NameIndex, resolve_name  # noqa: E402

# Bookmaker prefix -> (winner-price column, loser-price column). A property of
# the vendor's file layout, so it lives with the reader rather than in config.
BOOK_COLUMNS: dict[str, tuple[str, str]] = {
    "B365": ("B365W", "B365L"),
    "PS": ("PSW", "PSL"),
    "Max": ("MaxW", "MaxL"),
    "Avg": ("AvgW", "AvgL"),
    "BFE": ("BFEW", "BFEL"),
}

# Columns the join needs; a snapshot missing any of them is rejected up front
# rather than producing a mysteriously empty report.
REQUIRED_COLUMNS = ("Date", "Winner", "Loser", "Comment")


# ==========================================================================
# De-vigging (pure)
# ==========================================================================

def devig(odds_winner: float, odds_loser: float) -> float:
    """Remove the overround from a two-way decimal-odds market.

    Implied probabilities are the reciprocals ``1/oW`` and ``1/oL``; because the
    book prices in its margin they sum to more than 1 (the overround, or vig).
    Normalising by that sum, proportional, a.k.a. multiplicative de-vigging,
    gives ``p_winner = (1/oW) / (1/oW + 1/oL)``.

    Worked example from the ticket: ``1.50`` / ``2.80`` implies ``0.6667`` and
    ``0.3571``, summing to ``1.0238`` (a 2.38% overround); the de-vigged winner
    probability is ``0.6667 / 1.0238 = 28/43 = 0.651163``, and the loser takes
    ``15/43 = 0.348837``.

    Note the expression simplifies to ``oL / (oW + oL)``, but it is written in
    reciprocal form because that is the form the definition takes and the form
    that generalises past two outcomes.

    Args:
        odds_winner: Decimal odds offered on the player who won.
        odds_loser: Decimal odds offered on the player who lost.

    Returns:
        The de-vigged implied probability that the winner wins, in ``(0, 1)``.

    Raises:
        ValueError: If either price is not a finite number strictly above 1.0.
            Decimal odds of 1.0 imply a certainty (and a zero-profit bet); at or
            below that the quote is not a real price.
    """
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


# ==========================================================================
# Name resolution (reuses common/names.py, no second matcher)
# ==========================================================================

@dataclass(frozen=True)
class NameResolution:
    """Outcome of resolving one vendor-format name against the model's names.

    Attributes:
        raw: The vendor string, verbatim.
        name: The canonical model name, or ``None`` when unresolved.
        reason: ``None`` on success; otherwise ``"no_match"`` or ``"ambiguous"``.
        candidates: The ambiguous candidates, when ``reason == "ambiguous"``.
    """

    raw: str
    name: str | None
    reason: str | None = None
    candidates: tuple[str, ...] = ()


def market_name_query_forms(raw: str) -> list[str]:
    """Rewrite a vendor name into query forms ``common.names`` already understands.

    This is **formatting, not matching**, every candidate string produced here is
    handed to :func:`common.names.resolve_name`, which owns all four matching
    strategies. tennis-data.co.uk writes ``"Surname F."`` (and ``"Surname F.C."``
    for compound given names) whereas the model carries full names like
    ``"Carlos Alcaraz"``, so the surname-last form is transposed to the
    ``"F Surname"`` shape the resolver's *initials* strategy was written for.

    The forms are tried in order, most specific first:

    1. ``"F Surname"``, first initial + surname, feeding the initials strategy.
    2. ``"Surname"``, surname alone, feeding the substring strategy. This is what
       rescues compound given names (``"Struff J.L."`` → ``"Jan-Lennard Struff"``)
       and compound surnames the initial cannot reach. It is safe precisely
       because the resolver reports ambiguity as *data*: two Cerundolos or two
       Gomezes come back as candidates and are logged, never guessed.
    3. The raw string, so an already-canonical name still resolves.

    Args:
        raw: The vendor's name string.

    Returns:
        Query strings in priority order, de-duplicated.
    """
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
    """Resolve one vendor name to a canonical model name, or explain why not.

    Walks :func:`market_name_query_forms` and returns the first **unambiguous**
    match. An ambiguous form does not end the walk, a later, more specific form
    may still resolve, but if nothing resolves, the most informative failure is
    kept (ambiguity beats no-match, since it names candidates a human can act on).

    Args:
        raw: The vendor's name string.
        index: Names to resolve against (built from the model's own frame).
        cache: Optional memo; the same few hundred names recur across ~2,600 rows.

    Returns:
        A :class:`NameResolution`. Never raises, never prints, callers log.
    """
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


# ==========================================================================
# The join
# ==========================================================================

@dataclass
class JoinReport:
    """Everything the join produced, including every row it could not use.

    ``skipped`` is the point of this type: a benchmark whose join rate is low is
    not a benchmark, so nothing is dropped silently, every unusable market row
    carries its date, players and reason.
    """

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
    """Read the committed odds snapshot. Offline, never fetches.

    Args:
        path: CSV path; defaults to the committed snapshot.

    Returns:
        The frame with ``Date`` parsed to datetime.

    Raises:
        FileNotFoundError: With the manual-download instructions, since no
            automated path will produce this file.
        ValueError: If a column the join needs is missing.
    """
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
    """De-vigged market probability that **p1** wins, or ``None`` if unpriced.

    The vendor keys its price columns by result, so the de-vigged number always
    describes the actual winner; it is flipped onto p1 when p1 lost. That
    re-orientation is why the outcome leaking through the column names does not
    leak into the comparison: both sides are scored on p1, and p1 was chosen by
    ``preprocess.py``'s coin flip, not by who won.

    Args:
        row: One market row.
        p1_name: Unused except for readability at call sites.
        p1_is_winner: Whether the model's p1 is this row's ``Winner``.
        book: Key into :data:`BOOK_COLUMNS`.

    Returns:
        ``P(p1 wins)`` in ``(0, 1)``, or ``None`` when either price is absent or
        not a usable decimal quote.
    """
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
    """Join market rows onto the model's held-out-season rows.

    The key is the **unordered pair of resolved player names plus a date window**,
    not an exact date: the model's ``tourney_date`` is the tournament's *start*
    date while the vendor records the day the match was played, so an exact join
    would match almost nothing. Each model row is consumed at most once (a pair
    can meet twice in a season), nearest date first.

    Args:
        market: Frame from :func:`load_market_snapshot`.
        model_rows: Engineered held-out-season rows carrying ``p1_name``,
            ``p2_name``, ``tourney_date`` and ``target``.
        book: Bookmaker prefix; see :data:`BOOK_COLUMNS`.
        min_days: Most negative allowed ``market_date − tourney_date``.
        max_days: Most positive allowed ``market_date − tourney_date``.

    Returns:
        A :class:`JoinReport`; ``matched`` carries ``p_market``, ``outcome`` and
        the model index so probabilities can be attached per row.

    Raises:
        KeyError: If ``book`` is not a known bookmaker prefix.
    """
    if book not in BOOK_COLUMNS:
        raise KeyError(f"Unknown book {book!r}; known: {sorted(BOOK_COLUMNS)}")
    for col in BOOK_COLUMNS[book]:
        if col not in market.columns:
            raise KeyError(f"Market snapshot has no column {col!r} for book {book!r}")

    report = JoinReport(n_market_rows=len(market), n_model_rows=len(model_rows))

    names = sorted(set(model_rows["p1_name"]) | set(model_rows["p2_name"]))
    index = NameIndex.from_names(names)
    cache: dict[str, NameResolution] = {}

    # Candidate model rows per unordered name pair, so the scan is a dict lookup
    # rather than a full pass per market row.
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


# ==========================================================================
# Reporting
# ==========================================================================

def plot_comparison(
    model_result: calibrate.CalibrationResult,
    market_result: calibrate.CalibrationResult,
    model_brier: float,
    market_brier: float,
    n: int,
    book: str,
    save_path=config.BENCHMARK_PLOT,
) -> None:
    """Save one reliability chart carrying both curves.

    Deliberately a *second* plot rather than an edit of
    ``outputs/calibration.png``: that one is the model's calibration over the whole
    held-out season, this one is over the matched subset, and overwriting it would
    quietly change what the older artefact means.
    """
    import os

    import matplotlib

    matplotlib.use("Agg")  # headless, same as model/calibrate.py
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


# ==========================================================================
# Orchestration
# ==========================================================================

def run_benchmark(
    snapshot_path=config.BENCHMARK_ODDS_SNAPSHOT,
    book: str = config.BENCHMARK_DEFAULT_BOOK,
    classifier=None,
    model_rows: pd.DataFrame | None = None,
    top_unresolved: int = 20,
    save_plot: bool = True,
):
    """Run the whole comparison and write the plot + report.

    Args:
        snapshot_path: The static odds CSV.
        book: Bookmaker prefix to de-vig.
        classifier: Estimator exposing ``predict_proba``; the pinned model is
            loaded when ``None``.
        model_rows: Engineered held-out-season rows; built from the pipeline when
            ``None`` (~2 s). Injectable so tests never touch the vendored data.
        top_unresolved: How many unresolved names to list in the report.
        save_plot: Write the PNG (off in tests).

    Returns:
        ``(join_report, model_brier, market_brier, report_text)``.
    """
    market = load_market_snapshot(snapshot_path)

    if model_rows is None:
        # The model's own held-out-season selection, reused rather than re-derived, so
        # this benchmark scores exactly the rows outputs/calibration.png scores.
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

    # The comparison's whole premise: both probabilities and the outcomes describe
    # ONE identical set of matches. Checked explicitly rather than left to
    # brier_score's length check, which only catches a mismatch when the sizes
    # happen to differ and says nothing about the rows being the *same* ones.
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
