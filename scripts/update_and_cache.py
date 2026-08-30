"""Sequence a scheduled data refresh into retrain and simulation precompute, aborting at the first failed gate.

Gates fail closed so a cache is never published against half-downloaded data: coverage, refresh
completeness, file verification, sha256 change detection, forced retrain, then precompute via
``precompute_sim.main``. ``data/draws/`` is never touched; exit 0 only if every attempted phase succeeded.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Add both src/ and scripts/ to the path: this module imports from each tree.
_REPO_ROOT = Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT / "src", _REPO_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd  # noqa: E402

import config  # noqa: E402
import precompute_elo  # noqa: E402
import precompute_sim  # noqa: E402
import refresh_data  # noqa: E402
from api.deps import build_api_context  # noqa: E402
from api.registry import build_registry  # noqa: E402
from data.preprocess import SERVE_STAT_COLUMNS  # noqa: E402

# The columns data/preprocess.py indexes by name; their absence is a KeyError, so catch it here.
CORE_COLUMNS = (
    "tourney_date", "surface", "tourney_level", "round", "best_of", "score",
    "winner_id", "winner_name", "winner_rank", "winner_age",
    "loser_id", "loser_name", "loser_rank", "loser_age",
)
REQUIRED_COLUMNS = frozenset(CORE_COLUMNS) | {
    f"{side}_{stat}" for side in ("w", "l") for stat in SERVE_STAT_COLUMNS
}

# loader.py coerces unparseable tourney_date to NaT; a wholesale parse failure is a format change, not bad rows.
MIN_DATE_PARSE_RATIO = 0.99


class OrchestratorError(RuntimeError):
    """A phase failed. The message is the operator-facing explanation."""


@dataclass
class RunSummary:
    """Machine-readable run outcome; the workflow opens a PR only when ``data_changed`` and ``ok`` are both true."""

    ok: bool = False
    refreshed: bool = False
    years: list[int] = field(default_factory=list)
    data_changed: bool = False
    changed_years: list[int] = field(default_factory=list)
    retrained: bool = False
    estimator_class: str | None = None
    elo_regenerated: bool = False
    targets: list[str] = field(default_factory=list)
    succeeded: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    stopped_after: str | None = None
    message: str = ""
    started_at: str = ""
    finished_at: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2) + "\n"


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _log(message: str) -> None:
    """One-line progress, unbuffered so a CI log interleaves correctly."""
    print(message, flush=True)


def file_digest(path: Path) -> str | None:
    """sha256 of ``path``, or ``None`` if it does not exist yet."""
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_digests(raw_dir: Path, years: list[int]) -> dict[int, str | None]:
    """Digest every year's vendored file, missing ones included as ``None``."""
    return {
        year: file_digest(raw_dir / refresh_data.LOCAL_NAME.format(year=year))
        for year in years
    }


def run_refresh(years: list[int], raw_dir: Path) -> None:
    """Download every year in ``years``, or raise naming the ones missing from ``refresh_data.refresh``'s returned map."""
    written = refresh_data.refresh(years[0], years[-1], raw_dir)
    missing = [year for year in years if year not in written]
    if missing:
        raise OrchestratorError(
            "refresh failed for "
            + ", ".join(str(year) for year in missing)
            + f" ({len(missing)} of {len(years)} years). Nothing was retrained "
            "and no cache was regenerated. The vendored data on disk may be "
            "partially updated, so it is deliberately not committed. See the "
            "per-year failure lines above for each failure."
        )


def verify_year_file(
    path: Path, year: int, previous_rows: int | None = None, allow_shrink: bool = False
) -> list[str]:
    """Check one refreshed CSV is usable, returning every problem found (empty = ok); ``allow_shrink`` waives the row-count check."""
    problems: list[str] = []
    if not path.exists():
        return [f"{year}: {path} was not written"]
    if path.stat().st_size == 0:
        return [f"{year}: {path} is empty (0 bytes)"]

    try:
        frame = pd.read_csv(path)
    except Exception as err:  # noqa: BLE001, any parse failure is the same verdict
        return [f"{year}: {path} does not parse as CSV ({type(err).__name__}: {err})"]

    if frame.empty:
        problems.append(f"{year}: {path} parsed to 0 rows")
        return problems

    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        problems.append(
            f"{year}: {len(missing)} required column(s) missing: "
            + ", ".join(missing)
            + " (the vendor's schema may have changed)"
        )

    if "tourney_date" in frame.columns:
        parsed = pd.to_datetime(frame["tourney_date"], format="%Y%m%d", errors="coerce")
        ratio = float(parsed.notna().mean())
        if ratio < MIN_DATE_PARSE_RATIO:
            problems.append(
                f"{year}: only {ratio:.1%} of tourney_date values parse as "
                f"YYYYMMDD (need {MIN_DATE_PARSE_RATIO:.0%}). data/loader.py "
                "coerces failures to NaT, so this would corrupt the date "
                "ordering every leakage-safe feature depends on"
            )

    if previous_rows is not None and len(frame) < previous_rows and not allow_shrink:
        problems.append(
            f"{year}: row count dropped {previous_rows:,} → {len(frame):,}. "
            "Historical results only accumulate, so this is what a truncated "
            "or partial download looks like. Re-run; pass --allow-shrink if "
            "the vendor really did remove rows."
        )

    return problems


def verify_raw_data(
    years: list[int], raw_dir: Path, previous_rows: dict[int, int], allow_shrink: bool
) -> None:
    """Verify every refreshed year, or raise with all problems at once."""
    problems: list[str] = []
    for year in years:
        path = raw_dir / refresh_data.LOCAL_NAME.format(year=year)
        problems.extend(
            verify_year_file(path, year, previous_rows.get(year), allow_shrink)
        )

    if problems:
        raise OrchestratorError(
            f"the refreshed data failed verification ({len(problems)} problem(s)):\n"
            + "\n".join(f"     - {problem}" for problem in problems)
            + "\n   Nothing was retrained and no cache was regenerated."
        )
    _log(f"   verified {len(years)} year file(s): schema, dates, row counts")


def count_rows(raw_dir: Path, years: list[int]) -> dict[int, int]:
    """Row count per existing year file, the pre-refresh baseline for shrinkage."""
    counts: dict[int, int] = {}
    for year in years:
        path = raw_dir / refresh_data.LOCAL_NAME.format(year=year)
        if not path.exists():
            continue
        try:
            counts[year] = len(pd.read_csv(path))
        except Exception:  # noqa: BLE001, an unreadable baseline just has no opinion
            continue
    return counts


def train_model(model_path: Path = config.MODEL_PATH) -> str:
    """Delete the persisted classifier (no restore on failure) and retrain it, returning the estimator class name; the delete is load-bearing because ``predictor.py`` reloads a stale pickle with no staleness check."""
    if model_path.exists():
        _log(f"   removing stale {model_path} (no staleness check exists)")
        model_path.unlink()

    env = {**os.environ, "MPLBACKEND": "Agg"}  # headless: train.py writes PNGs
    result = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "src" / "predictor.py")],
        stdin=subprocess.DEVNULL,
        env=env,
        check=False,
    )
    if result.returncode != 0:
        raise OrchestratorError(
            f"model training failed (predictor.py exited {result.returncode}). "
            "No cache was regenerated."
        )
    if not model_path.exists() or model_path.stat().st_size == 0:
        raise OrchestratorError(
            f"model training exited 0 but wrote no model to {model_path}. "
            "No cache was regenerated."
        )

    import joblib  # local import: only this path needs it

    return type(joblib.load(model_path)).__name__


def regenerate_elo(start: int, end: int, cache_dir: str) -> None:
    """Recompute the Elo rankings cache from the refreshed data, or raise (fatal like a draw precompute)."""
    argv = ["--start", str(start), "--end", str(end), "--cache-dir", cache_dir]
    try:
        code = precompute_elo.main(argv)
    except Exception as err:  # noqa: BLE001, surface any failure as one verdict
        raise OrchestratorError(
            f"Elo rankings precompute crashed ({type(err).__name__}: {err}). "
            "Nothing is committed."
        ) from err
    if code != 0:
        raise OrchestratorError(
            f"Elo rankings precompute exited {code}. Nothing is committed."
        )


def discover_targets(draws_dir: str | Path) -> tuple[list[str], list[str]]:
    """Return ``(simulatable_ids, skipped_descriptions)`` from the same registry the API serves, keying on ``is_simulatable``."""
    context = build_api_context()
    registry = build_registry(context.skill_table, draws_dir)

    simulatable = [e.tournament_id for e in registry.valid if e.is_simulatable]
    skipped = [
        f"{e.tournament_id} (awaiting entrants: "
        f"{len(e.placeholder_slots)} placeholder slot(s))"
        for e in registry.valid
        if not e.is_simulatable
    ] + [f"{e.tournament_id} (draw file failed validation)" for e in registry.invalid]
    return simulatable, skipped


def precompute_draws(
    targets: list[str], runs: int, seed: int, draws_dir: str, cache_dir: str
) -> tuple[list[str], list[str]]:
    """Regenerate the cache for every target draw (all attempted even after one fails), returning ``(succeeded, failed)``."""
    succeeded: list[str] = []
    failed: list[str] = []

    for target in targets:
        _log(f"\nprecomputing {target} ({runs:,} runs, seed {seed})")
        argv = [
            "--draw", target,
            "--runs", str(runs),
            "--seed", str(seed),
            "--draws-dir", draws_dir,
            "--cache-dir", cache_dir,
        ]
        try:
            code = precompute_sim.main(argv)
        except Exception as err:  # noqa: BLE001, one draw must not kill the rest
            _log(f"{target}: {type(err).__name__}: {err}")
            failed.append(target)
            continue
        if code == 0:
            succeeded.append(target)
        else:
            _log(f"{target}: precompute_sim exited {code}")
            failed.append(target)

    return succeeded, failed


def orchestrate(args: argparse.Namespace) -> RunSummary:
    """Run the phases in order, stopping at the first gate that fails."""
    summary = RunSummary(started_at=_now())
    raw_dir = Path(args.raw_dir)
    years = list(range(args.start, args.end + 1))
    summary.years = years

    # Refreshing past config.END_YEAR would date a cache today while ignoring the new season.
    if args.end > config.END_YEAR:
        summary.stopped_after = "config-check"
        summary.message = (
            f"--end is {args.end} but config.END_YEAR is {config.END_YEAR}, so the "
            f"pipeline would load data only through {config.END_YEAR} and quietly "
            f"ignore {args.end}. Bump config.END_YEAR (and check TEST_YEAR, it is "
            "deliberately decoupled) before refreshing this range."
        )
        return summary

    before = snapshot_digests(raw_dir, years)
    previous_rows = count_rows(raw_dir, years) if args.refresh else {}

    # Refresh and verify (the only network in this script).
    if args.refresh:
        _log(f"\n═══ Phase 1/4: refreshing data/raw/ for {args.start}–{args.end}")
        try:
            run_refresh(years, raw_dir)
            _log("\n═══ Phase 2/4: verifying the refreshed files")
            verify_raw_data(years, raw_dir, previous_rows, args.allow_shrink)
        except OrchestratorError as err:
            summary.stopped_after = "refresh"
            summary.message = str(err)
            return summary
        except Exception as err:  # noqa: BLE001, a crash here is still a clean stop
            summary.stopped_after = "refresh"
            summary.message = (
                f"the refresh crashed ({type(err).__name__}: {err}). Nothing was "
                "retrained and no cache was regenerated."
            )
            return summary
        summary.refreshed = True
    else:
        _log("\n═══ Phase 1/4 SKIPPED (--no-refresh): using the vendored data as-is")

    after = snapshot_digests(raw_dir, years)
    summary.changed_years = [y for y in years if before[y] != after[y]]
    summary.data_changed = bool(summary.changed_years)

    if summary.data_changed:
        _log(
            "   vendored data changed in "
            + ", ".join(str(y) for y in summary.changed_years)
        )
    else:
        _log("   vendored data is byte-identical to what was already on disk")

    # Unchanged bytes: retraining would only move `generated_at`. Stop, successfully.
    if args.refresh and not summary.data_changed and not args.force:
        summary.ok = True
        summary.stopped_after = "no-change"
        summary.message = (
            "vendored data unchanged. Skipped retrain and precompute. "
            "Nothing to commit. Pass --force to regenerate anyway."
        )
        summary.finished_at = _now()
        return summary

    # Retrain.
    _log("\n═══ Phase 3/4: retraining the classifier")
    if args.retrain == "never":
        if not config.MODEL_PATH.exists():
            summary.stopped_after = "retrain"
            summary.message = (
                f"--retrain never was passed but {config.MODEL_PATH} does not "
                "exist, and the precompute cannot run without it."
            )
            return summary
        _log(f"   SKIPPED (--retrain never): reusing {config.MODEL_PATH}")
    elif args.retrain == "auto" and not args.refresh and config.MODEL_PATH.exists():
        # No refresh ran, so the existing model is as current as the data.
        _log(f"   SKIPPED (--no-refresh, model present): reusing {config.MODEL_PATH}")
    else:
        try:
            summary.estimator_class = train_model()
        except OrchestratorError as err:
            summary.stopped_after = "retrain"
            summary.message = str(err)
            return summary
        summary.retrained = True
        _log(f"   persisted estimator: {summary.estimator_class}")

    # Elo rankings (display feature, needs only the raw frame; failure stops the run).
    _log("\n═══ Phase 3b/4: regenerating the Elo rankings cache")
    try:
        regenerate_elo(args.start, args.end, args.cache_dir)
    except OrchestratorError as err:
        summary.stopped_after = "elo"
        summary.message = str(err)
        return summary
    summary.elo_regenerated = True

    # Precompute.
    _log("\n═══ Phase 4/4: precomputing the simulation cache")
    if args.draw:
        targets, skipped = list(args.draw), []
    else:
        try:
            targets, skipped = discover_targets(args.draws_dir)
        except Exception as err:  # noqa: BLE001, a broken pipeline is fatal here
            summary.stopped_after = "discover"
            summary.message = (
                f"could not read the draw registry ({type(err).__name__}: {err}). "
                "No cache was regenerated."
            )
            return summary
        for entry in skipped:
            _log(f"   skipping {entry}")
    summary.targets = targets
    summary.skipped = skipped

    if not targets:
        summary.stopped_after = "precompute"
        summary.message = (
            f"no simulatable draw found in {args.draws_dir}. Draw entrants are "
            "entered by hand; this workflow never edits data/draws/."
        )
        return summary

    summary.succeeded, summary.failed = precompute_draws(
        targets, args.runs, args.seed, args.draws_dir, args.cache_dir
    )
    summary.ok = not summary.failed
    if summary.failed:
        summary.stopped_after = "precompute"
        summary.message = (
            f"{len(summary.failed)} of {len(targets)} draw(s) failed to "
            f"precompute: {', '.join(summary.failed)}. Nothing is committed. "
            "The repository keeps the caches it already had rather than a mix "
            "of fresh and stale ones."
        )
    else:
        summary.message = (
            f"regenerated {len(summary.succeeded)} cache file(s): "
            f"{', '.join(summary.succeeded)}"
        )
    summary.finished_at = _now()
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="update_and_cache",
        description="Refresh data/raw/, retrain, and regenerate data/cache/. "
                    "Never touches data/draws/.",
    )
    parser.add_argument(
        "--start", type=int, default=config.START_YEAR,
        help=f"First season to refresh (default: {config.START_YEAR}).",
    )
    parser.add_argument(
        "--end", type=int, default=datetime.date.today().year,
        help="Last season to refresh, inclusive (default: the current calendar "
             "year, which must not exceed config.END_YEAR).",
    )
    parser.add_argument(
        "--no-refresh", dest="refresh", action="store_false",
        help="Skip the download and precompute against the data already on "
             "disk. The one path in this script that needs no network.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Retrain and precompute even when the refresh changed no bytes.",
    )
    parser.add_argument(
        "--allow-shrink", action="store_true",
        help="Accept a year file with fewer rows than the one it replaced "
             "(otherwise treated as a truncated download).",
    )
    parser.add_argument(
        "--draw", action="append", default=[],
        help="Tournament id to precompute; repeatable. Default: every "
             "simulatable draw in the registry.",
    )
    parser.add_argument(
        "--retrain", choices=("auto", "always", "never"), default="auto",
        help="auto (default): retrain whenever a refresh ran or no model "
             "exists. never: reuse the persisted model, and fail if there is "
             "none. NOTE: retraining DELETES "
             f"{config.MODEL_PATH} first and does not put it back if training "
             "then fails; you are left with no model, not a stale one "
             "(regenerate with `python src/predictor.py`, ~1 min). Use "
             "`never` to leave the persisted model untouched.",
    )
    parser.add_argument(
        "--runs", type=int, default=config.MC_RUNS,
        help=f"Monte Carlo runs per draw (default: {config.MC_RUNS}).",
    )
    parser.add_argument(
        "--seed", type=int, default=config.MC_SEED,
        help=f"Base seed (default: {config.MC_SEED}).",
    )
    parser.add_argument("--raw-dir", default=str(config.RAW_DATA_DIR))
    parser.add_argument("--draws-dir", default=str(config.DRAWS_DIR))
    parser.add_argument("--cache-dir", default=str(config.CACHE_DIR))
    parser.add_argument(
        "--summary-json", default=None,
        help="Write the machine-readable run summary here (the workflow reads "
             "it to decide whether to open a pull request).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)

    if args.start > args.end:
        print(f"--start {args.start} is after --end {args.end}.")
        return 1
    if args.runs < 1:
        print(f"--runs must be at least 1, got {args.runs}.")
        return 1

    summary = orchestrate(args)
    summary.finished_at = summary.finished_at or _now()

    _log("\n" + "─" * 70)
    if summary.ok:
        _log(f"{summary.message}")
    else:
        _log(f"FAILED after the {summary.stopped_after} phase:\n   {summary.message}")
    _log(
        f"   refreshed={summary.refreshed} data_changed={summary.data_changed} "
        f"retrained={summary.retrained} elo={summary.elo_regenerated} "
        f"succeeded={len(summary.succeeded)} failed={len(summary.failed)}"
    )

    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary.to_json(), encoding="utf-8")
        _log(f"   summary → {path}")

    return 0 if summary.ok else 1


if __name__ == "__main__":
    sys.exit(main())
