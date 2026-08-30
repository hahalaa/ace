"""Scheduled data refresh + simulation precompute, the orchestrator.

``.github/workflows/refresh-and-simulate.yml`` runs this; a human can run it by
hand identically::

    python scripts/update_and_cache.py                          # everything
    python scripts/update_and_cache.py --draw usopen_2026_atp   # one draw
    python scripts/update_and_cache.py --no-refresh --force     # re-sim only

It wraps two existing scripts and adds nothing to what either of them does:
``scripts/refresh_data.py`` (the project's only network call) and
``scripts/precompute_sim.py`` (the Monte Carlo the API serves from
``data/cache/``). What lives *here* is the **sequencing and the aborts**, the
part that is wrong by default if written inline in YAML.

--------------------------------------------------------------------------------
The safety property this file exists for
--------------------------------------------------------------------------------
A cache file states, in its own metadata, when it was generated and what it was
generated from. Publishing one that was computed against half-downloaded data
would be a *false* claim of freshness, not merely a stale one, and nothing
downstream could tell. So the phases are hard-gated, and every gate fails
closed:

0. **Coverage.** ``--end`` defaults to the calendar year, and the pipeline loads
   only through ``config.END_YEAR``. Refreshing past it would regenerate, date
   and commit a cache that ignores the new season entirely, so the run aborts
   before it touches the network. This is the one gate guaranteed to fire on its
   own: on 1 January it stops every run until ``END_YEAR`` is bumped, and says
   so in the message.
1. **Refresh.** ``refresh_data.refresh`` deliberately logs and skips a failing
   year, "one bad year never aborts the whole run", and its ``main()``
   returns ``None``, so **the script exits 0 after downloading nothing at all.**
   That is right for an interactive refresh and catastrophic for an unattended
   one. This orchestrator therefore ignores the exit code and reads the
   ``{year: path}`` map it returns: **every requested year must be in it**, or
   the run aborts here.
2. **Verify.** A 200 response is not data. Each year's file is re-read and
   checked for the columns the pipeline needs, for dates that actually parse
   (``loader`` parses with ``errors="coerce"``, so a vendor date-format change
   would silently produce an all-``NaT`` column), and for a row count that has
   not *shrunk* against the file it replaced. ``pd.read_csv(...,
   on_bad_lines="skip")`` in the loader means a truncated file is otherwise
   read without complaint.
3. **Change detection.** sha256 per file, before and after. Unchanged bytes ⇒
   nothing downstream can have changed either, so the run stops with exit 0 and
   ``data_changed: false``, and the workflow opens no pull request. This is the
   "don't create empty commits" guard, made at the only place that can tell:
   the cache file's ``generated_at`` moves on every single run, so a git diff of
   ``data/cache/`` can never answer the question.
4. **Retrain.** Not optional after a refresh, and not implicit. The skill table
   *is* implicit, ``features.engineering.add_features`` rebuilds it from the
   frame on every pipeline build, and nothing persists it, but
   ``outputs/tennis_model.pkl`` is persisted and ``predictor.py`` loads it
   whenever it exists **without any staleness check**. New data plus an old
   pickle is a silent mismatch, so the pickle is deleted and regenerated.
5. **Precompute.** Only now, and only via ``precompute_sim.main``, its id
   resolution, its placeholder refusal, its error ladder and its disclosure
   metadata, not a second copy of any of them.

**Draw files are never touched.** ``data/draws/*.json`` is hand-entered, there
is no scraper anywhere in this project, and this script only ever *reads* the
registry to decide what to simulate. Refreshing match data and regenerating
simulations is the whole of its remit. A draw whose entrants are still
placeholders (``Qualifier``/``Bye``) is skipped with a log line and is picked up
automatically by the next run once a human fills it in.

Exit code is 0 only if every phase attempted succeeded; any failure is 1, with
the reason named on stdout and in the ``--summary-json`` file the workflow
reads.
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

# scripts/ is not on pyproject's ``pythonpath = ["src"]``, and this module
# imports from both trees: src/ for config and the pipeline, scripts/ for the
# two scripts it orchestrates. Same bootstrap precompute_sim.py uses.
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

# --------------------------------------------------------------------------- #
# What a usable year file looks like.
# --------------------------------------------------------------------------- #
# The columns `data/preprocess.py` indexes by name, the subset whose absence is
# a KeyError rather than a NaN, so a vendor schema change is caught here instead
# of three phases later. Not the whole vendor schema.
CORE_COLUMNS = (
    "tourney_date", "surface", "tourney_level", "round", "best_of", "score",
    "winner_id", "winner_name", "winner_rank", "winner_age",
    "loser_id", "loser_name", "loser_rank", "loser_age",
)
REQUIRED_COLUMNS = frozenset(CORE_COLUMNS) | {
    f"{side}_{stat}" for side in ("w", "l") for stat in SERVE_STAT_COLUMNS
}

# `data/loader.py` parses tourney_date with `format="%Y%m%d", errors="coerce"`,
# so a vendor switch to ISO dates yields a silently all-NaT column and a frame
# that engineers features in arbitrary order. Rounded down from the observed
# 100% on every vendored file, a handful of bad rows is data, a wholesale
# failure to parse is a format change.
MIN_DATE_PARSE_RATIO = 0.99


class OrchestratorError(RuntimeError):
    """A phase failed. The message is the operator-facing explanation."""


# --------------------------------------------------------------------------- #
# Run summary, the machine-readable half of "clear logs".
# --------------------------------------------------------------------------- #
@dataclass
class RunSummary:
    """What happened, in the shape the workflow branches on.

    ``data_changed`` is the commit gate and ``ok`` is the publish gate: the
    workflow opens a pull request only when both are true, so neither a failed
    run nor a no-op run can put anything in front of a reviewer.
    """

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


# --------------------------------------------------------------------------- #
# Phase 1, refresh.
# --------------------------------------------------------------------------- #
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
    """Download every year in ``years``, or raise naming the ones that failed.

    ``refresh_data.refresh`` reports success by *omission*, a year it could not
    download is logged, skipped, and simply absent from the returned map, and
    the process still exits 0. Treating a short map as fatal is the entire point
    of this wrapper.

    Raises:
        OrchestratorError: If any requested year is missing from the result.
    """
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


# --------------------------------------------------------------------------- #
# Phase 2, verify.
# --------------------------------------------------------------------------- #
def verify_year_file(
    path: Path, year: int, previous_rows: int | None = None, allow_shrink: bool = False
) -> list[str]:
    """Check one refreshed CSV is usable. Returns a list of problems (empty = ok).

    Fail-loudly-but-complete, the same convention ``sim/draw.py`` uses: every
    problem with a file is collected so one run reports the whole picture.

    Args:
        path: The refreshed file.
        year: Its season, for the messages.
        previous_rows: Row count of the file this one replaced, if there was
            one. Historical results only ever accumulate, so a drop is the
            signature of a truncated or partial download.
        allow_shrink: Waive the row-count check (a vendor may legitimately
            *remove* rows in a correction; that should be a deliberate,
            human-taken decision, not a silent one).
    """
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
    """Verify every refreshed year, or raise with all problems at once.

    Raises:
        OrchestratorError: If any year's file fails a check.
    """
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


# --------------------------------------------------------------------------- #
# Phase 4, retrain.
# --------------------------------------------------------------------------- #
def train_model(model_path: Path = config.MODEL_PATH) -> str:
    """Delete the persisted classifier and retrain it from the refreshed data.

    The delete is the load-bearing half. ``predictor.py`` loads
    ``outputs/tennis_model.pkl`` whenever the file exists and **never checks
    whether the data behind it has moved on**, so simply re-running it after a
    refresh would reload the old model and publish new-data numbers from an old
    fit.

    **Destructive on a developer machine, and deliberately not undone.** The
    delete happens *before* training and there is no restore on failure: if the
    training run then fails, the machine is left with no model at all rather
    than with a stale one. On a runner that is free (the checkout never had a
    pickle, ``outputs/`` is gitignored). Locally it costs a ~1 min
    ``python src/predictor.py`` to regenerate. Keeping a backup and rolling it
    back on failure was rejected on purpose: restoring the stale pickle would
    hand the next caller exactly the silent new-data/old-fit mismatch this
    function exists to prevent, and a *missing* model fails loudly everywhere
    (``precompute_sim`` refuses to run without one) whereas a stale one does
    not. Pass ``--retrain never`` if you need the persisted model left alone.

    Run as a subprocess with stdin closed, exactly as the Dockerfile's
    ``artefacts`` stage does: ``predictor.py`` ends in the interactive REPL,
    which takes EOF as "input closed" and exits 0 cleanly.

    Returns:
        The class name of the persisted estimator, which of the best-of-four
        won on ``TEST_YEAR`` is data- and environment-dependent, so it is
        logged here and carried into every cache file's metadata.

    Raises:
        OrchestratorError: If training fails or writes no model.
    """
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


# --------------------------------------------------------------------------- #
# Phase 5a, Elo rankings (a display feature; regenerated on the same cadence).
# --------------------------------------------------------------------------- #
def regenerate_elo(start: int, end: int, cache_dir: str) -> None:
    """Recompute the Elo rankings cache from the refreshed data, or raise.

    Elo is a standalone *display* feature (``src/features/elo.py``): it feeds no
    model and no simulation, so it needs neither the trained pickle nor the skill
    table, only the raw winner/loser match frame, which by this point is the
    freshly refreshed data. It regenerates here so the Rankings screen stays on
    the same weekly cadence as the tournament caches rather than drifting behind.

    Fatal on failure like a draw precompute: the workflow commits ``data/cache/``
    as one unit, so a broken rankings file must stop the whole run rather than be
    published beside fresh tournament caches.

    Raises:
        OrchestratorError: If the precompute exits non-zero or raises.
    """
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


# --------------------------------------------------------------------------- #
# Phase 5, precompute.
# --------------------------------------------------------------------------- #
def discover_targets(draws_dir: str | Path) -> tuple[list[str], list[str]]:
    """Every draw that can actually be simulated, and every one that cannot.

    Reads the same registry the API serves from, so "what is simulatable" has
    one definition (``TournamentEntry.is_simulatable``) rather than a second
    copy here. Building it needs a skill table, hence the pipeline, which by
    this point is loading the freshly refreshed data and the freshly trained
    model, so it doubles as a load-bearing smoke test of both.

    ``precompute_sim.main`` then builds its own context per draw, so this
    pipeline build is paid twice (~2 s each, measured in ``api/deps.py``). That
    is deliberate: the alternative is re-implementing that script's id
    resolution, placeholder refusal, error ladder and disclosure metadata here,
    against a phase that already costs ~30 s per draw.

    Returns:
        ``(simulatable_ids, skipped_descriptions)``.
    """
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
    """Regenerate the cache for each target draw. Returns ``(succeeded, failed)``.

    **Every target is attempted even after one fails**, deliberately: a run that
    stops at the first failure reports one broken draw per run, and a human
    fixing them one round-trip at a time is exactly the wrong shape for a weekly
    unattended job. The failures are still fatal, the caller exits non-zero and
    the workflow commits *nothing*, so the repository never holds a mix of fresh
    and stale caches. Continuing buys diagnostics, not partial publication.

    That non-zero exit is also what makes a half-written cache unpublishable.
    ``precompute_sim`` writes its file with a single ``write_text``, not a
    tmp-and-rename, so a crash mid-write can leave truncated JSON on disk. It
    cannot reach the repository: the workflow's staging and commit steps are
    skipped the moment this function reports a failure. The atomicity lives in
    the exit code rather than in the write.
    """
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


# --------------------------------------------------------------------------- #
# The run.
# --------------------------------------------------------------------------- #
def orchestrate(args: argparse.Namespace) -> RunSummary:
    """Run the phases in order, stopping at the first gate that fails."""
    summary = RunSummary(started_at=_now())
    raw_dir = Path(args.raw_dir)
    years = list(range(args.start, args.end + 1))
    summary.years = years

    # A season the refresh fetches but the pipeline never loads is the same
    # falsely-fresh failure this script exists to prevent, just slower: the
    # cache would be regenerated, committed and dated today while ignoring the
    # new year entirely. `--end` defaults to the calendar year, so this fires on
    # the first run of a new season and names its own fix.
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

    # ---- Phase 1 + 2: refresh and verify. The only network in this script. ---
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

    # Nothing new upstream ⇒ retraining and re-simulating would burn ~5 minutes
    # to produce numbers that differ only in `generated_at`. Stop, successfully.
    if args.refresh and not summary.data_changed and not args.force:
        summary.ok = True
        summary.stopped_after = "no-change"
        summary.message = (
            "vendored data unchanged. Skipped retrain and precompute. "
            "Nothing to commit. Pass --force to regenerate anyway."
        )
        summary.finished_at = _now()
        return summary

    # ---- Phase 3: retrain. ---------------------------------------------------
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

    # ---- Phase 3b: Elo rankings (display feature, same cadence). -------------
    # Independent of the draws and the trained model, it only needs the raw
    # match frame, so it runs before the tournament precompute and its failure
    # stops the run the same way (data/cache/ is committed as one unit).
    _log("\n═══ Phase 3b/4: regenerating the Elo rankings cache")
    try:
        regenerate_elo(args.start, args.end, args.cache_dir)
    except OrchestratorError as err:
        summary.stopped_after = "elo"
        summary.message = str(err)
        return summary
    summary.elo_regenerated = True

    # ---- Phase 4: precompute. ------------------------------------------------
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
