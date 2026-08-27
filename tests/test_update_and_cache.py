"""Tests for scripts/update_and_cache.py, the refresh orchestrator.

No network, no training, no Monte Carlo: every expensive phase is a spy, and
what is under test is the **sequencing and the aborts**. That is the whole point
of the script, the two things it wraps (``refresh_data``, ``precompute_sim``)
are already covered by their own suites, and neither of them can tell that a
half-finished refresh must not become a published cache.

The load-bearing case is ``test_partial_refresh_aborts_before_precompute``:
``refresh_data.refresh`` reports failure by *omission* (it logs a bad year,
skips it, and its ``main()`` still exits 0), so a wrapper that trusted the exit
code would happily precompute against yesterday's data and stamp it with
today's timestamp.

``scripts/`` is not on pyproject's ``pythonpath = ["src"]``, so the sys.path
hack from ``tests/test_refresh_data.py`` is repeated here.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))

import config  # noqa: E402
import precompute_sim  # noqa: E402
import refresh_data  # noqa: E402
import update_and_cache as uac  # noqa: E402


# --------------------------------------------------------------------------- #
# Fixtures, the smallest CSV that passes verification.
# --------------------------------------------------------------------------- #
def _csv(rows: int = 3, *, date: str = "20260105", drop: str = "") -> str:
    """A vendored-shaped year file with ``rows`` matches."""
    columns = [c for c in sorted(uac.REQUIRED_COLUMNS) if c != drop]
    values = {"tourney_date": date, "surface": "Hard", "score": "6-4 6-4"}
    header = ",".join(columns)
    lines = [header]
    for i in range(rows):
        lines.append(",".join(str(values.get(c, i + 1)) for c in columns))
    return "\n".join(lines) + "\n"


def _write_years(raw_dir: Path, years, **kwargs) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        (raw_dir / refresh_data.LOCAL_NAME.format(year=year)).write_text(
            _csv(**kwargs), encoding="utf-8"
        )


def _args(tmp_path: Path, *extra: str):
    """Parsed args pointed entirely at ``tmp_path``, with one explicit draw.

    A ``--draw`` is always passed so no test reaches ``discover_targets`` (which
    builds the whole pipeline); target discovery is exercised separately.
    """
    return uac._parse_args(
        [
            "--start", "2025", "--end", "2026",
            "--raw-dir", str(tmp_path / "raw"),
            "--cache-dir", str(tmp_path / "cache"),
            "--draws-dir", str(tmp_path / "draws"),
            "--draw", "some_draw",
            *extra,
        ]
    )


@pytest.fixture
def spies(monkeypatch):
    """Replace every expensive phase with a call recorder.

    Returns the shared ``calls`` list, its *order* is what several tests
    assert, since "refresh before precompute" is the property under test.
    """
    calls: list[str] = []

    def fake_refresh(start, end, raw_dir):
        calls.append("refresh")
        _write_years(Path(raw_dir), range(start, end + 1))
        return {year: Path(raw_dir) for year in range(start, end + 1)}

    def fake_train(*_a, **_k):
        calls.append("train")
        return "FakeClassifier"

    def fake_precompute(argv):
        calls.append(f"precompute:{argv[argv.index('--draw') + 1]}")
        return 0

    def fake_elo(argv):
        # A no-op so the shared expensive-phase order assertions (which track
        # refresh/train/precompute) stay unchanged and hermetic; the Elo phase's
        # own ordering and fatal-on-failure behaviour is covered separately below.
        return 0

    monkeypatch.setattr(refresh_data, "refresh", fake_refresh)
    monkeypatch.setattr(uac, "train_model", fake_train)
    monkeypatch.setattr(precompute_sim, "main", fake_precompute)
    monkeypatch.setattr(uac.precompute_elo, "main", fake_elo)
    return calls


# --------------------------------------------------------------------------- #
# Phase 1, a failed refresh must abort the run.
# --------------------------------------------------------------------------- #
def test_partial_refresh_aborts_before_precompute(tmp_path, spies, monkeypatch):
    """The ticket's safety criterion, stated as a test.

    ``refresh_data.refresh`` swallows a failing year and returns a short map, so
    the abort has to come from the map, not from an exit code.
    """
    raw = tmp_path / "raw"
    _write_years(raw, [2025, 2026])

    def half_refresh(start, end, raw_dir):
        spies.append("refresh")
        return {2025: raw / "atp_matches_2025.csv"}  # 2026 "failed to download"

    monkeypatch.setattr(refresh_data, "refresh", half_refresh)

    summary = uac.orchestrate(_args(tmp_path))

    assert summary.ok is False
    assert summary.stopped_after == "refresh"
    assert "2026" in summary.message
    assert spies == ["refresh"], "nothing may run after a failed refresh"


def test_a_crashing_refresh_stops_the_run_cleanly(tmp_path, spies, monkeypatch):
    """Even an exception the refresh never contracted to raise stops phase 1."""
    def boom(*_a, **_k):
        spies.append("refresh")
        raise RuntimeError("connection reset")

    monkeypatch.setattr(refresh_data, "refresh", boom)

    summary = uac.orchestrate(_args(tmp_path))

    assert summary.ok is False
    assert summary.stopped_after == "refresh"
    assert "connection reset" in summary.message
    assert spies == ["refresh"]


# --------------------------------------------------------------------------- #
# Phase 2, a 200 response is not data.
# --------------------------------------------------------------------------- #
def test_truncated_download_is_caught_by_the_row_count_guard(tmp_path, spies, monkeypatch):
    raw = tmp_path / "raw"
    _write_years(raw, [2025, 2026], rows=50)

    def shrinking_refresh(start, end, raw_dir):
        spies.append("refresh")
        _write_years(Path(raw_dir), [2025], rows=50)
        _write_years(Path(raw_dir), [2026], rows=2)  # truncated
        return {y: Path(raw_dir) for y in (2025, 2026)}

    monkeypatch.setattr(refresh_data, "refresh", shrinking_refresh)

    summary = uac.orchestrate(_args(tmp_path))

    assert summary.ok is False
    assert summary.stopped_after == "refresh"
    assert "row count dropped" in summary.message
    assert spies == ["refresh"]


def test_allow_shrink_waives_the_row_count_guard(tmp_path, spies, monkeypatch):
    raw = tmp_path / "raw"
    _write_years(raw, [2025, 2026], rows=50)

    def shrinking_refresh(start, end, raw_dir):
        spies.append("refresh")
        _write_years(Path(raw_dir), [2025, 2026], rows=2)
        return {y: Path(raw_dir) for y in (2025, 2026)}

    monkeypatch.setattr(refresh_data, "refresh", shrinking_refresh)

    summary = uac.orchestrate(_args(tmp_path, "--allow-shrink"))

    assert summary.ok is True
    assert spies == ["refresh", "train", "precompute:some_draw"]


def test_a_vendor_schema_change_is_caught(tmp_path, spies, monkeypatch):
    """A column the preprocessor indexes by name going missing is fatal here."""
    def dropping_refresh(start, end, raw_dir):
        spies.append("refresh")
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        for year in (2025, 2026):
            (raw_dir / refresh_data.LOCAL_NAME.format(year=year)).write_text(
                _csv(drop="winner_name"), encoding="utf-8"
            )
        return {y: raw_dir for y in (2025, 2026)}

    monkeypatch.setattr(refresh_data, "refresh", dropping_refresh)

    summary = uac.orchestrate(_args(tmp_path))

    assert summary.ok is False
    assert "winner_name" in summary.message
    assert spies == ["refresh"]


def test_a_date_format_change_is_caught(tmp_path, spies, monkeypatch):
    """`data/loader.py` coerces unparseable dates to NaT, silently, and fatally.

    Every leakage-safe feature in the project is built by iterating in date
    order, so an all-NaT column is worse than a missing file.
    """
    def iso_refresh(start, end, raw_dir):
        spies.append("refresh")
        raw_dir = Path(raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        for year in (2025, 2026):
            (raw_dir / refresh_data.LOCAL_NAME.format(year=year)).write_text(
                _csv(date="2026-01-05"), encoding="utf-8"
            )
        return {y: raw_dir for y in (2025, 2026)}

    monkeypatch.setattr(refresh_data, "refresh", iso_refresh)

    summary = uac.orchestrate(_args(tmp_path))

    assert summary.ok is False
    assert "tourney_date" in summary.message
    assert spies == ["refresh"]


def test_an_empty_file_is_caught(tmp_path):
    path = tmp_path / "atp_matches_2026.csv"
    path.write_text("", encoding="utf-8")

    problems = uac.verify_year_file(path, 2026)

    assert len(problems) == 1 and "empty" in problems[0]


def test_verification_reports_every_problem_at_once(tmp_path):
    """Fail-loudly-but-complete, the convention `sim/draw.py` established."""
    path = tmp_path / "atp_matches_2026.csv"
    path.write_text(_csv(rows=1, date="nonsense", drop="surface"), encoding="utf-8")

    problems = uac.verify_year_file(path, 2026, previous_rows=99)

    assert len(problems) == 3
    assert any("surface" in p for p in problems)
    assert any("tourney_date" in p for p in problems)
    assert any("row count dropped" in p for p in problems)


# --------------------------------------------------------------------------- #
# Phase 3, change detection (the "no empty commits" gate).
# --------------------------------------------------------------------------- #
def test_unchanged_data_skips_retrain_and_precompute(tmp_path, spies):
    """Byte-identical downloads ⇒ nothing to retrain, nothing to publish.

    This is the only gate that can answer the question: the cache file's own
    `generated_at` moves on every run, so a git diff of data/cache/ is always
    dirty.
    """
    _write_years(tmp_path / "raw", [2025, 2026])

    summary = uac.orchestrate(_args(tmp_path))

    assert summary.ok is True
    assert summary.data_changed is False
    assert summary.stopped_after == "no-change"
    assert summary.succeeded == []
    assert spies == ["refresh"], "an unchanged refresh must not retrain or simulate"


def test_force_regenerates_when_the_data_is_unchanged(tmp_path, spies):
    _write_years(tmp_path / "raw", [2025, 2026])

    summary = uac.orchestrate(_args(tmp_path, "--force"))

    assert summary.ok is True
    assert summary.data_changed is False
    assert spies == ["refresh", "train", "precompute:some_draw"]


def test_changed_data_runs_every_phase_in_order(tmp_path, spies):
    """refresh → verify → retrain → precompute. The order is the contract."""
    _write_years(tmp_path / "raw", [2025, 2026], rows=2)  # refresh writes 3 rows

    summary = uac.orchestrate(_args(tmp_path))

    assert summary.ok is True
    assert summary.data_changed is True
    assert summary.changed_years == [2025, 2026]
    assert summary.retrained is True
    assert summary.estimator_class == "FakeClassifier"
    assert spies == ["refresh", "train", "precompute:some_draw"]


def test_a_brand_new_year_counts_as_changed(tmp_path, spies):
    """January: the season's file does not exist yet, so its digest is None."""
    _write_years(tmp_path / "raw", [2025])

    summary = uac.orchestrate(_args(tmp_path))

    assert summary.changed_years == [2026]
    assert summary.ok is True


# --------------------------------------------------------------------------- #
# --no-refresh, the offline path.
# --------------------------------------------------------------------------- #
def test_no_refresh_never_touches_the_network(tmp_path, spies, monkeypatch):
    """The manual "just re-simulate draw X" path must not re-download 13 seasons."""
    def explode(*_a, **_k):
        raise AssertionError("refresh must not run under --no-refresh")

    monkeypatch.setattr(refresh_data, "refresh", explode)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "model.pkl")
    (tmp_path / "model.pkl").write_text("pinned", encoding="utf-8")
    _write_years(tmp_path / "raw", [2025, 2026])

    summary = uac.orchestrate(_args(tmp_path, "--no-refresh"))

    assert summary.ok is True
    assert summary.refreshed is False
    assert summary.data_changed is False
    # Reuses the pinned model: no refresh ran, so it is as current as the data.
    assert spies == ["precompute:some_draw"]


def test_no_refresh_still_trains_when_no_model_exists(tmp_path, spies, monkeypatch):
    """A clean checkout (CI) has no pickle, `outputs/` is gitignored."""
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "absent.pkl")
    monkeypatch.setattr(refresh_data, "refresh", lambda *a, **k: pytest.fail("no net"))
    _write_years(tmp_path / "raw", [2025, 2026])

    summary = uac.orchestrate(_args(tmp_path, "--no-refresh"))

    assert spies == ["train", "precompute:some_draw"]
    assert summary.retrained is True


def test_retrain_never_without_a_model_aborts(tmp_path, spies, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "absent.pkl")
    _write_years(tmp_path / "raw", [2025, 2026], rows=2)

    summary = uac.orchestrate(_args(tmp_path, "--retrain", "never"))

    assert summary.ok is False
    assert summary.stopped_after == "retrain"
    assert spies == ["refresh"]


# --------------------------------------------------------------------------- #
# Phase 4, precompute, and what a partial failure means.
# --------------------------------------------------------------------------- #
def test_one_failing_draw_still_attempts_the_rest_and_fails_the_run(
    tmp_path, spies, monkeypatch
):
    """Continue for diagnostics; exit non-zero so nothing is committed.

    The workflow commits only on a zero exit, so "keep going" buys a complete
    failure list without ever publishing a mix of fresh and stale caches.
    """
    def failing_precompute(argv):
        draw = argv[argv.index("--draw") + 1]
        spies.append(f"precompute:{draw}")
        return 1 if draw == "bad" else 0

    monkeypatch.setattr(precompute_sim, "main", failing_precompute)
    _write_years(tmp_path / "raw", [2025, 2026], rows=2)

    args = uac._parse_args(
        [
            "--start", "2025", "--end", "2026",
            "--raw-dir", str(tmp_path / "raw"),
            "--cache-dir", str(tmp_path / "cache"),
            "--draws-dir", str(tmp_path / "draws"),
            "--draw", "bad", "--draw", "good",
        ]
    )
    summary = uac.orchestrate(args)

    assert summary.ok is False
    assert summary.failed == ["bad"] and summary.succeeded == ["good"]
    assert spies == ["refresh", "train", "precompute:bad", "precompute:good"]
    assert "Nothing is committed" in summary.message


def test_an_exception_from_one_draw_does_not_kill_the_others(tmp_path, spies, monkeypatch):
    def raising_precompute(argv):
        draw = argv[argv.index("--draw") + 1]
        spies.append(f"precompute:{draw}")
        if draw == "bad":
            raise MemoryError("out of memory")
        return 0

    monkeypatch.setattr(precompute_sim, "main", raising_precompute)
    _write_years(tmp_path / "raw", [2025, 2026], rows=2)

    args = uac._parse_args(
        [
            "--start", "2025", "--end", "2026",
            "--raw-dir", str(tmp_path / "raw"),
            "--cache-dir", str(tmp_path / "cache"),
            "--draws-dir", str(tmp_path / "draws"),
            "--draw", "bad", "--draw", "good",
        ]
    )
    summary = uac.orchestrate(args)

    assert summary.failed == ["bad"] and summary.succeeded == ["good"]
    assert summary.ok is False


def test_precompute_receives_the_run_parameters(tmp_path, spies, monkeypatch):
    seen: dict[str, list[str]] = {}

    def capture(argv):
        seen["argv"] = argv
        return 0

    monkeypatch.setattr(precompute_sim, "main", capture)
    _write_years(tmp_path / "raw", [2025, 2026], rows=2)

    uac.orchestrate(_args(tmp_path, "--runs", "7", "--seed", "11"))

    argv = seen["argv"]
    assert argv[argv.index("--runs") + 1] == "7"
    assert argv[argv.index("--seed") + 1] == "11"
    assert argv[argv.index("--cache-dir") + 1] == str(tmp_path / "cache")


# --------------------------------------------------------------------------- #
# The config-range guard.
# --------------------------------------------------------------------------- #
def test_refreshing_past_config_end_year_aborts(tmp_path, spies, monkeypatch):
    """Fetching a season the pipeline will not load is a falsely-fresh cache.

    `--end` defaults to the calendar year, so this fires on the first run of a
    new season rather than silently publishing a cache that ignores it.
    """
    monkeypatch.setattr(config, "END_YEAR", 2026)

    args = uac._parse_args(
        [
            "--start", "2025", "--end", "2027",
            "--raw-dir", str(tmp_path / "raw"),
            "--draw", "some_draw",
        ]
    )
    summary = uac.orchestrate(args)

    assert summary.ok is False
    assert summary.stopped_after == "config-check"
    assert "config.END_YEAR" in summary.message
    assert spies == []


# --------------------------------------------------------------------------- #
# train_model, the staleness fix that has to happen before precompute.
# --------------------------------------------------------------------------- #
def test_train_model_deletes_the_stale_pickle_before_retraining(tmp_path, monkeypatch):
    """`predictor.py` loads an existing pickle and never checks it (§4, seam 5).

    Retraining without the delete would reload the old fit and publish new-data
    numbers from it, silently.
    """
    joblib = pytest.importorskip("joblib")
    model = tmp_path / "tennis_model.pkl"
    joblib.dump({"stale": True}, model)
    observed: dict[str, bool] = {}

    class _Result:
        returncode = 0

    def fake_run(cmd, **kwargs):
        observed["existed_at_train_time"] = model.exists()
        observed["stdin_closed"] = kwargs.get("stdin") is not None
        joblib.dump([1, 2, 3], model)  # a "freshly trained" model
        return _Result()

    monkeypatch.setattr(uac.subprocess, "run", fake_run)

    name = uac.train_model(model)

    assert observed["existed_at_train_time"] is False, "the stale pickle must go first"
    assert observed["stdin_closed"] is True, "predictor.py ends in a REPL"
    assert name == "list"


def test_train_model_fails_when_training_writes_no_model(tmp_path, monkeypatch):
    class _Result:
        returncode = 0

    monkeypatch.setattr(uac.subprocess, "run", lambda *a, **k: _Result())

    with pytest.raises(uac.OrchestratorError, match="wrote no model"):
        uac.train_model(tmp_path / "never_written.pkl")


def test_train_model_fails_on_a_non_zero_exit(tmp_path, monkeypatch):
    class _Result:
        returncode = 3

    monkeypatch.setattr(uac.subprocess, "run", lambda *a, **k: _Result())

    with pytest.raises(uac.OrchestratorError, match="exited 3"):
        uac.train_model(tmp_path / "model.pkl")


# --------------------------------------------------------------------------- #
# The summary file the workflow branches on.
# --------------------------------------------------------------------------- #
def test_main_writes_a_summary_and_returns_zero_on_success(tmp_path, spies):
    _write_years(tmp_path / "raw", [2025, 2026], rows=2)
    summary_path = tmp_path / "summary.json"

    code = uac.main(
        [
            "--start", "2025", "--end", "2026",
            "--raw-dir", str(tmp_path / "raw"),
            "--cache-dir", str(tmp_path / "cache"),
            "--draws-dir", str(tmp_path / "draws"),
            "--draw", "some_draw",
            "--summary-json", str(summary_path),
        ]
    )

    assert code == 0
    payload = json.loads(summary_path.read_text())
    assert payload["ok"] is True
    assert payload["data_changed"] is True
    assert payload["succeeded"] == ["some_draw"]
    assert payload["retrained"] is True


def test_main_returns_one_and_still_writes_a_summary_on_failure(
    tmp_path, spies, monkeypatch
):
    """A failed run must be legible to the workflow, not just to a human."""
    monkeypatch.setattr(refresh_data, "refresh", lambda *a, **k: {})
    summary_path = tmp_path / "summary.json"

    code = uac.main(
        [
            "--start", "2025", "--end", "2026",
            "--raw-dir", str(tmp_path / "raw"),
            "--draw", "some_draw",
            "--summary-json", str(summary_path),
        ]
    )

    assert code == 1
    payload = json.loads(summary_path.read_text())
    assert payload["ok"] is False
    assert payload["succeeded"] == []
    assert payload["stopped_after"] == "refresh"


def test_bad_year_range_is_rejected(tmp_path):
    assert uac.main(["--start", "2026", "--end", "2020", "--draw", "x"]) == 1


# --------------------------------------------------------------------------- #
# Target discovery, reuses the API's registry, and never writes a draw file.
# --------------------------------------------------------------------------- #
def test_discover_targets_uses_the_registrys_own_simulatable_rule(monkeypatch):
    """One definition of "simulatable" (`TournamentEntry.is_simulatable`).

    The shipped placeholder draw must be *skipped and reported*, not silently
    dropped and not attempted: a human fills its entrants in, and the next run
    picks it up.
    """
    class _Entry:
        def __init__(self, tid, simulatable, placeholders=()):
            self.tournament_id = tid
            self.is_simulatable = simulatable
            self.placeholder_slots = placeholders

    class _Registry:
        valid = [_Entry("full_draw", True), _Entry("awaiting", False, ("Qualifier",))]
        invalid = [_Entry("broken_file", False)]

    monkeypatch.setattr(
        uac, "build_api_context", lambda: SimpleNamespace(skill_table=object())
    )
    monkeypatch.setattr(uac, "build_registry", lambda *a, **k: _Registry())

    targets, skipped = uac.discover_targets("data/draws")

    assert targets == ["full_draw"]
    assert any("awaiting" in s and "placeholder" in s for s in skipped)
    assert any("broken_file" in s and "failed validation" in s for s in skipped)


# --------------------------------------------------------------------------- #
# Phase 3b, Elo rankings (a display feature, regenerated on the same cadence).
# --------------------------------------------------------------------------- #
def test_elo_regenerates_after_retrain_and_before_the_draw_precompute(
    tmp_path, spies, monkeypatch
):
    """Elo sits between the retrain and the tournament precompute, and its cache
    dir is threaded through, it needs neither the model nor the draws, only the
    refreshed data, so it runs once the run is past the change gate."""
    captured: dict[str, list[str]] = {}

    def recording_elo(argv):
        spies.append("elo")
        captured["argv"] = list(argv)
        return 0

    monkeypatch.setattr(uac.precompute_elo, "main", recording_elo)
    _write_years(tmp_path / "raw", [2025, 2026])

    summary = uac.orchestrate(_args(tmp_path, "--force"))

    assert summary.ok is True
    assert summary.elo_regenerated is True
    assert spies == ["refresh", "train", "elo", "precompute:some_draw"]
    # The run's cache dir is passed through, so rankings land beside the sim cache.
    assert captured["argv"][captured["argv"].index("--cache-dir") + 1] == str(
        tmp_path / "cache"
    )


def test_a_failing_elo_precompute_stops_the_run_before_the_draw_precompute(
    tmp_path, spies, monkeypatch
):
    """data/cache/ is committed as one unit, so a broken rankings file must stop
    the whole run rather than be published beside fresh tournament caches."""
    monkeypatch.setattr(uac.precompute_elo, "main", lambda argv: 1)
    _write_years(tmp_path / "raw", [2025, 2026])

    summary = uac.orchestrate(_args(tmp_path, "--force"))

    assert summary.ok is False
    assert summary.stopped_after == "elo"
    assert summary.elo_regenerated is False
    # Nothing after the Elo phase ran, no draw was precomputed.
    assert not any(call.startswith("precompute:") for call in spies)


def test_a_crashing_elo_precompute_stops_the_run_cleanly(tmp_path, spies, monkeypatch):
    def boom(argv):
        raise RuntimeError("bad rating frame")

    monkeypatch.setattr(uac.precompute_elo, "main", boom)
    _write_years(tmp_path / "raw", [2025, 2026])

    summary = uac.orchestrate(_args(tmp_path, "--force"))

    assert summary.ok is False
    assert summary.stopped_after == "elo"
    assert "bad rating frame" in summary.message
