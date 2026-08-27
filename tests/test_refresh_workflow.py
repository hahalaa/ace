"""Regression guards for .github/workflows/refresh-and-simulate.yml.

Same remit as ``tests/test_ci_workflow.py``: these do not run the workflow, they
pin the handful of facts that are **silently** wrong when broken. This workflow
is the only one in the repo that can write to the default branch, unattended, so
the list is longer:

* ``git add data/cache`` without ``-f`` is a no-op against ``.gitignore:15``,
  the job goes green, opens a PR, and the "fresh" cache never leaves the runner.
* a commit gate keyed on a git diff would fire on every run, because every cache
  file's ``generated_at`` moves.
* ``data/draws/`` reaching an automated ``git add`` would mean a scheduled job
  editing hand-entered tournament entrants.
* an interpolated ``${{ inputs.tournament_id }}`` in a ``run:`` block is a
  script-injection sink.

Follows ``tests/test_ci_workflow.py``'s ``importorskip`` on PyYAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "refresh-and-simulate.yml"
JOB = "refresh"


def _text() -> str:
    assert WORKFLOW.exists(), f"{WORKFLOW.name} is missing"
    return WORKFLOW.read_text()


def _workflow() -> dict[str, Any]:
    yaml = pytest.importorskip(
        "yaml", reason="PyYAML (a uvicorn[standard] dependency) not installed"
    )
    return yaml.safe_load(_text())


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """The ``on:`` block, read under either spelling (YAML 1.1 makes it ``True``)."""
    for key in ("on", True):
        if key in workflow:
            return workflow[key] or {}
    raise AssertionError("the workflow declares no triggers")


def _steps() -> list[dict[str, Any]]:
    return _workflow()["jobs"][JOB]["steps"]


def _run_commands() -> str:
    return "\n".join(step["run"] for step in _steps() if "run" in step)


def test_it_runs_on_a_schedule_and_on_manual_dispatch() -> None:
    """The acceptance criterion: both, not one or the other."""
    triggers = _triggers(_workflow())

    assert "schedule" in triggers
    assert triggers["schedule"] and "cron" in triggers["schedule"][0]
    assert "workflow_dispatch" in triggers


def test_manual_dispatch_takes_a_tournament_id() -> None:
    inputs = _triggers(_workflow())["workflow_dispatch"]["inputs"]

    assert "tournament_id" in inputs
    # Blank must be legal: the default run is "every simulatable draw".
    assert not inputs["tournament_id"].get("required", False)
    assert inputs["tournament_id"].get("default", "") == ""


def test_refreshing_and_precomputing_are_separable_inputs() -> None:
    """"Pull new data" and "re-simulate draw X" are different jobs.

    Pairing them would force a 13-season re-download to re-run one Monte Carlo.
    """
    inputs = _triggers(_workflow())["workflow_dispatch"]["inputs"]

    assert inputs["refresh_data"]["default"] is True
    assert "--no-refresh" in _run_commands()


def test_the_cache_is_force_added_past_the_gitignore_rule() -> None:
    """`.gitignore:15` ignores `data/cache/*.json`; a plain add is a silent no-op.

    The rule is kept, it protects developer machines from committing a local
    cache by accident, so this workflow is the single deliberate exception.
    """
    commands = _run_commands()

    assert "git add -f data/cache" in commands
    gitignore = (REPO_ROOT / ".gitignore").read_text().splitlines()
    assert "data/cache/*.json" in gitignore, (
        "the ignore rule was removed, then the -f above is pointless and the "
        "protection it deliberately preserved is gone"
    )


def test_draw_files_are_never_staged() -> None:
    """Entrants are hand-entered. This workflow refreshes data and sims only."""
    commands = _run_commands()

    adds = [ln for ln in commands.splitlines() if ln.strip().startswith("git add")]
    assert adds, "nothing is staged at all"
    for line in adds:
        # Every add names its paths; none of them may be data/draws, and none
        # may be a bare `git add .`/`-A` that would sweep one in.
        assert "data/draws" not in line, "draw files must never be auto-committed"
        assert line.split()[-1] not in {".", "-A", "--all"}

    # ...and the workflow says so where a reviewer will read it.
    assert "data/draws" in _text()


def test_the_commit_is_guarded_by_an_actual_diff_check() -> None:
    """Belt and braces on top of the orchestrator's digest gate."""
    assert "git diff --cached --quiet" in _run_commands()


def test_the_commit_gate_is_not_keyed_on_a_cache_diff() -> None:
    """`generated_at` moves every run, so a cache diff always looks like change.

    The gate must come from the orchestrator's summary instead.
    """
    steps = _steps()
    gated = [s for s in steps if "regenerated" in str(s.get("if", ""))]

    assert gated, "no step is gated on the orchestrator's `regenerated` output"
    assert any("git add" in s.get("run", "") for s in gated)


def test_the_orchestrator_is_what_runs_not_the_two_scripts_inline() -> None:
    """The sequencing and the aborts live in Python, not in YAML."""
    commands = _run_commands()

    assert "scripts/update_and_cache.py" in commands
    assert "scripts/refresh_data.py" not in commands
    assert "scripts/precompute_sim.py" not in commands


def test_dispatch_inputs_reach_the_shell_through_the_environment() -> None:
    """`${{ inputs.… }}` spliced into a `run:` block is a script-injection sink."""
    for step in _steps():
        run = step.get("run", "")
        assert "${{ inputs." not in run
        assert "${{ github.event.inputs." not in run


def test_permissions_are_scoped_to_what_the_pr_flow_needs() -> None:
    permissions = _workflow()["permissions"]

    assert permissions["contents"] == "write"
    assert permissions["pull-requests"] == "write"
    assert set(permissions) == {"contents", "pull-requests"}


def test_it_opens_a_pull_request_rather_than_pushing_to_the_default_branch() -> None:
    """The design decision: a reviewable diff before the demo's numbers change."""
    commands = _run_commands()

    assert "gh pr create" in commands
    # Nothing may push to the checked-out default branch.
    assert "git push origin HEAD:refs/heads/main" not in commands
    assert "git push --force-with-lease origin \"HEAD:refs/heads/$BRANCH\"" in commands


def test_the_checkout_fetches_every_ref_so_the_lease_is_meaningful() -> None:
    """`fetch-depth: 0` is load-bearing, not a nicety, and cheap to "optimise" away.

    The push is `--force-with-lease`, which with no explicit value leases against
    the *remote-tracking ref* for the destination branch. A shallow/single-ref
    checkout has no `origin/automation/data-refresh`, so git treats the lease as
    "expecting this ref not to exist" and every run after the first is rejected
    with `stale info`. Reproduced against a local bare remote: run 1 creates the
    branch, run 2 from a clone lacking that ref fails the push and exits 1.
    """
    checkout = next(
        s for s in _steps() if s.get("uses", "").startswith("actions/checkout")
    )

    assert str(checkout["with"]["fetch-depth"]) == "0", (
        "the --force-with-lease push below needs the remote-tracking ref for "
        "the automation branch; without it the second run onwards is rejected "
        "with 'stale info'"
    )


def test_concurrency_does_not_cancel_a_run_in_flight() -> None:
    """Two runs push the same branch; a half-finished refresh is not an upgrade."""
    concurrency = _workflow()["concurrency"]

    assert concurrency["cancel-in-progress"] is False


def test_python_matches_the_rest_of_the_project() -> None:
    setup = next(
        s for s in _steps() if s.get("uses", "").startswith("actions/setup-python")
    )

    assert str(setup["with"]["python-version"]) == "3.11"
    assert setup["with"]["cache"] == "pip"


def test_the_suite_runs_before_a_pr_is_opened() -> None:
    """A GITHUB_TOKEN PR does not trigger ci.yml, so this job is the only check."""
    steps = _steps()
    test_index = next(i for i, s in enumerate(steps) if s.get("run", "").strip() == "pytest -q")
    pr_index = next(i for i, s in enumerate(steps) if "gh pr create" in s.get("run", ""))

    assert test_index < pr_index
