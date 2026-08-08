"""Regression guards for the GitHub Actions workflow (T5.2).

These do not run CI — that happens on GitHub. What they pin is the handful of
facts that are *silently* wrong when broken: a workflow that runs green while
type-checking nothing, or a frontend job made to "pass" by handing it a dummy
``VITE_API_BASE_URL``. That last one is the trap this file exists for: the
variable is inlined at build time, so a dummy value is not a harmless
placeholder — it removes the warning T4.5 deliberately kept as the signal that
nobody set the real value, and trains people to satisfy the check instead.

Follows ``tests/test_docker_config.py``'s pattern (T5.1), including its
``importorskip`` on PyYAML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

PYTHON_JOB = "python-tests"
FRONTEND_JOB = "frontend-check"


def _workflow() -> dict[str, Any]:
    yaml = pytest.importorskip(
        "yaml", reason="PyYAML (a uvicorn[standard] dependency) not installed"
    )
    assert WORKFLOW.exists(), f"{WORKFLOW.name} is missing — T5.2 creates it"
    return yaml.safe_load(WORKFLOW.read_text())


def _triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    """The ``on:`` block.

    PyYAML follows YAML 1.1, where a bare ``on`` key is the boolean ``True`` —
    so the block is read under either spelling rather than ``workflow["on"]``.
    """
    for key in ("on", True):
        if key in workflow:
            return workflow[key] or {}
    raise AssertionError("the workflow declares no triggers")


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    return job.get("steps", [])


def _run_commands(job: dict[str, Any]) -> str:
    return "\n".join(step["run"] for step in _steps(job) if "run" in step)


def _uses(job: dict[str, Any], action: str) -> dict[str, Any]:
    for step in _steps(job):
        if step.get("uses", "").startswith(action):
            return step
    raise AssertionError(f"{action} is not used by this job")


def test_ci_runs_on_both_push_and_pull_request() -> None:
    """The acceptance criterion: push *and* PR, not one or the other."""
    triggers = _triggers(_workflow())

    assert "push" in triggers
    assert "pull_request" in triggers


def test_both_jobs_exist() -> None:
    jobs = _workflow()["jobs"]

    assert PYTHON_JOB in jobs
    assert FRONTEND_JOB in jobs


def test_python_job_installs_requirements_and_runs_pytest() -> None:
    job = _workflow()["jobs"][PYTHON_JOB]
    commands = _run_commands(job)

    assert "pip install -r requirements.txt" in commands
    assert "pytest" in commands
    # 3.11 is the Dockerfile's base image; CI testing a different interpreter
    # than the one shipped is a silent divergence.
    assert str(_uses(job, "actions/setup-python")["with"]["python-version"]) == "3.11"


def test_frontend_job_type_checks_and_builds() -> None:
    """A build alone would not catch a type error the build does not reach."""
    commands = _run_commands(_workflow()["jobs"][FRONTEND_JOB])

    assert "npm ci" in commands
    assert "npm run typecheck" in commands
    assert "npm run build" in commands


def test_the_frontend_job_does_not_supply_a_dummy_api_base_url() -> None:
    """T4.5's warn-but-succeed behaviour is the point; do not paper over it.

    Checked structurally (env blocks + run commands), not as raw text, so the
    file may keep explaining *why* the variable is absent.
    """
    workflow = _workflow()
    var = "VITE_API_BASE_URL"

    scopes: list[dict[str, Any]] = [workflow.get("env") or {}]
    for job in workflow["jobs"].values():
        scopes.append(job.get("env") or {})
        for step in _steps(job):
            scopes.append(step.get("env") or {})
    for scope in scopes:
        assert var not in scope, f"{var} must not be set anywhere in CI"

    for job in workflow["jobs"].values():
        assert f"{var}=" not in _run_commands(job), f"{var} must not be set inline"


def test_pip_and_npm_are_cached() -> None:
    jobs = _workflow()["jobs"]

    assert _uses(jobs[PYTHON_JOB], "actions/setup-python")["with"]["cache"] == "pip"
    assert _uses(jobs[FRONTEND_JOB], "actions/setup-node")["with"]["cache"] == "npm"
