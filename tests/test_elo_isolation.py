"""The wall around the Elo feature, enforced by static analysis.

The hard constraint on the elo-ratings work: the Elo rating is a **display-only**
feature and must never feed ``config.MODEL_FEATURES``, the classifier, the point
model, reconciliation or any simulation path. This file proves that structurally,
the same way ``tests/test_benchmark_vs_market.py`` guards the odds
benchmark, an AST walk over the repo, not a promise in prose.

Two directions are checked:

  * **Nothing in the model/simulation path imports ``features.elo``.** Serving it
    (``api/main.py``) and precomputing it (``scripts/precompute_elo.py``,
    ``scripts/update_and_cache.py``) are the *intended* consumers, Elo is meant
    to be shown, so those are allowed. Everything that builds features, trains,
    reconciles or simulates is not.
  * **Nothing in that path names a ``config.ELO_*`` knob.** Elo's tunable
    constants live in ``config.py`` (the project keeps magic numbers in one
    place), next to ``BENCHMARK_*`` and guarded the same way: ``config.py`` is
    imported by ``preprocess.py``, so the names are *reachable* from the training
    path, and this file asserts by source scan that nothing there reaches for
    them. ``MODEL_FEATURES`` also names no Elo column.
"""

from __future__ import annotations

import ast
from pathlib import Path

import config

REPO_ROOT = Path(__file__).resolve().parent.parent

# The one Elo module everything else is walled off from.
ELO_MODULE_STEM = "elo"

# Modules that ARE allowed to import features.elo, the serving and precompute
# layers, which exist precisely to publish the feature. Keyed by path relative to
# the repo root.
ALLOWED_IMPORTERS = frozenset(
    {
        "src/features/elo.py",  # the module itself
        "src/api/main.py",  # serves GET /rankings (reads the cache path only)
        "scripts/precompute_elo.py",  # writes the cache
        "scripts/update_and_cache.py",  # imports precompute_elo to run it on refresh
    }
)


def _python_sources() -> list[Path]:
    files: list[Path] = []
    for root in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" not in path.parts:
                files.append(path)
    return files


def _imports_elo(path: Path) -> bool:
    """True if ``path`` imports ``features.elo`` (or ``from features import elo``)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "features.elo":
                return True
            if node.module == "features" and any(
                alias.name == ELO_MODULE_STEM for alias in node.names
            ):
                return True
        elif isinstance(node, ast.Import):
            if any(alias.name == "features.elo" for alias in node.names):
                return True
    return False


def test_the_scan_sees_the_repo():
    """Guard the guard: an empty file list would make the checks below vacuous."""
    stems = {p.stem for p in _python_sources()}
    for expected in ("preprocess", "engineering", "tournament", "train", "reconcile", "elo"):
        assert expected in stems


def test_only_the_serving_and_precompute_layers_import_elo():
    """The model/simulation path must not reach for Elo."""
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in _python_sources()
        if _imports_elo(path)
        and str(path.relative_to(REPO_ROOT)) not in ALLOWED_IMPORTERS
    ]
    assert offenders == [], (
        f"{offenders} import features.elo. Elo is a display feature and must never "
        "reach the model or the simulator, only the serving/precompute layers may "
        "import it (see ALLOWED_IMPORTERS)."
    )


def test_the_feature_engineers_and_simulator_specifically_stay_clean():
    """A named, explicit list of the modules that most matter, so a refactor that
    renamed a file could not quietly slip Elo past the broad scan above."""
    for relative in (
        "src/data/preprocess.py",
        "src/features/engineering.py",
        "src/features/rolling.py",
        "src/features/serve.py",
        "src/common/classifier_adapter.py",
        "src/sim/reconcile.py",
        "src/sim/tournament.py",
        "src/sim/points.py",
        "src/model/train.py",
    ):
        assert not _imports_elo(REPO_ROOT / relative), (
            f"{relative} imports features.elo, the Elo wall is breached."
        )


def test_model_features_names_no_elo_column():
    assert not any("elo" in feature.lower() for feature in config.MODEL_FEATURES)


def test_elo_config_constants_are_read_only_by_the_elo_feature():
    """The sharpest version of the wall: nothing in the model/simulation path
    even names a ``config.ELO_*`` constant.

    ``config.py`` is imported by ``preprocess.py``, so the ELO_* names are
    technically reachable from the training path. This asserts nothing there
    reaches for them, mirroring
    ``tests/test_benchmark_vs_market.py::test_benchmark_config_constants_are_read_only_by_the_benchmark``.
    Only the serving/precompute layers (ALLOWED_IMPORTERS) and config.py itself
    may name them.
    """
    constants = [name for name in dir(config) if name.startswith("ELO_")]
    assert constants, "expected the Elo constants to live in config.py"

    offenders: list[str] = []
    for path in _python_sources():
        relative = str(path.relative_to(REPO_ROOT))
        if path.name == "config.py" or relative in ALLOWED_IMPORTERS:
            continue
        text = path.read_text()
        for name in constants:
            if name in text:
                offenders.append(f"{relative}: {name}")
    assert offenders == [], (
        f"config.ELO_* referenced outside the Elo feature: {offenders}. Elo is a "
        "display feature; the model and simulator must not read its knobs."
    )
