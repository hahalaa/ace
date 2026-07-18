"""Shared pytest fixtures.

``pythonpath = ["src"]`` in pyproject.toml puts ``src/`` on the import path, so
tests import project modules the same way the runtime pipeline does
(``import config``, ``import data.loader``) — no per-file sys.path hacks.
"""
from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_matches_csv() -> Path:
    """Path to the hand-written TML-schema sample matches CSV.

    A handful of rows using the real TML-Database column names, covering the
    documented edge cases: rows that land on both sides of the p1/p2 swap, a
    retirement score (``6-4 2-0 RET``), and a row with missing serve stats.
    """
    return FIXTURES_DIR / "sample_matches.csv"


@pytest.fixture
def sample_matches_df(sample_matches_csv: Path) -> pd.DataFrame:
    """The sample matches CSV loaded as a raw winner/loser DataFrame."""
    return pd.read_csv(sample_matches_csv)
