"""Tests for src/data/loader.py — offline vendored-file reading (T0.2).

No network access: the loader only reads local ``data/raw/`` CSVs. Tests point
``config.RAW_DATA_DIR`` at a tmp dir of hand-written fixtures.
"""
import pandas as pd
import pytest

import config
import data.loader as loader


def _write_year(raw_dir, year, rows):
    """Write a tiny per-year vendored CSV named like the real files."""
    path = raw_dir / loader.RAW_FILENAME.format(year=year)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def raw_dir(tmp_path, monkeypatch):
    """A tmp data/raw/ dir wired into config for the duration of a test."""
    d = tmp_path / "raw"
    d.mkdir()
    monkeypatch.setattr(config, "RAW_DATA_DIR", d)
    return d


def test_concatenates_and_adds_year(raw_dir):
    _write_year(raw_dir, 2014, [
        {"tourney_date": 20140113, "winner_name": "A", "loser_name": "B"},
    ])
    _write_year(raw_dir, 2015, [
        {"tourney_date": 20150119, "winner_name": "C", "loser_name": "D"},
        {"tourney_date": 20150615, "winner_name": "E", "loser_name": "F"},
    ])

    df = loader.load_atp_data(2014, 2015)

    # Concatenated: 1 + 2 rows.
    assert len(df) == 3
    # year column added per source file.
    assert sorted(df["year"].unique().tolist()) == [2014, 2015]
    assert (df["year"] == 2015).sum() == 2


def test_tourney_date_parsed_to_datetime(raw_dir):
    _write_year(raw_dir, 2014, [
        {"tourney_date": 20140113, "winner_name": "A", "loser_name": "B"},
    ])

    df = loader.load_atp_data(2014, 2014)

    assert pd.api.types.is_datetime64_any_dtype(df["tourney_date"])
    assert df["tourney_date"].iloc[0] == pd.Timestamp("2014-01-13")


def test_missing_year_raises_actionable_error(raw_dir):
    _write_year(raw_dir, 2014, [
        {"tourney_date": 20140113, "winner_name": "A", "loser_name": "B"},
    ])
    # 2015 file is absent.

    with pytest.raises(FileNotFoundError) as exc:
        loader.load_atp_data(2014, 2015)

    msg = str(exc.value)
    assert "2015" in msg
    assert "refresh_data.py" in msg
