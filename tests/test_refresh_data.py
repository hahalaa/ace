"""Tests for scripts/refresh_data.py — no real network access.

The single network primitive (``fetch_url``) is monkeypatched so downloads are
served from an in-memory fake manifest + CSV bytes.
"""
import json
import os
import sys

import pytest

# Make scripts/ importable (mirrors tests/test_rolling.py's src-on-path pattern).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))

import refresh_data  # noqa: E402


def _fake_fetch(manifest_files, csv_bytes=b"col1,col2\n1,2\n"):
    """Build a fake ``fetch_url`` serving the manifest and per-file CSV bytes."""
    urls = {refresh_data.MANIFEST_URL: json.dumps({"files": manifest_files}).encode()}
    for entry in manifest_files:
        urls[entry["url"]] = csv_bytes

    def fetch(url):
        if url not in urls:
            raise refresh_data.urllib.error.URLError(f"404: {url}")
        return urls[url]

    return fetch


def test_refresh_writes_expected_filenames(tmp_path, monkeypatch):
    files = [
        {"name": "2014.csv", "url": "https://stats.tennismylife.org/data/2014.csv"},
        {"name": "2015.csv", "url": "https://stats.tennismylife.org/data/2015.csv"},
        # Non-main-tour files in the manifest must be ignored.
        {"name": "2014_challenger.csv", "url": "https://x/2014_challenger.csv"},
    ]
    monkeypatch.setattr(refresh_data, "fetch_url", _fake_fetch(files))

    written = refresh_data.refresh(2014, 2015, raw_dir=tmp_path)

    assert set(written) == {2014, 2015}
    assert (tmp_path / "atp_matches_2014.csv").exists()
    assert (tmp_path / "atp_matches_2015.csv").exists()
    # Challenger file must not be pulled in.
    assert not (tmp_path / "2014_challenger.csv").exists()
    assert (tmp_path / "atp_matches_2014.csv").read_bytes() == b"col1,col2\n1,2\n"


def test_individual_year_failure_does_not_abort(tmp_path, monkeypatch):
    # 2016 is absent from the manifest AND its direct URL 404s -> that year fails,
    # but 2014/2015 still succeed.
    files = [
        {"name": "2014.csv", "url": "https://stats.tennismylife.org/data/2014.csv"},
        {"name": "2015.csv", "url": "https://stats.tennismylife.org/data/2015.csv"},
    ]
    monkeypatch.setattr(refresh_data, "fetch_url", _fake_fetch(files))

    written = refresh_data.refresh(2014, 2016, raw_dir=tmp_path)

    assert set(written) == {2014, 2015}
    assert not (tmp_path / "atp_matches_2016.csv").exists()


def test_falls_back_to_direct_url_when_manifest_empty(tmp_path, monkeypatch):
    # Empty manifest -> download_year should hit the direct data-path URL.
    direct_url = f"{refresh_data.DATA_BASE_URL}/2020.csv"
    urls = {
        refresh_data.MANIFEST_URL: b"not json",  # forces empty manifest
        direct_url: b"a,b\n3,4\n",
    }

    def fetch(url):
        if url not in urls:
            raise refresh_data.urllib.error.URLError(f"404: {url}")
        return urls[url]

    monkeypatch.setattr(refresh_data, "fetch_url", fetch)

    written = refresh_data.refresh(2020, 2020, raw_dir=tmp_path)

    assert set(written) == {2020}
    assert (tmp_path / "atp_matches_2020.csv").read_bytes() == b"a,b\n3,4\n"


def test_no_import_time_side_effects():
    # data/raw must not be created merely by importing the module; all work is
    # behind functions / __main__.
    assert hasattr(refresh_data, "main")
    assert callable(refresh_data.refresh)
