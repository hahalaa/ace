"""Refresh the vendored raw match data from TML-Database (the only network fetch in the project)."""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
import config  # noqa: E402  (import after sys.path tweak)

# The data-files API serves a JSON manifest; main-tour files are named "YYYY.csv".
MANIFEST_URL = "https://stats.tennismylife.org/api/data-files"
DATA_BASE_URL = "https://stats.tennismylife.org/data"

RAW_DATA_DIR = _REPO_ROOT / "data" / "raw"
LOCAL_NAME = "atp_matches_{year}.csv"

_USER_AGENT = "ace-tennis-sim/refresh_data (non-commercial research)"


def fetch_url(url: str) -> bytes:
    """Fetch the raw bytes at ``url`` (the single network primitive here; tests monkeypatch it)."""
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch_manifest() -> dict[str, str]:
    """Fetch the TML data-files manifest as a ``{name: url}`` map (empty if unavailable)."""
    try:
        payload = json.loads(fetch_url(MANIFEST_URL).decode("utf-8"))
        return {entry["name"]: entry["url"] for entry in payload.get("files", [])}
    except (urllib.error.URLError, ValueError, KeyError, TypeError) as err:
        print(f"Could not fetch manifest ({err}); falling back to direct URLs")
        return {}


def download_year(year: int, raw_dir: Path, manifest: dict[str, str]) -> Path:
    """Download one year's main-tour CSV to ``raw_dir``, resolving the URL via ``manifest`` or the direct pattern."""
    remote_name = f"{year}.csv"
    url = manifest.get(remote_name, f"{DATA_BASE_URL}/{remote_name}")

    data = fetch_url(url)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / LOCAL_NAME.format(year=year)
    dest.write_bytes(data)
    return dest


def refresh(start_year: int, end_year: int, raw_dir: Path = RAW_DATA_DIR) -> dict[int, Path]:
    """Download main-tour CSVs for ``start_year..end_year`` into ``raw_dir``, returning a ``{year: path}`` map of successes (a failed year is logged, skipped, and simply absent from the map)."""
    print(f"Refreshing TML-Database match data for {start_year}–{end_year}...")
    manifest = fetch_manifest()

    written: dict[int, Path] = {}
    for year in range(start_year, end_year + 1):
        try:
            dest = download_year(year, raw_dir, manifest)
            size = dest.stat().st_size
            print(f"   {year}: {size:,} bytes -> {dest}")
            written[year] = dest
        except Exception as err:  # noqa: BLE001, continue on any single-year failure
            print(f"   Failed to refresh {year}: {err}")

    print(f"\nRefreshed {len(written)}/{end_year - start_year + 1} years into {raw_dir}")
    # Self-contained terms notice: printed to whoever just downloaded the data.
    print(
        "Data source: Tennismylife TML-Database (stats.tennismylife.org),\n"
        "   offered in partnership with CanalTenis (canaltenis.com).\n"
        "   TML-Database ships no formal licence file; the binding terms are its\n"
        "   own repository notice: the data is for educational, analytical and\n"
        "   research purposes, all use is non-commercial unless explicitly\n"
        "   permitted, and redistributing or selling the raw database without\n"
        "   permission from TennisMyLife and/or the ATP may infringe copyright.\n"
        "   Acknowledge both sources in anything you publish from it.\n"
        "   (TML-Database was originally inspired by Jeff Sackmann's tennis_atp,\n"
        "   which is CC BY-NC-SA 4.0, that is *his* dataset's licence and does\n"
        "   not govern the files downloaded here.)"
    )
    return written


def main() -> None:
    """CLI entry point. Parses ``--start``/``--end`` and refreshes the data."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--start",
        type=int,
        default=config.START_YEAR,
        help=f"First year to fetch (default: config.START_YEAR = {config.START_YEAR})",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=datetime.date.today().year,
        help="Last year to fetch, inclusive (default: current calendar year)",
    )
    args = parser.parse_args()
    refresh(args.start, args.end)


if __name__ == "__main__":
    main()
