"""Refresh the vendored raw match data from TML-Database.

This script is the **only** place in the project that hits the network for match
data. It is run **manually** (or by CI, T5.3) to (re)populate ``data/raw/`` with
one CSV per year. Simulation, training, and API code must never fetch at request
time — they read the vendored files this script produces.

Source: Tennismylife's ``TML-Database`` (confirmed as the primary source in T0.0;
see ``docs/ace-02-data-schema.md``). Match data is served from the website's
data-files API, not Jeff Sackmann's ``tennis_atp`` (which is currently unreachable
and remains a documented fallback only — it is deliberately not a fetch target here).

Usage::

    python scripts/refresh_data.py --start 2014 --end 2026
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

# --- src on path so we can read shared constants (START_YEAR) --------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
import config  # noqa: E402  (import after sys.path tweak)

# --- TML-Database endpoints (see docs/ace-02-data-schema.md, T0.0) ---------
# The data-files API returns a JSON manifest of every available file with its
# download URL. Main-tour per-year files are named "YYYY.csv" (NOT
# "atp_matches_YYYY.csv" — that was a Sackmann-derived assumption). We resolve
# each year's URL from the manifest, falling back to the direct data path.
MANIFEST_URL = "https://stats.tennismylife.org/api/data-files"
DATA_BASE_URL = "https://stats.tennismylife.org/data"

# Vendored per-year files are saved here. Kept as an internal local name; the
# loader (T0.2) reads this same pattern. RAW_DATA_DIR moves to config in T0.2.
RAW_DATA_DIR = _REPO_ROOT / "data" / "raw"
LOCAL_NAME = "atp_matches_{year}.csv"

_USER_AGENT = "ace-tennis-sim/refresh_data (non-commercial research)"


def fetch_url(url: str) -> bytes:
    """Fetch the raw bytes at ``url``. The single network primitive in this module.

    Tests monkeypatch this to avoid real network access.
    """
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def fetch_manifest() -> dict[str, str]:
    """Fetch the TML data-files manifest and return a ``{name: url}`` map.

    Returns an empty dict if the manifest can't be fetched/parsed, so callers can
    fall back to the direct data-path URL pattern.
    """
    try:
        payload = json.loads(fetch_url(MANIFEST_URL).decode("utf-8"))
        return {entry["name"]: entry["url"] for entry in payload.get("files", [])}
    except (urllib.error.URLError, ValueError, KeyError, TypeError) as err:
        print(f"⚠️  Could not fetch manifest ({err}); falling back to direct URLs")
        return {}


def download_year(year: int, raw_dir: Path, manifest: dict[str, str]) -> Path:
    """Download a single year's main-tour CSV and write it to ``raw_dir``.

    Resolves the download URL from ``manifest`` (self-validating that the year
    exists), falling back to the direct ``{DATA_BASE_URL}/{year}.csv`` pattern.

    Args:
        year: Calendar year to fetch.
        raw_dir: Destination directory for the vendored file.
        manifest: ``{name: url}`` map from :func:`fetch_manifest` (may be empty).

    Returns:
        The path the file was written to.
    """
    remote_name = f"{year}.csv"
    url = manifest.get(remote_name, f"{DATA_BASE_URL}/{remote_name}")

    data = fetch_url(url)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / LOCAL_NAME.format(year=year)
    dest.write_bytes(data)
    return dest


def refresh(start_year: int, end_year: int, raw_dir: Path = RAW_DATA_DIR) -> dict[int, Path]:
    """Download main-tour CSVs for ``start_year..end_year`` (inclusive) into ``raw_dir``.

    Downloads are sequential and polite. Individual year failures are logged and
    skipped — one bad year never aborts the whole run.

    Returns:
        A ``{year: path}`` map of the years that were successfully written.
    """
    print(f"⬇️  Refreshing TML-Database match data for {start_year}–{end_year}...")
    manifest = fetch_manifest()

    written: dict[int, Path] = {}
    for year in range(start_year, end_year + 1):
        try:
            dest = download_year(year, raw_dir, manifest)
            size = dest.stat().st_size
            print(f"   ✓ {year}: {size:,} bytes -> {dest}")
            written[year] = dest
        except Exception as err:  # noqa: BLE001 — continue on any single-year failure
            print(f"   ✗ Failed to refresh {year}: {err}")

    print(f"\n✅ Refreshed {len(written)}/{end_year - start_year + 1} years into {raw_dir}")
    print(
        "ℹ️  Data source: Tennismylife TML-Database (stats.tennismylife.org),\n"
        "   in partnership with CanalTenis (canaltenis.com). Non-commercial use\n"
        "   only unless explicitly permitted; acknowledge the source. See\n"
        "   docs/ace-02-data-schema.md for the full terms."
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
