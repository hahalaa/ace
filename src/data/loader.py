"""
Data loading utilities for ATP tennis match data.
Handles downloading from GitHub and local caching.
"""
import pandas as pd
from pathlib import Path

def load_atp_data(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Downloads ATP match data from Jeff Sackmann's GitHub repository
    for a given range of years as CSV files (inclusive).

    Returns a single pandas DataFrame containing all matches.
    """
    BASE_URL = (
        "https://raw.githubusercontent.com/"
        "JeffSackmann/tennis_atp/master/atp_matches_{}.csv"
    )

    yearly_dfs = []
    print(f"⬇️  Downloading ATP data from {start_year} to {end_year}...")

    for year in range(start_year, end_year + 1):
        try:
            url = BASE_URL.format(year)
            df = pd.read_csv(url, on_bad_lines="skip")
            df["year"] = year
            yearly_dfs.append(df)

            print(f"   ✓ Loaded {year}: {len(df)} matches")

        except Exception as err:
            print(f"   ✗ Failed to load {year}: {err}")

    return pd.concat(yearly_dfs, ignore_index=True)

def load_cached_data(
    path: Path,
    start_year: int,
    end_year: int
) -> pd.DataFrame | None:
    """
    Load cached ATP data if it exists and covers the required year range.
    Returns None if cache is missing or outdated.
    """
    if not path.exists():
        return None

    print(f"📂 Loading cached data from {path}...")
    df = pd.read_csv(path)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format="%Y%m%d", errors="coerce")

    cached_min = df['year'].min()
    cached_max = df['year'].max()

    if cached_min > start_year or cached_max < end_year:
        print(f"⚠️  Cache outdated (Have {cached_min}-{cached_max}, need {start_year}-{end_year})")
        return None

    return df