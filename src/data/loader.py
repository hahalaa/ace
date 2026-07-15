"""
Data loading utilities for ATP tennis match data.

Reads the vendored per-year CSVs under ``data/raw/`` (see ``config.RAW_DATA_DIR``).
This module performs **no network access**: refreshing the raw files is an
explicit, separate action handled by ``scripts/refresh_data.py`` (T0.1). The
runtime pipeline only ever reads the local vendored files.
"""
import config
import pandas as pd

# Vendored per-year filename pattern, matching scripts/refresh_data.py's LOCAL_NAME.
RAW_FILENAME = "atp_matches_{year}.csv"


def load_atp_data(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Load ATP match data for a range of years (inclusive) from the vendored
    ``data/raw/`` CSVs, concatenate them, add a ``year`` column, and return a
    single DataFrame.

    ``tourney_date`` is parsed to datetime here — once, centrally — so every
    downstream caller sees a consistent dtype (see ace-04-current-state.md §6).

    Args:
        start_year: First year to load (inclusive).
        end_year: Last year to load (inclusive).

    Returns:
        Concatenated DataFrame of all matches in the range.

    Raises:
        FileNotFoundError: If any year's vendored file is missing. The data is
            offline by contract, so the loader never fetches it — run
            ``scripts/refresh_data.py`` instead.
    """
    yearly_dfs = []
    print(f"📂 Loading vendored ATP data from {start_year} to {end_year}...")

    for year in range(start_year, end_year + 1):
        path = config.RAW_DATA_DIR / RAW_FILENAME.format(year=year)

        if not path.exists():
            raise FileNotFoundError(
                f"Missing vendored data file for {year}: {path}. "
                f"The pipeline reads local data only — refresh it with "
                f"`python scripts/refresh_data.py --start {start_year} --end {end_year}`."
            )

        df = pd.read_csv(path, on_bad_lines="skip")
        df["year"] = year
        yearly_dfs.append(df)
        print(f"   ✓ Loaded {year}: {len(df)} matches")

    combined = pd.concat(yearly_dfs, ignore_index=True)

    # Parse once, centrally (ace-04-current-state.md §6): downstream of the loader
    # tourney_date is always datetime, so features/train never depend on the raw int.
    combined["tourney_date"] = pd.to_datetime(
        combined["tourney_date"], format="%Y%m%d", errors="coerce"
    )

    return combined
