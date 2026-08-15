"""
Rolling window feature computation.
Calculates recent form statistics (win rate, games/sets won/lost)
over a sliding window of past matches.

Two entry points, sharing one player-centric view of the data:

* :func:`compute_rolling_features` — the training-pipeline pass. For **every**
  match row it computes each player's form over their previous ``N`` matches
  (``shift(1)`` before the window, so the current match can never leak in).
* :func:`build_rolling_form_table` — the *as-of-now* snapshot accessor (T3.5).
  It answers "what is player X's rolling form **right now**", i.e. the values a
  hypothetical next match for X would receive, for a player named outside any
  particular match row. That question has no answer in the pass above, which is
  why ``cli/interactive.build_feature_row`` filled those 20 columns with
  synthetic constants (``ace-04-current-state.md §7`` seam 7).

The two cannot drift apart: both read :func:`build_player_match_frame` for the
long-form per-player history and :data:`ROLLING_METRIC_COLUMNS` for the metric →
column-name mapping, so the snapshot is the same computation the pipeline runs,
evaluated one row past the end of a player's history instead of at each row.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

# The per-match metrics the rolling windows average, mapped to the feature-column
# name each produces. ONE definition, consumed by the training pass and by the
# as-of-now snapshot below — a second copy of these names is how the two would
# silently start disagreeing about what "recent form" means.
ROLLING_METRIC_COLUMNS: dict[str, str] = {
    'won': 'recent_win_rate_{window}',
    'games_won': 'recent_games_won_avg_{window}',
    'games_lost': 'recent_games_lost_avg_{window}',
    'sets_won': 'recent_sets_won_avg_{window}',
    'sets_lost': 'recent_sets_lost_avg_{window}',
}

ROLLING_METRICS: tuple[str, ...] = tuple(ROLLING_METRIC_COLUMNS)


def rolling_feature_names(windows: list[int] | None = None) -> list[str]:
    """The ``recent_*`` column names produced for ``windows`` (unprefixed).

    Window-major order, matching the order :func:`compute_rolling_features`
    creates them in.
    """
    windows = config.RECENT_FORM_WINDOWS if windows is None else windows
    return [
        template.format(window=window)
        for window in windows
        for template in ROLLING_METRIC_COLUMNS.values()
    ]


def build_player_match_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Explode the p1/p2 match frame into one row per player per match.

    This is step 1 of :func:`compute_rolling_features`, lifted out unchanged so
    the snapshot accessor reads the *same* history — same p1/p2 stacking, same
    ``won``-inversion for the p2 side, and (critically) the same
    ``['player', 'date', 'match_index']`` sort, which is the order the rolling
    windows are taken in.

    Returns:
        A frame with columns ``date``, ``player``, ``opponent``, ``won``,
        ``games_won``, ``games_lost``, ``sets_won``, ``sets_lost``,
        ``match_index``, ``is_p1``, sorted by player then date.
    """
    # Player 1 perspective
    p1_cols = {
        'tourney_date': 'date',
        'p1_name': 'player',
        'p2_name': 'opponent',
        'target': 'won', # 1 if p1 won
        'p1_games_won': 'games_won',
        'p1_games_lost': 'games_lost',
        'p1_sets_won': 'sets_won',
        'p1_sets_lost': 'sets_lost'
    }

    df_p1 = df[list(p1_cols.keys())].rename(columns=p1_cols)
    df_p1['match_index'] = df.index
    df_p1['is_p1'] = True

    # Player 2 perspective
    p2_cols = {
        'tourney_date': 'date',
        'p2_name': 'player',
        'p1_name': 'opponent',
        # if target=1 (p1 won), then p2 lost (0). if target=0 (p1 lost), then p2 won (1)
        'won_inverse': 'won',
        'p2_games_won': 'games_won',
        'p2_games_lost': 'games_lost',
        'p2_sets_won': 'sets_won',
        'p2_sets_lost': 'sets_lost'
    }

    df_p2 = df.copy()
    df_p2['match_index'] = df.index
    df_p2['won_inverse'] = 1 - df_p2['target']
    df_p2 = df_p2[list(p2_cols.keys()) + ['match_index']] # need match_index from original
    df_p2 = df_p2.rename(columns=p2_cols)
    df_p2['is_p1'] = False

    # Concatenate to get full history
    player_df = pd.concat([df_p1, df_p2], ignore_index=True)

    # Sort by player and date to ensure correct rolling window
    return player_df.sort_values(['player', 'date', 'match_index'])


@dataclass(frozen=True)
class RollingFormTable:
    """Every player's *latest* rolling-form state — the T3.5 snapshot accessor.

    :func:`compute_rolling_features` answers "what was this player's form before
    match *i*" for every row of the training frame. This answers the question the
    classifier adapter actually asks: "what is this player's form **now**,
    outside any match row" — the values a match played immediately after the last
    row of the data would carry.

    It is not a reimplementation of the rolling logic. It stores the tail of each
    player's per-match metric history (:func:`build_player_match_frame`'s output,
    in the same sort order) and averages the last ``window`` entries, which is
    exactly what ``shift(1).rolling(window, min_periods=1).mean()`` evaluates to
    at a hypothetical next row: ``shift(1)`` means "the window ends at the
    previous match", and for a new match that is the player's whole recent past.
    ``tests/test_rolling.py`` pins the equality against the pipeline itself.

    Attributes:
        form: ``player -> metric -> `` the player's last ``max(windows)`` values,
            oldest first.
        windows: The recent-form windows, from ``config.RECENT_FORM_WINDOWS``.
    """

    form: dict[str, dict[str, np.ndarray]]
    windows: tuple[int, ...]

    def __contains__(self, player: str) -> bool:
        return player in self.form

    def n_matches(self, player: str) -> int:
        """How many recent matches are stored for ``player`` (capped at ``max(windows)``)."""
        history = self.form.get(player)
        return 0 if history is None else len(history[ROLLING_METRICS[0]])

    def latest(self, player: str) -> dict[str, float]:
        """This player's rolling-form features as of now.

        Args:
            player: Display name, as it appears in ``p1_name``/``p2_name``.

        Returns:
            ``{column_name: value}`` for every window × metric, e.g.
            ``{"recent_win_rate_5": 0.6, "recent_games_won_avg_5": 9.0, ...}``.

        Raises:
            KeyError: If the player has no match history. Deliberately loud: the
                training pipeline's NaN-fill defaults exist for a player's
                *first* match, and an adapter asking for a snapshot has already
                established the player has a history (it needs a rank and an age
                from the same frame). Silently returning neutral constants here
                is the seam-7 defect this class exists to remove.
        """
        history = self.form[player]
        values: dict[str, float] = {}
        for window in self.windows:
            for metric, template in ROLLING_METRIC_COLUMNS.items():
                tail = history[metric][-window:]
                # nanmean, not mean: pandas' rolling(...).mean() averages the
                # non-NaN entries of its window, so this must too.
                values[template.format(window=window)] = float(np.nanmean(tail))
        return values


def build_rolling_form_table(df: pd.DataFrame) -> RollingFormTable:
    """Build the as-of-now :class:`RollingFormTable` for every player in ``df``.

    **Granularity note.** The snapshot is "after the last row of ``df``", and
    that is the only instant it answers for. It is *not* a general as-of query:
    ``tourney_date`` is the tournament's start date, so 46% of player-date groups
    in the vendored data hold more than one match (up to 7 — a full run at one
    event), and the training pass orders those by row index within the day.
    Truncating a frame by date therefore cannot reproduce a mid-tournament row,
    while the end-of-frame snapshot this builds is exact — every stored match is
    genuinely prior to the hypothetical next one.

    Args:
        df: A preprocessed match frame — the same input
            :func:`compute_rolling_features` takes (it needs ``tourney_date``,
            ``p1_name``/``p2_name``, ``target`` and the per-player games/sets
            columns). An engineered frame works too; the extra columns are
            ignored.

    Returns:
        The populated table. Measured at **0.08 s** over the full 2014–2026 frame
        (35,319 matches, 1,408 players); a :meth:`RollingFormTable.latest` lookup
        is ~66 µs.
    """
    player_df = build_player_match_frame(df)
    windows = tuple(config.RECENT_FORM_WINDOWS)
    longest = max(windows) if windows else 0

    # groupby(sort=False).tail(n) keeps the frame's existing (player, date,
    # match_index) order, so each player's slice is already chronological.
    recent = player_df.groupby('player', sort=False).tail(longest)

    form: dict[str, dict[str, np.ndarray]] = {}
    for player, group in recent.groupby('player', sort=False):
        form[player] = {
            metric: group[metric].to_numpy(dtype=float) for metric in ROLLING_METRICS
        }

    return RollingFormTable(form=form, windows=windows)


def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes rolling statistics for each player based on their last N matches.
    Matches must be strictly prior to the current match.

    Features computed (for both p1 and p2):
    - recent_win_rate_N
    - recent_games_won_avg_N
    - recent_games_lost_avg_N
    - recent_sets_won_avg_N
    - recent_sets_lost_avg_N
    """
    print("   Computing rolling features...")

    # 1. One row per player per match, sorted so the windows are chronological.
    player_df = build_player_match_frame(df)

    # 2. Compute Rolling Stats
    windows = config.RECENT_FORM_WINDOWS
    metrics = list(ROLLING_METRICS)

    # Group by player
    grouped = player_df.groupby('player')[metrics]

    for window in windows:
        # We shift(1) to ensure we only use PAST matches, not the current one.
        # Then rolling(window, min_periods=1).mean()

        # Calculate rolling means
        rolling_stats = grouped.apply(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

        # Rename columns
        # e.g. won -> recent_win_rate_5
        # games_won -> recent_games_won_avg_5

        for metric, template in ROLLING_METRIC_COLUMNS.items():
            player_df[template.format(window=window)] = (
                rolling_stats[metric].reset_index(0, drop=True)
            )

    # Fill NaNs for players with insufficient history
    # Win rate defaults to 0.5 (neutral)
    win_cols = [c for c in player_df.columns if 'win_rate' in c]
    player_df[win_cols] = player_df[win_cols].fillna(config.DEFAULT_WIN_PCT)
    
    # Other stats default to the global mean (neutral performance)
    stat_cols = [c for c in player_df.columns if 'recent_' in c and 'win_rate' not in c]
    for col in stat_cols:
         player_df[col] = player_df[col].fillna(player_df[col].mean())
    


    # 3. Merge back to original DataFrame
    # We need to pull the features back for p1 and p2 separately
    
    # Extract p1 features
    p1_features = player_df[player_df['is_p1']].set_index('match_index')
    p1_cols_map = {c: f'p1_{c}' for c in player_df.columns if 'recent_' in c}
    p1_features = p1_features.rename(columns=p1_cols_map)
    
    # Extract p2 features
    p2_features = player_df[~player_df['is_p1']].set_index('match_index')
    p2_cols_map = {c: f'p2_{c}' for c in player_df.columns if 'recent_' in c}
    p2_features = p2_features.rename(columns=p2_cols_map)
    
    # Join
    df = df.join(p1_features[list(p1_cols_map.values())])
    df = df.join(p2_features[list(p2_cols_map.values())])
    
    return df
