"""
Rolling window feature computation.
Calculates recent form statistics (win rate, games/sets won/lost)
over a sliding window of past matches.
"""
import pandas as pd
import config

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
    print("   ⏳ Computing rolling features...")
    
    # 1. Create a player-centric DataFrame
    # We need one row per player per match to sort and shift correctly
    
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
    player_df = player_df.sort_values(['player', 'date', 'match_index'])
    
    # 2. Compute Rolling Stats
    windows = config.RECENT_FORM_WINDOWS
    metrics = ['won', 'games_won', 'games_lost', 'sets_won', 'sets_lost']
    
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
        
        player_df[f'recent_win_rate_{window}'] = rolling_stats['won'].reset_index(0, drop=True)
        player_df[f'recent_games_won_avg_{window}'] = rolling_stats['games_won'].reset_index(0, drop=True)
        player_df[f'recent_games_lost_avg_{window}'] = rolling_stats['games_lost'].reset_index(0, drop=True)
        player_df[f'recent_sets_won_avg_{window}'] = rolling_stats['sets_won'].reset_index(0, drop=True)
        player_df[f'recent_sets_lost_avg_{window}'] = rolling_stats['sets_lost'].reset_index(0, drop=True)

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
