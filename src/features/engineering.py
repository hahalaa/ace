"""
Feature engineering module.
Generates time-aware features from match data, including:
- Surface-specific win rates
- Head-to-Head (H2H) history
- Recent form (rolling) features
"""
import config
import pandas as pd
import features.rolling as rolling

def add_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    """
    Engineer features using only past match data to prevent leakage.
    
    Returns:
        - DataFrame with new feature columns
        - surface_history dict
        - h2h_history dict
    """
    print("⚙️  Engineering features...")

    df = df.copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"], format="%Y%m%d", errors="coerce")
    df = df.sort_values("tourney_date").reset_index(drop=True)

    # History containers
    # surface_history: { 'Player': { 'Hard': [Wins, Total] } }
    # h2h_history:     { tuple('P1', 'P2'): [P1_wins, P2_wins] }
    surface_history = {} 
    h2h_history = {}     

    p1_surface_pct = []
    p2_surface_pct = []
    h2h_diff = []

    def get_surface_win_pct(player: str, surface: str) -> float:
        """Calculate historical win % for a player on a surface."""
        if surface == "Unknown":
            return config.DEFAULT_WIN_PCT

        if player in surface_history and surface in surface_history[player]:
            wins, total = surface_history[player][surface]
            if total > 0:
                return wins / total
            
        return config.DEFAULT_WIN_PCT

    def update_surface_history(player: str, surface: str, won: bool) -> None:
        """Update historical record for a player on a surface."""
        if player not in surface_history:
            surface_history[player] = {}
        if surface not in surface_history[player]:
            surface_history[player][surface] = [0, 0]
            
        surface_history[player][surface][1] += 1
        if won:
            surface_history[player][surface][0] += 1

    for _, row in df.iterrows():
        p1 = row["p1_name"]
        p2 = row["p2_name"]
        surface = row["surface"] if pd.notna(row["surface"]) else "Unknown"
        p1_won = row["target"] == 1

        # Surface features
        p1_surface_pct.append(get_surface_win_pct(p1, surface))
        p2_surface_pct.append(get_surface_win_pct(p2, surface))

        # H2H feature
        pair = tuple(sorted([p1, p2]))

        if pair in h2h_history:
            wins_a, wins_b = h2h_history[pair]

            if p1 == pair[0]:
                diff = wins_a - wins_b
            else:
                diff = wins_b - wins_a
        else:
            diff = 0

        h2h_diff.append(diff)

        # Update player surface histories
        update_surface_history(p1, surface, p1_won)
        update_surface_history(p2, surface, not p1_won)

        if pair not in h2h_history:
            h2h_history[pair] = [0, 0]

        # Compute winner index
        if (p1_won and p1 == pair[0]) or (not p1_won and p1 != pair[0]):
            winner_index = 0
        else:
            winner_index = 1

        h2h_history[pair][winner_index] += 1

    # Attach features
    df["p1_surface_win_pct"] = p1_surface_pct
    df["p2_surface_win_pct"] = p2_surface_pct
    df["h2h_diff"] = h2h_diff

    # Add rolling features
    df = rolling.compute_rolling_features(df)

    return df, surface_history, h2h_history
