from pathlib import Path

# ==========================================
# PATHS & CONFIGURATION
# ==========================================
OUTPUT_DIR = Path("outputs")
DATA_PATH = Path("atp_tennis_data.csv") # Keep data in root for now as per gitignore
MODEL_PATH = OUTPUT_DIR / "tennis_model.pkl"
ACCURACY_PLOT = OUTPUT_DIR / "accuracy_comparison.png"
FEATURE_IMPORTANCE_PLOT = OUTPUT_DIR / "feature_importance.png"

# ==========================================
# MODEL PARAMETERS
# ==========================================
START_YEAR = 2014
END_YEAR = 2024
DEFAULT_RANK = 2000
DEFAULT_WIN_PCT = 0.5
VALID_SURFACES = {"Hard", "Clay", "Grass"}

# Recent Form Windows (N matches)
RECENT_FORM_WINDOWS = [5, 10]

# Features used for training and prediction
MODEL_FEATURES = [
    'p1_rank', 'p2_rank', 
    'p1_age', 'p2_age', 
    'p1_surface_win_pct', 'p2_surface_win_pct', 
    'h2h_diff',
    # Recent Form Features
    'p1_recent_win_rate_5', 'p2_recent_win_rate_5',
    'p1_recent_win_rate_10', 'p2_recent_win_rate_10',
    'p1_recent_games_won_avg_5', 'p2_recent_games_won_avg_5',
    'p1_recent_games_won_avg_10', 'p2_recent_games_won_avg_10',
    'p1_recent_games_lost_avg_5', 'p2_recent_games_lost_avg_5',
    'p1_recent_games_lost_avg_10', 'p2_recent_games_lost_avg_10',
    'p1_recent_sets_won_avg_5', 'p2_recent_sets_won_avg_5',
    'p1_recent_sets_won_avg_10', 'p2_recent_sets_won_avg_10',
    'p1_recent_sets_lost_avg_5', 'p2_recent_sets_lost_avg_5',
    'p1_recent_sets_lost_avg_10', 'p2_recent_sets_lost_avg_10',
]
