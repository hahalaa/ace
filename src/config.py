from pathlib import Path

# ==========================================
# PATHS & CONFIGURATION
# ==========================================
OUTPUT_DIR = Path("outputs")
RAW_DATA_DIR = Path("data/raw")         # Vendored per-year CSVs (T0.1); read offline by the loader (T0.2)
MODEL_PATH = OUTPUT_DIR / "tennis_model.pkl"
ACCURACY_PLOT = OUTPUT_DIR / "accuracy_comparison.png"
FEATURE_IMPORTANCE_PLOT = OUTPUT_DIR / "feature_importance.png"

# ==========================================
# MODEL PARAMETERS
# ==========================================
START_YEAR = 2014
END_YEAR = 2026            # Data now vendored through 2026 (partial season)
TEST_YEAR = 2025          # Held-out test season, decoupled from END_YEAR (2026 is partial)
DEFAULT_RANK = 2000
# Fallback win % for an unknown player (no history) and the >0.5 favourite
# boundary in the CLI. Kept distinct from the two 0.5s below (T0.5): they happen
# to share the value today but mean different things, so changing one must not
# silently move the others. See ace-04-current-state.md §2 Gotcha 1.
DEFAULT_WIN_PCT = 0.5
# preprocess.py's p1/p2 coin-flip: a row is swapped when rng.random() > threshold.
PLAYER_SWAP_THRESHOLD = 0.5
# Seed for that swap's RNG (was hardcoded in preprocess.py before T0.5).
PLAYER_SWAP_SEED = 42
# y-axis minimum of train.py's model-accuracy bar plot.
ACCURACY_PLOT_YMIN = 0.5
VALID_SURFACES = {"Hard", "Clay", "Grass"}

# difflib cutoff for the fuzzy (4th-strategy) fallback in the name resolver
# (common/names.py, T0.6). Promoted here from a hardcoded 0.6 in the CLI.
FUZZY_MATCH_CUTOFF = 0.6

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
