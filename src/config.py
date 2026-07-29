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

# ==========================================
# SERVE/RETURN SKILL TABLE (T1.1)
# ==========================================
# Exponential-decay half-life (days) for recency-weighting serve/return stats:
# a match this many days older counts half as much. ~1 season.
SERVE_RECENCY_HALFLIFE_DAYS = 365.0
# Empirical-Bayes shrinkage strength, in serve/return-point units: a player's
# rate is blended with the surface baseline μ as (n·rate + k·μ)/(n + k). Acts as
# the low-sample control (subsumes a hard MIN_SERVE_POINTS threshold — a single
# knob rather than two, since EB shrinkage is continuous). k points of prior.
SERVE_SHRINKAGE_K = 200.0
# Data-derived tour-average serve-points-won (μ) per surface — the point-model
# baseline (ace-03-tennis-math.md §1). Canonical home is the table in
# ace-02-data-schema.md; recompute with features.serve.compute_surface_mu if the
# vendored data range changes. Values below computed from data/raw 2014–2026.
SURFACE_MU = {
    "Hard": 0.6423,
    "Clay": 0.6196,
    "Grass": 0.6606,
}

# ==========================================
# POINT-WIN PROBABILITY MODEL (T1.2)
# ==========================================
# Clamp bounds for the opponent-adjusted server point-win probability
# (ace-03-tennis-math.md §1). Noisy/small-sample skill estimates can push the raw
# spw − rpw + (1 − μ) outside a sane band; clamping to [P_MIN, P_MAX] avoids
# degenerate matches (near-certain holds/breaks). §1 suggests [0.50, 0.90].
P_MIN = 0.50
P_MAX = 0.90
# Skill-gap amplification factor for the point-win model (T1.9b). The §1 additive
# model maps genuinely-modest season-average serve/return differences to point-win
# gaps of mean |pA−pB| ≈ 0.05 — about half the ≈0.10 that reproduces the historical
# Slam straight-set share (T1.9 finding). A constant-p i.i.d. match structurally
# under-produces match-level dominance (the momentum/consistency effect v1 omits),
# so we amplify each server's deviation from the surface baseline μ before clamping:
#   P = μ + γ·(spw_server − rpw_returner + (1 − μ) − μ)
# γ = 1 recovers the pure §1 formula. γ was calibrated against scripts/validate_sim.py
# to bring the bo5/bo3 set-count distributions into T1.9's tolerance bands while
# keeping games/set and tiebreak frequency in range. Investigation & sweep: T1.9b.
POINT_GAP_GAMMA = 1.9

# ==========================================
# MATCH SIMULATION (T1.6+)
# ==========================================
# Tiebreak target for a *non-deciding* set — the standard 7-point tiebreak at
# 6–6 (ace-03-tennis-math.md §3/§4). The deciding set instead uses the
# per-match final_set_rule (7 / 10 / advantage). Kept here rather than hardcoded
# in sim/match.py so the "standard" and "final-set" targets are both nameable
# and independently changeable.
STANDARD_TIEBREAK_TARGET = 7
# Canonical deciding-set rule enum: rule name -> tiebreak target, where None
# means "advantage set" (no tiebreak, win by 2 games). Lives here rather than in
# sim/match.py so that everything needing the allowed values — the match layer
# (sim/match.FINAL_SET_TB_TARGET is an alias of this), sim/reconcile.py, the
# draw validator (T2.1, whose JSON field is spelled `final_set_tiebreak`) and
# the CLIs' argparse choices — reads one definition instead of re-listing the
# three string literals. See ace-03-tennis-math.md §4.
FINAL_SET_TB_TARGET: dict[str, int | None] = {
    "7pt_at_6_6": 7,
    "10pt_at_6_6": 10,
    "advantage": None,
}

# ==========================================
# RECONCILIATION + CALIBRATION (T1.8)
# ==========================================
# How sim/reconcile.py fuses the classifier's match-win probability (P_clf) with
# the point model's (P_point). See ace-03-tennis-math.md §6.
#   "classifier_anchor" — trust P_clf for the winner; the point model only shapes
#                         the scoreline (serve probs are shifted by δ so the
#                         simulated win rate reproduces P_clf).
#   "blend"             — P = w·P_clf + (1−w)·P_point, then the same δ shift is
#                         solved to reproduce the blended P for scorelines.
RECONCILE_MODE = "classifier_anchor"
# Blend weight w on the classifier in "blend" mode (ignored by "classifier_anchor").
RECONCILE_BLEND_WEIGHT = 0.5
# Convergence tolerance (in match-win-probability units) for the δ bisection
# solver in reconcile.solve_delta.
RECONCILE_DELTA_TOL = 1e-4

# Calibration artefact (reliability curve) produced on the held-out TEST_YEAR.
CALIBRATION_PLOT = OUTPUT_DIR / "calibration.png"
# Number of equal-width probability bins for the reliability curve.
CALIBRATION_BINS = 10

# ==========================================
# SINGLE-MATCH SIMULATION CLI (T1.10)
# ==========================================
# Monte Carlo runs the single-match CLI averages to report "A wins X% of
# simulations". ~1,000 is the ticket's figure: enough for a ±1.5pp standard
# error while keeping a manual sanity run to a couple of seconds.
SIM_CLI_MC_RUNS = 1000
# Default match format for that CLI. Best-of-5 with a 10-point deciding-set
# tiebreak is the current Grand Slam standard, which is what this simulator is
# aimed at; override per run with --best-of / --final-set-rule.
SIM_CLI_BEST_OF = 5
SIM_CLI_FINAL_SET_RULE = "10pt_at_6_6"
# Default surface when the CLI isn't told one.
SIM_CLI_SURFACE = "Hard"
# Reconciliation mode for *this CLI only* — deliberately NOT RECONCILE_MODE.
# The CLI's ClassifierProb adapter builds its feature row with
# cli/interactive.build_feature_row, which synthesises 20 of the 27
# MODEL_FEATURES (every recent-form rolling column), so the P_clf it emits is
# form-blind and collapses toward 0.5 (ace-04-current-state.md §7 seam 7).
# Under "classifier_anchor" that flattened P_clf *is* the target δ is solved to
# reproduce, which would anchor every CLI scoreline near a coin flip regardless
# of real skill gap; "blend" dilutes it with the point model instead. This is a
# mitigation for the CLI's default, not a fix for the underlying feature gap —
# override per run with --reconcile-mode. config.RECONCILE_MODE stays the
# system-wide default for callers with real engineered rows.
SIM_CLI_RECONCILE_MODE = "blend"

# ==========================================
# TOURNAMENT DRAWS (T2.1)
# ==========================================
# Hand-entered draw JSON files, one per event (ace-02-data-schema.md
# "Tournament draw JSON schema"); read by sim/draw.py.
DRAWS_DIR = Path("data/draws")
# Allowed `draw_size` values — the powers of two a real singles draw uses. A
# power-of-2 size is what makes the bracket halve cleanly round by round.
VALID_DRAW_SIZES = frozenset({8, 16, 32, 64, 128})
# Allowed `best_of` values for a draw's match format.
VALID_BEST_OF = frozenset({3, 5})
# Allowed `final_set_tiebreak` values. DERIVED from the canonical rule enum
# above (FINAL_SET_TB_TARGET) — never re-list the literals here, so the draw
# validator cannot drift out of sync with what sim/match.py actually accepts.
VALID_FINAL_SET_TIEBREAKS = frozenset(FINAL_SET_TB_TARGET)
# Entrant strings that name a *slot* rather than a player. These are not
# resolved against the skill table; they receive SkillTable.default(surface) —
# the surface-baseline profile (spw = μ, rpw = 1 − μ). Compared lower-cased and
# whitespace-stripped, and a "/"-joined entrant ("Qualifier/Lucky Loser") counts
# as a placeholder when every part does.
DRAW_PLACEHOLDER_ENTRANTS = frozenset(
    {"qualifier", "q", "lucky loser", "ll", "bye", "tbd", "alternate"}
)

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
