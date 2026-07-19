import config
import sys
import pandas as pd

from common.names import MatchStrategy, NameIndex, resolve_name

# ==========================================
# 5. INTERACTIVE PREDICTION LOOP
# ==========================================
def interactive_prediction_loop(model, data, surf_hist, h2h_hist):
    """
    Runs a REPL-style loop for predicting tennis matches.
    Prompts for two player names and a surface, then predicts
    the winner using the provided model and historical stats.
    """
    print("\n" + "="*40)
    print(" 🎾  TENNIS MATCH PREDICTOR  🎾")
    print("="*40)
    print("Type 'exit' to quit. Use Ctrl+C to stop safely.\n")

    # Pre-fetch all unique player names for fuzzy matching
    all_players = pd.concat([data['p1_name'], data['p2_name']]).unique().tolist()

    while True:
        try:
            p1_input = input("Enter Player 1 (e.g. Carlos Alcaraz): ").strip()
            if p1_input.lower() == 'exit': break
            
            p1 = resolve_player_name(p1_input, all_players)
            if p1 is None:
                print(f"❌ Player '{p1_input}' not found. Please try again.\n")
                continue
            if p1 != p1_input:
                print(f"   -> Deduced: {p1}")

            p2_input = input("Enter Player 2 (e.g. Jannik Sinner): ").strip()
            if p2_input.lower() == 'exit': break

            p2 = resolve_player_name(p2_input, all_players)
            if p2 is None:
                print(f"❌ Player '{p2_input}' not found. Please try again.\n")
                continue
            if p2 != p2_input:
                print(f"   -> Deduced: {p2}")

            surf_input = input("Enter Surface (Hard, Clay, Grass): ")
            surf = validate_surface(surf_input)

            if surf is None:
                print("❌ Invalid surface. Choose Hard, Clay, or Grass.\n")
                continue

            p1_stats = get_latest(p1, data)
            p2_stats = get_latest(p2, data)

            if not p1_stats[0] or not p2_stats[0]:
                print(f"❌ Error: Player not found in database.\n")
                continue
                
            p1_rank, p1_age = p1_stats
            p2_rank, p2_age = p2_stats
            
            p1_w, p1_t = get_surf_record(p1, surf, surf_hist)
            p2_w, p2_t = get_surf_record(p2, surf, surf_hist)

            p1_pct = p1_w/p1_t if p1_t > 0 else config.DEFAULT_WIN_PCT
            p2_pct = p2_w/p2_t if p2_t > 0 else config.DEFAULT_WIN_PCT

            diff, h2h_msg = compute_h2h(p1, p2, h2h_hist)

            # Predict
            input_data = build_feature_row(
                p1_rank, p2_rank,
                p1_age, p2_age,
                p1_pct, p2_pct,
                diff
            )
            prob = model.predict_proba(input_data)[0][1] # Probability P1 wins

            display_matchup(p1, p2, surf, p1_rank, p2_rank, p1_age, p2_age, p1_pct, p2_pct, p1_w, p1_t, p2_w, p2_t, h2h_msg, prob)

        except KeyboardInterrupt:
            print("\n\n👋 Exiting. Thank you!")
            sys.exit()
        except EOFError:
            # Closed/empty stdin (piped input, redirected file, Ctrl-D). Without
            # this, input() raises EOFError every iteration, the broad handler
            # below swallows it, and the loop spins forever. Leave cleanly.
            print("\n👋 Input closed. Exiting.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

# =========================
# Helper functions
# =========================
def validate_surface(s: str) -> str | None:
    s = s.strip().capitalize()
    return s if s in config.VALID_SURFACES else None

def resolve_player_name(input_name: str, all_names: list[str]) -> str | None:
    """
    Fuzzy match a player name against the list of known names.
    Supports:
      - Exact match (case-insensitive)
      - "F. Lastname" or "F Lastname" format
      - Partial/Fuzzy string matching

    Thin CLI wrapper over the pure resolver in ``common.names`` (T0.6): the
    matching lives there; this keeps the friendly ``print``-based UX by
    rendering ambiguity itself and returning a plain matched name (or ``None``).
    """
    match = resolve_name(input_name, NameIndex.from_names(all_names))
    if match is None:
        return None
    if match.is_ambiguous:
        _print_ambiguity(input_name, match)
        return None
    return match.name

def _print_ambiguity(input_name: str, match) -> None:
    """Render an ambiguous NameMatch exactly as the old resolver did."""
    if match.strategy is MatchStrategy.FUZZY:
        print(f"Did you mean: {', '.join(match.candidates)}?")
    elif match.strategy is MatchStrategy.SUBSTRING:
        shown = ', '.join(match.candidates[:5])
        suffix = '...' if len(match.candidates) > 5 else ''
        print(f"Ambiguous: Multiple players match '{input_name}': {shown}{suffix}. Please be more specific.")
    else:  # INITIALS
        print(f"Ambiguous: Multiple players match '{input_name}': {', '.join(match.candidates)}. Please be more specific.")

def get_latest(name: str, data: pd.DataFrame):
    """
    Get the most recent match for a player and return their rank and age.

    Returns:
        (rank, age) if found, else (None, None)
    """
    player_mask = (data['p1_name'] == name) | (data['p2_name'] == name)
    player_matches = data.loc[player_mask]

    if player_matches.empty:
        return None, None

    latest_match = player_matches.sort_values('tourney_date', ascending=False).iloc[0]

    if latest_match['p1_name'] == name:
        return latest_match['p1_rank'], latest_match['p1_age']
    else:
        return latest_match['p2_rank'], latest_match['p2_age']

def get_surf_record(player: str, surf: str, surf_hist: dict) -> tuple[int,int]:
    """
    Get a player's historical surface record (wins, total matches).
    Returns (wins, total) or (0,0) if no history.
    """
    if player in surf_hist and surf in surf_hist[player]:
        return surf_hist[player][surf]
    return 0, 0

def compute_h2h(p1: str, p2: str, h2h_hist: dict) -> tuple[int,str]:
    """
    Returns (diff, message) for head-to-head history between p1 and p2.
    """
    key = tuple(sorted([p1, p2]))
    if key not in h2h_hist:
        return 0, "No prior matches"
    
    w1, w2 = h2h_hist[key]
    p1_wins = w1 if p1 == key[0] else w2
    p2_wins = w2 if p1 == key[0] else w1
    diff = p1_wins - p2_wins

    if p1_wins == p2_wins:
        msg = f"Tied {p1_wins}-{p2_wins}"
    else:
        leader = p1 if p1_wins > p2_wins else p2
        msg = f"{leader} leads {max(p1_wins, p2_wins)}-{min(p1_wins, p2_wins)}"
    
    return diff, msg

def display_matchup(
    p1: str, p2: str, surf: str, p1_rank: float, p2_rank: float, 
    p1_age: float, p2_age: float, p1_pct: float, p2_pct: float,
    p1_w: int, p1_t: int, p2_w: int, p2_t: int, h2h_msg: str, prob: float
) -> None:
    print(f"\n📊 MATCHUP STATS: {surf} Court")
    print(f"{'':<20} {p1:<20} {p2:<20}")
    print(f"{'Rank':<20} #{int(p1_rank):<19} #{int(p2_rank):<19}")
    print(f"{'Age':<20} {p1_age:.1f}y{'':<18} {p2_age:.1f}y")
    print(f"{'Surface Rec':<20} {p1_pct:.0%} ({p1_w}-{p1_t-p1_w}){'':<12} {p2_pct:.0%} ({p2_w}-{p2_t-p2_w})")
    print(f"{'Head-to-Head':<20} {h2h_msg}")
    print("-" * 60)
    if prob > config.DEFAULT_WIN_PCT:
        print(f"🏆 WINNER PREDICTION: {p1} ({prob:.1%} confidence)")
    else:
        print(f"🏆 WINNER PREDICTION: {p2} ({1-prob:.1%} confidence)")
    print("-" * 60 + "\n")

def build_feature_row(
    p1_rank: float,
    p2_rank: float,
    p1_age: float,
    p2_age: float,
    p1_pct: float,
    p2_pct: float,
    h2h_diff: int
) -> pd.DataFrame:
    """
    Build a single-row DataFrame matching MODEL_FEATURES order.
    Fills missing rolling features with defaults (neutral form).
    """
    # Base features that we can calculate interactively
    features = {
        'p1_rank': p1_rank,
        'p2_rank': p2_rank,
        'p1_age': p1_age,
        'p2_age': p2_age,
        'p1_surface_win_pct': p1_pct,
        'p2_surface_win_pct': p2_pct,
        'h2h_diff': h2h_diff
    }
    
    # Create DataFrame with all model features
    df = pd.DataFrame(columns=config.MODEL_FEATURES)
    
    # Fill known features
    for col, val in features.items():
        if col in df.columns:
            df.loc[0, col] = val
            
    # Fill remaining (rolling) features with defaults
    # Win rates -> 0.5, counts -> 0 (or reasonably neutral values)
    # Since we can't easily fetch the *latest* rolling stats without more complex logic,
    # we accept this limitation for the CLI tool.
    
    df = df.fillna(0) # Initialize with 0
    
    # Set neutral win rates
    win_cols = [c for c in df.columns if 'win_rate' in c]
    df[win_cols] = config.DEFAULT_WIN_PCT
    
    return df