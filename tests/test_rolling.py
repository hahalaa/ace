import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import features.rolling as rolling
import config

class TestRollingFeatures(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataframe representing a sequence of matches for "Player A"
        # We'll simulate Player A playing 6 matches.
        # Window sizes are default [5, 10]. We focus on 5.
        
        data = {
            'tourney_date': pd.to_datetime([
                '2023-01-01', '2023-01-02', '2023-01-03', 
                '2023-01-04', '2023-01-05', '2023-01-06'
            ]),
            'p1_name': ['A', 'B', 'A', 'C', 'A', 'D'],
            'p2_name': ['X', 'A', 'Y', 'A', 'Z', 'A'],
            'target': [1, 1, 0, 0, 1, 1], # 1 if p1 won
            
            # Outcome for A:
            # Match 1 (As P1): Won.
            # Match 2 (As P2): Lost (Since P1=B won).
            # Match 3 (As P1): Lost.
            # Match 4 (As P2): Won (Since P1=C lost).
            # Match 5 (As P1): Won.
            # Match 6 (As P2): Lost.
            
            # Stats for A (Games Won - Games Lost)
            # M1: 6-4, 6-4 (A won 12, A lost 8)
            # M2: 6-0, 6-0 (B won 12, A lost 12, A won 0)
            # M3: 6-3, 6-3 (Y won 12, A lost 12, A won 6)
            # M4: 6-4, 4-6, 6-4 (C lost? No C lost means A won. Scores usually official winner first. 
            # In preprocess, we produce p1_games_won etc.
            # Let's fill pre-processed columns manually to match logic.
            
            'p1_games_won':  [12, 12, 6,  10, 12, 12],
            'p1_games_lost': [8,  0, 12, 15, 8,  8],
            'p1_sets_won':   [2,  2, 0,  1,  2,  2],
            'p1_sets_lost':  [0,  0, 2,  2,  0,  0],
            
            'p2_games_won':  [8,  0, 12, 15, 8,  8],
            'p2_games_lost': [12, 12, 6,  10, 12, 12],
            'p2_sets_won':   [0,  0, 2,  2,  0,  0],
            'p2_sets_lost':  [2,  2, 0,  1,  2,  2],
        }
        self.df = pd.DataFrame(data)
        
        # Manually derive A's stats per match
        # M1 (P1): Won=1, GW=12, GL=8
        # M2 (P2): Won=0, GW=0,  GL=12  (Since B(p1) won, A(p2) lost)
        # M3 (P1): Won=0, GW=6,  GL=12
        # M4 (P2): Won=1, GW=15, GL=10  (Since C(p1) lostTarget=0, A(p2) won)
        # M5 (P1): Won=1, GW=12, GL=8
        # M6 (P2): Won=0, GW=8,  GL=12
        
        self.expected_a_stats = [
            {'won': 1, 'gw': 12, 'gl': 8},
            {'won': 0, 'gw': 0,  'gl': 12},
            {'won': 0, 'gw': 6,  'gl': 12},
            {'won': 1, 'gw': 15, 'gl': 10},
            {'won': 1, 'gw': 12, 'gl': 8},
            {'won': 0, 'gw': 8,  'gl': 12}
        ]

    def test_rolling_features(self):
        # Override config windows if necessary, but we rely on [5, 10]
        
        # DEBUG: Print the relevant columns for visual inspection
        res = rolling.compute_rolling_features(self.df)
        
        # Reconstruct player_df logic partially to see what happened or trust print inside rolling if enabled
        # Better: let's modify rolling.py to print? No, let's just inspect 'res' carefully
        
        # Filter for matches involving A
        print("\nMatches involving A:")
        for idx, row in res.iterrows():
            if row['p1_name'] == 'A' or row['p2_name'] == 'A':
                print(f"Match {idx}: {row['tourney_date']} {row['p1_name']} vs {row['p2_name']} | Target: {row['target']}")
                if row['p1_name'] == 'A':
                    print(f"  > A (P1) Recent WinRate 5: {row['p1_recent_win_rate_5']}")
                    print(f"  > A (P1) Recent GW Avg 5:  {row['p1_recent_games_won_avg_5']}")
                else:
                    print(f"  > A (P2) Recent WinRate 5: {row['p2_recent_win_rate_5']}")
                    print(f"  > A (P2) Recent GW Avg 5:  {row['p2_recent_games_won_avg_5']}")

        match_6_row = res.iloc[5]
        
        # A is P2 in match 6
        self.assertEqual(match_6_row['p2_name'], 'A')
        
        print(f"Computed Win Rate: {match_6_row['p2_recent_win_rate_5']}")
        print(f"Computed GW Avg: {match_6_row['p2_recent_games_won_avg_5']}")

        self.assertAlmostEqual(match_6_row['p2_recent_win_rate_5'], 0.6)
        self.assertAlmostEqual(match_6_row['p2_recent_games_won_avg_5'], 9.0)
        self.assertAlmostEqual(match_6_row['p2_recent_games_lost_avg_5'], 10.0)

    def test_leakage(self):
        # Verify Match 1 features for P1 (A) don't use Match 1 results
        res = rolling.compute_rolling_features(self.df)
        match_1_row = res.iloc[0]
        
        # Should be default values because no prior history
        # Config default win pct is 0.5
        self.assertEqual(match_1_row['p1_recent_win_rate_5'], 0.5)
        
        # Games won should be global mean or 0 (we used mean fillna)
        # Hard to check exact mean without calc, but shouldn't be 12 (match 1 value)
        # If it was 12, leakage!
        # Calculating mean of 'games_won' column in the internal player_df
        # In this small dataset: (12+0+6+15+12+8)/6 = 53/6 = 8.833
        # Match 1 GW for A is 12.
        
        self.assertNotEqual(match_1_row['p1_recent_games_won_avg_5'], 12)

if __name__ == '__main__':
    unittest.main()
