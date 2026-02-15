import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src directory to Python path so we can import project modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import features.rolling as rolling
import config

class TestRollingFeatures(unittest.TestCase):
    def setUp(self):
        """
        Create a small controlled dataset of 6 matches involving Player A.
        We mainly test behaviour for the 5-match rolling window.
        """

        # Player A appears in all matches, sometimes as P1 and sometimes as P2.
        # Match outcomes for Player A:
        # M1 (P1): Win
        # M2 (P2): Loss
        # M3 (P1): Loss
        # M4 (P2): Win
        # M5 (P1): Win
        # M6 (P2): Loss
        
        data = {
            'tourney_date': pd.to_datetime([
                '2023-01-01', '2023-01-02', '2023-01-03', 
                '2023-01-04', '2023-01-05', '2023-01-06'
            ]),
            'p1_name': ['A', 'B', 'A', 'C', 'A', 'D'],
            'p2_name': ['X', 'A', 'Y', 'A', 'Z', 'A'],

            # target = 1 if p1 won
            'target': [1, 1, 0, 0, 1, 1],
            
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
        
        # Manually derived stats for Player A in each match.
        # Used for reasoning about expected rolling values.
        # Format: {won, games won (gw), games lost (gl)}
        
        self.expected_a_stats = [
            {'won': 1, 'gw': 12, 'gl': 8},
            {'won': 0, 'gw': 0,  'gl': 12},
            {'won': 0, 'gw': 6,  'gl': 12},
            {'won': 1, 'gw': 15, 'gl': 10},
            {'won': 1, 'gw': 12, 'gl': 8},
            {'won': 0, 'gw': 8,  'gl': 12}
        ]

    def test_rolling_features(self):
        """
        Validate rolling statistics for Match 6 using a 5-match window.
        Ensures win rate and game averages are computed correctly
        and that player roles (P1/P2) are handled properly.
        """

        res = rolling.compute_rolling_features(self.df)

        # Match 6 corresponds to index 5
        match_6_row = res.iloc[5]
        
        # Player A is P2 in Match 6
        self.assertEqual(match_6_row['p2_name'], 'A')

        # Rolling window uses Matches 1–5 for Player A.
        # Wins: 1,0,0,1,1 -> 3/5 = 0.6
        # Games won: 12,0,6,15,12 -> 45/5 = 9.0
        # Games lost: 8,12,12,10,8 -> 50/5 = 10.0

        self.assertAlmostEqual(match_6_row['p2_recent_win_rate_5'], 0.6)
        self.assertAlmostEqual(match_6_row['p2_recent_games_won_avg_5'], 9.0)
        self.assertAlmostEqual(match_6_row['p2_recent_games_lost_avg_5'], 10.0)

    def test_leakage(self):
        """
        Ensure rolling features do not include the current match.
        Match 1 has no prior history, so default values should be used.
        """

        res = rolling.compute_rolling_features(self.df)
        match_1_row = res.iloc[0]
        
        # No prior history -> win rate should equal the default (0.5)
        self.assertEqual(match_1_row['p1_recent_win_rate_5'], 0.5)
        
        # The average games won should not equal 12 (Match 1 value).
        # If it does, current-match data has leaked into the feature.
        self.assertNotEqual(match_1_row['p1_recent_games_won_avg_5'], 12)

if __name__ == '__main__':
    unittest.main()
