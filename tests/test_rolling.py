import unittest
import pandas as pd
import numpy as np

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


class TestRollingFormTable(unittest.TestCase):
    """The as-of-now snapshot accessor.

    ``build_rolling_form_table`` exists so a caller outside the training pass,
    the classifier adapter, can ask "what is player X's rolling form right
    now". The property that makes it trustworthy is that it is not a second
    implementation: it must equal what ``compute_rolling_features`` produces for
    a match played immediately after the last row of the data.
    """

    def setUp(self):
        TestRollingFeatures.setUp(self)

    def _appended_match_row(self, player: str) -> pd.Series:
        """Pipeline features for a hypothetical next match with ``player`` as p1."""
        extra = self.df.iloc[[0]].copy()
        extra['tourney_date'] = self.df['tourney_date'].max() + pd.Timedelta(days=1)
        extra['p1_name'] = player
        extra['p2_name'] = 'Newcomer'
        extended = pd.concat([self.df, extra], ignore_index=True)
        return rolling.compute_rolling_features(extended).iloc[-1]

    def test_snapshot_equals_what_the_pipeline_would_compute_next(self):
        """The accessor reproduces the training pass, one row past the end."""
        table = rolling.build_rolling_form_table(self.df)
        snapshot = table.latest('A')
        expected = self._appended_match_row('A')

        for column in rolling.rolling_feature_names():
            self.assertAlmostEqual(snapshot[column], expected[f'p1_{column}'], msg=column)

    def test_snapshot_covers_every_window_and_metric(self):
        table = rolling.build_rolling_form_table(self.df)
        self.assertEqual(
            sorted(table.latest('A')), sorted(rolling.rolling_feature_names())
        )
        # 5 metrics x 2 windows, the 20 features (10 per player) MODEL_FEATURES needs.
        self.assertEqual(len(rolling.rolling_feature_names()), 10)

    def test_snapshot_uses_only_the_last_window_matches(self):
        """Player A played 6 matches; the 5-window must drop the oldest."""
        table = rolling.build_rolling_form_table(self.df)
        # A's results in order: W L L W W L -> last 5 = L L W W L = 2/5.
        self.assertAlmostEqual(table.latest('A')['recent_win_rate_5'], 0.4)
        # All six fall inside the 10-window: 3/6.
        self.assertAlmostEqual(table.latest('A')['recent_win_rate_10'], 0.5)

    def test_unknown_player_raises_rather_than_defaulting(self):
        """A silent neutral default here is the seam-7 defect in miniature."""
        table = rolling.build_rolling_form_table(self.df)
        self.assertNotIn('Nobody', table)
        with self.assertRaises(KeyError):
            table.latest('Nobody')

    def test_both_entry_points_read_the_same_player_frame(self):
        """No second stacking/sorting implementation to drift from."""
        frame = rolling.build_player_match_frame(self.df)
        self.assertEqual(len(frame), 2 * len(self.df))
        self.assertEqual(
            list(frame.columns[:2]), ['date', 'player']
        )
        a_rows = frame[frame['player'] == 'A']
        self.assertTrue(a_rows['date'].is_monotonic_increasing)


if __name__ == '__main__':
    unittest.main()
