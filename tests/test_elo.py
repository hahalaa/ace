"""Tests for the Elo rating engine (``features/elo.py``), a display feature.

Four properties the ticket calls out explicitly, plus the surface-track and
K-shrink behaviour:

  * a hand-computable worked example, verified to the arithmetic (not just
    "runs without erroring");
  * an upset moves a rating more than an expected result does;
  * chronological order is *enforced* in code, not assumed, the assert fires on
    out-of-order input, and the public entry point is order-insensitive;
  * the stale-player exclusion flag.
"""

from __future__ import annotations

import pandas as pd
import pytest

from features import elo
from features.elo import (
    ELO_INITIAL_RATING,
    Leaderboard,
    compute_elo,
    dynamic_k,
    prepare_matches,
)

FIXED_K = lambda n: 32.0  # noqa: E731, a constant K makes the arithmetic hand-checkable


def _match(winner, loser, date, surface="Hard", rnd="R32", num=1) -> dict:
    """One raw-loader-shaped match row (winner/loser, before p1/p2 randomising)."""
    return {
        "tourney_date": pd.Timestamp(date),
        "winner_id": winner,
        "winner_name": f"Player {winner}",
        "loser_id": loser,
        "loser_name": f"Player {loser}",
        "surface": surface,
        "round": rnd,
        "match_num": num,
    }


def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _rating(result, track: str, player_id: str) -> float:
    board: Leaderboard = result.leaderboards[track]
    for entry in board.ratings:
        if entry.player_id == player_id:
            return entry.rating
    raise KeyError(f"{player_id} not in {track} board")


# --------------------------------------------------------------------------- #
# Worked example, fixed K = 32, hand-computed.
# --------------------------------------------------------------------------- #
class TestWorkedExample:
    def test_a_single_win_between_two_debutants_is_exact(self):
        """Both start at 1500, E = 0.5, so the winner gains 32*0.5 = 16 exactly."""
        result = compute_elo(_frame([_match("A", "B", "2020-01-06")]), k_factor=FIXED_K)
        assert _rating(result, "overall", "A") == pytest.approx(1516.0)
        assert _rating(result, "overall", "B") == pytest.approx(1484.0)

    def test_three_match_sequence_matches_the_hand_calculation(self):
        """A beats B, A beats B, then B upsets A. Independently hand-computed:

            M1  A 1516.0000  B 1484.0000   (E=0.5, ±16)
            M2  A 1530.5305  B 1469.4695   (E_A=0.545922, A +14.5305)
            M3  B beats A: E_B=0.413..., B +18.7849 → A 1511.7471  B 1488.2529
        """
        rows = [
            _match("A", "B", "2020-01-06", num=1),
            _match("A", "B", "2020-01-13", num=2),
            _match("B", "A", "2020-01-20", num=3),  # upset
        ]
        result = compute_elo(_frame(rows), k_factor=FIXED_K)
        # Ratings are rounded to 2 dp for display, so allow one rounding step.
        assert _rating(result, "overall", "A") == pytest.approx(1511.7471, abs=1e-2)
        assert _rating(result, "overall", "B") == pytest.approx(1488.2529, abs=1e-2)

    def test_ratings_are_zero_sum_under_a_constant_k(self):
        """With one K, every point A gains B loses, the total stays 2*1500."""
        rows = [
            _match("A", "B", "2020-01-06"),
            _match("A", "B", "2020-01-13"),
            _match("B", "A", "2020-01-20"),
        ]
        result = compute_elo(_frame(rows), k_factor=FIXED_K)
        total = _rating(result, "overall", "A") + _rating(result, "overall", "B")
        assert total == pytest.approx(2 * ELO_INITIAL_RATING)


# --------------------------------------------------------------------------- #
# An upset swings a rating more than an expected result does.
# --------------------------------------------------------------------------- #
class TestUpsetSwing:
    def test_upset_moves_the_favourite_more_than_an_expected_result(self):
        """Build a gap (A beats B eight times), then compare the ninth match:
        A winning again (expected) versus B winning (upset). The upset must move
        A's rating by more."""
        history = [_match("A", "B", "2020-01-06", num=i) for i in range(8)]
        base = compute_elo(_frame(history), k_factor=FIXED_K)
        base_a = _rating(base, "overall", "A")
        assert base_a > 1500, "A should be the clear favourite after eight wins"

        expected = compute_elo(
            _frame(history + [_match("A", "B", "2020-02-01", num=9)]), k_factor=FIXED_K
        )
        upset = compute_elo(
            _frame(history + [_match("B", "A", "2020-02-01", num=9)]), k_factor=FIXED_K
        )

        expected_swing = abs(_rating(expected, "overall", "A") - base_a)
        upset_swing = abs(_rating(upset, "overall", "A") - base_a)
        assert upset_swing > expected_swing


# --------------------------------------------------------------------------- #
# Chronological order is enforced, not assumed.
# --------------------------------------------------------------------------- #
class TestChronology:
    def test_compute_elo_is_order_insensitive_because_it_sorts(self):
        """The public entry point sorts, so a shuffled frame gives identical
        ratings, the same-day round order is the only tiebreak."""
        rows = [
            _match("A", "B", "2020-01-06", num=1),
            _match("B", "C", "2020-02-06", num=1),
            _match("A", "C", "2020-03-06", num=1),
            _match("C", "A", "2020-04-06", num=1),
        ]
        ordered = compute_elo(_frame(rows), k_factor=FIXED_K)
        shuffled = compute_elo(_frame(list(reversed(rows))), k_factor=FIXED_K)
        for pid in ("A", "B", "C"):
            assert _rating(ordered, "overall", pid) == pytest.approx(
                _rating(shuffled, "overall", pid)
            )

    def test_the_pass_rejects_an_out_of_order_frame(self):
        """The load-bearing assert: ``_elo_pass`` refuses a descending-date frame,
        proving the ordering requirement is real and not decorative."""
        # Prepared (so it has the columns) but then reversed into the wrong order.
        prepared = prepare_matches(
            _frame(
                [
                    _match("A", "B", "2020-01-06"),
                    _match("A", "B", "2020-02-06"),
                ]
            )
        )
        backwards = prepared.iloc[::-1].reset_index(drop=True)
        with pytest.raises(AssertionError, match="chronological"):
            elo._elo_pass(
                backwards, initial_rating=ELO_INITIAL_RATING, k_factor=FIXED_K
            )

    def test_assume_sorted_still_asserts_the_claim(self):
        """A caller that claims ``assume_sorted`` but hands unsorted rows is
        rejected, not silently miscomputed."""
        prepared = prepare_matches(
            _frame(
                [
                    _match("A", "B", "2020-01-06"),
                    _match("A", "B", "2020-02-06"),
                ]
            )
        )
        with pytest.raises(AssertionError):
            compute_elo(
                prepared.iloc[::-1].reset_index(drop=True),
                k_factor=FIXED_K,
                assume_sorted=True,
            )

    def test_same_day_matches_apply_in_round_order(self):
        """Within one tournament date, an earlier round is applied first. Feeding
        the final before the semi (same date) must still rate the semi first."""
        rows = [
            # Deliberately listed final-first; prepare_matches must reorder them.
            _match("A", "C", "2020-01-06", rnd="F", num=2),
            _match("A", "B", "2020-01-06", rnd="SF", num=1),
        ]
        prepared = prepare_matches(_frame(rows))
        assert list(prepared["round"]) == ["SF", "F"]


# --------------------------------------------------------------------------- #
# Stale-player exclusion.
# --------------------------------------------------------------------------- #
class TestStaleExclusion:
    def test_a_player_who_stopped_long_ago_is_flagged_inactive(self):
        """OLD's last match predates the active window; NEW's is recent."""
        rows = [
            # OLD plays FOE, long before the reference date, then never again.
            _match("OLD", "FOE", "2018-01-06"),
            # NEW plays FOE recently, this is also the data's latest date.
            _match("NEW", "FOE", "2026-06-06"),
        ]
        result = compute_elo(_frame(rows), k_factor=FIXED_K, active_window_days=365)
        board = {e.player_id: e for e in result.leaderboards["overall"].ratings}
        assert result.as_of == pd.Timestamp("2026-06-06").date()
        assert board["OLD"].is_active is False
        assert board["NEW"].is_active is True
        assert board["FOE"].is_active is True  # played recently against NEW

    def test_activity_is_tour_wide_not_per_surface(self):
        """A player active overall but whose last CLAY match is old is still
        active on the clay board, activity is a whole-player property."""
        rows = [
            _match("CLAYER", "FOE", "2024-05-06", surface="Clay"),
            _match("CLAYER", "FOE", "2026-06-06", surface="Hard"),  # recent, hard
        ]
        result = compute_elo(_frame(rows), k_factor=FIXED_K, active_window_days=365)
        clay = {e.player_id: e for e in result.leaderboards["Clay"].ratings}
        # Last clay match is 2024 (stale on clay), but the player is active overall.
        assert clay["CLAYER"].last_played == pd.Timestamp("2024-05-06").date()
        assert clay["CLAYER"].is_active is True


# --------------------------------------------------------------------------- #
# Surface tracks.
# --------------------------------------------------------------------------- #
class TestSurfaceTracks:
    def test_a_clay_match_moves_clay_and_overall_but_not_hard_or_grass(self):
        result = compute_elo(
            _frame([_match("A", "B", "2020-01-06", surface="Clay")]), k_factor=FIXED_K
        )
        assert _rating(result, "overall", "A") == pytest.approx(1516.0)
        assert _rating(result, "Clay", "A") == pytest.approx(1516.0)
        # A never played hard/grass, so appears on neither board.
        assert all(e.player_id != "A" for e in result.leaderboards["Hard"].ratings)
        assert all(e.player_id != "A" for e in result.leaderboards["Grass"].ratings)

    def test_carpet_and_unknown_surface_move_only_the_overall_track(self):
        rows = [
            _match("A", "B", "2020-01-06", surface="Carpet"),
            _match("A", "B", "2020-02-06", surface=None),
        ]
        result = compute_elo(_frame(rows), k_factor=FIXED_K)
        # Two overall matches happened; A is on the overall board...
        a_overall = next(
            e for e in result.leaderboards["overall"].ratings if e.player_id == "A"
        )
        assert a_overall.matches == 2
        # ...but on none of the three surface boards.
        for surface in ("Hard", "Clay", "Grass"):
            assert all(
                e.player_id != "A" for e in result.leaderboards[surface].ratings
            )


# --------------------------------------------------------------------------- #
# Dynamic K.
# --------------------------------------------------------------------------- #
class TestDynamicK:
    def test_k_shrinks_as_matches_accumulate(self):
        """The stabiliser: an established player's K is far below a debutant's."""
        assert dynamic_k(0) > dynamic_k(50) > dynamic_k(400)

    def test_default_k_is_the_dynamic_one(self):
        """A first match under the default K moves the winner by the debutant K/2
        (E=0.5), not by the constant-32 amount the worked example pins."""
        result = compute_elo(_frame([_match("A", "B", "2020-01-06")]))
        gain = _rating(result, "overall", "A") - ELO_INITIAL_RATING
        assert gain == pytest.approx(dynamic_k(0) * 0.5, abs=1e-2)


# --------------------------------------------------------------------------- #
# Cleaning.
# --------------------------------------------------------------------------- #
class TestPreparation:
    def test_rows_missing_an_id_are_dropped(self):
        rows = [
            _match("A", "B", "2020-01-06"),
            {**_match("C", "D", "2020-02-06"), "winner_id": None},
        ]
        prepared = prepare_matches(_frame(rows))
        assert len(prepared) == 1

    def test_empty_after_cleaning_raises(self):
        rows = [{**_match("A", "B", "2020-01-06"), "winner_id": None}]
        with pytest.raises(ValueError, match="no usable matches"):
            compute_elo(_frame(rows))

    def test_missing_required_column_raises_keyerror(self):
        frame = _frame([_match("A", "B", "2020-01-06")]).drop(columns=["winner_id"])
        with pytest.raises(KeyError, match="winner_id"):
            prepare_matches(frame)
