"""Tests for src/sim/match.py — the T1.3 game layer.

Covers ace-03-tennis-math.md §2 (analytic hold probability) and §5
(point-by-point game simulation). The analytic form is checked against an
**independent** enumeration of game-winning paths (combinatorial counts of the
non-deuce wins plus a geometric sum over deuce cycles), not merely a re-derived
closed form, and cross-checked against seeded Monte Carlo.
"""
from math import comb

import numpy as np
import pytest

from sim.match import (
    GameResult,
    TiebreakResult,
    _tiebreak_server,
    hold_prob,
    simulate_game,
    simulate_tiebreak,
)


def enumerate_hold_prob(p: float, deuce_cycles: int = 400) -> float:
    """Independent brute-force enumeration of P(server holds).

    Deliberately does **not** reuse §2's grouped form ``(1 + 4q + 10q²)`` or the
    closed-form deuce term ``p²/(p²+q²)``:

    - Non-deuce wins (server wins 4–k for k = 0,1,2): the last point is the
      server's, so the earlier ``3 + k`` points contain the ``k`` returner points
      in ``C(3+k, k)`` orders, each with probability ``p⁴ q^k``.
    - Deuce is reached at 3–3 in ``C(6, 3)`` orders (prob ``C(6,3) p³q³``). From
      deuce, the server wins after ``n`` split cycles (each a 1–1 exchange, prob
      ``2pq``) followed by winning two straight (``p²``) — summed as a truncated
      geometric series rather than collapsed algebraically.

    Args:
        p: Point-win probability in ``(0, 1)``.
        deuce_cycles: Number of deuce split-cycles to sum (truncation of the
            geometric series; 400 is far beyond numerical relevance).

    Returns:
        The enumerated hold probability.
    """
    q = 1.0 - p
    total = 0.0
    for k in range(3):  # server wins 4–0, 4–1, 4–2
        total += comb(3 + k, k) * p**4 * q**k
    p_reach_deuce = comb(6, 3) * p**3 * q**3
    split = 2.0 * p * q
    deuce_win = sum((split**n) * p**2 for n in range(deuce_cycles))
    total += p_reach_deuce * deuce_win
    return total


# ---------------------------------------------------------------------------
# hold_prob — analytic §2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", [0.5, 0.55, 0.6, 0.62, 0.66, 0.7, 0.75, 0.8, 0.9])
def test_hold_prob_matches_independent_enumeration(p):
    """§2 closed form equals the independent path enumeration."""
    assert hold_prob(p) == pytest.approx(enumerate_hold_prob(p), abs=1e-12)


def test_hold_prob_half_is_half():
    """At p = 0.5 the game is symmetric, so the server holds exactly half the time."""
    assert hold_prob(0.5) == pytest.approx(0.5, abs=1e-12)


def test_hold_prob_monotonic_increasing():
    """A better server holds more often."""
    ps = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    holds = [hold_prob(p) for p in ps]
    assert holds == sorted(holds)
    assert all(a < b for a, b in zip(holds, holds[1:]))


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_hold_prob_rejects_out_of_range(bad):
    """p must be strictly inside (0, 1)."""
    with pytest.raises(ValueError):
        hold_prob(bad)


# ---------------------------------------------------------------------------
# simulate_game — point-by-point §5
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("p", [0.3, 0.5, 0.62, 0.75])
def test_simulate_game_produces_valid_scores(p):
    """Winner has >=4 points and leads by >=2; the result is self-consistent."""
    rng = np.random.default_rng(20260720)
    for _ in range(2000):
        res = simulate_game(p, rng)
        assert isinstance(res, GameResult)
        winner_pts = max(res.server_pts, res.returner_pts)
        loser_pts = min(res.server_pts, res.returner_pts)
        assert winner_pts >= 4
        assert winner_pts - loser_pts >= 2
        assert res.server_won == (res.server_pts > res.returner_pts)


@pytest.mark.parametrize("p", [0.5, 0.62, 0.75])
def test_empirical_hold_rate_matches_analytic(p):
    """Seeded Monte Carlo hold rate ≈ hold_prob(p) — ties §5 back to §2."""
    rng = np.random.default_rng(7)
    n = 60000
    wins = sum(simulate_game(p, rng).server_won for _ in range(n))
    empirical = wins / n
    # ~4 standard errors of a proportion at this n (worst-case SE ≈ 0.002 at
    # hold ≈ 0.5): tight enough to catch a wrong hold_prob, loose enough not to flake.
    assert empirical == pytest.approx(hold_prob(p), abs=0.008)


def test_simulate_game_is_deterministic():
    """Same seed → identical game outcome (value equality on the dataclass)."""
    r1 = simulate_game(0.63, np.random.default_rng(123))
    r2 = simulate_game(0.63, np.random.default_rng(123))
    assert r1 == r2


def test_different_seeds_can_differ():
    """Sanity: the RNG actually drives the outcome (not a constant)."""
    results = {
        simulate_game(0.6, np.random.default_rng(s)) for s in range(50)
    }
    # Expect a spread of scorelines across seeds, not a single fixed result.
    assert len(results) > 1


# ---------------------------------------------------------------------------
# simulate_tiebreak — point-by-point §4
# ---------------------------------------------------------------------------

def test_tiebreak_serve_schedule_matches_1_2_2_2():
    """§4: first server serves point 0, then serve alternates every 2 points.

    Asserts the server per point directly (not just the final score) for the
    first 12 points, for both possible first servers — the real A, BB, AA, BB, …
    pattern.
    """
    # first_server = 0 -> A BB AA BB AA BB
    expected_a_first = [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
    assert [_tiebreak_server(n, 0) for n in range(12)] == expected_a_first
    # first_server = 1 is the mirror image.
    expected_b_first = [1 - s for s in expected_a_first]
    assert [_tiebreak_server(n, 1) for n in range(12)] == expected_b_first


def test_tiebreak_win_by_two_and_target_reached():
    """Winner has >= target points and leads by exactly >= 2; totals consistent."""
    rng = np.random.default_rng(20260720)
    for target in (7, 10):
        for _ in range(3000):
            res = simulate_tiebreak(0.62, 0.6, first_server=0, target=target, rng=rng)
            assert isinstance(res, TiebreakResult)
            hi = max(res.pts_first, res.pts_other)
            lo = min(res.pts_first, res.pts_other)
            assert hi >= target
            assert hi - lo >= 2
            # winner index is consistent with the point totals and first_server=0.
            assert res.winner == (0 if res.pts_first > res.pts_other else 1)


def test_tiebreak_can_end_exactly_at_target():
    """A tiebreak can end at exactly `target` points (e.g. 7–0..7–5), not only
    beyond it — pins the win condition at `>= target` and rejects `> target`
    (first-to-8), which would never produce a winner on exactly 7 points."""
    rng = np.random.default_rng(20260720)
    saw_exact = False
    for _ in range(2000):
        res = simulate_tiebreak(0.62, 0.6, first_server=0, target=7, rng=rng)
        if max(res.pts_first, res.pts_other) == 7:
            saw_exact = True
            break
    assert saw_exact, "no tiebreak ever ended with the winner on exactly 7 points"


def test_tiebreak_long_breaker_possible():
    """A long win-by-2 tiebreak (e.g. 12–10) can occur — target is not a cap."""
    rng = np.random.default_rng(1)
    max_points = 0
    for _ in range(20000):
        res = simulate_tiebreak(0.6, 0.6, first_server=0, target=7, rng=rng)
        max_points = max(max_points, res.pts_first + res.pts_other)
        if max(res.pts_first, res.pts_other) >= 12:
            break
    else:
        raise AssertionError(
            f"never saw a tiebreak reaching 12 points (max total {max_points})"
        )


def test_tiebreak_first_server_index_labels_winner():
    """winner == first_server when the first server wins, else 1 - first_server."""
    # A strong first server who never misses on serve and a weak opponent: the
    # first server should take it, and winner must equal first_server.
    res = simulate_tiebreak(
        0.99, 0.01, first_server=1, target=7, rng=np.random.default_rng(3)
    )
    assert res.winner == 1
    assert res.pts_first > res.pts_other


def test_tiebreak_symmetry_equal_p_is_fifty_fifty():
    """With p_server_first == p_other, each player wins ~50% over many sims."""
    rng = np.random.default_rng(2026)
    n = 30000
    first_wins = 0
    for _ in range(n):
        res = simulate_tiebreak(0.63, 0.63, first_server=0, target=7, rng=rng)
        first_wins += res.winner == 0
    assert first_wins / n == pytest.approx(0.5, abs=0.02)


def test_tiebreak_supports_target_10():
    """target=10 is honoured — the winner needs at least 10 points."""
    rng = np.random.default_rng(99)
    res = simulate_tiebreak(0.6, 0.6, first_server=0, target=10, rng=rng)
    assert max(res.pts_first, res.pts_other) >= 10


def test_tiebreak_is_deterministic():
    """Same seed → identical tiebreak outcome (value equality on the dataclass)."""
    r1 = simulate_tiebreak(0.62, 0.58, 0, 7, np.random.default_rng(456))
    r2 = simulate_tiebreak(0.62, 0.58, 0, 7, np.random.default_rng(456))
    assert r1 == r2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_server_first": 0.0},
        {"p_server_first": 1.0},
        {"p_other": 0.0},
        {"p_other": 1.0},
        {"first_server": 2},
        {"first_server": -1},
        {"target": 0},
    ],
)
def test_tiebreak_rejects_bad_args(kwargs):
    """Degenerate probabilities, bad server index, or non-positive target raise."""
    base = {
        "p_server_first": 0.6,
        "p_other": 0.6,
        "first_server": 0,
        "target": 7,
        "rng": np.random.default_rng(0),
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        simulate_tiebreak(**base)
