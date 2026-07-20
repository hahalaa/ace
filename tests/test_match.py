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

from sim.match import GameResult, hold_prob, simulate_game


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
