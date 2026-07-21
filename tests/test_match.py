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

import config
import sim.match as match
from sim.match import (
    FINAL_SET_TB_TARGET,
    GameResult,
    MatchResult,
    SetResult,
    TiebreakResult,
    _set_server,
    _tiebreak_server,
    hold_prob,
    simulate_game,
    simulate_match_bo3,
    simulate_set,
    simulate_tiebreak,
)

# Every legal final game score for a set, keyed as (winner_games, loser_games).
VALID_SET_GAME_SCORES = {(6, 0), (6, 1), (6, 2), (6, 3), (6, 4), (7, 5), (7, 6)}


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


# ---------------------------------------------------------------------------
# _set_server — game-by-game serve rotation (§3) + the cross-set contract
# ---------------------------------------------------------------------------

def test_set_server_alternates_each_game():
    """Server flips every game; the first server owns the even-numbered games."""
    # A serves first: A on 0,2,4,…  B on 1,3,5,…
    assert [_set_server(g, True) for g in range(6)] == [0, 1, 0, 1, 0, 1]
    # B serves first is the mirror image.
    assert [_set_server(g, False) for g in range(6)] == [1, 0, 1, 0, 1, 0]


def test_set_server_at_game_12_is_the_set_first_server():
    """At 6–6 (12 games played) the due server is the set's first server.

    This is exactly the index :func:`simulate_set` derives for the tiebreak, and
    the value the cross-set contract uses for a 7–6 set (13 games → flip).
    """
    assert _set_server(12, True) == 0
    assert _set_server(12, False) == 1
    # A 7–6 set has 13 games → the next set's first server flips.
    assert _set_server(13, True) == 1
    assert _set_server(13, False) == 0


def test_cross_set_serve_continuity_formula():
    """The documented cross-set contract holds for an even- and an odd-total set.

    The module docstring specifies the next set's first server as
    ``next_a_serves_first = (_set_server(games_a + games_b, a_serves_first) == 0)``.
    Under continuous rotation this must flip iff the set's total games is odd.
    Exercise it on a genuine even-total set (a forced 6–0, total 6) and a genuine
    odd-total set (a real 7–6 tiebreak, total 13) — enough because ``_set_server``
    is a fully characterized parity function.
    """
    def next_a_serves_first(res, a_serves_first):
        total = res.games_a + res.games_b
        return _set_server(total, a_serves_first) == 0

    # Even total (6–0): continuous rotation keeps the same first server.
    even_set = simulate_set(0.99, 0.01, True, tb_target=7, rng=np.random.default_rng(1))
    assert (even_set.games_a + even_set.games_b) % 2 == 0
    assert next_a_serves_first(even_set, True) is True
    assert next_a_serves_first(even_set, False) is False

    # Odd total (7–6, tiebreak game counts as one): the first server flips.
    _, odd_set = _find_tiebreak_set(a_serves_first=True)
    assert (odd_set.games_a + odd_set.games_b) % 2 == 1
    assert next_a_serves_first(odd_set, True) is False
    assert next_a_serves_first(odd_set, False) is True


# ---------------------------------------------------------------------------
# simulate_set — point-by-point set (§3/§5)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a_serves_first", [True, False])
def test_simulate_set_ends_on_a_legal_score(a_serves_first):
    """Every set ends 6–x (x≤4), 7–5, or 7–6 — and never beyond."""
    rng = np.random.default_rng(20260720)
    for _ in range(4000):
        res = simulate_set(0.63, 0.6, a_serves_first, tb_target=7, rng=rng)
        assert isinstance(res, SetResult)
        hi = max(res.games_a, res.games_b)
        lo = min(res.games_a, res.games_b)
        assert (hi, lo) in VALID_SET_GAME_SCORES
        # winner index is consistent with the game score.
        assert res.winner == (0 if res.games_a > res.games_b else 1)


def test_simulate_set_extreme_p_gives_near_shutout():
    """p_a_serving≈1, p_b_serving≈0 → A wins every game → 6–0, winner A.

    A wins on serve (0.99) and breaks B's serve (B only wins 0.01 of its points),
    so A takes six straight regardless of who serves first.
    """
    rng = np.random.default_rng(1)
    for a_serves_first in (True, False):
        for _ in range(50):
            res = simulate_set(0.99, 0.01, a_serves_first, tb_target=7, rng=rng)
            assert (res.games_a, res.games_b) == (6, 0)
            assert res.winner == 0
            assert res.tb_score is None


def test_simulate_set_reverse_extreme_p_gives_near_shutout_for_b():
    """Symmetric check: p_b_serving≈1, p_a_serving≈0 → B wins 6–0."""
    rng = np.random.default_rng(2)
    for a_serves_first in (True, False):
        for _ in range(50):
            res = simulate_set(0.01, 0.99, a_serves_first, tb_target=7, rng=rng)
            assert (res.games_a, res.games_b) == (0, 6)
            assert res.winner == 1
            assert res.tb_score is None


def _find_tiebreak_set(a_serves_first, tb_target=7, max_seeds=500):
    """Return the first (seed, SetResult) whose set reached a tiebreak.

    High, equal hold probabilities (0.95 on both serves) make breaks rare, so a
    6–6 tiebreak turns up quickly across seeds.
    """
    for seed in range(max_seeds):
        res = simulate_set(
            0.95, 0.95, a_serves_first, tb_target, np.random.default_rng(seed)
        )
        if res.tb_score is not None:
            return seed, res
    raise AssertionError("no tiebreak set found across the searched seeds")


@pytest.mark.parametrize("a_serves_first", [True, False])
def test_simulate_set_tiebreak_at_6_6_records_score(a_serves_first):
    """A 6–6 set is decided by a tiebreak: game score is 7–6 and tb_score is set.

    Also checks the tiebreak point totals are internally consistent — the set
    winner is the player with more tiebreak points, and the loser's total is
    ``min(tb_score)`` (the ``7-6(x)`` parenthetical).
    """
    _, res = _find_tiebreak_set(a_serves_first)
    assert res.tb_score is not None
    hi, lo = max(res.games_a, res.games_b), min(res.games_a, res.games_b)
    assert (hi, lo) == (7, 6)
    pts_a, pts_b = res.tb_score
    # Tiebreak winner (≥7, win by 2) matches the set winner.
    assert max(pts_a, pts_b) >= 7
    assert abs(pts_a - pts_b) >= 2
    assert res.winner == (0 if pts_a > pts_b else 1)
    # Winner's game count is 7; loser's is 6.
    winner_games = res.games_a if res.winner == 0 else res.games_b
    assert winner_games == 7


def test_tb_score_populated_iff_set_went_to_tiebreak():
    """tb_score is non-None exactly when the game score is 7–6, else None."""
    rng = np.random.default_rng(555)
    for _ in range(5000):
        res = simulate_set(0.7, 0.68, True, tb_target=7, rng=rng)
        went_to_tb = {res.games_a, res.games_b} == {7, 6}
        assert (res.tb_score is not None) == went_to_tb


def test_tiebreak_first_server_is_the_due_server_not_hardcoded(monkeypatch):
    """At 6–6 the tiebreak's first server is derived from whose turn it is.

    A naive bug would hardcode ``first_server=0`` (always A) or pass the *last*
    game's server (game 11 → the wrong player). We spy on ``simulate_tiebreak``
    to capture the actual ``first_server`` argument in a real 6–6 set and assert
    it flips with ``a_serves_first``: A (0) when A served first, B (1) otherwise.
    """
    captured = {}
    real_tb = match.simulate_tiebreak

    def spy(p_first, p_other, first_server, target, rng):
        captured["first_server"] = first_server
        return real_tb(p_first, p_other, first_server, target, rng)

    monkeypatch.setattr(match, "simulate_tiebreak", spy)

    for a_serves_first, expected in [(True, 0), (False, 1)]:
        captured.clear()
        for seed in range(500):
            res = simulate_set(
                0.95, 0.95, a_serves_first, 7, np.random.default_rng(seed)
            )
            if res.tb_score is not None:
                break
        assert res.tb_score is not None, "expected to reach a tiebreak"
        assert captured["first_server"] == expected


def test_simulate_set_server_alternates_via_captured_probabilities(monkeypatch):
    """Each game is drawn with the current server's p, alternating game to game.

    Spying on ``simulate_game`` records the point-win probability handed to each
    game. With distinct per-server probabilities, the recorded sequence must
    alternate ``p_a, p_b, p_a, …`` (A serving first) — proving serve alternates
    correctly and that the right player's probability is used each game.
    """
    seen_p = []
    real_game = match.simulate_game

    def spy(p, rng):
        seen_p.append(p)
        return real_game(p, rng)

    monkeypatch.setattr(match, "simulate_game", spy)
    simulate_set(0.71, 0.62, a_serves_first=True, tb_target=7,
                 rng=np.random.default_rng(20260720))
    expected = [0.71 if i % 2 == 0 else 0.62 for i in range(len(seen_p))]
    assert seen_p == expected


def test_simulate_set_is_deterministic():
    """Same seed → identical SetResult (value equality on the dataclass)."""
    r1 = simulate_set(0.66, 0.61, True, 7, np.random.default_rng(2024))
    r2 = simulate_set(0.66, 0.61, True, 7, np.random.default_rng(2024))
    assert r1 == r2


def test_simulate_set_tb_target_10_every_winner_reaches_ten():
    """tb_target=10 is threaded into the tiebreak (deciding-set variant).

    Collects many 6–6 tiebreak sets and asserts the *winner's* tiebreak points
    are ≥10 on **every** one. A genuine target-10 breaker can never end with a
    winner below 10 (e.g. 7–5 is impossible), whereas a target-7 breaker ends
    7–x constantly — so this kills a ``tb_target``-hardcoded-to-7 mutant, which
    a single ``max(tb_score) >= 10`` check does not (a long 7-target win-by-2
    breaker can coincidentally reach 10+).
    """
    tb_sets = []
    for seed in range(500):
        res = simulate_set(0.95, 0.95, True, 10, np.random.default_rng(seed))
        if res.tb_score is not None:
            tb_sets.append(res)
        if len(tb_sets) >= 30:
            break
    assert len(tb_sets) >= 30, "not enough tiebreak sets to exercise tb_target=10"
    for res in tb_sets:
        winner_pts = res.tb_score[res.winner]
        assert winner_pts >= 10


def test_simulate_set_advantage_mode_never_tiebreaks(monkeypatch):
    """``tb_target=None`` → no tiebreak; play continues past 6–6, win by 2.

    Spies on ``simulate_tiebreak`` to prove it is *never* called, and searches
    for a genuine extended set (winner on ≥8 games) that a 6–6 tiebreak would
    have prevented. Every advantage set must end on an exactly-2-game margin with
    ``tb_score is None``.
    """
    monkeypatch.setattr(
        match,
        "simulate_tiebreak",
        lambda *a, **k: pytest.fail("advantage set must not call simulate_tiebreak"),
    )
    saw_extended = False
    for seed in range(3000):
        res = simulate_set(0.9, 0.9, True, tb_target=None, rng=np.random.default_rng(seed))
        assert res.tb_score is None
        hi, lo = max(res.games_a, res.games_b), min(res.games_a, res.games_b)
        assert hi >= 6 and hi - lo >= 2
        if lo >= 6:  # extended past 6–6: must end on an exact 2-game margin
            assert hi - lo == 2
        if hi >= 8:
            saw_extended = True
    assert saw_extended, "never saw an advantage set extend past 7 games"


@pytest.mark.parametrize("tb_target", [7, None], ids=["tiebreak", "advantage"])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"p_a_serving": 0.0},
        {"p_a_serving": 1.0},
        {"p_b_serving": 0.0},
        {"p_b_serving": 1.0},
    ],
)
def test_simulate_set_rejects_degenerate_p(tb_target, kwargs):
    """Degenerate 0/1 serving probabilities raise for both players, both modes.

    Mirrors :func:`simulate_tiebreak`'s ``(0, 1)`` guard. Applied regardless of
    ``tb_target`` so the int (tiebreak) path is protected too — and, crucially,
    so an advantage set (``tb_target=None``) rejects a degenerate ``p`` up front
    rather than looping forever (no tiebreak exists to terminate it).
    """
    base = {
        "p_a_serving": 0.6,
        "p_b_serving": 0.6,
        "a_serves_first": True,
        "tb_target": tb_target,
        "rng": np.random.default_rng(0),
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        simulate_set(**base)


# ---------------------------------------------------------------------------
# simulate_match_bo3 — best-of-3 match (§5, T1.6)
# ---------------------------------------------------------------------------

def _fake_set_sequence(monkeypatch, winners, captured):
    """Replace ``simulate_set`` with a fake returning ``winners`` in order.

    Records each call's ``(a_serves_first, tb_target)`` into ``captured`` and
    returns a clean 6–0 / 0–6 set for the scheduled winner, so a match's set
    sequence (and thus whether a deciding set is reached) is forced
    deterministically, independent of the RNG.
    """
    it = iter(winners)

    def fake_set(p_a, p_b, a_serves_first, tb_target, rng):
        captured.append({"a_serves_first": a_serves_first, "tb_target": tb_target})
        w = next(it)
        return SetResult(winner=w, games_a=6 if w == 0 else 0,
                         games_b=6 if w == 1 else 0)

    monkeypatch.setattr(match, "simulate_set", fake_set)


def test_match_ends_when_a_player_reaches_two_sets():
    """A Bo3 is always 2 or 3 sets, and the winner wins exactly 2 of them."""
    rng = np.random.default_rng(20260722)
    for _ in range(500):
        res = simulate_match_bo3(0.63, 0.6, 0, "10pt_at_6_6", rng)
        assert isinstance(res, MatchResult)
        assert res.best_of == 3
        assert len(res.sets) in (2, 3)
        winner_sets = sum(s.winner == res.winner for s in res.sets)
        assert winner_sets == 2
        # The loser never reached 2 sets.
        assert len(res.sets) - winner_sets < 2


def test_extreme_p_wins_in_straight_sets():
    """p_a≈1, p_b≈0 → A wins 2–0 with two 6–0 sets; symmetric for B."""
    for seed in range(50):
        res = simulate_match_bo3(0.99, 0.01, 0, "10pt_at_6_6", np.random.default_rng(seed))
        assert res.winner == 0
        assert len(res.sets) == 2
        for s in res.sets:
            assert (s.games_a, s.games_b) == (6, 0)
    for seed in range(50):
        res = simulate_match_bo3(0.01, 0.99, 0, "10pt_at_6_6", np.random.default_rng(seed))
        assert res.winner == 1
        assert len(res.sets) == 2
        for s in res.sets:
            assert (s.games_a, s.games_b) == (0, 6)


def test_balanced_p_is_fifty_fifty_with_realistic_set_split():
    """Equal players → ~50/50 winner and ~50% straight-sets (2–0) matches.

    With independent 50/50 sets, P(2–0) = P(same player wins the first two
    sets) = 0.5 exactly, and P(2–1) = 0.5 — a realistic, analytically-pinned
    split, not a hand-tuned constant.
    """
    rng = np.random.default_rng(20260722)
    n = 20000
    a_wins = 0
    straight = 0
    for _ in range(n):
        res = simulate_match_bo3(0.63, 0.63, 0, "10pt_at_6_6", rng)
        a_wins += res.winner == 0
        if len(res.sets) == 2:
            straight += 1
    assert a_wins / n == pytest.approx(0.5, abs=0.02)
    assert straight / n == pytest.approx(0.5, abs=0.03)


def test_final_set_rule_applied_only_to_deciding_set(monkeypatch):
    """Each rule's target reaches the *3rd* set only; sets 1–2 use the standard.

    Forces a 2–1 match (A, B, A) with a fake ``simulate_set`` and asserts the
    ``tb_target`` handed to each set: the standard target for the first two, the
    rule-specific target for the deciding third — for every enum value including
    ``"advantage"`` (``None``). A bug applying the rule to an earlier set, or the
    standard target to the decider, is caught here.
    """
    for rule, expected_target in FINAL_SET_TB_TARGET.items():
        captured: list[dict] = []
        _fake_set_sequence(monkeypatch, [0, 1, 0], captured)
        res = simulate_match_bo3(0.6, 0.6, 0, rule, np.random.default_rng(0))
        assert [c["tb_target"] for c in captured] == [
            config.STANDARD_TIEBREAK_TARGET,
            config.STANDARD_TIEBREAK_TARGET,
            expected_target,
        ]
        assert res.winner == 0
        assert len(res.sets) == 3


def test_advantage_deciding_set_is_a_real_no_tiebreak_set():
    """A real ``"advantage"`` match: the deciding set never tiebreaks and can
    extend past 7 games (e.g. 8–6, 10–8), while a *non-deciding* set in the same
    format still uses the standard 7-point tiebreak (so ``tb_score`` can appear
    there). Not an untested enum pass-through — the deciding set is exercised.
    """
    saw_extended = False
    checked = 0
    for seed in range(6000):
        res = simulate_match_bo3(0.9, 0.9, 0, "advantage", np.random.default_rng(seed))
        if len(res.sets) < 3:
            continue
        checked += 1
        deciding = res.sets[2]
        assert deciding.tb_score is None  # advantage set: no tiebreak, ever
        hi, lo = max(deciding.games_a, deciding.games_b), min(deciding.games_a, deciding.games_b)
        assert hi >= 6 and hi - lo >= 2
        if lo >= 6:  # extended past 6–6 must end on an exact 2-game margin
            assert hi - lo == 2
        if hi >= 8:
            saw_extended = True
    assert checked > 0, "no 3-set advantage match found to exercise the decider"
    assert saw_extended, "advantage decider never extended past 7 games"


def test_deciding_set_10pt_tiebreak_is_recorded():
    """A 10-point deciding tiebreak is applied and its score is recorded.

    Confirms ``MatchResult`` carries tiebreak scores where relevant and that the
    ``"10pt_at_6_6"`` target actually reaches the decider: the winner's tiebreak
    points are ≥10 (impossible under a 7-point breaker's 7–x endings short of
    deuce).
    """
    for seed in range(6000):
        res = simulate_match_bo3(0.9, 0.9, 0, "10pt_at_6_6", np.random.default_rng(seed))
        if len(res.sets) == 3 and res.sets[2].tb_score is not None:
            deciding = res.sets[2]
            assert max(deciding.tb_score) >= 10
            assert deciding.tb_score[deciding.winner] == max(deciding.tb_score)
            break
    else:
        raise AssertionError("no 3-set match with a 10-point deciding tiebreak found")


@pytest.mark.parametrize("first_server", [0, 1])
def test_serve_continuity_across_set_boundaries_uses_t15_formula(first_server, monkeypatch):
    """The a_serves_first handed to each set follows the exact T1.5 contract.

    Wraps the real ``simulate_set`` to record ``(a_serves_first, SetResult)`` per
    set, then asserts the first set starts with ``first_server`` and every
    subsequent set's first server equals
    ``_set_server(prev.games_a + prev.games_b, prev_a_serves_first) == 0`` — the
    documented formula, reused rather than re-derived. A Bo3 is always ≥2 sets,
    so at least one boundary is exercised for each ``first_server``.
    """
    calls: list[tuple[bool, SetResult]] = []
    real_set = match.simulate_set

    def spy(p_a, p_b, a_serves_first, tb_target, rng):
        res = real_set(p_a, p_b, a_serves_first, tb_target, rng)
        calls.append((a_serves_first, res))
        return res

    monkeypatch.setattr(match, "simulate_set", spy)
    simulate_match_bo3(0.63, 0.6, first_server, "10pt_at_6_6",
                       np.random.default_rng(20260722))

    assert calls[0][0] is (first_server == 0)
    for (a_first_prev, res_prev), (a_first_next, _) in zip(calls, calls[1:]):
        total = res_prev.games_a + res_prev.games_b
        assert a_first_next == (_set_server(total, a_first_prev) == 0)


def test_simulate_match_bo3_is_deterministic():
    """Same seed → identical MatchResult (value equality through the set list)."""
    r1 = simulate_match_bo3(0.64, 0.6, 0, "10pt_at_6_6", np.random.default_rng(2024))
    r2 = simulate_match_bo3(0.64, 0.6, 0, "10pt_at_6_6", np.random.default_rng(2024))
    assert r1 == r2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"first_server": 2},
        {"first_server": -1},
        {"final_set_rule": "bogus"},
        {"final_set_rule": "7pt"},
    ],
)
def test_simulate_match_bo3_rejects_bad_args(kwargs):
    """Bad server index or unrecognised final_set_rule raises ValueError."""
    base = {
        "pA": 0.6,
        "pB": 0.6,
        "first_server": 0,
        "final_set_rule": "10pt_at_6_6",
        "rng": np.random.default_rng(0),
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        simulate_match_bo3(**base)
