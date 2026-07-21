"""Game and tiebreak simulation, plus the analytic hold-probability helper.

The point-by-point match layer (``ace-03-tennis-math.md``). Landed so far:

- :func:`hold_prob` — the closed-form probability the server holds (wins a game)
  at a constant point-win probability ``p`` (``§2``, T1.3). Use this when only
  the outcome probability is needed, at Monte Carlo scale.
- :func:`simulate_game` — a point-by-point game draw returning the actual game
  score, for building real scorelines (``§5``, T1.3).
- :func:`simulate_tiebreak` — a point-by-point tiebreak with the real
  ``1-2-2-2`` alternating serve schedule (``§4``, T1.4).
- :func:`simulate_set` — a point-by-point set: serve alternates by game, first
  to 6 win by 2, 7–5, or a tiebreak at 6–6 (``§3``/``§5``, T1.5). With
  ``tb_target=None`` it plays a **no-tiebreak advantage set** — win by 2 games
  indefinitely — used by the ``"advantage"`` deciding-set rule (T1.6).
- :func:`simulate_match_bo3` — a best-of-3 match: first to 2 sets, threading the
  running server across sets and applying the ``final_set_rule`` only in the
  deciding (3rd) set (``§5``, T1.6).

The best-of-5 match layer (T1.7) extends this module later; do not add it here.

**Two decisions taken in T1.6 (documented at their call sites too):**

1. *Standard (non-deciding-set) tiebreak target.* Added
   ``config.STANDARD_TIEBREAK_TARGET = 7`` rather than hardcoding ``7`` in this
   module — threading a bare magic number through the match layer is worse, and
   the ticket sanctions the small config addition. The deciding set instead
   takes its target from the per-match ``final_set_rule``.
2. *"advantage" deciding sets.* Implemented by **extending
   :func:`simulate_set`** with ``tb_target=None`` ("no tiebreak, keep playing
   games until a 2-game lead") rather than duplicating a game loop in the match
   layer. The existing win-by-2 game check already loops indefinitely at 6–6, so
   ``None`` simply suppresses the 6–6 tiebreak branch; ``tb_score`` stays
   ``None`` for advantage sets.

**Serve-continuity contract across sets (decided in T1.5 — T1.6/T1.7 depend on
this).** Serve alternates *continuously* game-by-game across the set boundary;
it is **not** reset per set. The match layer maintains a single running
"who serves next" state and passes it into each set as ``a_serves_first``.
:func:`simulate_set` deliberately does **not** store or return any next-server
field — the next set's first server is *derived* by the caller from the set it
just played, because it is fully determined by the running rotation::

    next_a_serves_first = (_set_server(games_a + games_b, a_serves_first) == 0)

i.e. whoever is due to serve game number ``games_a + games_b`` (the game that
would come next) serves first in the following set. This is exact for every set
type because the tiebreak counts as one game, so a 7–6 set has
``games_a + games_b == 13`` (odd → the first server flips), and a normal set of
``T`` games flips the first server iff ``T`` is odd. Note this is the *general*
continuous-rotation rule; ``§3``'s shorthand "the player who received first in
the previous set serves first in the next set" is only the special case that
holds for odd-game sets (all tiebreak sets, plus 6–1/6–3/7–5/…), and this
contract intentionally supersedes that shorthand.

This module is **pure** apart from the RNG explicitly threaded into
:func:`simulate_game`: no pandas, no file/network I/O, and never the global
``np.random`` (see the determinism rule in ``CLAUDE.md``). Its only project
import is ``config`` (for ``STANDARD_TIEBREAK_TARGET``); it belongs to the
``sim/`` core and must not import from ``cli/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config


@dataclass(frozen=True)
class GameResult:
    """Outcome of a single simulated service game.

    Attributes:
        server_won: ``True`` if the server won the game.
        server_pts: Points the server won (0-based count, e.g. 4 for a 4–1 game).
        returner_pts: Points the returner won.
    """

    server_won: bool
    server_pts: int
    returner_pts: int


def hold_prob(p: float) -> float:
    """Probability the server holds (wins a game) at constant point-win prob ``p``.

    Closed form from ``ace-03-tennis-math.md §2``::

        P(game) = p⁴·(1 + 4q + 10q²) + 20·p³q³·[ p² / (p² + q²) ]

    where ``q = 1 − p``. The first term sums the win-to-love (``p⁴``), 4–1
    (``4p⁴q``), and 4–2 (``10p⁴q²``) paths; the second is the probability of
    reaching deuce (3–3, ``20p³q³``) times the closed-form deuce win prob
    ``p²/(p²+q²)``.

    Args:
        p: Probability the server wins a single point, strictly in ``(0, 1)``.

    Returns:
        The probability the server wins the game, in ``(0, 1)``.

    Raises:
        ValueError: If ``p`` is not strictly inside ``(0, 1)``.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be strictly in (0, 1), got {p!r}")
    q = 1.0 - p
    win_by_two = p**4 * (1.0 + 4.0 * q + 10.0 * q**2)  # §2: love/15/30 paths
    deuce = 20.0 * p**3 * q**3 * (p**2 / (p**2 + q**2))  # §2: reach deuce, then win
    return win_by_two + deuce


def simulate_game(p: float, rng: np.random.Generator) -> GameResult:
    """Simulate one service game point-by-point (``ace-03-tennis-math.md §5``).

    Draws points ~ Bernoulli(``p``) until one player reaches at least 4 points
    with a lead of at least 2 (deuce/advantage continues indefinitely otherwise).

    Args:
        p: Probability the server wins a single point.
        rng: A ``numpy`` ``Generator`` (from ``numpy.random.default_rng(seed)``).
            Passed explicitly for determinism — the global ``np.random`` is never
            used.

    Returns:
        A :class:`GameResult` with the winner and the final point score.
    """
    server_pts = 0
    returner_pts = 0
    while True:
        if rng.random() < p:
            server_pts += 1
        else:
            returner_pts += 1
        if max(server_pts, returner_pts) >= 4 and abs(server_pts - returner_pts) >= 2:
            break
    return GameResult(
        server_won=server_pts > returner_pts,
        server_pts=server_pts,
        returner_pts=returner_pts,
    )


@dataclass(frozen=True)
class TiebreakResult:
    """Outcome of a single simulated tiebreak (``ace-03-tennis-math.md §4``).

    Player identity is expressed as an index in ``{0, 1}``. The caller labels one
    player ``0`` and the other ``1`` and passes ``first_server`` (the index of the
    player who serves the very first point) into :func:`simulate_tiebreak`. The
    two point totals are keyed by **serving role**, and because ``pts_first`` is
    the first server's tally, it maps to player ``first_server``:

    Attributes:
        winner: Index (``0`` or ``1``) of the player who won the tiebreak. Equals
            ``first_server`` when the first server won, else ``1 - first_server``.
        pts_first: Points won by the player who served the first point
            (i.e. player ``first_server``).
        pts_other: Points won by the other player (``1 - first_server``).

    To render a set line, the loser's total is the parenthetical: e.g. a
    ``7-6(5)`` set is a game score of 7–6 with a tiebreak whose loser won 5
    points — ``min(pts_first, pts_other) == 5`` here.
    """

    winner: int
    pts_first: int
    pts_other: int


def _tiebreak_server(point_index: int, first_server: int) -> int:
    """Index of the player serving point ``point_index`` (0-based) of a tiebreak.

    Implements the ``§4`` serving schedule: ``first_server`` serves point 0, then
    serve alternates every two points — ``A, BB, AA, BB, …`` (the real
    ``1-2-2-2`` pattern). Ends change every 6 points, which is cosmetic and not
    modelled.

    Args:
        point_index: Zero-based index of the point within the tiebreak.
        first_server: Index (``0`` or ``1``) of the player who serves point 0.

    Returns:
        The index (``0`` or ``1``) of the player serving that point.
    """
    # first_server serves point 0; blocks of two thereafter. The block parity of
    # (point_index + 1) // 2 flips ownership: 0->first, 1,2->other, 3,4->first, …
    first_serving = ((point_index + 1) // 2) % 2 == 0
    return first_server if first_serving else 1 - first_server


def simulate_tiebreak(
    p_server_first: float,
    p_other: float,
    first_server: int,
    target: int,
    rng: np.random.Generator,
) -> TiebreakResult:
    """Simulate one tiebreak point-by-point (``ace-03-tennis-math.md §4``).

    Points are drawn ~ Bernoulli until one player reaches at least ``target``
    points with a lead of at least 2 (win by 2; long tiebreaks such as 12–10 are
    possible). Each point uses the *current server's* point-win probability,
    following the ``§4`` schedule via :func:`_tiebreak_server`: ``p_server_first``
    when the first server is serving, ``p_other`` when the other player is.

    Args:
        p_server_first: Probability the first server wins a point **on their own
            serve**. Strictly in ``(0, 1)``.
        p_other: Probability the other player wins a point **on their own
            serve**. Strictly in ``(0, 1)``.
        first_server: Index (``0`` or ``1``) of the player who serves the first
            point of the tiebreak.
        target: The target score — ``7`` for a standard tiebreak, ``10`` for a
            deciding-set match tiebreak. Win by 2 applies regardless.
        rng: A ``numpy`` ``Generator`` (from ``numpy.random.default_rng(seed)``).
            Passed explicitly for determinism — the global ``np.random`` is never
            used, and the generator is never reseeded.

    Returns:
        A :class:`TiebreakResult` with the winning player index and the two
        serving-role point totals.

    Raises:
        ValueError: If either probability is not strictly inside ``(0, 1)``
            (a degenerate 0/1 probability can prevent the win-by-2 condition from
            ever being met), if ``first_server`` is not ``0`` or ``1``, or if
            ``target`` is not positive.
    """
    if not (0.0 < p_server_first < 1.0):
        raise ValueError(
            f"p_server_first must be strictly in (0, 1), got {p_server_first!r}"
        )
    if not (0.0 < p_other < 1.0):
        raise ValueError(f"p_other must be strictly in (0, 1), got {p_other!r}")
    if first_server not in (0, 1):
        raise ValueError(f"first_server must be 0 or 1, got {first_server!r}")
    if target < 1:
        raise ValueError(f"target must be a positive integer, got {target!r}")

    pts_first = 0
    pts_other = 0
    point_index = 0
    while True:
        server = _tiebreak_server(point_index, first_server)
        # Probability the point goes to the first server, whoever is serving.
        p_first_wins = p_server_first if server == first_server else 1.0 - p_other
        if rng.random() < p_first_wins:
            pts_first += 1
        else:
            pts_other += 1
        point_index += 1
        if max(pts_first, pts_other) >= target and abs(pts_first - pts_other) >= 2:
            break

    winner = first_server if pts_first > pts_other else 1 - first_server
    return TiebreakResult(winner=winner, pts_first=pts_first, pts_other=pts_other)


@dataclass(frozen=True)
class SetResult:
    """Outcome of a single simulated set (``ace-03-tennis-math.md §3``/``§5``).

    Player identity follows the same convention as the game/tiebreak layers:
    the two players are ``0`` (referred to as A below) and ``1`` (B). ``games_a``
    / ``games_b`` are A's and B's final game counts — for a tiebreak set the
    winner's count includes the tiebreak game, so a 7–6 set stores ``7`` and
    ``6``.

    Attributes:
        winner: Index (``0`` = A, ``1`` = B) of the player who won the set.
        games_a: Games won by player A (e.g. ``7`` in a 7–5 or 7–6 set).
        games_b: Games won by player B.
        tb_score: The tiebreak point score as ``(points_a, points_b)`` keyed by
            player (A, B) — **only** populated when the set actually went to a
            tiebreak (6–6). ``None`` for every non-tiebreak set. To render a
            ``7-6(x)`` line, take the loser's total, ``min(tb_score)``.
    """

    winner: int
    games_a: int
    games_b: int
    tb_score: tuple[int, int] | None = None


def _set_server(game_index: int, a_serves_first: bool) -> int:
    """Index (``0`` = A, ``1`` = B) of the player serving game ``game_index``.

    Serve alternates every game within a set (``§3``): if A serves first (game 0)
    then A serves the even-numbered games and B the odd ones, and vice versa.

    The match layer reuses this with ``game_index = games_a + games_b`` (the game
    that *would* be played next) to derive the following set's first server —
    see the serve-continuity contract in the module docstring.

    Args:
        game_index: Zero-based index of the game within the set.
        a_serves_first: ``True`` if player A serves the first game of the set.

    Returns:
        The index (``0`` or ``1``) of the player serving that game.
    """
    a_serving = (game_index % 2 == 0) == a_serves_first
    return 0 if a_serving else 1


def simulate_set(
    p_a_serving: float,
    p_b_serving: float,
    a_serves_first: bool,
    tb_target: int | None,
    rng: np.random.Generator,
) -> SetResult:
    """Simulate one set point-by-point (``ace-03-tennis-math.md §3``/``§5``).

    Games are played until one player reaches 6 with a ≥2 lead (6–0…6–4), wins
    7–5, or the set reaches 6–6 and is decided by a tiebreak. Serve alternates
    each game starting with ``a_serves_first``; each game is drawn by
    :func:`simulate_game` using the *current server's* point-win probability
    (``p_a_serving`` on A's serve, ``p_b_serving`` on B's serve).

    With ``tb_target=None`` the set is an **advantage set** (``§4``): there is no
    6–6 tiebreak — play simply continues until one player leads by two games
    (e.g. 8–6, 12–10, …). This is driven by the ``"advantage"`` deciding-set
    rule; ``tb_score`` is ``None`` for such sets.

    At 6–6 the tiebreak's first server is the player *due to serve the next
    game* — ``_set_server(12, a_serves_first)`` — which, because 12 games have
    been played, is the set's first server. It is derived here rather than
    hardcoded; the set's serving probabilities carry into the tiebreak via
    :func:`simulate_tiebreak`, and the tiebreak game is credited to its winner
    (making the game score 7–6).

    See the module docstring for the cross-set serve-continuity contract: this
    function does not expose a next-server value; the match layer derives it
    from ``games_a + games_b`` and ``a_serves_first``.

    Args:
        p_a_serving: Probability player A wins a point on A's own serve.
        p_b_serving: Probability player B wins a point on B's own serve.
        a_serves_first: ``True`` if A serves the first game of the set.
        tb_target: The tiebreak target — ``7`` for a standard 6–6 tiebreak,
            ``10`` for a deciding-set match tiebreak, or ``None`` for a
            no-tiebreak advantage set. Only used if 6–6 is reached.
        rng: A ``numpy`` ``Generator`` (from ``numpy.random.default_rng(seed)``).
            Passed explicitly for determinism — the global ``np.random`` is never
            used, and the generator is never reseeded.

    Returns:
        A :class:`SetResult` with the winner, the game score, and ``tb_score``
        populated iff the set went to a tiebreak.

    Raises:
        ValueError: If either serving probability is not strictly inside
            ``(0, 1)``. A degenerate 0/1 probability can make the win-by-2
            condition unreachable — most acutely in an advantage set
            (``tb_target is None``), where equal degenerate holds would loop
            forever without a tiebreak to terminate the set.
    """
    if not (0.0 < p_a_serving < 1.0):
        raise ValueError(
            f"p_a_serving must be strictly in (0, 1), got {p_a_serving!r}"
        )
    if not (0.0 < p_b_serving < 1.0):
        raise ValueError(
            f"p_b_serving must be strictly in (0, 1), got {p_b_serving!r}"
        )

    games_a = 0
    games_b = 0
    game_index = 0
    while True:
        server = _set_server(game_index, a_serves_first)
        p = p_a_serving if server == 0 else p_b_serving
        game = simulate_game(p, rng)
        game_winner = server if game.server_won else 1 - server
        if game_winner == 0:
            games_a += 1
        else:
            games_b += 1
        game_index += 1

        # 6–6 → tiebreak, unless this is an advantage set (tb_target is None), in
        # which case the standard win-by-2 game check below just keeps looping.
        # The first server is whoever is due to serve the next game (game index
        # 12), derived — not assumed to be A. §3/§4.
        if tb_target is not None and games_a == 6 and games_b == 6:
            first_server = _set_server(game_index, a_serves_first)
            if first_server == 0:
                p_first, p_other = p_a_serving, p_b_serving
            else:
                p_first, p_other = p_b_serving, p_a_serving
            tb = simulate_tiebreak(p_first, p_other, first_server, tb_target, rng)
            # Map serving-role tallies back to players (A, B).
            if first_server == 0:
                pts_a, pts_b = tb.pts_first, tb.pts_other
            else:
                pts_a, pts_b = tb.pts_other, tb.pts_first
            # The tiebreak game goes to its winner → 7–6.
            if tb.winner == 0:
                games_a += 1
            else:
                games_b += 1
            return SetResult(
                winner=tb.winner,
                games_a=games_a,
                games_b=games_b,
                tb_score=(pts_a, pts_b),
            )

        # Standard set win: reach 6 with a ≥2 lead (covers 6–0…6–4 and 7–5). In
        # an advantage set (tb_target is None) this also covers 8–6, 12–10, …
        if max(games_a, games_b) >= 6 and abs(games_a - games_b) >= 2:
            winner = 0 if games_a > games_b else 1
            return SetResult(winner=winner, games_a=games_a, games_b=games_b)


# Deciding-set tiebreak target per final_set_rule (ace-03-tennis-math.md §4).
# ``None`` selects the no-tiebreak advantage mode of :func:`simulate_set`. All
# four current Grand Slams use ``"10pt_at_6_6"``; ``"advantage"`` is historical
# but kept for completeness.
FINAL_SET_TB_TARGET: dict[str, int | None] = {
    "7pt_at_6_6": 7,
    "10pt_at_6_6": 10,
    "advantage": None,
}


@dataclass(frozen=True)
class MatchResult:
    """Outcome of a single simulated match (``ace-03-tennis-math.md §5``).

    Player identity follows the game/set/tiebreak convention: ``0`` = A (whose
    serve carries ``pA``), ``1`` = B.

    Attributes:
        winner: Index (``0`` = A, ``1`` = B) of the player who won the match.
        sets: Every set played, in order, each a :class:`SetResult` (which
            carries its game score and, for a 6–6 set, its ``tb_score``).
        best_of: The match format — ``3`` for :func:`simulate_match_bo3`.
    """

    winner: int
    sets: list[SetResult]
    best_of: int = 3


def simulate_match_bo3(
    pA: float,
    pB: float,
    first_server: int,
    final_set_rule: str,
    rng: np.random.Generator,
) -> MatchResult:
    """Simulate a best-of-3 match point-by-point (``ace-03-tennis-math.md §5``).

    Sets are played until one player wins 2 (so 2–0 or 2–1). ``pA``/``pB`` are
    the two players' point-win probabilities on their own serve (T1.2 output) and
    are held **constant across the whole match** — no momentum/fatigue (``§5``).

    Serve continuity across sets follows the T1.5 contract exactly: the running
    "who serves next" state is carried from one set into the next via
    :func:`_set_server` on ``games_a + games_b`` (the game that would come next),
    rather than being reset per set.

    The ``final_set_rule`` is applied **only** to the deciding set — the 3rd,
    reached at one set all. Non-deciding sets use the standard tiebreak target
    ``config.STANDARD_TIEBREAK_TARGET``. The rule maps to a
    :func:`simulate_set` ``tb_target`` via :data:`FINAL_SET_TB_TARGET`:
    ``"7pt_at_6_6"`` → ``7``, ``"10pt_at_6_6"`` → ``10``, ``"advantage"`` →
    ``None`` (no-tiebreak advantage set).

    Args:
        pA: Probability player A wins a point on A's own serve.
        pB: Probability player B wins a point on B's own serve.
        first_server: Index (``0`` = A, ``1`` = B) of the player who serves the
            first game of the match.
        final_set_rule: One of ``"7pt_at_6_6"``, ``"10pt_at_6_6"``,
            ``"advantage"`` — applied only to the deciding set.
        rng: A ``numpy`` ``Generator`` (from ``numpy.random.default_rng(seed)``).
            Passed explicitly for determinism — the global ``np.random`` is never
            used, and the generator is never reseeded.

    Returns:
        A :class:`MatchResult` with the winner, every :class:`SetResult` in
        order, and ``best_of=3``.

    Raises:
        ValueError: If ``first_server`` is not ``0`` or ``1``, or
            ``final_set_rule`` is not a recognised rule.
    """
    if first_server not in (0, 1):
        raise ValueError(f"first_server must be 0 or 1, got {first_server!r}")
    if final_set_rule not in FINAL_SET_TB_TARGET:
        raise ValueError(
            f"final_set_rule must be one of {sorted(FINAL_SET_TB_TARGET)}, "
            f"got {final_set_rule!r}"
        )

    sets: list[SetResult] = []
    sets_a = 0
    sets_b = 0
    a_serves_first = first_server == 0
    while sets_a < 2 and sets_b < 2:
        # The deciding set is the 3rd, reached only at one set all.
        deciding = sets_a == 1 and sets_b == 1
        tb_target = (
            FINAL_SET_TB_TARGET[final_set_rule]
            if deciding
            else config.STANDARD_TIEBREAK_TARGET
        )
        result = simulate_set(pA, pB, a_serves_first, tb_target, rng)
        sets.append(result)
        if result.winner == 0:
            sets_a += 1
        else:
            sets_b += 1
        # Serve continuity (T1.5 contract): the next set's first server is
        # whoever is due to serve the game after this set's last one.
        a_serves_first = (
            _set_server(result.games_a + result.games_b, a_serves_first) == 0
        )

    return MatchResult(winner=0 if sets_a == 2 else 1, sets=sets, best_of=3)
