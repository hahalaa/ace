"""Game and tiebreak simulation, plus the analytic hold-probability helper.

The point-by-point match layer (``ace-03-tennis-math.md``). Landed so far:

- :func:`hold_prob` — the closed-form probability the server holds (wins a game)
  at a constant point-win probability ``p`` (``§2``, T1.3). Use this when only
  the outcome probability is needed, at Monte Carlo scale.
- :func:`simulate_game` — a point-by-point game draw returning the actual game
  score, for building real scorelines (``§5``, T1.3).
- :func:`simulate_tiebreak` — a point-by-point tiebreak with the real
  ``1-2-2-2`` alternating serve schedule (``§4``, T1.4).

Later tickets extend this module with the set and match layers (T1.5+); do not
add them here.

This module is **pure** apart from the RNG explicitly threaded into
:func:`simulate_game`: no pandas, no file/network I/O, and never the global
``np.random`` (see the determinism rule in ``CLAUDE.md``). It belongs to the
``sim/`` core and must not import from ``cli/``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
