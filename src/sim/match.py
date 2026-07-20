"""Game simulation and the analytic hold-probability helper.

The first module of the point-by-point match layer (``ace-03-tennis-math.md``).
This ticket (T1.3) covers only the **game** level:

- :func:`hold_prob` — the closed-form probability the server holds (wins a game)
  at a constant point-win probability ``p`` (``§2``). Use this when only the
  outcome probability is needed, at Monte Carlo scale.
- :func:`simulate_game` — a point-by-point game draw returning the actual game
  score, for building real scorelines (``§5``).

Later tickets extend this module with the tiebreak, set, and match layers
(T1.4+); do not add them here.

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
