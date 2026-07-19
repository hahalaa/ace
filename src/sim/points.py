"""Opponent-adjusted point-win probability (``ace-03-tennis-math.md §1``).

Turns two players' serve/return skill on a surface into the probability the
**server** wins a single point on their own serve — the one number every layer
of the simulator (games, sets, tiebreaks, matches) is built on.

This module is **pure**: deterministic numeric transformations only, no pandas,
no file/network I/O, and — because §1 is not stochastic — no RNG. It consumes the
T1.1 :class:`~features.serve.PlayerSkill` rates and the surface baseline ``μ``;
it does not compute or store those (that is T1.1). See the layering rule in
``CLAUDE.md`` and the purity requirement in ``ace-01-architecture.md``.
"""

from __future__ import annotations

import config
from features.serve import PlayerSkill


def point_win_prob(
    server_spw: float,
    returner_rpw: float,
    mu_surface: float,
    p_min: float = config.P_MIN,
    p_max: float = config.P_MAX,
) -> float:
    """Probability the server wins a point on serve, clamped to ``[p_min, p_max]``.

    Implements ``ace-03-tennis-math.md §1``::

        P = spw_server − rpw_returner + (1 − μ)

    Baseline is ``μ``; the server's serve-skill deviation (``spw − μ``) is added
    and the returner's return-skill deviation (``rpw − (1 − μ)``) subtracted,
    which reduces to the expression above. An average server (``spw = μ``) facing
    an average returner (``rpw = 1 − μ``) yields exactly ``μ``.

    Args:
        server_spw: Server's serve-points-won rate on the surface.
        returner_rpw: Returner's return-points-won rate on the surface.
        mu_surface: Tour-average serve-points-won for the surface (``config.SURFACE_MU``).
        p_min: Lower clamp bound (default ``config.P_MIN``).
        p_max: Upper clamp bound (default ``config.P_MAX``).

    Returns:
        The clamped point-win probability in ``[p_min, p_max]``.
    """
    p = server_spw - returner_rpw + (1.0 - mu_surface)  # §1
    # Clamp to guard against noisy/small-sample skill estimates producing
    # degenerate (near-certain hold/break) points. §1.
    return min(max(p, p_min), p_max)


def matchup_point_probs(
    skill_a: PlayerSkill,
    skill_b: PlayerSkill,
    mu_surface: float,
    p_min: float = config.P_MIN,
    p_max: float = config.P_MAX,
) -> tuple[float, float]:
    """Both servers' point-win probabilities for a matchup on one surface.

    §1 must be computed **twice per match** — once with each player serving —
    because the two players' serve/return profiles differ. Returns the pair in
    ``(A serving, B serving)`` order.

    Args:
        skill_a: Player A's :class:`~features.serve.PlayerSkill` on the surface.
        skill_b: Player B's :class:`~features.serve.PlayerSkill` on the surface.
        mu_surface: Tour-average serve-points-won for the surface.
        p_min: Lower clamp bound (default ``config.P_MIN``).
        p_max: Upper clamp bound (default ``config.P_MAX``).

    Returns:
        ``(p_a_serving, p_b_serving)``: P(A wins a point on A's serve) and
        P(B wins a point on B's serve), each clamped to ``[p_min, p_max]``.
    """
    p_a_serving = point_win_prob(skill_a.spw, skill_b.rpw, mu_surface, p_min, p_max)
    p_b_serving = point_win_prob(skill_b.spw, skill_a.rpw, mu_surface, p_min, p_max)
    return p_a_serving, p_b_serving
