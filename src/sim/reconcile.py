"""Reconcile the point model with the classifier, and expose one authoritative
match-win probability plus a scoreline generator consistent with it (T1.8).

Implements ``ace-03-tennis-math.md §6``. The existing classifier yields
``P_clf(A beats B)`` from rank/form/H2H/surface features; the point model yields
``P_point(A beats B)`` from serve/return skills. They will not agree exactly.
This module fuses them (config-selectable) and produces a
:class:`~sim.match.MatchResult` whose winner distribution matches the fused
probability.

Three pieces, in dependency order:

1. **The name/id join (the single seam, ``ace-04-current-state.md §7``).**
   :func:`simulate_reconciled_match` receives player **names**, resolves each to a
   ``player_id`` *exactly once* via ``SkillTable.resolve_name`` (which delegates to
   ``common/names.py``), and uses the ids for the point model while the classifier
   keeps its name-keyed view. A name that does not resolve raises immediately — a
   resolution failure must never surface downstream as a ``KeyError`` or a silent
   wrong-player bug.

2. **The analytic match-win probability.** :func:`match_win_prob_point` composes
   the closed-form hold (``§2``), tiebreak (``§4``) and set (``§3``) probabilities
   directly — **deterministic, no RNG, no games/points simulated**. This is a hard
   performance requirement: T2.2/T2.3's Monte-Carlo bracket runs call it via
   ``simulate_bracket(..., outcome_only=True)`` on the order of hundreds of
   thousands of match evaluations per pass, and that budget only holds if this
   stays non-stochastic. It deliberately does **not** call
   :func:`~sim.match.simulate_match_bo3` / ``_bo5`` (there is no
   ``simulate_match(best_of=...)`` dispatcher on disk, and those functions
   *simulate*). A separate MC estimator, :func:`match_win_prob_point_mc`, exists
   **only** for cross-validating the analytic composition (T1.9) — it is never on
   the hot path and ``outcome_only`` runs must never call it.

3. **Reconciliation + scoreline.** :func:`solve_delta` finds the symmetric serve
   shift ``δ`` such that ``match_win_prob_point(pA+δ, pB−δ, …) ≈`` a target
   probability, by bisection on a function proven monotone below. The two modes
   (``config.RECONCILE_MODE``) pick the target: ``"classifier_anchor"`` uses
   ``P_clf`` directly; ``"blend"`` uses ``w·P_clf + (1−w)·P_point``.
   :func:`simulate_reconciled_match` ties it together and is what Phase 2 calls.

**On the ``classifier`` argument (design decision, flagged for review).** The
classifier's ``predict_proba`` needs a *name-keyed feature row* built from the
pipeline's ``surface_history`` / ``h2h_history`` / rank-age data — state that lives
in the classifier/CLI layer and that ``sim/`` must not import (the layering rule in
``CLAUDE.md``). The ticket's signature does not thread those histories in, so
``sim/reconcile`` cannot faithfully rebuild that feature row itself. We therefore
model ``classifier`` as a **callable adapter** :class:`ClassifierProb` —
``classifier(player_a, player_b, surface) -> P(A beats B)`` — which *is* the
opaque ``predict_proba`` boundary, just wrapped so the feature-row plumbing stays
in the layer that owns it (``predictor.py`` / the T1.10 CLI, where importing
``cli/`` is allowed). :func:`load_pinned_classifier` loads and *pins* the persisted
estimator (recording which of the best-of-four it is; see
``ace-04-current-state.md §4``) for that wiring and for the calibration check.

This module belongs to the ``sim/`` core: its only project imports are ``config``,
``sim.match``, ``sim.points`` and (for skills/typing) ``features.serve`` — never
``cli/``. It is not on the pure-by-construction list that ``sim/points.py`` and the
scoring functions of ``sim/match.py`` are; :func:`load_pinned_classifier` does load
a file, but only once at wiring time, never in the per-match hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import numpy as np

import config
from sim.match import (
    FINAL_SET_TB_TARGET,
    MatchResult,
    _set_server,
    _tiebreak_server,
    hold_prob,
    simulate_match_bo3,
    simulate_match_bo5,
)
from sim.points import matchup_point_probs

if TYPE_CHECKING:  # pragma: no cover - typing only
    from features.serve import SkillTable


class ClassifierProb(Protocol):
    """The opaque classifier boundary: names + surface → ``P(A beats B)``.

    ``simulate_reconciled_match`` feeds this its **name-keyed** inputs (the two
    display names and the surface) and expects the probability that player A beats
    player B. A real adapter wraps the pinned ``predict_proba`` estimator plus the
    pipeline's feature-row builder; tests pass a plain callable. See the module
    docstring for why the classifier is a callable rather than the raw estimator.
    """

    def __call__(self, player_a: str, player_b: str, surface: str) -> float: ...


# ---------------------------------------------------------------------------
# Analytic match-win probability (the hot path — deterministic, no RNG).
# ---------------------------------------------------------------------------
# The closed-form composition §2→§4→§3. It reuses the *exact* serve schedule
# helpers of the point-by-point simulator (``_tiebreak_server`` / ``_set_server``)
# so the analytic model and the simulation agree on who serves each point/game —
# which is what lets a δ solved on the analytic function reproduce the target as a
# *simulated* win rate (the T1.8 acceptance criterion). All functions here are
# deterministic and make no RNG draws.


def _tiebreak_win_prob(pA: float, pB: float, first_server: int, target: int) -> float:
    """Exact P(A wins a tiebreak), by memoised recursion over the point score.

    ``§4`` is analytically fiddly because serving alternates 1-2-2-2, so we walk
    the score lattice, using :func:`~sim.match._tiebreak_server` to know who serves
    each point (A wins a point with prob ``pA`` on A's serve, ``1 − pB`` on B's).
    The unbounded win-by-2 tail is collapsed with a closed form: at any tie
    ``(k, k)`` with ``k ≥ target − 1`` the next two points are one on each player's
    serve, so ``P(A wins from the tie) = pA(1−pB) / (pA(1−pB) + (1−pA)pB)``.

    Deterministic; no RNG.
    """
    win_from_tie_num = pA * (1.0 - pB)
    win_from_tie_den = win_from_tie_num + (1.0 - pA) * pB
    win_from_tie = win_from_tie_num / win_from_tie_den if win_from_tie_den > 0 else 0.5

    memo: dict[tuple[int, int], float] = {}

    def rec(a: int, b: int) -> float:
        if a >= target and a - b >= 2:
            return 1.0
        if b >= target and b - a >= 2:
            return 0.0
        if a == b and a >= target - 1:
            return win_from_tie
        key = (a, b)
        cached = memo.get(key)
        if cached is not None:
            return cached
        server = _tiebreak_server(a + b, first_server)  # index of the server
        pa_point = pA if server == 0 else 1.0 - pB
        val = pa_point * rec(a + 1, b) + (1.0 - pa_point) * rec(a, b + 1)
        memo[key] = val
        return val

    return rec(0, 0)


def _set_win_prob(
    a_hold: float, b_hold: float, a_serves_first: bool, at_6_6: float
) -> float:
    """Exact P(A wins a set), by memoised recursion over the game score (``§3``).

    Games alternate serve (:func:`~sim.match._set_server`): A wins a game she
    serves with prob ``a_hold`` and a game B serves with prob ``1 − b_hold``. The
    set ends at 6–x (x≤4), 7–5, or 6–6 → ``at_6_6`` (the tiebreak win prob, or the
    advantage-set win-by-2 closed form supplied by the caller).

    Deterministic; no RNG.
    """
    memo: dict[tuple[int, int], float] = {}

    def rec(ga: int, gb: int) -> float:
        if ga == 6 and gb == 6:
            return at_6_6
        if ga >= 6 and ga - gb >= 2:  # 6–0…6–4, 7–5
            return 1.0
        if gb >= 6 and gb - ga >= 2:
            return 0.0
        key = (ga, gb)
        cached = memo.get(key)
        if cached is not None:
            return cached
        server = _set_server(ga + gb, a_serves_first)  # index of the server
        pa_game = a_hold if server == 0 else 1.0 - b_hold
        val = pa_game * rec(ga + 1, gb) + (1.0 - pa_game) * rec(ga, gb + 1)
        memo[key] = val
        return val

    return rec(0, 0)


def _advantage_6_6(a_hold: float, b_hold: float) -> float:
    """P(A wins an advantage set from 6–6): win by two games, alternating serve.

    From 6–6 the next two games are one on each serve, so by the same tie argument
    as the tiebreak, ``P = a_hold(1−b_hold) / (a_hold(1−b_hold) + (1−a_hold)b_hold)``.
    """
    num = a_hold * (1.0 - b_hold)
    den = num + (1.0 - a_hold) * b_hold
    return num / den if den > 0 else 0.5


def _avg_set_prob(
    pA: float,
    pB: float,
    a_hold: float,
    b_hold: float,
    tb_target: int | None,
    adv_6_6: float,
) -> float:
    """Representative per-set P(A wins), averaged over who serves first.

    The match layer alternates who serves first from set to set (serve continuity),
    and over a match each configuration occurs roughly equally often, so we average
    the two set-win probabilities. This is the deliberate i.i.d.-sets approximation
    the analytic composition rests on (see :func:`match_win_prob_point`). The 6–6
    win prob depends on the tiebreak's first server, which is derived from the
    set's first server exactly as :func:`~sim.match.simulate_set` derives it
    (``_set_server(12, a_serves_first)``). ``tb_target is None`` selects the
    advantage-set closed form instead of a tiebreak.
    """
    total = 0.0
    for a_first in (True, False):
        if tb_target is None:
            at_6_6 = adv_6_6
        else:
            tb_first = _set_server(12, a_first)
            at_6_6 = _tiebreak_win_prob(pA, pB, tb_first, tb_target)
        total += _set_win_prob(a_hold, b_hold, a_first, at_6_6)
    return total / 2.0


def _match_win_prob_from_sets(s_std: float, s_dec: float, best_of: int) -> float:
    """Combine i.i.d. set-win probs into P(A wins the match).

    Non-deciding sets carry win prob ``s_std``; the single deciding set carries
    ``s_dec`` (it differs only through the final-set tiebreak rule). Enumerates the
    set-count paths — the deciding-set prob applies only to the path that actually
    reaches it (2–1 in Bo3, 3–2 in Bo5).
    """
    if best_of == 3:
        # 2–0: win both non-deciding sets. 2–1: split them, then win the decider.
        return s_std**2 + 2.0 * s_std * (1.0 - s_std) * s_dec
    if best_of == 5:
        s = s_std
        # 3–0 + 3–1 (decider never reached) + 3–2 (win the decider).
        return s**3 + 3.0 * s**3 * (1.0 - s) + 6.0 * s**2 * (1.0 - s) ** 2 * s_dec
    raise ValueError(f"best_of must be 3 or 5, got {best_of!r}")


def match_win_prob_point(
    pA: float, pB: float, best_of: int, final_set_rule: str
) -> float:
    """Analytic P(A beats B) from the point model — the hot path (``§2``–``§4``).

    Composes the closed-form hold (``§2``), tiebreak (``§4``) and set (``§3``)
    probabilities into a match-win probability **without simulating** and **without
    any RNG draws**: it is fully deterministic in its four inputs. This is the
    function T2.2/T2.3's ``outcome_only=True`` bracket runs call hundreds of
    thousands of times per Monte-Carlo pass; keeping it non-stochastic is a hard
    requirement, not a preference. For a stochastic cross-check use
    :func:`match_win_prob_point_mc` (never on the hot path).

    Modelling note — the composition treats sets as i.i.d. with the per-set win
    prob :func:`_avg_set_prob` (averaged over who serves first), i.e. it ignores
    the second-order correlation introduced by cross-set serve continuity. The
    error this introduces is small (serve-first is worth a fraction of a percent
    per set) and is validated against :func:`match_win_prob_point_mc` in the tests;
    the *scoreline* generator (:func:`simulate_reconciled_match`) still honours
    serve continuity exactly.

    Args:
        pA: P(A wins a point on A's serve), in ``(0, 1)``.
        pB: P(B wins a point on B's serve), in ``(0, 1)``.
        best_of: ``3`` or ``5``.
        final_set_rule: One of ``"7pt_at_6_6"``/``"10pt_at_6_6"``/``"advantage"``
            (applied to the deciding set only, via
            :data:`~sim.match.FINAL_SET_TB_TARGET`).

    Returns:
        P(A beats B) in ``[0, 1]``.

    Raises:
        ValueError: If ``best_of`` is not 3/5, ``final_set_rule`` is unknown, or a
            probability is outside ``(0, 1)`` (via :func:`~sim.match.hold_prob`).
    """
    if best_of not in (3, 5):
        raise ValueError(f"best_of must be 3 or 5, got {best_of!r}")
    if final_set_rule not in FINAL_SET_TB_TARGET:
        raise ValueError(
            f"final_set_rule must be one of {sorted(FINAL_SET_TB_TARGET)}, "
            f"got {final_set_rule!r}"
        )
    a_hold = hold_prob(pA)  # §2 (also guards pA, pB ∈ (0, 1))
    b_hold = hold_prob(pB)
    adv_6_6 = _advantage_6_6(a_hold, b_hold)

    s_std = _avg_set_prob(
        pA, pB, a_hold, b_hold, config.STANDARD_TIEBREAK_TARGET, adv_6_6
    )
    s_dec = _avg_set_prob(
        pA, pB, a_hold, b_hold, FINAL_SET_TB_TARGET[final_set_rule], adv_6_6
    )
    return _match_win_prob_from_sets(s_std, s_dec, best_of)


def match_win_prob_point_mc(
    pA: float,
    pB: float,
    best_of: int,
    final_set_rule: str,
    n_sims: int,
    rng: np.random.Generator,
) -> float:
    """MC estimate of P(A beats B) by simulating ``n_sims`` matches.

    **Cross-validation only** (T1.9): this exists to sanity-check the analytic
    :func:`match_win_prob_point` against the actual point-by-point simulator. It is
    stochastic and slow; it must **never** be what an ``outcome_only=True`` bracket
    run calls. Each simulated match draws a fresh first server from ``rng``, mirror-
    ing :func:`simulate_reconciled_match`.

    Args:
        pA, pB: The two servers' point-win probabilities.
        best_of: ``3`` or ``5``.
        final_set_rule: Deciding-set rule.
        n_sims: Number of matches to simulate.
        rng: A ``numpy`` ``Generator`` (never the global ``np.random``).

    Returns:
        The fraction of simulated matches won by A.
    """
    if best_of == 3:
        sim = simulate_match_bo3
    elif best_of == 5:
        sim = simulate_match_bo5
    else:
        raise ValueError(f"best_of must be 3 or 5, got {best_of!r}")
    wins = 0
    for _ in range(n_sims):
        first_server = int(rng.integers(2))
        result = sim(pA, pB, first_server, final_set_rule, rng)
        if result.winner == 0:
            wins += 1
    return wins / n_sims


# ---------------------------------------------------------------------------
# δ solver + reconciliation modes.
# ---------------------------------------------------------------------------


def solve_delta(
    target: float,
    pA: float,
    pB: float,
    best_of: int,
    final_set_rule: str,
    tol: float = config.RECONCILE_DELTA_TOL,
    max_iter: int = 100,
) -> float:
    """Symmetric serve shift ``δ`` with ``match_win_prob_point(pA+δ, pB−δ) ≈ target``.

    We adjust the two players' serve probabilities in opposite directions (``pA+δ``,
    ``pB−δ``) so the *simulated* match-win rate reproduces ``target`` while leaving
    the point model to shape the scoreline. Solved by bisection.

    **Monotonicity (the property bisection relies on).** Restrict ``δ`` to the
    window that keeps both ``pA+δ`` and ``pB−δ`` inside ``[P_MIN, P_MAX]``:

        ``δ ∈ [max(P_MIN−pA, pB−P_MAX), min(P_MAX−pA, pB−P_MIN)]``.

    ``δ = 0`` always lies in it because ``pA``/``pB`` come from
    ``matchup_point_probs`` already clamped to ``[P_MIN, P_MAX]``. On that window
    ``f(δ) = match_win_prob_point(pA+δ, pB−δ, …)`` is **strictly increasing**:
    ``hold_prob`` is strictly increasing in ``p``, so raising ``pA`` raises A's hold
    and lowering ``pB`` raises A's break rate, both of which raise every set-win
    prob and hence the match-win prob. Crucially, because the window keeps both
    probabilities strictly inside ``[P_MIN, P_MAX]``, the ``[P_MIN, P_MAX]`` clamp
    inside :func:`~sim.points.point_win_prob` **never fires during the solve** — so
    the clamp cannot introduce a flat region that would break strict monotonicity
    (the boundary flat regions live *outside* the window, where we saturate on
    purpose). A target beyond the window's reach saturates to the nearest endpoint.

    Args:
        target: Desired P(A beats B).
        pA, pB: Base point-win probabilities (already in ``[P_MIN, P_MAX]``).
        best_of: ``3`` or ``5``.
        final_set_rule: Deciding-set rule.
        tol: Convergence tolerance in probability units.
        max_iter: Bisection iteration cap.

    Returns:
        ``δ`` (which may be negative). ``pA+δ`` / ``pB−δ`` are guaranteed to stay in
        ``[P_MIN, P_MAX]``.
    """
    d_lo = max(config.P_MIN - pA, pB - config.P_MAX)
    d_hi = min(config.P_MAX - pA, pB - config.P_MIN)
    if d_lo >= d_hi:
        # Degenerate window (e.g. both probs already pinned to the same bound):
        # no room to move, so no adjustment is possible.
        return 0.0

    def f(delta: float) -> float:
        return match_win_prob_point(pA + delta, pB - delta, best_of, final_set_rule)

    f_lo = f(d_lo)
    f_hi = f(d_hi)
    # Saturate when the target is outside the achievable range.
    if target <= f_lo:
        return d_lo
    if target >= f_hi:
        return d_hi

    lo, hi = d_lo, d_hi
    mid = 0.5 * (lo + hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid - target) < tol:
            return mid
        if f_mid < target:  # f strictly increasing → move the low bound up
            lo = mid
        else:
            hi = mid
    return mid


def reconciled_prob(
    p_clf: float,
    p_point: float,
    mode: str = config.RECONCILE_MODE,
    w: float = config.RECONCILE_BLEND_WEIGHT,
) -> float:
    """The authoritative match-win probability under ``mode`` (``§6``).

    - ``"classifier_anchor"``: return ``p_clf`` unchanged (the well-validated
      winner model stays authoritative; the point model only shapes scorelines).
    - ``"blend"``: return ``w·p_clf + (1−w)·p_point``.

    Raises:
        ValueError: If ``mode`` is not a recognised reconciliation mode.
    """
    if mode == "classifier_anchor":
        return p_clf
    if mode == "blend":
        return w * p_clf + (1.0 - w) * p_point
    raise ValueError(
        f"RECONCILE_MODE must be 'classifier_anchor' or 'blend', got {mode!r}"
    )


def simulate_reconciled_match(
    player_a: str,
    player_b: str,
    surface: str,
    best_of: int,
    final_set_rule: str,
    skill_table: "SkillTable",
    classifier: ClassifierProb,
    rng: np.random.Generator,
    mode: str = config.RECONCILE_MODE,
    w: float = config.RECONCILE_BLEND_WEIGHT,
) -> MatchResult:
    """Simulate one reconciled match — the entry point Phase 2 calls.

    Resolves both names to ids **once** (the single name/id join), reads the
    id-keyed point-win probabilities, obtains ``P_clf`` from the name-keyed
    classifier, reconciles them per ``mode``, solves the serve shift ``δ`` that
    reproduces the reconciled probability, and simulates the actual scoreline with
    the adjusted probabilities. Player A is index ``0`` in the returned
    :class:`~sim.match.MatchResult`, so ``P(winner == 0)`` matches the reconciled
    probability over many runs.

    Args:
        player_a, player_b: Display names (name-keyed side of the seam).
        surface: One of ``config.SURFACE_MU`` (``"Hard"``/``"Clay"``/``"Grass"``).
        best_of: ``3`` or ``5``.
        final_set_rule: Deciding-set rule for :func:`~sim.match.simulate_match_bo3`
            / ``_bo5``.
        skill_table: The id-keyed serve/return :class:`~features.serve.SkillTable`.
        classifier: A :class:`ClassifierProb` returning ``P(A beats B)`` from names
            + surface.
        rng: A ``numpy`` ``Generator`` (never the global ``np.random``); also draws
            the opening server.
        mode, w: Reconciliation mode / blend weight (default from ``config``).

    Returns:
        A :class:`~sim.match.MatchResult` for the simulated match.

    Raises:
        ValueError: If either name fails to resolve (fail loudly — no silent
            fallback), the surface is unknown, or ``best_of`` is not 3/5.
    """
    # 1. Name → id, exactly once each. Fail loudly rather than let a bad name
    #    become a KeyError or a silent default-profile (wrong-player) result.
    id_a = skill_table.resolve_name(player_a)
    if id_a is None:
        raise ValueError(
            f"Could not resolve player name {player_a!r} to a player id "
            f"(unknown or ambiguous)."
        )
    id_b = skill_table.resolve_name(player_b)
    if id_b is None:
        raise ValueError(
            f"Could not resolve player name {player_b!r} to a player id "
            f"(unknown or ambiguous)."
        )

    if surface not in config.SURFACE_MU:
        raise ValueError(
            f"Unknown surface {surface!r}; expected one of {sorted(config.SURFACE_MU)}"
        )
    mu = config.SURFACE_MU[surface]

    # 2. Point model on the id side.
    skill_a = skill_table.get(id_a, surface)
    skill_b = skill_table.get(id_b, surface)
    pA, pB = matchup_point_probs(skill_a, skill_b, mu)

    # 3. Classifier on the name side.
    p_clf = float(classifier(player_a, player_b, surface))

    # 4. Reconcile → target, then solve the serve shift reproducing it.
    p_point = (
        match_win_prob_point(pA, pB, best_of, final_set_rule)
        if mode == "blend"
        else 0.0  # unused by classifier_anchor
    )
    target = reconciled_prob(p_clf, p_point, mode, w)
    delta = solve_delta(target, pA, pB, best_of, final_set_rule)

    # solve_delta keeps these inside [P_MIN, P_MAX]; clamp defensively regardless.
    pA_adj = min(max(pA + delta, config.P_MIN), config.P_MAX)
    pB_adj = min(max(pB - delta, config.P_MIN), config.P_MAX)

    # 5. Simulate the real scoreline (serve continuity honoured by the sim layer).
    first_server = int(rng.integers(2))
    if best_of == 3:
        return simulate_match_bo3(pA_adj, pB_adj, first_server, final_set_rule, rng)
    if best_of == 5:
        return simulate_match_bo5(pA_adj, pB_adj, first_server, final_set_rule, rng)
    raise ValueError(f"best_of must be 3 or 5, got {best_of!r}")


# ---------------------------------------------------------------------------
# Model pinning (ace-04-current-state.md §4).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PinnedClassifier:
    """A loaded classifier plus the metadata that pins the reproducibility risk.

    The persisted model is whichever of four estimators won on the test year, so
    its *type* can change across data refreshes (``ace-04-current-state.md §4``).
    Recording the estimator class and feature count alongside the estimator lets
    downstream code (calibration, published probabilities) note exactly what it
    used. The estimator is treated opaquely — only ``predict_proba`` is assumed.

    Attributes:
        estimator: The loaded scikit-learn/xgboost estimator (opaque).
        estimator_class: e.g. ``"RandomForestClassifier"``.
        path: The artefact path it was loaded from.
        n_features_in: The estimator's expected feature count, if it exposes one.
    """

    estimator: object
    estimator_class: str
    path: str
    n_features_in: int | None


def load_pinned_classifier(path=config.MODEL_PATH) -> PinnedClassifier:
    """Load and pin the persisted classifier (not on the hot path).

    Loads the artefact once — at wiring/calibration time, never per match — and
    records which best-of-four estimator it is. Raises ``FileNotFoundError`` if the
    artefact is missing (run ``python src/predictor.py`` to train and persist it).
    """
    import joblib

    estimator = joblib.load(path)
    return PinnedClassifier(
        estimator=estimator,
        estimator_class=type(estimator).__name__,
        path=str(path),
        n_features_in=getattr(estimator, "n_features_in_", None),
    )
