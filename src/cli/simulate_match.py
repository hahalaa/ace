"""Single-match simulation CLI (T1.10).

Simulate one named matchup end-to-end and print a real scoreline plus a Monte
Carlo win probability::

    python src/cli/simulate_match.py "Carlos Alcaraz" "Jannik Sinner" \
        --surface Clay --best-of 5 --seed 7

This is the first entry point that drives the whole Phase 1 stack together:
skill table (T1.1) → point-win model (T1.2) → point-by-point scoring engine
(T1.3–T1.7) → reconciliation with the classifier (T1.8). It is a manual sanity
tool and a demo, and it is also where the **ClassifierProb adapter** T1.8
deliberately deferred finally gets built (see below).

**The ClassifierProb adapter (the real work here).** ``sim/reconcile.py`` takes
its classifier as an injected callable ``(player_a, player_b, surface) -> P_clf``
rather than the raw estimator, because ``predict_proba`` needs a *name-keyed*
feature row assembled from the pipeline's ``surface_history`` / ``h2h_history``
and latest rank/age — state owned by the classifier/CLI layer, which ``sim/``
must never import (the layering rule in ``CLAUDE.md``). T1.8's audit recorded
that wiring as "owed by predictor.py/T1.10"; :func:`make_classifier_prob` is it.
The dependency inversion runs one way only: **this module imports from ``sim/``
and constructs the adapter; ``sim/reconcile.py`` never reaches back into
``cli/``** — it only receives the callable as an argument.

The adapter reuses the REPL's helpers verbatim (``get_latest``,
``get_surf_record``, ``compute_h2h``, ``build_feature_row`` in
``cli/interactive.py``) so the feature row it feeds ``predict_proba`` is built
exactly the way the interactive predictor builds it — the leakage-safe history
structures come from ``features.engineering.add_features``, which is run **once**
at startup by :func:`build_context`, not per simulated match. The adapter also
memoises per ``(player_a, player_b, surface)``: the feature row is constant for a
matchup, so the ~1,000 MC runs cost exactly one ``predict_proba`` call.

**Reconciliation mode.** This CLI defaults to ``config.SIM_CLI_RECONCILE_MODE``
(``"blend"``), *not* the system-wide ``config.RECONCILE_MODE``. The adapter's
feature row synthesises the 20 recent-form columns, which flattens ``P_clf``
toward 0.5 (``ace-04-current-state.md §7`` seam 7); under
``"classifier_anchor"`` that flattened value would be the target every simulated
scoreline is solved to reproduce. ``--reconcile-mode classifier_anchor`` still
opts in explicitly, with a printed caveat. This is a mitigation of the default,
not a fix — the underlying rolling-feature gap is untouched.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Running this file directly (`python src/cli/simulate_match.py`) puts src/cli/
# on sys.path[0], not src/, so the src-relative imports below would fail. Add
# src/ explicitly — same bootstrap scripts/validate_sim.py uses. A no-op under
# pytest, which already has src/ on the path via pyproject's pythonpath.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import cli.interactive as interactive  # noqa: E402
import config  # noqa: E402
import data.loader as loader  # noqa: E402
import data.preprocess as preprocess  # noqa: E402
import features.engineering as engineering  # noqa: E402
from common.names import NameIndex, resolve_name  # noqa: E402
from features.serve import SkillTable  # noqa: E402
from sim.match import MatchResult, SetResult  # noqa: E402
from sim.reconcile import (  # noqa: E402
    ClassifierProb,
    load_pinned_classifier,
    simulate_reconciled_match,
)


# --------------------------------------------------------------------------- #
# Matchup stats — the one place the REPL's history helpers are read.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MatchupStats:
    """Name-keyed classifier inputs for one matchup, plus display extras.

    Exactly the quantities ``cli/interactive.py``'s REPL gathers before calling
    ``build_feature_row``/``display_matchup``, collected once so the adapter and
    the printed matchup table cannot drift apart.
    """

    a_rank: float
    b_rank: float
    a_age: float
    b_age: float
    a_pct: float
    b_pct: float
    a_wins: int
    a_total: int
    b_wins: int
    b_total: int
    h2h_diff: int
    h2h_msg: str


def matchup_stats(
    player_a: str,
    player_b: str,
    surface: str,
    data: pd.DataFrame,
    surface_history: dict,
    h2h_history: dict,
) -> MatchupStats:
    """Gather the classifier's name-keyed inputs for one matchup.

    Reuses the REPL's helpers rather than reimplementing them: ``get_latest``
    (latest rank/age), ``get_surf_record`` (leakage-safe surface record built by
    ``features.engineering.add_features``) and ``compute_h2h``.

    Raises:
        ValueError: If either player has no match history in ``data`` — the
            classifier cannot be given a feature row for an unknown player, and
            failing here is better than silently predicting off defaults.
    """
    a_rank, a_age = interactive.get_latest(player_a, data)
    b_rank, b_age = interactive.get_latest(player_b, data)
    if a_rank is None or b_rank is None:
        missing = player_a if a_rank is None else player_b
        raise ValueError(
            f"No match history for {missing!r}; cannot build a classifier feature row."
        )

    a_wins, a_total = interactive.get_surf_record(player_a, surface, surface_history)
    b_wins, b_total = interactive.get_surf_record(player_b, surface, surface_history)
    a_pct = a_wins / a_total if a_total > 0 else config.DEFAULT_WIN_PCT
    b_pct = b_wins / b_total if b_total > 0 else config.DEFAULT_WIN_PCT

    h2h_diff, h2h_msg = interactive.compute_h2h(player_a, player_b, h2h_history)

    return MatchupStats(
        a_rank=a_rank,
        b_rank=b_rank,
        a_age=a_age,
        b_age=b_age,
        a_pct=a_pct,
        b_pct=b_pct,
        a_wins=a_wins,
        a_total=a_total,
        b_wins=b_wins,
        b_total=b_total,
        h2h_diff=h2h_diff,
        h2h_msg=h2h_msg,
    )


def make_classifier_prob(
    estimator,
    data: pd.DataFrame,
    surface_history: dict,
    h2h_history: dict,
) -> ClassifierProb:
    """Wrap the pinned classifier as the ``ClassifierProb`` adapter T1.8 needs.

    Returns a callable ``(player_a, player_b, surface) -> P(A beats B)``: it
    builds the name-keyed feature row with :func:`matchup_stats` +
    ``interactive.build_feature_row`` (the same row the REPL predicts on) and
    calls ``estimator.predict_proba``. The histories are captured by closure and
    built once by :func:`build_context`, never per call.

    Results are memoised per ``(player_a, player_b, surface)``. The feature row
    is constant for a matchup, so a 1,000-run Monte Carlo pass makes exactly one
    ``predict_proba`` call.

    Args:
        estimator: The opaque persisted estimator — only ``predict_proba`` is
            assumed (it is whichever of four classifiers won on the test year;
            see ``ace-04-current-state.md §4``).
        data: The engineered match frame (for latest rank/age).
        surface_history: Name-keyed surface records from ``add_features``.
        h2h_history: Name-keyed H2H records from ``add_features``.
    """
    cache: dict[tuple[str, str, str], float] = {}

    def classifier_prob(player_a: str, player_b: str, surface: str) -> float:
        key = (player_a, player_b, surface)
        cached = cache.get(key)
        if cached is not None:
            return cached
        stats = matchup_stats(
            player_a, player_b, surface, data, surface_history, h2h_history
        )
        row = interactive.build_feature_row(
            stats.a_rank,
            stats.b_rank,
            stats.a_age,
            stats.b_age,
            stats.a_pct,
            stats.b_pct,
            stats.h2h_diff,
        )
        prob = float(estimator.predict_proba(row)[0][1])  # P(player A wins)
        cache[key] = prob
        return prob

    return classifier_prob


# --------------------------------------------------------------------------- #
# Startup context — everything expensive, paid once.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SimContext:
    """Everything the simulator needs, loaded once at CLI startup.

    Building this runs the full offline pipeline (load → preprocess → engineer
    features → build the skill table) and loads the pinned classifier. It is
    deliberately *not* per-match work: the adapter and the skill table are then
    reused across the storybook match and every Monte Carlo run.
    """

    data: pd.DataFrame
    surface_history: dict
    h2h_history: dict
    skill_table: SkillTable
    classifier: ClassifierProb
    estimator_class: str
    player_names: list[str]

    @property
    def name_index(self) -> NameIndex:
        """Resolvable index of every display name in the data (T0.6)."""
        return NameIndex.from_names(self.player_names)


def build_context(
    start_year: int = config.START_YEAR,
    end_year: int = config.END_YEAR,
    model_path=config.MODEL_PATH,
) -> SimContext:
    """Run the offline pipeline once and wire up the classifier adapter.

    Raises:
        FileNotFoundError: If a vendored data year or the persisted model is
            missing (run ``scripts/refresh_data.py`` / ``python src/predictor.py``).
    """
    raw = loader.load_atp_data(start_year, end_year)
    processed = preprocess.preprocess_data(raw)
    data, surface_history, h2h_history, skill_table = engineering.add_features(processed)

    pinned = load_pinned_classifier(model_path)
    classifier = make_classifier_prob(
        pinned.estimator, data, surface_history, h2h_history
    )

    names = pd.concat([data["p1_name"], data["p2_name"]]).unique().tolist()

    return SimContext(
        data=data,
        surface_history=surface_history,
        h2h_history=h2h_history,
        skill_table=skill_table,
        classifier=classifier,
        estimator_class=pinned.estimator_class,
        player_names=names,
    )


# --------------------------------------------------------------------------- #
# Name resolution (reuses the shared T0.6 resolver + the REPL's messages).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NameResolution:
    """A resolved display name, or ``None`` plus the message to show the user.

    ``message`` carries the shared resolver's candidate list when the query was
    ambiguous — rendered by ``interactive.ambiguity_message``, so the suggestion
    text is byte-identical to the REPL's.
    """

    name: str | None
    message: str


def resolve_display_name(query: str, index: NameIndex) -> NameResolution:
    """Resolve a user-typed name to a canonical display name.

    Delegates matching to the shared, UI-free resolver (``common/names.py``,
    T0.6) — no new matching logic here — and turns its structured result into
    something printable.

    Note the canonical *name* is what the caller wants: ``sim/reconcile.py``
    performs the authoritative name→``player_id`` join itself (via the skill
    table), and the classifier adapter is name-keyed. This pass exists to give
    the user candidates and deduction feedback before that join happens.
    """
    match = resolve_name(query, index)
    if match is None:
        return NameResolution(
            None,
            f"❌ Player '{query}' not found. "
            f"Try a fuller name (e.g. 'Carlos Alcaraz') or a surname.",
        )
    if match.is_ambiguous:
        return NameResolution(None, interactive.ambiguity_message(query, match))
    note = "" if match.name == query else f"   -> Deduced: {match.name}"
    return NameResolution(match.name, note)


# --------------------------------------------------------------------------- #
# Simulation + rendering.
# --------------------------------------------------------------------------- #
# Shown only in "classifier_anchor" mode, where P_clf is the authoritative
# target δ reproduces — so the adapter's synthetic rolling features drive the
# scoreline directly (ace-04-current-state.md §7 seam 7). "blend" already
# dilutes that, so it is not warned about.
ANCHOR_MODE_CAVEAT = (
    "⚠️  classifier_anchor: this tool's rolling-form features are "
    "synthetic-filled, so P_clf is less informative here than in blend mode."
)


def format_set(s: SetResult) -> str:
    """Render one set from A's perspective, e.g. ``6-4`` or ``7-6(5)``."""
    line = f"{s.games_a}-{s.games_b}"
    if s.tb_score is not None:
        line += f"({min(s.tb_score)})"  # the loser's tiebreak points
    return line


def format_scoreline(result: MatchResult) -> str:
    """Render a full match scoreline from A's perspective, e.g. ``6-4 3-6 7-6(5) 6-2``."""
    return " ".join(format_set(s) for s in result.sets)


@dataclass(frozen=True)
class MatchSimulation:
    """One storybook match plus the Monte Carlo win probability behind it."""

    player_a: str
    player_b: str
    surface: str
    best_of: int
    final_set_rule: str
    result: MatchResult
    scoreline: str
    winner: str
    win_prob_a: float
    n_sims: int
    p_clf: float
    stats: MatchupStats
    reconcile_mode: str = config.SIM_CLI_RECONCILE_MODE


def simulate_named_match(
    player_a: str,
    player_b: str,
    surface: str,
    ctx: SimContext,
    best_of: int = config.SIM_CLI_BEST_OF,
    final_set_rule: str = config.SIM_CLI_FINAL_SET_RULE,
    n_sims: int = config.SIM_CLI_MC_RUNS,
    seed: int | None = None,
    reconcile_mode: str = config.SIM_CLI_RECONCILE_MODE,
) -> MatchSimulation:
    """Simulate one reconciled match and estimate the win probability by MC.

    The single seeded ``Generator`` is threaded through the storybook match and
    every Monte Carlo run, so a given ``seed`` reproduces **both** the scoreline
    and the MC estimate exactly.

    Args:
        player_a, player_b: Canonical display names (already resolved).
        surface: ``"Hard"``/``"Clay"``/``"Grass"``.
        ctx: The startup context (skill table + classifier adapter + histories).
        best_of: ``3`` or ``5``.
        final_set_rule: ``"7pt_at_6_6"``/``"10pt_at_6_6"``/``"advantage"``.
        n_sims: Monte Carlo runs behind the reported win %.
        seed: Seed for ``numpy.random.default_rng``.
        reconcile_mode: ``"blend"``/``"classifier_anchor"``. Defaults to
            ``config.SIM_CLI_RECONCILE_MODE`` (``"blend"``), **not** the
            system-wide ``config.RECONCILE_MODE``: this CLI's ``P_clf`` is
            form-blind and flattened toward 0.5 (``ace-04-current-state.md §7``
            seam 7), and anchoring on it would push every scoreline toward a
            coin flip. Blending dilutes that; it does not fix it.

    Returns:
        A :class:`MatchSimulation`.

    Raises:
        ValueError: If a name does not resolve to a ``player_id`` in the skill
            table, the surface is unknown, or ``best_of`` is not 3/5 (raised by
            ``sim.reconcile.simulate_reconciled_match``).
    """
    rng = np.random.default_rng(seed)

    def run() -> MatchResult:
        return simulate_reconciled_match(
            player_a,
            player_b,
            surface,
            best_of,
            final_set_rule,
            ctx.skill_table,
            ctx.classifier,
            rng,
            mode=reconcile_mode,
        )

    result = run()  # the storybook match — one full scoreline
    wins_a = sum(1 for _ in range(n_sims) if run().winner == 0)

    return MatchSimulation(
        player_a=player_a,
        player_b=player_b,
        surface=surface,
        best_of=best_of,
        final_set_rule=final_set_rule,
        result=result,
        scoreline=format_scoreline(result),
        winner=player_a if result.winner == 0 else player_b,
        win_prob_a=wins_a / n_sims if n_sims > 0 else float("nan"),
        n_sims=n_sims,
        p_clf=ctx.classifier(player_a, player_b, surface),
        stats=matchup_stats(
            player_a, player_b, surface, ctx.data, ctx.surface_history, ctx.h2h_history
        ),
        reconcile_mode=reconcile_mode,
    )


def print_simulation(sim: MatchSimulation) -> None:
    """Print the matchup table (reusing the REPL's ``display_matchup``) + result."""
    s = sim.stats
    interactive.display_matchup(
        sim.player_a,
        sim.player_b,
        sim.surface,
        s.a_rank,
        s.b_rank,
        s.a_age,
        s.b_age,
        s.a_pct,
        s.b_pct,
        s.a_wins,
        s.a_total,
        s.b_wins,
        s.b_total,
        s.h2h_msg,
        sim.win_prob_a,
    )
    print(f"🎾 SIMULATED MATCH (best of {sim.best_of}, {sim.final_set_rule})")
    print(f"   {sim.winner} def. "
          f"{sim.player_b if sim.winner == sim.player_a else sim.player_a}  "
          f"{sim.scoreline}")
    print(
        f"   {sim.player_a} wins {sim.win_prob_a:.1%} of {sim.n_sims:,} simulations "
        f"(classifier: {sim.p_clf:.1%}, mode: {sim.reconcile_mode})"
    )
    if sim.reconcile_mode == "classifier_anchor":
        print(f"   {ANCHOR_MODE_CAVEAT}")
    print("-" * 60 + "\n")


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="simulate_match",
        description="Simulate a single tennis matchup and print the scoreline.",
    )
    parser.add_argument("player_a", nargs="?", help="First player's name.")
    parser.add_argument("player_b", nargs="?", help="Second player's name.")
    parser.add_argument(
        "--surface",
        default=config.SIM_CLI_SURFACE,
        help=f"Court surface (default: {config.SIM_CLI_SURFACE}).",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        choices=(3, 5),
        default=config.SIM_CLI_BEST_OF,
        help=f"Match format (default: {config.SIM_CLI_BEST_OF}).",
    )
    parser.add_argument(
        "--final-set-rule",
        # Choices come from the canonical enum (T2.1) so they cannot drift from
        # what sim/match.py accepts.
        choices=sorted(config.FINAL_SET_TB_TARGET),
        default=config.SIM_CLI_FINAL_SET_RULE,
        help=f"Deciding-set rule (default: {config.SIM_CLI_FINAL_SET_RULE}).",
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=config.SIM_CLI_MC_RUNS,
        help=f"Monte Carlo runs behind the win %% (default: {config.SIM_CLI_MC_RUNS}).",
    )
    parser.add_argument(
        "--reconcile-mode",
        choices=("blend", "classifier_anchor"),
        default=config.SIM_CLI_RECONCILE_MODE,
        help=f"How P_clf and the point model are fused (default: "
             f"{config.SIM_CLI_RECONCILE_MODE}). This CLI defaults to 'blend' "
             f"rather than config.RECONCILE_MODE because its P_clf is built "
             f"from a partly synthetic feature row; 'classifier_anchor' is "
             f"available for comparison and prints a caveat.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducibility — the same seed replays the same "
             "scoreline and MC estimate.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)

    surface = interactive.validate_surface(args.surface)
    if surface is None:
        print(f"❌ Invalid surface '{args.surface}'. Choose Hard, Clay, or Grass.")
        return 2

    ctx = build_context()
    index = ctx.name_index

    resolved: list[str] = []
    for label, supplied in (("Player A", args.player_a), ("Player B", args.player_b)):
        if supplied:
            query = supplied
        else:
            try:
                query = input(f"Enter {label}: ").strip()
            except EOFError:
                # Closed/empty stdin — leave cleanly rather than crash (the
                # pre-existing REPL defect in ace-04-current-state.md §8).
                print("\n👋 Input closed. Exiting.")
                return 1
        resolution = resolve_display_name(query, index)
        if resolution.name is None:
            print(resolution.message)
            return 1
        if resolution.message:
            print(resolution.message)
        resolved.append(resolution.name)

    try:
        sim = simulate_named_match(
            resolved[0],
            resolved[1],
            surface,
            ctx,
            best_of=args.best_of,
            final_set_rule=args.final_set_rule,
            n_sims=args.sims,
            seed=args.seed,
            reconcile_mode=args.reconcile_mode,
        )
    except ValueError as exc:
        print(f"❌ {exc}")
        return 1

    print_simulation(sim)
    return 0


if __name__ == "__main__":
    sys.exit(main())
