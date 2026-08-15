"""Tournament simulation CLI (T2.5).

Run a whole draw, either as a Monte Carlo title board or as one storybook
tournament played out with real scorelines::

    python src/cli/simulate_tournament.py --draw data/draws/example_usopen_2024_full.json \
        --mode montecarlo --runs 2000
    python src/cli/simulate_tournament.py --draw data/draws/example_usopen_2024_full.json \
        --mode storybook --seed 42

This module is **plumbing, not machinery**. Everything it drives already exists
and has been audited: the draw loader/validator (T2.1, ``sim/draw.py``), the
bracket simulator (T2.2), the Monte Carlo runner (T2.3) and the storybook run +
renderer (T2.4), all in ``sim/tournament.py``. The startup context — the offline
pipeline, the skill table and the ``ClassifierProb`` adapter — is
``cli/simulate_match.build_context`` (T1.10), reused verbatim so this CLI and the
single-match CLI cannot drift apart in how they build ``P_clf``. What is new
here is argument parsing, the Monte Carlo table's layout, and turning the two
failure modes a user actually hits (a missing or malformed draw file, and a draw
that validates but contains placeholder entrants) into readable messages.

**Reconciliation mode — ``config.SIM_CLI_RECONCILE_MODE`` (``"blend"``), not
``config.RECONCILE_MODE``.** This is not a fresh decision; it applies the same
value the other entry points use to a second CLI sharing the *same* adapter.
T2.5 inherited it as T1.10's seam-7 mitigation: back then
``cli/simulate_match.make_classifier_prob`` built its row with
``cli/interactive.build_feature_row``, which synthesised 20 of the 27
``config.MODEL_FEATURES``, so ``P_clf`` was form-blind and collapsed toward 0.5.
**T3.5 fixed that** — the shared ``common/classifier_adapter.py`` builds all 27
from the pipeline's own leakage-safe state (``ace-04-current-state.md §7`` seam 7,
closed) — and ``"blend"`` is **kept** as a modelling choice rather than a
workaround: it lets the point model contribute half the match-win probability,
which is the configuration T1.9/T1.9b's scoreline-realism validation ran under.
``sim/tournament.py`` still keeps ``mode`` a pass-through defaulting to the
system-wide constant and says the layer that builds the adapter owns the choice;
this module is that layer. The Monte Carlo output still carries a caveat line,
now for the narrower reason: the model state is an as-of-now snapshot, so a
simulation of an already-played draw is retrospective rather than a forecast.

**No ``--workers`` flag (deliberate, deferred).** ``sim.tournament.monte_carlo``
supports ``workers > 1``, and this CLI is the caller its docstring anticipates.
T2.5 named two blockers (``ace-04-current-state.md §7`` seam 8): an unpicklable
adapter, and a spawn-safe entry point on macOS/Windows. **The first is gone** —
T3.5's ``common.classifier_adapter.ClassifierProbAdapter`` is a module-level
class with a ``__call__`` and pickles (asserted in
``tests/test_classifier_adapter.py``), so ``_require_picklable`` would no longer
refuse. What is still owed is the rest: the spawn-safe entry point, and
re-measuring the speedup with an adapter that calls ``predict_proba`` *inside* a
worker. Both are out of scope here, so this module always calls
``monte_carlo(..., workers=1)`` and does not offer the knob. That costs little:
the measured 128 × 5,000 job is 38.5 s single-threaded, inside the
``ace-01-architecture.md`` budget, and four workers only bought ~1.3× because
each re-pays its own cache fill.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Running this file directly (`python src/cli/simulate_tournament.py`) puts
# src/cli/ on sys.path[0], not src/, so the src-relative imports below would
# fail. Add src/ explicitly — the same bootstrap scripts/validate_sim.py and
# cli/simulate_match.py use. A no-op under pytest, which already has src/ on the
# path via pyproject's pythonpath.
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import config  # noqa: E402
from cli.simulate_match import SimContext, build_context  # noqa: E402
from sim.draw import Draw, DrawValidationError, load_draw  # noqa: E402
from sim.tournament import (  # noqa: E402
    MonteCarloResult,
    StorybookResult,
    monte_carlo,
    render_storybook,
    storybook_run,
)

MODES = ("montecarlo", "storybook")

# Printed under the Monte Carlo table. Any caller publishing Monte Carlo
# probabilities must state the model's limitation alongside the numbers. Until
# T3.5 that limitation was the seam-7 defect (20 of 27 features synthetic); the
# adapter is fixed, and what this now discloses is the caveat that remains — the
# model state is an as-of-now snapshot, so an already-played draw is simulated
# with knowledge from after the event. Deliberately the same claim
# api.schemas.CLASSIFIER_LIMITATION makes to API consumers. The storybook output
# is not annotated: render_storybook already prints the reconciliation mode, and
# a single narrative bracket does not read as a published probability.
#
# Like CLASSIFIER_LIMITATION, this is **user-facing and must stay
# self-contained** — it is printed to whoever ran the command, who has no
# internal design notes to look anything up in. It carried an
# "(ace-04-current-state.md §7)" citation until the 2026-08-09 audit.
ADAPTER_CAVEAT = (
    "Not a forecast. This draw already happened, so the model is scoring a "
    "result it could have seen. These probabilities come from an as-of-now "
    "snapshot of the loaded data (rankings, surface records, head-to-heads and "
    "recent form as they stand at the end of the vendored seasons), fused with "
    f"the point model by '{config.SIM_CLI_RECONCILE_MODE}' reconciliation."
)


# --------------------------------------------------------------------------- #
# Draw loading — the file-level failure modes, rendered rather than raised.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DrawLoad:
    """A loaded draw, or ``None`` plus the message to show the user.

    Mirrors ``cli/simulate_match.NameResolution``: the CLI's error paths return
    structured results and let ``main`` do the printing, so they stay testable
    without capturing stdout.
    """

    draw: Draw | None
    message: str


def _available_draws() -> str:
    """A ``", "``-joined list of the draw files shipped under ``config.DRAWS_DIR``."""
    draws_dir = Path(config.DRAWS_DIR)
    if not draws_dir.is_dir():
        return ""
    return ", ".join(sorted(str(p) for p in draws_dir.glob("*.json")))


def load_tournament_draw(path: str | Path, skill_table) -> DrawLoad:
    """Load and validate a draw file, turning both failure modes into text.

    Reuses ``sim.draw.load_draw`` (T2.1) unchanged — no second validator — and
    renders its :class:`~sim.draw.DrawValidationError` as the accumulated
    problem list it already collected, one bullet per problem, so a
    hand-entered draw can be fixed in one pass. A missing file is reported with
    the draws directory's actual contents rather than a traceback.

    Args:
        path: Path to the draw JSON file.
        skill_table: T1.1 skill table used to resolve entrant names to ids.

    Returns:
        A :class:`DrawLoad` — ``draw`` set on success, otherwise ``message``.
    """
    try:
        return DrawLoad(load_draw(path, skill_table), "")
    except FileNotFoundError:
        available = _available_draws()
        hint = f" Available draws: {available}" if available else ""
        return DrawLoad(
            None,
            f"Draw file not found: {path}\n"
            f"   Draw files are JSON, conventionally under {config.DRAWS_DIR}/.{hint}",
        )
    except IsADirectoryError:
        return DrawLoad(
            None, f"{path} is a directory, not a draw file."
        )
    except DrawValidationError as exc:
        problems = "\n".join(f"     - {problem}" for problem in exc.problems)
        return DrawLoad(
            None,
            f"Invalid draw file: {exc.source}\n"
            f"   {len(exc.problems)} problem(s) found:\n{problems}",
        )


# --------------------------------------------------------------------------- #
# Running — thin wrappers over sim/tournament.py.
# --------------------------------------------------------------------------- #
def run_monte_carlo(
    draw: Draw,
    ctx: SimContext,
    n_runs: int = config.MC_RUNS,
    seed: int = config.MC_SEED,
    reconcile_mode: str = config.SIM_CLI_RECONCILE_MODE,
) -> MonteCarloResult:
    """Run the Monte Carlo job for ``draw`` using the CLI's startup context.

    A pass-through to :func:`sim.tournament.monte_carlo` — the aggregation, the
    per-run seeding and the whole-job ``prob_cache`` all live there. The two
    things this wrapper decides are the ones the calling layer owns: the
    reconciliation mode (see the module docstring) and ``workers=1``.

    Args:
        draw: A validated, placeholder-free draw.
        ctx: The startup context, built once (skill table + classifier adapter).
        n_runs: Number of bracket simulations.
        seed: Base seed; run *i*'s RNG is spawned from it.
        reconcile_mode: ``"blend"``/``"classifier_anchor"``.

    Returns:
        A :class:`~sim.tournament.MonteCarloResult`, players sorted by title
        probability descending.

    Raises:
        ValueError: If the draw contains placeholder entrants, or ``n_runs < 1``.
    """
    return monte_carlo(
        draw,
        ctx.skill_table,
        ctx.classifier,
        n_runs=n_runs,
        seed=seed,
        # Single-process only, on purpose — see the module docstring and
        # ace-04-current-state.md §7 seam 8. T3.5's shared adapter *is* picklable,
        # so this is no longer a refusal we would hit anyway; exposing workers > 1
        # still owns the spawn-safe entry point and its own re-measurement.
        workers=1,
        mode=reconcile_mode,
    )


def run_storybook(
    draw: Draw,
    ctx: SimContext,
    seed: int = config.MC_SEED,
    reconcile_mode: str = config.SIM_CLI_RECONCILE_MODE,
) -> StorybookResult:
    """Play one full tournament with scorelines, using the CLI's context.

    A pass-through to :func:`sim.tournament.storybook_run`; render the result
    with :func:`sim.tournament.render_storybook`. Deterministic given ``seed``.

    Raises:
        ValueError: If the draw contains placeholder entrants.
    """
    return storybook_run(
        draw,
        ctx.skill_table,
        ctx.classifier,
        seed=seed,
        mode=reconcile_mode,
    )


# --------------------------------------------------------------------------- #
# Monte Carlo table rendering.
# --------------------------------------------------------------------------- #
# Column widths. The player column is the only elastic one in spirit, but a
# fixed width keeps every row aligned without a second pass over the data.
_PLAYER_WIDTH = 24
_PCT_WIDTH = 7


def _reconciliation_label(mode: str, w: float) -> str:
    """``"blend (w=0.5)"`` / ``"classifier_anchor"`` — as render_storybook writes it."""
    return f"{mode} (w={w:g})" if mode == "blend" else mode


def _truncate(name: str, width: int = _PLAYER_WIDTH) -> str:
    """Clip a display name to ``width``, marking the clip with ``…``."""
    return name if len(name) <= width else name[: width - 1] + "…"


def format_montecarlo_table(
    result: MonteCarloResult,
    top_n: int = config.SIM_TOURNAMENT_CLI_TOP_N,
) -> str:
    """Render a Monte Carlo aggregate as a sorted title-probability table.

    Layout (one header block, the table, one footer):

    * ``#`` — rank in title-probability order (``result.players`` is already
      sorted, ties broken by bracket position, so the rank is deterministic).
    * ``Seed`` — the entrant's tournament seed as ``[1]``, blank if unseeded.
    * ``Player`` — display name, clipped to 24 characters.
    * One percentage column per round **after the first**, headed with the
      draw's own label from ``result.round_labels``: P(the entrant contests that
      round). The first round is omitted because every entrant contests it by
      construction — the column would read ``100.0%`` for all of them. An
      8-draw therefore shows ``SF``/``F``; a 128-draw ``R64 … F``.
    * ``Title`` — P(wins the tournament); the sort key.
    * ``E[W]`` — mean matches won per run (``expected_rounds_won``), the one
      column that summarises a run in a single number.

    Percentages are shown to one decimal place; ``E[W]`` to two. Nothing is
    computed here that :class:`~sim.tournament.PlayerOutcome` does not already
    expose.

    Args:
        result: The completed Monte Carlo aggregate.
        top_n: How many entrants to list. Values ``< 1`` or beyond the field
            size list everyone.

    Returns:
        The rendered table, newline-separated and without a trailing newline.
    """
    survival_labels = result.round_labels[1:]  # round 1 is 100% for everyone
    shown = (
        result.players
        if top_n < 1 or top_n >= len(result.players)
        else result.players[:top_n]
    )

    lines = [
        f"{result.name} — {result.surface}, best of {result.best_of}, "
        f"{result.draw_size} entrants",
        f"{result.n_runs:,} runs · seed {result.seed} · reconciliation "
        f"{_reconciliation_label(result.mode, result.w)} · workers {result.workers}",
        "",
    ]

    header = (
        f"{'#':>3}  {'Seed':>4}  {'Player':<{_PLAYER_WIDTH}}"
        + "".join(f"{label:>{_PCT_WIDTH}}" for label in survival_labels)
        + f"{'Title':>{_PCT_WIDTH}}{'E[W]':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for rank, player in enumerate(shown, start=1):
        seed = f"[{player.seed}]" if player.seed is not None else ""
        lines.append(
            f"{rank:>3}  {seed:>4}  {_truncate(player.player):<{_PLAYER_WIDTH}}"
            + "".join(
                f"{player.p_reach(label):>{_PCT_WIDTH}.1%}"
                for label in survival_labels
            )
            + f"{player.p_title:>{_PCT_WIDTH}.1%}"
            + f"{player.expected_rounds_won:>7.2f}"
        )

    lines.append("")
    if len(shown) < len(result.players):
        lines.append(
            f"Showing the top {len(shown)} of {len(result.players)} entrants."
        )
    lines.append(ADAPTER_CAVEAT)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="simulate_tournament",
        description="Simulate a tournament draw: Monte Carlo probabilities or "
                    "one storybook run with full scorelines.",
    )
    parser.add_argument(
        "--draw",
        required=True,
        help=f"Path to a draw JSON file (conventionally under {config.DRAWS_DIR}/).",
    )
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="montecarlo",
        help="'montecarlo' prints a title-probability table; 'storybook' plays "
             "one bracket out with real scorelines (default: montecarlo).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,  # sentinel: lets storybook mode notice a pointless --runs
        help=f"Monte Carlo runs (montecarlo mode only; default: {config.MC_RUNS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.MC_SEED,
        help=f"Base seed — the same seed replays the same tournament "
             f"(default: {config.MC_SEED}).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=config.SIM_TOURNAMENT_CLI_TOP_N,
        help=f"Entrants to list in montecarlo mode; 0 lists all "
             f"(default: {config.SIM_TOURNAMENT_CLI_TOP_N}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)

    if args.mode == "storybook" and args.runs is not None:
        print("--runs is ignored in storybook mode (it plays exactly one bracket).")
    n_runs = config.MC_RUNS if args.runs is None else args.runs

    # Fail on an obviously bad path *before* paying for the pipeline: build_context
    # loads 13 seasons, engineers features and unpickles the model.
    if not Path(args.draw).exists():
        print(load_tournament_draw(args.draw, None).message)
        return 1

    ctx = build_context()

    loaded = load_tournament_draw(args.draw, ctx.skill_table)
    if loaded.draw is None:
        print(loaded.message)
        return 1

    try:
        if args.mode == "montecarlo":
            result = run_monte_carlo(loaded.draw, ctx, n_runs=n_runs, seed=args.seed)
            print(format_montecarlo_table(result, args.top))
        else:
            print(render_storybook(run_storybook(loaded.draw, ctx, seed=args.seed)))
    except ValueError as exc:
        # Chiefly the placeholder-entrant refusal (T2.2), which names every
        # offending slot — worth showing in full, not as a traceback.
        print(f"{exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
