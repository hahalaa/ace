"""Offline Monte Carlo precompute (T3.3) — fills the cache the API serves.

A full 128 × 5,000 job takes ~40 s, and Phase 3's global rules forbid running
one inside a request handler. So the expensive half happens here, offline, and
``GET /tournaments/{id}/simulate`` only ever reads the file this script writes::

    python scripts/precompute_sim.py --draw usopen_2024_atp_full --runs 5000 --seed 0

``--draw`` takes a **tournament id**, the same id ``GET /tournaments`` lists and
``/tournaments/{id}/bracket`` accepts, resolved through ``api/registry.py`` — the
module kept free of FastAPI imports precisely so an offline script could use it
without importing the web app. The output goes to
``data/cache/<tournament_id>.json`` (``api.registry.cache_path_for``, the one
definition of that mapping).

**Everything expensive is paid once**, in the same order the CLIs and the API pay
it: ``api.deps.build_api_context`` runs the offline pipeline (load → preprocess →
engineer features → skill table) and pins the classifier (~2 s), then the draw
registry is built from the skill table that produced. Nothing inside the run loop
reloads anything; ``sim.tournament.monte_carlo`` owns the per-job ``prob_cache``.

--------------------------------------------------------------------------------
The classifier adapter, and why this script discloses rather than fixes (seam 7)
--------------------------------------------------------------------------------
``api/deps.py`` deliberately builds **no** ``ClassifierProb`` adapter: the only
one that exists lives in ``cli/``, which ``api/`` may not import, and
``ace-04-current-state.md §7`` seam 6 requires each new caller to assess seam 7
for itself. This script is that caller, and it is not in ``api/`` — it is an
outermost-layer entry point like ``cli/`` and ``scripts/`` generally, so it may
import ``cli.simulate_match.make_classifier_prob`` and does. The API package
still imports nothing from ``cli/``; the adapter is constructed *here* and its
output reaches the API only as numbers in a JSON file.

That adapter carries a known defect. ``cli/interactive.build_feature_row``
populates 7 of the 27 ``config.MODEL_FEATURES`` from real state and fills the
other 20 — every recent-form column — with synthetic constants, so ``P_clf`` is
form-blind and collapses toward 0.5 (measured: 0.5091 where a real engineered row
gives 0.3800). **The ticket's requirement is to either build an adapter with
populated rolling features or state the limitation alongside the numbers. This
script does the latter, deliberately:**

* Building a real-feature adapter means computing each matchup's latest rolling
  state (the 20 columns ``features/rolling.py`` derives per player per match)
  and re-validating the resulting probabilities against T1.8's calibration — a
  substantial piece of modelling work in its own right, comparable to T1.10's
  whole adapter build, and beyond a "Size: M" endpoint ticket. Attempting it
  half-way would be worse than disclosing: a partially-populated row is
  *silently* out-of-distribution, whereas a documented one is not.
* The defect belongs to the adapter, not to this ticket, and it is tracked as an
  open seam with an owner decision pending. Publishing under disclosure keeps the
  numbers honest without pre-empting that decision.

Disclosure is therefore structural, not a comment: ``mode``/``w``, an
``is_forecast`` flag and a plain-language ``classifier_limitation`` are
**required** fields of ``api.schemas.SimulationMetadata``, so they are written
into every cache file and served in every API response, and a cache file lacking
them fails validation instead of publishing bare probabilities.

The mode is ``config.SIM_CLI_RECONCILE_MODE`` (``"blend"``) for the same reason
both CLIs use it: under ``"classifier_anchor"`` the flattened ``P_clf`` *is* the
target δ is solved to reproduce, so it would drive every match of every bracket.
Blending dilutes that; it does not fix it — hence the disclosure above.
``--reconcile-mode`` overrides it, and whatever is used is recorded in the file.

**Placeholder draws are refused, by T2.2's own check.** ``monte_carlo`` rejects a
draw containing ``Qualifier``/``Bye`` slots and names every offending slot; this
script prints that message and exits non-zero rather than writing a cache file.
The API's matching behaviour is a 409 — see ``api/main.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# scripts/ is not on pyproject's ``pythonpath = ["src"]``; add src/ so this
# script imports project modules the way the runtime pipeline does. Same
# bootstrap as scripts/validate_sim.py and the two src/cli/ entry points.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import config  # noqa: E402
from api.deps import ApiContext, build_api_context  # noqa: E402
from api.registry import TournamentRegistry, build_registry, cache_path_for  # noqa: E402
from api.schemas import (  # noqa: E402
    SimulationMetadata,
    SimulationPlayer,
    SimulationResponse,
)
from cli.simulate_match import make_classifier_prob  # noqa: E402
from sim.tournament import MonteCarloResult, monte_carlo  # noqa: E402

# The adapter this script builds, named in every cache file's metadata so a
# consumer can tell which model produced the numbers (see the module docstring).
ADAPTER = "cli.simulate_match.make_classifier_prob"

# The seam-7 disclosure, written verbatim into every cache file and served with
# every /simulate response. Deliberately plain-language and self-contained: an
# API consumer will not have read ace-04-current-state.md.
CLASSIFIER_LIMITATION = (
    "Demonstration, not a forecast. The classifier behind these probabilities is "
    "fed a feature row in which 20 of its 27 features — every recent-form "
    "feature — are synthetic constants rather than real form, which pushes its "
    "match-win probability toward 0.5 (ace-04-current-state.md §7 seam 7). The "
    "reconciliation mode in this metadata dilutes that effect but does not "
    "remove it. Treat these numbers as a demonstration of the simulation "
    "machinery, not as a prediction of the event."
)


# --------------------------------------------------------------------------- #
# Cache payload — the schema in api/schemas.py is the format.
# --------------------------------------------------------------------------- #
def build_payload(
    result: MonteCarloResult,
    *,
    data_through_year: int,
    estimator_class: str,
    source: str,
    generated_at: datetime | None = None,
    adapter: str = ADAPTER,
    classifier_limitation: str = CLASSIFIER_LIMITATION,
    is_forecast: bool = False,
) -> SimulationResponse:
    """Turn a finished Monte Carlo run into the cache/wire payload.

    Presentation over already-built data: every number comes from
    :class:`~sim.tournament.MonteCarloResult` and its
    :class:`~sim.tournament.PlayerOutcome`\\ s (counts, ``p_title``,
    ``p_reach``, ``expected_rounds_won``, the draw's own ``round_labels``) — the
    same renderer/data separation T2.4 and T2.5 keep. Nothing is recomputed and
    no new statistic is invented here.

    Args:
        result: The completed aggregate, players already sorted by title
            probability.
        data_through_year: Latest season in the data behind the model.
        estimator_class: Class name of the pinned classifier, for provenance.
        source: Draw file basename the run was against.
        generated_at: Write timestamp; defaults to now (UTC).
        adapter: Which ``ClassifierProb`` adapter produced ``P_clf``.
        classifier_limitation: The seam-7 disclosure text.
        is_forecast: Whether these numbers may be presented as a prediction.
            ``False`` while the adapter carries the seam-7 defect.

    Returns:
        A validated :class:`~api.schemas.SimulationResponse`.
    """
    return SimulationResponse(
        tournament_id=result.tournament_id,
        name=result.name,
        surface=result.surface,
        best_of=result.best_of,
        final_set_tiebreak=result.final_set_tiebreak,
        draw_size=result.draw_size,
        round_labels=list(result.round_labels),
        count=len(result.players),
        players=[
            SimulationPlayer(
                position=outcome.position,
                player=outcome.player,
                player_id=outcome.player_id,
                seed=outcome.seed,
                titles=outcome.titles,
                matches_won=outcome.matches_won,
                p_title=outcome.p_title,
                expected_rounds_won=outcome.expected_rounds_won,
                reached=dict(outcome.reached),
                p_reach={
                    label: outcome.p_reach(label) for label in result.round_labels
                },
            )
            for outcome in result.players
        ],
        metadata=SimulationMetadata(
            runs=result.n_runs,
            seed=result.seed,
            workers=result.workers,
            mode=result.mode,
            w=result.w,
            generated_at=generated_at or datetime.now(timezone.utc),
            data_through_year=data_through_year,
            estimator_class=estimator_class,
            adapter=adapter,
            is_forecast=is_forecast,
            classifier_limitation=classifier_limitation,
            source=source,
        ),
    )


def write_cache(
    payload: SimulationResponse, cache_dir: Path | str = config.CACHE_DIR
) -> Path:
    """Write ``payload`` to its cache file, creating the directory if needed.

    The file is exactly ``payload.model_dump_json()``, so what the endpoint parses
    back is what was validated here.

    Raises:
        ValueError: If the payload's ``tournament_id`` cannot be a filename (see
            :func:`api.registry.cache_path_for`).
    """
    path = cache_path_for(payload.tournament_id, cache_dir)
    if path is None:
        raise ValueError(
            f"tournament id {payload.tournament_id!r} cannot be used as a cache "
            f"filename (it contains a path separator)."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Id resolution — the registry's own catalogue, rendered for a terminal.
# --------------------------------------------------------------------------- #
def _known_ids_message(registry: TournamentRegistry) -> str:
    """List what *is* addressable, grouped by what can actually be simulated.

    The groups are kept apart on purpose, and by the same rule the API's 404
    uses (``TournamentEntry.is_simulatable``). Lumping a draw that failed
    validation — or one still holding ``Qualifier`` slots — in with the working
    ones reads as though this script could run it, when in fact nothing but
    editing the file will make it so.
    """
    groups = [
        ("Available draws", [e.tournament_id for e in registry.valid if e.is_simulatable]),
        (
            "Draws awaiting entrants (fill their placeholder slots first)",
            [e.tournament_id for e in registry.valid if not e.is_simulatable],
        ),
        (
            "Draw files that failed validation (fix the file first)",
            [e.tournament_id for e in registry.invalid],
        ),
    ]
    lines = [f"   {label}: " + ", ".join(ids) for label, ids in groups if ids]
    if not lines:
        lines.append(f"   No draw files found in {registry.draws_dir}.")
    return "\n".join(lines)


def _build_classifier(context: ApiContext):
    """The ``ClassifierProb`` adapter, built once from the startup context.

    ``api/deps.py`` loads the estimator and both name-keyed histories but stops
    short of building an adapter (the API may not import ``cli/``); this script
    may, so the last step happens here. Cheap — ``make_classifier_prob`` returns
    a memoising closure and does no work until first called.
    """
    return make_classifier_prob(
        context.estimator,
        context.data,
        context.surface_history,
        context.h2h_history,
    )


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="precompute_sim",
        description="Run a tournament Monte Carlo offline and cache the result "
                    "for GET /tournaments/{id}/simulate.",
    )
    parser.add_argument(
        "--draw",
        required=True,
        help="Tournament id, as listed by GET /tournaments (not a file path).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=config.MC_RUNS,
        help=f"Number of bracket simulations (default: {config.MC_RUNS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.MC_SEED,
        help=f"Base seed — the same seed reproduces the file (default: "
             f"{config.MC_SEED}).",
    )
    parser.add_argument(
        "--reconcile-mode",
        choices=("blend", "classifier_anchor"),
        default=config.SIM_CLI_RECONCILE_MODE,
        help=f"Reconciliation mode; recorded in the cache metadata (default: "
             f"{config.SIM_CLI_RECONCILE_MODE} — see the module docstring).",
    )
    parser.add_argument(
        "--draws-dir",
        default=str(config.DRAWS_DIR),
        help=f"Directory of draw files (default: {config.DRAWS_DIR}).",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(config.CACHE_DIR),
        help=f"Where to write the cache file (default: {config.CACHE_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)

    if args.runs < 1:
        print(f"❌ --runs must be at least 1, got {args.runs}.")
        return 1

    print("Loading data, skill table and classifier…")
    context = build_api_context()
    classifier = _build_classifier(context)
    registry = build_registry(context.skill_table, args.draws_dir)

    entry = registry.get(args.draw)
    if entry is None:
        print(f"❌ No tournament with id {args.draw!r}.")
        print(_known_ids_message(registry))
        return 1
    if entry.draw is None:
        problems = "\n".join(f"     - {problem}" for problem in entry.problems)
        print(
            f"❌ Draw file {entry.source} failed validation with "
            f"{len(entry.problems)} problem(s):\n{problems}"
        )
        return 1

    print(
        f"Simulating {entry.draw.name} — {args.runs:,} runs, seed {args.seed}, "
        f"reconciliation {args.reconcile_mode}…"
    )
    try:
        result = monte_carlo(
            entry.draw,
            context.skill_table,
            classifier,
            n_runs=args.runs,
            seed=args.seed,
            # Single-process: the adapter is a closure and cannot be pickled
            # (ace-04-current-state.md §7 seam 8), and the measured 128 × 5,000
            # job is ~40 s single-threaded anyway.
            workers=1,
            mode=args.reconcile_mode,
        )
    except ValueError as exc:
        # Chiefly T2.2's placeholder refusal, which names every offending slot.
        print(f"❌ {exc}")
        return 1

    payload = build_payload(
        result,
        data_through_year=context.data_through_year,
        estimator_class=context.estimator_class,
        source=entry.source,
    )
    path = write_cache(payload, args.cache_dir)

    champion = payload.players[0]
    print(f"✅ Wrote {path}")
    print(
        f"   {payload.count} entrants · {payload.metadata.runs:,} runs · "
        f"favourite: {champion.player} ({champion.p_title:.1%})"
    )
    print(f"ℹ️  {payload.metadata.classifier_limitation}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
