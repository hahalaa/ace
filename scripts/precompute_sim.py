"""Run a tournament Monte Carlo offline into the cache GET /tournaments/{id}/simulate reads.

``--draw`` takes a tournament id, resolved through ``api/registry.py``. The pipeline and
classifier are built once via ``api.deps``; the adapter is ``common/classifier_adapter.py``.
Disclosure fields (``mode``/``w``/``is_forecast``/``classifier_limitation``) are required
metadata. A placeholder-slot draw is refused by ``monte_carlo`` and no cache file is written.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# scripts/ is not on pyproject's pythonpath; add src/ so imports resolve like the runtime.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import config  # noqa: E402
from api.deps import ApiContext, build_api_context  # noqa: E402
from api.registry import TournamentRegistry, build_registry, cache_path_for  # noqa: E402
from api.schemas import (  # noqa: E402
    CLASSIFIER_LIMITATION,
    CLASSIFIER_LIMITATION_DETAIL,
    IS_FORECAST,
    SimulationMetadata,
    SimulationPlayer,
    SimulationResponse,
)
from common.classifier_adapter import make_classifier_prob  # noqa: E402
from sim.tournament import MonteCarloResult, monte_carlo  # noqa: E402

# Must stay equal to api/main.adapter_name()'s value; pinned by tests/test_api_storybook.py.
ADAPTER = "common.classifier_adapter.make_classifier_prob"

# Disclosure text and honesty flag come from api.schemas so /storybook says the same thing.


# Cache payload; the schema in api/schemas.py is the format.
def build_payload(
    result: MonteCarloResult,
    *,
    data_through_year: int,
    estimator_class: str,
    source: str,
    generated_at: datetime | None = None,
    adapter: str = ADAPTER,
    classifier_limitation: str = CLASSIFIER_LIMITATION,
    classifier_limitation_detail: str = CLASSIFIER_LIMITATION_DETAIL,
    is_forecast: bool = IS_FORECAST,
) -> SimulationResponse:
    """Turn a finished Monte Carlo run into a validated ``SimulationResponse`` (presentation only, nothing recomputed)."""
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
            classifier_limitation_detail=classifier_limitation_detail,
            source=source,
        ),
    )


def write_cache(
    payload: SimulationResponse, cache_dir: Path | str = config.CACHE_DIR
) -> Path:
    """Write ``payload`` (exactly ``model_dump_json()``) to its cache file; raises ValueError if the id cannot be a filename."""
    path = cache_path_for(payload.tournament_id, cache_dir)
    if path is None:
        raise ValueError(
            f"tournament id {payload.tournament_id!r} cannot be used as a cache "
            f"filename (it contains a path separator)."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _known_ids_message(registry: TournamentRegistry) -> str:
    """List addressable draws, grouped by ``TournamentEntry.is_simulatable`` (the rule the API's 404 uses)."""
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
    """The ``ClassifierProb`` adapter, built once from the startup context (form table and predict_proba deferred to first use)."""
    return make_classifier_prob(
        context.estimator,
        context.data,
        context.surface_history,
        context.h2h_history,
    )


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
        help=f"Base seed; the same seed reproduces the file (default: "
             f"{config.MC_SEED}).",
    )
    parser.add_argument(
        "--reconcile-mode",
        choices=("blend", "classifier_anchor"),
        default=config.SIM_CLI_RECONCILE_MODE,
        help=f"Reconciliation mode; recorded in the cache metadata (default: "
             f"{config.SIM_CLI_RECONCILE_MODE}; see the module docstring).",
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
        print(f"--runs must be at least 1, got {args.runs}.")
        return 1

    print("Loading data, skill table and classifier…")
    context = build_api_context()
    classifier = _build_classifier(context)
    registry = build_registry(context.skill_table, args.draws_dir)

    entry = registry.get(args.draw)
    if entry is None:
        print(f"No tournament with id {args.draw!r}.")
        print(_known_ids_message(registry))
        return 1
    if entry.draw is None:
        problems = "\n".join(f"     - {problem}" for problem in entry.problems)
        print(
            f"Draw file {entry.source} failed validation with "
            f"{len(entry.problems)} problem(s):\n{problems}"
        )
        return 1

    print(
        f"Simulating {entry.draw.name}: {args.runs:,} runs, seed {args.seed}, "
        f"reconciliation {args.reconcile_mode}…"
    )
    try:
        result = monte_carlo(
            entry.draw,
            context.skill_table,
            classifier,
            n_runs=args.runs,
            seed=args.seed,
            # workers>1 would need its own re-verification; 128 x 5,000 is 29 s single-threaded.
            workers=1,
            mode=args.reconcile_mode,
        )
    except ValueError as exc:
        # Chiefly the simulator's placeholder refusal, which names every offending slot.
        print(f"{exc}")
        return 1

    payload = build_payload(
        result,
        data_through_year=context.data_through_year,
        estimator_class=context.estimator_class,
        source=entry.source,
    )
    path = write_cache(payload, args.cache_dir)

    champion = payload.players[0]
    print(f"Wrote {path}")
    print(
        f"   {payload.count} entrants · {payload.metadata.runs:,} runs · "
        f"favourite: {champion.player} ({champion.p_title:.1%})"
    )
    print(f"{payload.metadata.classifier_limitation}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
