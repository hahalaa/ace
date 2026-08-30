"""Compute overall + per-surface Elo leaderboards offline into the cache GET /rankings reads.

Reads the raw loader frame directly (Elo needs explicit winner/loser, pre-p1/p2), never the
classifier or skill table. Output is a validated api.schemas.RankingsResponse.
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
from api.schemas import (  # noqa: E402
    RANKINGS_NOTE,
    RankedPlayer,
    RankingsMetadata,
    RankingsResponse,
    RankingsTrack,
)
from data.loader import load_atp_data  # noqa: E402
from features.elo import (  # noqa: E402
    ELO_ACTIVE_WINDOW_DAYS,
    ELO_TRACKS,
    EloResult,
    compute_elo,
    elo_cache_path,
)


def build_payload(
    result: EloResult, *, generated_at: datetime | None = None
) -> RankingsResponse:
    """Turn a finished Elo run into the cache/wire payload (presentation only, nothing recomputed)."""
    tracks = [
        RankingsTrack(
            track=track_name,
            players=[
                RankedPlayer(
                    rank=rank,
                    player_id=entry.player_id,
                    player_name=entry.player_name,
                    rating=entry.rating,
                    matches=entry.matches,
                    last_played=entry.last_played,
                    is_active=entry.is_active,
                )
                for rank, entry in enumerate(
                    result.leaderboards[track_name].ratings, start=1
                )
            ],
        )
        for track_name in ELO_TRACKS
    ]
    return RankingsResponse(
        tracks=tracks,
        metadata=RankingsMetadata(
            generated_at=generated_at or datetime.now(timezone.utc),
            as_of=result.as_of,
            active_cutoff=result.active_cutoff,
            active_window_days=ELO_ACTIVE_WINDOW_DAYS,
            data_through_year=result.data_through_year,
            n_matches=result.n_matches,
            note=RANKINGS_NOTE,
        ),
    )


def write_cache(
    payload: RankingsResponse, cache_dir: Path | str = config.CACHE_DIR
) -> Path:
    """Write ``payload`` to the ratings cache file, creating the directory if needed."""
    path = elo_cache_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="precompute_elo",
        description="Compute overall + per-surface Elo leaderboards offline and "
                    "cache them for GET /rankings.",
    )
    parser.add_argument(
        "--start", type=int, default=config.START_YEAR,
        help=f"First season to rate over (default: {config.START_YEAR}).",
    )
    parser.add_argument(
        "--end", type=int, default=config.END_YEAR,
        help=f"Last season to rate over, inclusive (default: {config.END_YEAR}).",
    )
    parser.add_argument(
        "--cache-dir", default=str(config.CACHE_DIR),
        help=f"Where to write the cache file (default: {config.CACHE_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _parse_args(argv)
    if args.start > args.end:
        print(f"--start {args.start} is after --end {args.end}.")
        return 1

    print(f"Loading match data {args.start}–{args.end}…")
    df = load_atp_data(args.start, args.end)

    print("Computing Elo ratings (overall + per surface)…")
    result = compute_elo(df)

    payload = build_payload(result)
    path = write_cache(payload, args.cache_dir)

    overall = payload.tracks[0]
    leader = next((p for p in overall.players if p.is_active), None)
    print(f"Wrote {path}")
    print(
        f"   {len(overall.players)} overall entries · {result.n_matches:,} matches · "
        f"as of {result.as_of}"
    )
    if leader is not None:
        print(f"   current #1 (overall, active): {leader.player_name} ({leader.rating:.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
