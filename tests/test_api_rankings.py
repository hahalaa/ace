"""Tests for ``GET /rankings`` and the Elo precompute's payload builder.

Nothing here touches the vendored CSVs: the app is pointed at a fixture cache
directory, and the precompute's payload builder is exercised over a synthetic
:class:`~features.elo.EloResult`. The endpoint is a pure cache reader (like
``/simulate``), so the two things worth proving are that it round-trips a written
cache file and that it fails informatively when the cache is missing or corrupt.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.deps import ApiContext
from api.main import create_app
from api.schemas import RANKINGS_NOTE, RankingsResponse
from features.elo import (
    EloResult,
    Leaderboard,
    PlayerRating,
    elo_cache_path,
)
from features.serve import SkillTable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))

import precompute_elo  # noqa: E402


# --------------------------------------------------------------------------- #
# A synthetic EloResult, so the payload builder needs no vendored data.
# --------------------------------------------------------------------------- #
def _rating(pid, name, r, matches, last, active) -> PlayerRating:
    return PlayerRating(
        player_id=pid,
        player_name=name,
        rating=r,
        matches=matches,
        last_played=date.fromisoformat(last),
        is_active=active,
    )


@pytest.fixture
def elo_result() -> EloResult:
    overall = Leaderboard(
        track="overall",
        ratings=(
            _rating("A", "Ada Ace", 2100.0, 300, "2026-06-01", True),
            _rating("L", "Old Legend", 2050.0, 900, "2020-06-01", False),
            _rating("B", "Ben Base", 1800.0, 120, "2026-05-01", True),
        ),
    )
    per_surface = {
        surface: Leaderboard(
            track=surface,
            ratings=(_rating("A", "Ada Ace", 1900.0, 80, "2026-06-01", True),),
        )
        for surface in ("Hard", "Clay", "Grass")
    }
    return EloResult(
        leaderboards={"overall": overall, **per_surface},
        as_of=date(2026, 6, 1),
        active_cutoff=date(2025, 6, 1),
        n_matches=1234,
        data_through_year=2026,
    )


class TestPayloadBuilder:
    def test_builds_a_validated_rankings_response_with_ranks_and_note(self, elo_result):
        fixed = datetime(2026, 6, 2, tzinfo=timezone.utc)
        payload = precompute_elo.build_payload(elo_result, generated_at=fixed)

        assert isinstance(payload, RankingsResponse)
        assert [t.track for t in payload.tracks] == ["overall", "Clay", "Grass", "Hard"]
        overall = payload.tracks[0]
        # Ranks are 1-based and follow the sorted order the engine produced.
        assert [p.rank for p in overall.players] == [1, 2, 3]
        assert overall.players[0].player_name == "Ada Ace"
        # The stale legend is kept in the list, flagged, not dropped.
        assert overall.players[1].is_active is False
        assert payload.metadata.note == RANKINGS_NOTE
        assert payload.metadata.as_of == date(2026, 6, 1)
        assert payload.metadata.n_matches == 1234

    def test_write_and_reread_roundtrips(self, elo_result, tmp_path):
        payload = precompute_elo.build_payload(elo_result)
        path = precompute_elo.write_cache(payload, tmp_path)
        assert path == elo_cache_path(tmp_path)
        reread = RankingsResponse.model_validate_json(path.read_text())
        assert reread.model_dump() == payload.model_dump()


# --------------------------------------------------------------------------- #
# The endpoint, a cache reader over an injected, minimal context.
# --------------------------------------------------------------------------- #
@pytest.fixture
def context() -> ApiContext:
    """A minimal context. The rankings handler reads only the cache dir, so the
    rest is stubbed just enough for the app to start."""
    skill_table = SkillTable({}, {}, dict(_MU := {"Hard": 0.64, "Clay": 0.62, "Grass": 0.66}), cutoff=pd.Timestamp("2026-01-01"))
    return ApiContext(
        data=pd.DataFrame({"tourney_date": pd.to_datetime(["2026-01-01"])}),
        surface_history={},
        h2h_history={},
        skill_table=skill_table,
        estimator=object(),
        estimator_class="StubClassifier",
        data_through_year=2026,
    )


def _client(context, cache_dir):
    app = create_app(
        context_factory=lambda: context,
        draws_dir=cache_dir,  # unused by /rankings; any path is fine
        cache_dir=cache_dir,
        classifier_factory=lambda *a, **k: (lambda pa, pb, s: 0.5),
    )
    return app


class TestEndpoint:
    def test_serves_a_written_cache_file(self, context, elo_result, tmp_path):
        payload = precompute_elo.build_payload(elo_result)
        precompute_elo.write_cache(payload, tmp_path)
        with TestClient(_client(context, tmp_path)) as client:
            response = client.get("/rankings")
        assert response.status_code == 200
        body = response.json()
        assert [t["track"] for t in body["tracks"]] == ["overall", "Clay", "Grass", "Hard"]
        assert body["tracks"][0]["players"][0]["player_name"] == "Ada Ace"
        assert body["metadata"]["note"] == RANKINGS_NOTE

    def test_missing_cache_is_a_425_naming_the_command(self, context, tmp_path):
        with TestClient(_client(context, tmp_path)) as client:
            response = client.get("/rankings")
        assert response.status_code == 425
        detail = response.json()["detail"]
        assert detail["reason"] == "rankings_missing"
        assert "precompute_elo" in detail["command"]

    def test_corrupt_cache_is_a_422(self, context, tmp_path):
        elo_cache_path(tmp_path).write_text("{ not json", encoding="utf-8")
        with TestClient(_client(context, tmp_path)) as client:
            response = client.get("/rankings")
        assert response.status_code == 422
        assert response.json()["detail"]["reason"] == "rankings_unreadable"
