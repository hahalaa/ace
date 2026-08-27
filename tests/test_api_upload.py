"""Tests for the draw upload feature, ``POST /tournaments/upload`` and the
serving of uploaded draws through ``/bracket`` and ``/storybook``.

Like ``tests/test_api_storybook.py``, nothing here touches the vendored CSVs, the
shipped draws or the real classifier: the app is pointed at
``tests/fixtures/draws/`` and given a deterministic stub ``ClassifierProb``. The
uploaded draws are built inline from the same toy skill table.

What this file exists to prove, beyond the happy path:

* **Uploads never touch disk** (``test_upload_writes_nothing_to_disk``), the
  whole point of the in-memory design, since the deploy target has no persistent
  disk.
* **Uploads are ephemeral**, a fresh app instance (a restart) has forgotten
  every prior upload, and an over-capacity store evicts the oldest.
* **Validation is the draw schema's**, surfaced as the same accumulated problem list every
  other draw-loading path uses.
* **is_forecast reads true end to end** for a genuinely future-dated draw, live
  through a storybook run, the first time in the project it legitimately can.
* **The rate limiter engages** on repeated uploads.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import config
from api.deps import ApiContext
from api.main import create_app
from api.schemas import (
    CLASSIFIER_LIMITATION,
    CONTENT_NOTE_UPLOAD,
    CONTENT_SOURCE_CURATED,
    CONTENT_SOURCE_UPLOAD,
    FORECAST_CLASSIFIER_LIMITATION,
    StorybookMetadata,
    UploadResponse,
)
from api.uploads import UploadStore
from features.serve import PlayerSkill, SkillTable

FIXTURE_DRAWS = Path(__file__).parent / "fixtures" / "draws"
FIXTURE_CACHE = Path(__file__).parent / "fixtures" / "cache"

# The eight toy entrants, same ids as tests/test_api_storybook.py.
TOY_PLAYERS = {
    "Alice Ace": "A1",
    "Bob Baseline": "B2",
    "Cara Chip": "C3",
    "Dan Drop": "D4",
    "Eve Emphatic": "E5",
    "Frank Fault": "F6",
    "Gina Grip": "G7",
    "Hank Half": "H8",
}

# A fully-resolvable 8-slot draw over the toy players, no placeholders.
BRACKET_ORDER = [
    "Alice Ace",
    "Hank Half",
    "Cara Chip",
    "Dan Drop",
    "Eve Emphatic",
    "Frank Fault",
    "Gina Grip",
    "Bob Baseline",
]


def valid_draw(event_date: str | None = None) -> dict:
    """A valid uploadable 8-slot draw; optionally carrying an ``event_date``."""
    draw = {
        "tournament_id": "my_upload_open_2026",
        "name": "My Upload Open 2026",
        "surface": "Clay",
        "best_of": 3,
        "final_set_tiebreak": "7pt_at_6_6",
        "draw_size": 8,
        "seeds": {"Alice Ace": 1, "Bob Baseline": 2},
        "bracket": [
            {"position": i + 1, "player": name} for i, name in enumerate(BRACKET_ORDER)
        ],
    }
    if event_date is not None:
        draw[config.UPLOAD_EVENT_DATE_FIELD] = event_date
    return draw


def placeholder_draw() -> dict:
    """A loadable 8-slot draw that still holds a placeholder, not simulatable."""
    draw = valid_draw()
    draw["bracket"][1] = {"position": 2, "player": "Qualifier"}
    draw["seeds"] = {"Alice Ace": 1}
    return draw


@pytest.fixture
def skill_table() -> SkillTable:
    """Toy id-keyed skills covering TOY_PLAYERS on all three surfaces."""
    skills = {}
    for index, player_id in enumerate(TOY_PLAYERS.values()):
        for surface in config.VALID_SURFACES:
            mu = config.SURFACE_MU[surface]
            skills[(player_id, surface)] = PlayerSkill(
                spw=mu + 0.01 * index,
                rpw=1.0 - mu + 0.01 * index,
                n_serve_pts=1000.0,
                n_return_pts=1000.0,
            )
    return SkillTable(
        skills,
        dict(TOY_PLAYERS),
        dict(config.SURFACE_MU),
        cutoff=pd.Timestamp("2026-01-01"),
    )


@pytest.fixture
def context(skill_table: SkillTable) -> ApiContext:
    """A fixture context with a real skill table and a stub estimator."""
    return ApiContext(
        data=pd.DataFrame({"tourney_date": pd.to_datetime(["2026-01-01"])}),
        surface_history={},
        h2h_history={},
        skill_table=skill_table,
        estimator=object(),
        estimator_class="StubClassifier",
        data_through_year=2026,
    )


class StubClassifier:
    """Deterministic ``(a, b, surface) -> P_clf``; skews on name order."""

    def __call__(self, player_a: str, player_b: str, surface: str) -> float:
        return 0.55 if player_a < player_b else 0.45


def stub_classifier_factory(estimator, data, surface_history, h2h_history):
    return StubClassifier()


def make_app(context: ApiContext, upload_store: UploadStore | None = None):
    return create_app(
        context_factory=lambda: context,
        draws_dir=FIXTURE_DRAWS,
        cache_dir=FIXTURE_CACHE,
        classifier_factory=stub_classifier_factory,
        upload_store=upload_store,
    )


@pytest.fixture
def client(context: ApiContext):
    with TestClient(make_app(context)) as test_client:
        yield test_client


def upload(client: TestClient, draw: dict):
    """POST a draw as JSON, returning the raw response."""
    return client.post("/tournaments/upload", json=draw)


# --------------------------------------------------------------------------- #
# Acceptance: a valid draw is accepted, stored, and simulable
# --------------------------------------------------------------------------- #
def test_valid_draw_is_accepted_and_addressable(client):
    response = upload(client, valid_draw())
    assert response.status_code == 201

    body = response.json()
    UploadResponse.model_validate(body)
    assert set(body) == set(UploadResponse.model_fields)

    assert body["tournament_id"].startswith(config.UPLOAD_ID_PREFIX)
    assert body["name"] == "My Upload Open 2026"
    assert body["draw_size"] == 8
    assert body["is_simulatable"] is True
    assert body["placeholder_count"] == 0
    assert body["is_forecast"] is False
    assert body["ephemeral"] is True
    assert body["content_note"] == CONTENT_NOTE_UPLOAD


def test_uploaded_bracket_is_served_like_a_curated_one(client):
    upload_id = upload(client, valid_draw()).json()["tournament_id"]

    response = client.get(f"/tournaments/{upload_id}/bracket")
    assert response.status_code == 200
    body = response.json()
    # The response reports the addressable upload id, not the draw's own id.
    assert body["tournament_id"] == upload_id
    assert body["draw_size"] == 8
    assert [slot["player"] for slot in body["slots"]] == BRACKET_ORDER
    assert all(slot["is_placeholder"] is False for slot in body["slots"])


def test_uploaded_draw_runs_a_live_storybook(client):
    upload_id = upload(client, valid_draw()).json()["tournament_id"]

    response = client.get(f"/tournaments/{upload_id}/storybook?seed=7")
    assert response.status_code == 200
    body = response.json()
    assert body["tournament_id"] == upload_id
    assert body["match_count"] == 7
    assert [r["label"] for r in body["rounds"]] == ["QF", "SF", "F"]
    assert body["metadata"]["seed"] == 7


def test_uploaded_storybook_is_deterministic(client):
    upload_id = upload(client, valid_draw()).json()["tournament_id"]
    first = client.get(f"/tournaments/{upload_id}/storybook?seed=7")
    second = client.get(f"/tournaments/{upload_id}/storybook?seed=7")
    assert first.content == second.content


# --------------------------------------------------------------------------- #
# Disclosure: distinguishing user-submitted content
# --------------------------------------------------------------------------- #
def test_uploaded_storybook_discloses_user_submitted_content(client):
    """The content note is distinct from the model's is_forecast framing."""
    upload_id = upload(client, valid_draw()).json()["tournament_id"]
    body = client.get(f"/tournaments/{upload_id}/storybook?seed=1").json()

    metadata = body["metadata"]
    # The schema round-trips and carries exactly the declared fields.
    StorybookMetadata.model_validate(metadata)
    assert set(metadata) == set(StorybookMetadata.model_fields)

    assert metadata["content_source"] == CONTENT_SOURCE_UPLOAD
    assert metadata["content_note"] == CONTENT_NOTE_UPLOAD
    assert metadata["source"] == upload_id
    # The two concerns are separate: content authenticity is not the model caveat.
    assert metadata["content_note"] != metadata["classifier_limitation"]


def test_curated_storybook_discloses_curated_content(client):
    """A curated draw reports curated provenance, not the upload note."""
    body = client.get("/tournaments/toy_open_2026/storybook?seed=1").json()
    metadata = body["metadata"]
    assert metadata["content_source"] == CONTENT_SOURCE_CURATED
    assert metadata["content_note"] != CONTENT_NOTE_UPLOAD


# --------------------------------------------------------------------------- #
# is_forecast: true, end to end, for a genuinely future-dated draw
# --------------------------------------------------------------------------- #
def test_future_dated_upload_is_a_forecast_end_to_end(client):
    """A future event_date flips is_forecast true, the first legitimate case.

    Confirmed live through a storybook run, not only on the upload ack: the flag
    and the forecast wording must reach the simulated response's metadata.
    """
    future = (date.today() + timedelta(days=400)).isoformat()
    ack = upload(client, valid_draw(event_date=future)).json()
    assert ack["is_forecast"] is True

    upload_id = ack["tournament_id"]
    metadata = client.get(f"/tournaments/{upload_id}/storybook?seed=3").json()["metadata"]
    assert metadata["is_forecast"] is True
    assert metadata["classifier_limitation"] == FORECAST_CLASSIFIER_LIMITATION
    # Even a forecast is still user-submitted content.
    assert metadata["content_source"] == CONTENT_SOURCE_UPLOAD


def test_past_dated_upload_is_not_a_forecast(client):
    past = (date.today() - timedelta(days=30)).isoformat()
    ack = upload(client, valid_draw(event_date=past)).json()
    assert ack["is_forecast"] is False

    upload_id = ack["tournament_id"]
    metadata = client.get(f"/tournaments/{upload_id}/storybook?seed=3").json()["metadata"]
    assert metadata["is_forecast"] is False
    assert metadata["classifier_limitation"] == CLASSIFIER_LIMITATION


def test_undated_upload_defaults_to_not_a_forecast(client):
    ack = upload(client, valid_draw()).json()
    assert ack["is_forecast"] is False


def test_malformed_event_date_is_rejected(client):
    response = upload(client, valid_draw(event_date="next tuesday"))
    assert response.status_code == 422
    assert response.json()["detail"]["reason"] == "invalid_event_date"


# --------------------------------------------------------------------------- #
# Rejection paths, each a distinguishable, structured error
# --------------------------------------------------------------------------- #
def test_malformed_draw_returns_the_accumulated_problem_list(client):
    bad = valid_draw()
    del bad["surface"]  # missing required field
    bad["best_of"] = 4  # invalid enum
    bad["bracket"][0]["player"] = "Nobody Whoever"  # unresolvable name

    response = upload(client, bad)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason"] == "draw_invalid"
    problems = detail["problems"]
    # Every problem accumulated, not just the first, the draw contract.
    joined = " ".join(problems)
    assert "surface" in joined
    assert "best_of" in joined
    assert any("Nobody Whoever" in p for p in problems)
    assert len(problems) >= 3


def test_non_json_content_type_is_rejected(client):
    response = client.post(
        "/tournaments/upload",
        content=b"position,player\n1,Alice",
        headers={"content-type": "text/csv"},
    )
    assert response.status_code == 415
    assert response.json()["detail"]["reason"] == "unsupported_media_type"


def test_invalid_json_body_is_rejected(client):
    response = client.post(
        "/tournaments/upload",
        content=b"{not valid json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["reason"] == "invalid_json"


def test_deeply_nested_json_is_rejected_without_a_500(context):
    """Adversarial deeply-nested JSON is a clean 422, not an uncaught 500.

    A few thousand open brackets (tiny, far under the size cap) overflows the
    JSON decoder's recursion. Left uncaught it escapes the handler as a 500 with
    a full stack trace logged on every request, a cheap error/log-spam vector on
    a public endpoint. It must come back as ``invalid_json`` like any other
    malformed body, and the server must stay up either way.

    ``raise_server_exceptions=True`` (the TestClient default) is load-bearing
    here: if this regressed to an unhandled 500, the client would re-raise the
    server exception and this test would error rather than see a 422.
    """
    with TestClient(make_app(context)) as c:
        nested = "[" * 50000  # ~50 KB, no closing brackets: nesting, not size
        response = c.post(
            "/tournaments/upload",
            content=nested,
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["reason"] == "invalid_json"
        # Server is still up: a normal request right after still succeeds.
        assert upload(c, valid_draw()).status_code == 201


def test_oversized_body_is_rejected(client):
    oversized = valid_draw()
    oversized["note"] = "x" * (config.API_UPLOAD_MAX_BYTES + 1)
    response = upload(client, oversized)
    assert response.status_code == 413
    assert response.json()["detail"]["reason"] == "payload_too_large"


def test_placeholder_upload_lists_but_is_not_simulatable(client):
    ack = upload(client, placeholder_draw()).json()
    assert ack["is_simulatable"] is False
    assert ack["placeholder_count"] == 1

    upload_id = ack["tournament_id"]
    # Bracket serves fine (placeholders flagged, not dropped).
    bracket = client.get(f"/tournaments/{upload_id}/bracket")
    assert bracket.status_code == 200
    # Storybook refuses it, same 409 shape as a curated placeholder draw.
    story = client.get(f"/tournaments/{upload_id}/storybook")
    assert story.status_code == 409
    assert story.json()["detail"]["reason"] == "draw_not_simulatable"


def test_monte_carlo_is_unavailable_for_uploads(client):
    upload_id = upload(client, valid_draw()).json()["tournament_id"]
    response = client.get(f"/tournaments/{upload_id}/simulate")
    assert response.status_code == 409
    assert response.json()["detail"]["reason"] == "monte_carlo_unavailable_for_upload"


def test_unknown_upload_id_is_a_404(client):
    """An id in the upload namespace but never stored (or evicted/restarted)."""
    fake = f"{config.UPLOAD_ID_PREFIX}deadbeefdeadbeef"
    bracket = client.get(f"/tournaments/{fake}/bracket")
    assert bracket.status_code == 404
    assert bracket.json()["detail"]["reason"] == "upload_not_found"
    story = client.get(f"/tournaments/{fake}/storybook")
    assert story.status_code == 404
    assert story.json()["detail"]["reason"] == "upload_not_found"


# --------------------------------------------------------------------------- #
# The no-persistent-disk design, under test
# --------------------------------------------------------------------------- #
def test_upload_writes_nothing_to_disk(client, monkeypatch):
    """Upload + serve must not write to disk, the core platform constraint.

    ``precompute_sim.write_cache`` (the only cache writer in the codebase) uses
    ``Path.write_text``; spying on the write primitives and asserting they are
    never called proves the upload path persists nothing.
    """
    write_text = MagicMock(side_effect=AssertionError("wrote to disk via write_text"))
    write_bytes = MagicMock(side_effect=AssertionError("wrote to disk via write_bytes"))
    monkeypatch.setattr(Path, "write_text", write_text)
    monkeypatch.setattr(Path, "write_bytes", write_bytes)

    upload_id = upload(client, valid_draw()).json()["tournament_id"]
    assert client.get(f"/tournaments/{upload_id}/bracket").status_code == 200
    assert client.get(f"/tournaments/{upload_id}/storybook?seed=5").status_code == 200

    write_text.assert_not_called()
    write_bytes.assert_not_called()


def test_uploads_do_not_survive_a_restart(context):
    """A fresh app instance (a restart) has forgotten every prior upload.

    This is the ephemeral contract asserted the only way a single process can:
    two apps in sequence do not share the in-memory store, so an id minted by the
    first is unknown to the second, exactly what a real restart produces.
    """
    with TestClient(make_app(context)) as first:
        upload_id = upload(first, valid_draw()).json()["tournament_id"]
        assert first.get(f"/tournaments/{upload_id}/bracket").status_code == 200

    # A brand-new app == a restarted process: its store starts empty.
    with TestClient(make_app(context)) as second:
        response = second.get(f"/tournaments/{upload_id}/bracket")
        assert response.status_code == 404
        assert response.json()["detail"]["reason"] == "upload_not_found"


def test_store_evicts_oldest_when_capacity_is_reached(context):
    """The concurrently-held cap bounds memory: the oldest upload is evicted."""
    store = UploadStore(max_uploads=2)
    with TestClient(make_app(context, upload_store=store)) as client:
        first = upload(client, valid_draw()).json()["tournament_id"]
        second = upload(client, valid_draw()).json()["tournament_id"]
        third = upload(client, valid_draw()).json()["tournament_id"]

        # Three uploaded, cap is two: the first is gone, the last two remain.
        assert client.get(f"/tournaments/{first}/bracket").status_code == 404
        assert client.get(f"/tournaments/{second}/bracket").status_code == 200
        assert client.get(f"/tournaments/{third}/bracket").status_code == 200
        assert len(store) == 2


# --------------------------------------------------------------------------- #
# Abuse prevention: the rate limiter engages
# --------------------------------------------------------------------------- #
def test_upload_rate_limiter_engages(context, monkeypatch):
    """Repeated uploads from one client hit the per-IP limit."""
    monkeypatch.setattr(config, "API_UPLOAD_RATE_LIMIT", "2/minute")
    with TestClient(make_app(context)) as client:
        assert upload(client, valid_draw()).status_code == 201
        assert upload(client, valid_draw()).status_code == 201
        limited = upload(client, valid_draw())
        assert limited.status_code == 429
        assert limited.json()["detail"]["reason"] == "rate_limited"
        assert "Retry-After" in limited.headers
