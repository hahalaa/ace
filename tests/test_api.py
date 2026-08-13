"""Tests for the FastAPI app (T3.1) — ``/health`` and ``/players``.

Everything runs against a **fixture** :class:`~api.deps.ApiContext` injected via
``create_app(context_factory=...)``: a nine-player toy skill table, no vendored
CSVs, no pickled model. Same standard as T1.10/T2.5's tests — the suite must stay
fast, and the endpoints' behaviour does not depend on the size of the dataset.

Covers the ticket's acceptance + testing criteria, plus the two decisions this
ticket owns:
  * the context is built **once per app**, at startup, not per request;
  * ``/players`` answers an ambiguous or unresolvable query with a **list**
    (200, possibly empty), not the CLI's error.

``build_api_context`` itself is the one thing a fixture context cannot cover, so
it gets a dedicated test over stubbed pipeline steps — see
``test_build_api_context_reports_the_observed_max_year``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import config
import api.deps as deps
from api.deps import ApiContext
from api.main import create_app
from api.schemas import (
    HealthResponse,
    PlayerSearchResponse,
    PlayerSummary,
    SurfaceSkill,
)
from features.serve import PlayerSkill, SkillTable

# Two "Alcaraz" entries so a substring query is genuinely ambiguous, and two
# "C <lastname>" players so the initials strategy can be ambiguous too.
TOY_PLAYERS = {
    "Carlos Alcaraz": "A1",
    "Alvaro Alcaraz": "A2",
    "Jannik Sinner": "S1",
    "Novak Djokovic": "D1",
    "Casper Ruud": "R1",
    "Carlos Ruud": "R2",
    "Taylor Fritz": "F1",
    "Grigor Dimitrov": "G1",
    # No serve rows for this one — resolves to an id, falls back to the default.
    "Unmeasured Newcomer": "N1",
}

# Ordered (not a set) so the toy skill spread below is identical run to run.
MEASURED = ("A1", "A2", "S1", "D1", "R1", "R2", "F1", "G1")


@pytest.fixture
def skill_table() -> SkillTable:
    """Toy id-keyed skills on all three surfaces, with a real spread."""
    skills = {}
    for index, player_id in enumerate(MEASURED):
        for surface in config.VALID_SURFACES:
            mu = config.SURFACE_MU[surface]
            edge = 0.005 * (len(MEASURED) - index)
            skills[(player_id, surface)] = PlayerSkill(
                spw=mu + edge,
                rpw=1.0 - mu + edge,
                n_serve_pts=1000.0 + index,
                n_return_pts=900.0 + index,
            )
    return SkillTable(
        skills,
        dict(TOY_PLAYERS),
        dict(config.SURFACE_MU),
        cutoff=pd.Timestamp("2026-01-01"),
    )


@pytest.fixture
def context(skill_table: SkillTable) -> ApiContext:
    """A fixture context: real skill table, stub estimator, tiny frame."""
    return ApiContext(
        data=pd.DataFrame({"tourney_date": pd.to_datetime(["2026-01-01"])}),
        surface_history={},
        h2h_history={},
        skill_table=skill_table,
        estimator=object(),  # never called by any T3.1 endpoint
        estimator_class="StubClassifier",
        data_through_year=2026,
    )


@pytest.fixture
def client(context: ApiContext):
    """TestClient over an app whose startup factory returns the fixture context.

    ``with`` is required: Starlette only runs the lifespan inside the context
    manager, and the lifespan is what populates ``app.state.context``.
    """
    app = create_app(context_factory=lambda: context)
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# Startup loading
# --------------------------------------------------------------------------- #
def test_context_is_built_once_across_many_requests(context: ApiContext):
    """The factory runs exactly once per app, not once per request."""
    calls = []

    def counting_factory() -> ApiContext:
        calls.append(1)
        return context

    app = create_app(context_factory=counting_factory)
    assert calls == []  # constructing the app loads nothing

    with TestClient(app) as client:
        for _ in range(5):
            assert client.get("/health").status_code == 200
            assert client.get("/players", params={"query": "sinner"}).status_code == 200

    assert len(calls) == 1


def test_startup_populates_inspectable_app_state(context: ApiContext):
    """The loaded context is inspectable on ``app.state`` and is the same object."""
    app = create_app(context_factory=lambda: context)
    with TestClient(app):
        assert app.state.context is context
        assert app.state.context.estimator_class == "StubClassifier"


def test_startup_logs_the_load(context: ApiContext, caplog):
    """Startup emits a record proving the load happened (acceptance criterion)."""
    app = create_app(context_factory=lambda: context)
    with caplog.at_level("INFO", logger="api.deps"):
        with TestClient(app):
            pass
    assert any("API startup" in record.message for record in caplog.records)


def test_importing_main_loads_no_data():
    """The module-level ``app`` must not run the pipeline at import time."""
    import api.main as main

    assert getattr(main.app.state, "context", None) is None


def test_build_api_context_reports_the_observed_max_year(monkeypatch, skill_table):
    """``data_through_year`` is read off the frame, not copied from ``END_YEAR``.

    Every other test injects a ready-made :class:`ApiContext`, which leaves the
    real orchestration — and this decision in particular — unexercised. The stub
    frame's latest season is deliberately **not** ``config.END_YEAR``: with the
    two equal (they are, on today's vendored data) a context that simply echoed
    the configured year would be indistinguishable from a correct one.
    """
    observed_year = config.END_YEAR - 7
    frame = pd.DataFrame(
        {
            "tourney_date": pd.to_datetime(
                [f"{observed_year - 1}-06-01", f"{observed_year}-03-01"]
            )
        }
    )
    assert observed_year != config.END_YEAR  # the test would be vacuous otherwise

    calls: list[str] = []

    def fake_load(start_year, end_year):
        calls.append(f"load({start_year},{end_year})")
        return "raw"

    def fake_preprocess(raw):
        calls.append(f"preprocess({raw})")
        return "processed"

    def fake_add_features(processed):
        calls.append(f"add_features({processed})")
        return frame, {"surf": 1}, {"h2h": 2}, skill_table

    def fake_pin(path):
        calls.append(f"pin({path})")
        return SimpleNamespace(estimator="ESTIMATOR", estimator_class="StubClassifier")

    monkeypatch.setattr(deps.loader, "load_atp_data", fake_load)
    monkeypatch.setattr(deps.preprocess, "preprocess_data", fake_preprocess)
    monkeypatch.setattr(deps.engineering, "add_features", fake_add_features)
    monkeypatch.setattr(deps, "load_pinned_classifier", fake_pin)

    context = deps.build_api_context()

    assert context.data_through_year == observed_year
    assert context.data_through_year != config.END_YEAR
    # The core chain runs in order, and nothing else is invented.
    assert calls == [
        f"load({config.START_YEAR},{config.END_YEAR})",
        "preprocess(raw)",
        "add_features(processed)",
        f"pin({config.MODEL_PATH})",
    ]
    assert context.estimator == "ESTIMATOR"
    assert context.estimator_class == "StubClassifier"
    assert context.surface_history == {"surf": 1}
    assert context.h2h_history == {"h2h": 2}
    assert context.n_players == len(TOY_PLAYERS)


def test_health_year_comes_from_the_context_not_the_config(context):
    """End to end: a context whose year differs from ``END_YEAR`` is reported."""
    off_config = ApiContext(
        data=context.data,
        surface_history={},
        h2h_history={},
        skill_table=context.skill_table,
        estimator=object(),
        estimator_class="StubClassifier",
        data_through_year=config.END_YEAR - 7,
    )
    with TestClient(create_app(context_factory=lambda: off_config)) as client:
        assert client.get("/health").json()["data_through_year"] == config.END_YEAR - 7


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #
def test_health_reports_status_and_dataset_info(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "data_through_year": 2026,
        "n_players": len(TOY_PLAYERS),
    }


# --------------------------------------------------------------------------- #
# /players
# --------------------------------------------------------------------------- #
def test_players_query_returns_matches_with_skill_fields(client):
    """The ticket's worked example: ?query=alcaraz returns players + skills."""
    body = client.get("/players", params={"query": "alcaraz"}).json()

    assert body["query"] == "alcaraz"
    assert body["count"] == 2
    names = {p["name"] for p in body["players"]}
    assert names == {"Carlos Alcaraz", "Alvaro Alcaraz"}

    player = next(p for p in body["players"] if p["name"] == "Carlos Alcaraz")
    assert player["player_id"] == "A1"
    assert set(player["skills"]) == config.VALID_SURFACES
    for surface, skill in player["skills"].items():
        assert 0.0 < skill["spw"] < 1.0
        assert 0.0 < skill["rpw"] < 1.0
        assert skill["n_serve_pts"] > 0
        assert skill["spw"] != config.SURFACE_MU[surface]  # a measured rate


def test_players_exact_query_returns_one_player(client):
    body = client.get("/players", params={"query": "Jannik Sinner"}).json()
    assert body["count"] == 1
    assert body["strategy"] == "exact"
    assert body["players"][0]["player_id"] == "S1"


def test_players_ambiguous_query_returns_all_candidates_not_an_error(client):
    """Ambiguity is a result, not a 404/409 — the documented HTTP-search shape."""
    response = client.get("/players", params={"query": "C Ruud"})
    assert response.status_code == 200

    body = response.json()
    assert body["strategy"] == "initials"
    assert body["count"] == 2
    assert {p["name"] for p in body["players"]} == {"Casper Ruud", "Carlos Ruud"}
    # Every candidate is a full summary, not a bare name.
    assert all(p["player_id"] and p["skills"] for p in body["players"])


def test_players_unresolvable_query_returns_empty_list(client):
    """No match is an empty successful search, not a 404."""
    response = client.get("/players", params={"query": "zzzzzzzz"})
    assert response.status_code == 200
    assert response.json() == {
        "query": "zzzzzzzz",
        "count": 0,
        "strategy": None,
        "players": [],
    }


@pytest.mark.parametrize(
    "params", [{}, {"query": ""}, {"query": " "}, {"query": "   "}, {"query": "\t"}]
)
def test_players_requires_a_non_empty_query(client, params):
    """A missing/blank/whitespace-only query is malformed, not an empty search.

    Whitespace matters more than it looks: ``resolve_name`` strips its input, and
    the substring strategy's ``"" in name`` then matches *every* player, so a
    ``" "`` that got past validation would dump the whole 1,408-player universe
    (~470 KB) instead of erroring. Validation strips before length-checking.
    """
    assert client.get("/players", params=params).status_code == 422


def test_players_query_is_echoed_stripped(client):
    """Surrounding whitespace is trimmed, and the echo reports what was searched."""
    body = client.get("/players", params={"query": "  Jannik Sinner  "}).json()
    assert body["query"] == "Jannik Sinner"
    assert body["count"] == 1
    assert body["strategy"] == "exact"


def test_players_unmeasured_player_falls_back_to_baseline_skills(client):
    """A resolvable player with no serve rows gets the surface default, flagged
    by a zero sample size rather than by omission.

    The surface-key assertion is the point: ``skills`` carries the same keys for
    everyone, so a client never has to distinguish "no data" from "key absent".
    Iterating alone would pass vacuously against a handler that omitted the
    defaults entirely.
    """
    body = client.get("/players", params={"query": "Unmeasured Newcomer"}).json()

    assert body["count"] == 1
    player = body["players"][0]
    assert player["player_id"] == "N1"
    assert set(player["skills"]) == config.VALID_SURFACES
    for surface in config.VALID_SURFACES:
        skill = player["skills"][surface]
        assert skill["spw"] == pytest.approx(config.SURFACE_MU[surface])
        assert skill["rpw"] == pytest.approx(1.0 - config.SURFACE_MU[surface])
        assert skill["n_serve_pts"] == 0.0
        assert skill["n_return_pts"] == 0.0


def test_players_response_is_schema_typed(client):
    """Responses parse as the declared Pydantic models (no raw dicts)."""
    HealthResponse.model_validate(client.get("/health").json())
    parsed = PlayerSearchResponse.model_validate(
        client.get("/players", params={"query": "alcaraz"}).json()
    )
    assert all(p.skills for p in parsed.players)


def test_every_route_declares_a_response_model():
    """Each app route carries ``response_model=``, checked by introspection.

    This is the enforceable half of "no raw dicts". A handler that *returns* a
    dict whose keys and types already match its ``response_model`` is
    indistinguishable over HTTP — FastAPI validates and re-serialises it to the
    identical bytes, and with ``extra="forbid"`` any deviation 500s — so that
    part of the convention is style, not behaviour. What is observable, and what
    this pins, is the ``response_model`` itself: drop it and the schema stops
    guarding the response at all.
    """
    from fastapi.routing import APIRoute

    app = create_app(context_factory=lambda: None)
    routes = [r for r in app.routes if isinstance(r, APIRoute)]
    assert {r.path for r in routes} == {
        "/health",
        "/players",
        "/tournaments",
        "/tournaments/upload",
        "/tournaments/{tournament_id}/bracket",
        "/tournaments/{tournament_id}/simulate",
        "/tournaments/{tournament_id}/storybook",
    }
    for route in routes:
        assert route.response_model is not None, f"{route.path} has no response_model"


def test_response_bodies_carry_exactly_the_declared_fields(client):
    """No handler may put a field on the wire that its schema does not declare.

    ``model_validate`` alone cannot enforce this — Pydantic v2 drops undeclared
    keys by default, so a handler returning a raw dict with an extra field would
    still round-trip cleanly. The schemas set ``extra="forbid"`` and this test
    pins the exact key set at every level, which together make "responses are
    typed, no raw dicts" an assertion rather than a convention.
    """
    health = client.get("/health").json()
    assert set(health) == set(HealthResponse.model_fields)

    body = client.get("/players", params={"query": "alcaraz"}).json()
    assert set(body) == set(PlayerSearchResponse.model_fields)
    assert body["players"], "need a populated response for the nested checks"
    for player in body["players"]:
        assert set(player) == set(PlayerSummary.model_fields)
        assert player["skills"], "need populated skills for the nested check"
        for skill in player["skills"].values():
            assert set(skill) == set(SurfaceSkill.model_fields)


# --------------------------------------------------------------------------- #
# CORS
# --------------------------------------------------------------------------- #
def test_cors_allows_a_configured_origin(client):
    origin = config.API_ALLOWED_ORIGINS[0]
    response = client.get("/health", headers={"Origin": origin})
    assert response.headers["access-control-allow-origin"] == origin


def test_cors_preflight_is_answered_for_a_configured_origin(client):
    response = client.options(
        "/players",
        headers={
            "Origin": config.API_ALLOWED_ORIGINS[0],
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert "access-control-allow-origin" in response.headers


def test_cors_rejects_an_unconfigured_origin(client):
    """No wildcard: an origin outside the allow-list gets no CORS grant."""
    response = client.get("/health", headers={"Origin": "https://evil.example"})
    assert response.status_code == 200  # the request itself still succeeds
    assert "access-control-allow-origin" not in response.headers
    assert "*" not in config.API_ALLOWED_ORIGINS


# --------------------------------------------------------------------------- #
# Layering
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "module", ["__init__", "deps", "schemas", "main", "registry"]
)
def test_api_modules_do_not_import_cli(module):
    """The api/ layering rule, asserted rather than assumed.

    ``api/deps.py`` deliberately re-orchestrates the pipeline instead of reusing
    ``cli.simulate_match.build_context``; this is the test that keeps a future
    "just reuse it" from silently reintroducing ``api/ → cli/``.
    """
    from pathlib import Path

    source = Path(f"src/api/{module}.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    assert "import cli" not in code
    assert "from cli" not in code
