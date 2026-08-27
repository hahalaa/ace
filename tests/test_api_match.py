"""Tests for POST /match/simulate (single-match live simulation).

Like ``test_api_storybook.py``, this never loads the vendored data: the app is
given a fixture ``ApiContext`` (a real toy skill table over unusable data) and a
deterministic stub ``ClassifierProb`` through ``create_app``'s injection seams,
so the endpoint's own logic, resolution, aggregation, disclosure, rate limiting
is what is under test.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import config
from api.deps import ApiContext
from api.main import create_app
from features.serve import PlayerSkill, SkillTable

FIXTURE_DRAWS = Path(__file__).parent / "fixtures" / "draws"
FIXTURE_CACHE = Path(__file__).parent / "fixtures" / "cache"

TOY_PLAYERS = {
    "Alice Ace": "A1",
    "Bob Baseline": "B2",
    "Cara Chip": "C3",
    "Dan Drop": "D4",
}


@pytest.fixture
def skill_table() -> SkillTable:
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
        skills, dict(TOY_PLAYERS), dict(config.SURFACE_MU),
        cutoff=pd.Timestamp("2026-01-01"),
    )


@pytest.fixture
def context(skill_table: SkillTable) -> ApiContext:
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
        return 0.6 if player_a < player_b else 0.4


def stub_classifier_factory(estimator, data, surface_history, h2h_history):
    return StubClassifier()


def make_app(context: ApiContext, factory=stub_classifier_factory):
    return create_app(
        context_factory=lambda: context,
        draws_dir=FIXTURE_DRAWS,
        cache_dir=FIXTURE_CACHE,
        classifier_factory=factory,
    )


@pytest.fixture
def client(context: ApiContext):
    with TestClient(make_app(context)) as test_client:
        yield test_client


def _body(**overrides):
    body = {
        "player_a": "Alice Ace",
        "player_b": "Bob Baseline",
        "surface": "Hard",
        "best_of": 5,
        "seed": 7,
    }
    body.update(overrides)
    return body


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #
def test_returns_a_full_structured_result(client):
    r = client.post("/match/simulate", json=_body())
    assert r.status_code == 200
    j = r.json()
    assert j["player_a"] == "Alice Ace"
    assert j["player_b"] == "Bob Baseline"
    assert j["surface"] == "Hard"
    assert j["best_of"] == 5
    assert j["final_set_rule"] == config.SIM_CLI_FINAL_SET_RULE
    assert 0.0 <= j["win_prob_a"] <= 1.0
    assert j["win_prob_a"] + j["win_prob_b"] == pytest.approx(1.0)


def test_distributions_are_present_and_consistent(client):
    j = client.post("/match/simulate", json=_body()).json()
    runs = j["metadata"]["runs"]
    assert runs == config.API_MATCH_SIM_RUNS
    assert sum(o["count"] for o in j["set_scores"]) == runs
    for o in j["set_scores"]:
        assert max(o["sets_a"], o["sets_b"]) == 3  # best-of-5 winner
    tg = j["total_games"]
    assert sum(c for _, c in tg["distribution"]) == runs
    assert tg["minimum"] <= tg["mean"] <= tg["maximum"]


def test_best_of_three_uses_three_set_scores(client):
    j = client.post("/match/simulate", json=_body(best_of=3)).json()
    assert j["best_of"] == 3
    for o in j["set_scores"]:
        assert max(o["sets_a"], o["sets_b"]) == 2


def test_a_fragment_name_resolves_to_the_canonical_player(client):
    # "Alice" is a substring match for "Alice Ace".
    j = client.post("/match/simulate", json=_body(player_a="Alice")).json()
    assert j["player_a"] == "Alice Ace"


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_same_seed_is_byte_identical(client):
    a = client.post("/match/simulate", json=_body(seed=42))
    b = client.post("/match/simulate", json=_body(seed=42))
    assert a.json() == b.json()


def test_different_seeds_can_differ(client):
    a = client.post("/match/simulate", json=_body(seed=1)).json()
    b = client.post("/match/simulate", json=_body(seed=2)).json()
    # Aggregates over 3,000 runs of different seeds should not be identical.
    assert a["set_scores"] != b["set_scores"] or a["win_prob_a"] != b["win_prob_a"]


def test_default_seed_is_used_and_echoed(client):
    j = client.post("/match/simulate", json=_body(seed=None)).json()
    assert j["metadata"]["seed"] == config.MC_SEED


# --------------------------------------------------------------------------- #
# Disclosure
# --------------------------------------------------------------------------- #
def test_discloses_the_model_limitation(client):
    meta = client.post("/match/simulate", json=_body()).json()["metadata"]
    assert meta["is_forecast"] is False
    assert meta["mode"] == config.SIM_CLI_RECONCILE_MODE
    assert meta["classifier_limitation"]  # non-empty required field
    assert meta["classifier_limitation_detail"]
    assert "betting" in meta["classifier_limitation"].lower()
    assert meta["estimator_class"] == "StubClassifier"
    assert meta["adapter"].endswith("stub_classifier_factory")


def test_response_body_has_exactly_the_declared_keys(client):
    from api.schemas import MatchSimulateResponse

    j = client.post("/match/simulate", json=_body()).json()
    # extra="forbid" round-trip: a leaked field would fail validation.
    MatchSimulateResponse.model_validate(j)
    assert set(j) == set(MatchSimulateResponse.model_fields)


# --------------------------------------------------------------------------- #
# Player resolution failures, 422, never a silent default
# --------------------------------------------------------------------------- #
def test_unknown_player_is_422(client):
    r = client.post("/match/simulate", json=_body(player_a="Zzz Nobody"))
    assert r.status_code == 422
    assert r.json()["detail"]["reason"] == "player_not_found"


def test_ambiguous_player_is_422_with_candidates(client):
    # Every toy player's surname starts with a distinct letter, but a bare
    # first-initial-style query can match several. "a" substring-matches
    # "Alice Ace", "Bob Baseline", "Cara Chip", "Dan Drop" (all contain 'a').
    r = client.post("/match/simulate", json=_body(player_a="a"))
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["reason"] == "player_ambiguous"
    assert len(detail["candidates"]) > 1


# --------------------------------------------------------------------------- #
# Request validation, closed value domains
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [{"surface": "Carpet"}, {"best_of": 4}, {"seed": -1}])
def test_invalid_request_is_422(client, bad):
    r = client.post("/match/simulate", json=_body(**bad))
    assert r.status_code == 422


def test_seed_over_the_cap_is_422(client):
    r = client.post("/match/simulate", json=_body(seed=config.API_MATCH_SEED_MAX + 1))
    assert r.status_code == 422
    assert r.json()["detail"]["reason"] == "seed_out_of_range"


# --------------------------------------------------------------------------- #
# Rate limiting + security headers
# --------------------------------------------------------------------------- #
def test_rate_limiter_engages(context):
    limit, _ = config.API_MATCH_RATE_LIMIT.split("/")
    with TestClient(make_app(context)) as client:
        last = None
        for _ in range(int(limit) + 1):
            last = client.post("/match/simulate", json=_body())
        assert last.status_code == 429
        assert last.json()["detail"]["reason"] == "rate_limited"
        assert "Retry-After" in last.headers


def test_security_headers_cover_the_endpoint(client):
    r = client.post("/match/simulate", json=_body())
    headers = {k.lower() for k in r.headers}
    assert "content-security-policy" in headers
    assert "x-content-type-options" in headers
