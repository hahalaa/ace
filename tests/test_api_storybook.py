"""Tests for ``GET /tournaments/{id}/storybook`` (T3.4).

Nothing here touches the vendored CSVs, the shipped draws or the real
classifier: the app is pointed at ``tests/fixtures/draws/`` and given a
deterministic stub ``ClassifierProb`` through ``create_app``'s injection seam.

Four things this file exists to prove, beyond the happy path:

* **Determinism survives the API layer.** T2.4 already proved ``storybook_run``
  replays a seed; what is new here is everything wrapped around it — seed
  defaulting, serialisation, a metadata block — any of which could make two
  identical requests differ. So the assertion is on raw response *bytes*, not
  on a parsed subset.
* **One run per request.** The endpoint is allowed to simulate precisely because
  a storybook is one bracket; a handler that quietly ran one per round would
  satisfy every other test here and violate the reason it is allowed at all.
* **The seam-7 disclosure is the same one ``/simulate`` serves**, field for field
  and word for word — inherited from ``api.schemas.ModelDisclosure``, not a
  second format that happens to look similar today.
* **The placeholder refusal is the same check**, not a second opinion: the 409
  body is compared against ``/simulate``'s for the same draw.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import config
import sim.tournament as tournament
from api import main as api_main
from api.deps import ApiContext
from api.main import create_app
from api.schemas import (
    CLASSIFIER_LIMITATION,
    ModelDisclosure,
    SimulationMetadata,
    StorybookMatch,
    StorybookMetadata,
    StorybookPlayerRun,
    StorybookResponse,
    StorybookRound,
)
from features.serve import PlayerSkill, SkillTable

# scripts/ is not on pyproject's ``pythonpath = ["src"]`` — same hack
# tests/test_api_simulate.py uses.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))

import precompute_sim  # noqa: E402

FIXTURE_DRAWS = Path(__file__).parent / "fixtures" / "draws"
FIXTURE_CACHE = Path(__file__).parent / "fixtures" / "cache"

# Entrants of the fixture draws (see tests/test_api_tournaments.py).
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
    """A fixture context: real skill table, stub everything the adapter would use.

    ``data``/``estimator`` are deliberately unusable, which is safe precisely
    because the classifier is injected below — if the injection seam ever stopped
    being honoured and the real adapter were built from this context instead, the
    first simulated match would raise rather than quietly produce numbers.
    """
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
    """A deterministic ``(a, b, surface) -> P_clf`` stand-in.

    Skews on name order so the simulated field is not a uniform coin flip.
    """

    def __call__(self, player_a: str, player_b: str, surface: str) -> float:
        return 0.55 if player_a < player_b else 0.45


def stub_classifier_factory(estimator, data, surface_history, h2h_history):
    """A ``ClassifierFactory`` with ``make_classifier_prob``'s signature.

    A named function rather than a lambda so ``metadata.adapter`` reports
    something a human can look up — the endpoint derives that string from the
    factory that actually ran.
    """
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
    """TestClient over an app with the fixture draws and a stub classifier."""
    with TestClient(make_app(context)) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# Acceptance: a full structured storybook
# --------------------------------------------------------------------------- #
def test_storybook_returns_a_full_structured_story(client):
    """Acceptance: rounds, scorelines and a champion, for the echoed seed."""
    response = client.get("/tournaments/toy_open_2026/storybook?seed=7")
    assert response.status_code == 200

    body = response.json()
    assert body["tournament_id"] == "toy_open_2026"
    assert body["name"] == "Toy Open 2026"
    assert body["surface"] == "Clay"
    assert body["best_of"] == 3
    assert body["final_set_tiebreak"] == "7pt_at_6_6"
    assert body["draw_size"] == 8

    # An 8-draw is three rounds and seven matches — every slot played.
    assert [r["label"] for r in body["rounds"]] == ["QF", "SF", "F"]
    assert [len(r["matches"]) for r in body["rounds"]] == [4, 2, 1]
    assert body["match_count"] == 7 == sum(len(r["matches"]) for r in body["rounds"])
    assert body["metadata"]["seed"] == 7


def test_storybook_every_match_carries_a_real_scoreline(client):
    """The point of the endpoint: played-out matches, not just winners."""
    body = client.get("/tournaments/toy_open_2026/storybook?seed=7").json()

    for storybook_round in body["rounds"]:
        for match in storybook_round["matches"]:
            sets = match["scoreline"].split()
            # best-of-3 → two or three sets, each "x-y" possibly with a "(n)".
            assert 2 <= len(sets) <= 3, match["scoreline"]
            for set_score in sets:
                games = set_score.split("(")[0]
                assert "-" in games and all(g.isdigit() for g in games.split("-"))
            assert match["line"] == (
                f"{match['winner']} def. {match['loser']} {match['scoreline']}"
            )
            assert match["winner"] != match["loser"]


def test_storybook_champion_agrees_with_the_final_and_the_runs(client):
    """The champion is one fact, reported in three places that must not disagree."""
    body = client.get("/tournaments/toy_open_2026/storybook?seed=7").json()

    champion = body["champion"]
    assert champion["player"] == body["rounds"][-1]["matches"][0]["winner"]
    assert champion["player_id"] == TOY_PLAYERS[champion["player"]]

    champions = [run for run in body["runs"] if run["is_champion"]]
    assert len(champions) == 1
    assert champions[0]["player"] == champion["player"]
    assert champions[0]["eliminated_by"] is None
    assert champions[0]["matches_won"] == 3
    assert champions[0]["furthest_round"] == "F"


def test_storybook_runs_cover_every_entrant_best_finish_first(client):
    """``runs`` is T2.4's own per-entrant summary, ordered as T2.4 orders it."""
    body = client.get("/tournaments/toy_open_2026/storybook?seed=7").json()

    runs = body["runs"]
    assert len(runs) == 8
    assert {run["player"] for run in runs} == set(TOY_PLAYERS)
    assert [run["matches_won"] for run in runs] == sorted(
        [run["matches_won"] for run in runs], reverse=True
    )
    # A first-round loser beat nobody; everyone but the champion was eliminated.
    for run in runs:
        assert len(run["beat"]) == run["matches_won"]
        assert (run["eliminated_by"] is None) is run["is_champion"]
    assert sum(run["matches_won"] for run in runs) == 7


def test_storybook_response_is_schema_typed_and_exact(client):
    body = client.get("/tournaments/toy_open_2026/storybook").json()
    StorybookResponse.model_validate(body)

    assert set(body) == set(StorybookResponse.model_fields)
    assert set(body["metadata"]) == set(StorybookMetadata.model_fields)
    assert set(body["rounds"][0]) == set(StorybookRound.model_fields)
    assert set(body["rounds"][0]["matches"][0]) == set(StorybookMatch.model_fields)
    assert set(body["runs"][0]) == set(StorybookPlayerRun.model_fields)


# --------------------------------------------------------------------------- #
# Acceptance: determinism, through the API layer
# --------------------------------------------------------------------------- #
def test_storybook_same_seed_is_byte_identical(client):
    """Acceptance: same seed → identical response.

    Compared as raw bytes rather than as a parsed subset: T2.4 already proved the
    simulator replays a seed, so what this pins is that *the endpoint* adds
    nothing non-reproducible on top of it — no timestamp, no ordering that
    depends on dict iteration, no state carried between requests.
    """
    first = client.get("/tournaments/toy_open_2026/storybook?seed=7")
    second = client.get("/tournaments/toy_open_2026/storybook?seed=7")

    assert first.status_code == second.status_code == 200
    assert first.content == second.content


def test_storybook_same_seed_is_identical_across_app_instances(context):
    """...and not merely within one process's warmed-up state.

    A second app repeats the whole path — fresh registry, fresh adapter, empty
    memo table — so this catches a response that only *happened* to repeat
    because something was cached from the first call.
    """
    with TestClient(make_app(context)) as first_client:
        first = first_client.get("/tournaments/toy_open_2026/storybook?seed=7")
    with TestClient(make_app(context)) as second_client:
        second = second_client.get("/tournaments/toy_open_2026/storybook?seed=7")

    assert first.content == second.content


def test_storybook_different_seeds_tell_different_stories(client):
    """The seed is real, not decoration — otherwise determinism is trivial."""
    baseline = client.get("/tournaments/toy_open_2026/storybook?seed=7").content
    others = [
        client.get(f"/tournaments/toy_open_2026/storybook?seed={seed}").content
        for seed in (1, 2, 3, 4, 5)
    ]
    assert any(other != baseline for other in others)


def test_storybook_default_seed_is_fixed_and_echoed(client):
    """Acceptance: a default seed, echoed — and reproducible on retry.

    A time-derived default would make a bare ``/storybook`` return a different
    story on refresh, which is exactly what the shareable-URL contract rules out.
    So the default is a config constant, and the response says which seed ran.
    """
    default = client.get("/tournaments/toy_open_2026/storybook")
    assert default.status_code == 200
    assert default.json()["metadata"]["seed"] == config.API_STORYBOOK_SEED

    # Echoing it is only useful if the echoed URL reproduces the body.
    explicit = client.get(
        f"/tournaments/toy_open_2026/storybook?seed={config.API_STORYBOOK_SEED}"
    )
    assert explicit.content == default.content
    assert client.get("/tournaments/toy_open_2026/storybook").content == default.content


@pytest.mark.parametrize("bad_seed", ["-1", "abc", str(config.API_STORYBOOK_SEED_MAX + 1)])
def test_storybook_rejects_an_unusable_seed(client, bad_seed):
    """A seed outside the documented range is a malformed request, not a run."""
    assert client.get(
        f"/tournaments/toy_open_2026/storybook?seed={bad_seed}"
    ).status_code == 422


# --------------------------------------------------------------------------- #
# Exactly one live run per request
# --------------------------------------------------------------------------- #
def test_storybook_calls_storybook_run_exactly_once_per_request(client, monkeypatch):
    """The same "exactly once" standard T2.4 holds its own bracket call to.

    A storybook is allowed to run inside a request handler *because* it is one
    bracket. Re-running it per round, or once to find the champion and again to
    fill the rounds, would multiply the cost the Phase 3 rule bounds — and no
    other assertion in this file would notice.

    The spy is installed on ``api.main``'s reference, which is the one the
    handler resolves (T2.2/T2.3's patching discipline), and delegates to the real
    function so the response under test is still a real storybook.
    """
    real = api_main.storybook_run
    calls = []

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(api_main, "storybook_run", spy)

    assert client.get("/tournaments/toy_open_2026/storybook?seed=7").status_code == 200
    assert len(calls) == 1
    assert client.get("/tournaments/toy_open_2026/storybook?seed=8").status_code == 200
    assert len(calls) == 2


def test_storybook_does_not_render_the_cli_text_format(client):
    """Structured JSON, not ``render_storybook``'s plain text.

    The T2.4 renderer composes the same ``StorybookResult`` into a terminal
    layout; this endpoint is the other consumer of that structure. Calling the
    renderer here would bake the CLI's presentation into the HTTP contract.
    """
    source = Path("src/api/main.py").read_text(encoding="utf-8")
    # The docstrings name it in prose; a call carries the opening parenthesis —
    # the same convention tests/test_api_simulate.py uses for its traps.
    assert "render_storybook(" not in source
    assert "import render_storybook" not in source

    body = client.get("/tournaments/toy_open_2026/storybook?seed=7").json()
    assert isinstance(body["rounds"], list)  # structure, not a rendered blob


def test_the_classifier_adapter_is_built_once_at_startup(context):
    """Startup state, like the context and the registry — never per request.

    Per-request construction would also throw away the adapter's memo table,
    which is what keeps a repeat matchup from re-paying ``predict_proba``.
    """
    calls = []

    def counting_factory(estimator, data, surface_history, h2h_history):
        calls.append(1)
        return StubClassifier()

    app = make_app(context, factory=counting_factory)
    assert calls == []  # constructing the app builds nothing

    with TestClient(app) as client:
        for seed in range(4):
            assert client.get(
                f"/tournaments/toy_open_2026/storybook?seed={seed}"
            ).status_code == 200

    assert len(calls) == 1


# --------------------------------------------------------------------------- #
# Seam-7 disclosure — the same block /simulate serves
# --------------------------------------------------------------------------- #
def test_storybook_discloses_the_classifier_limitation(client):
    """The seam-7 requirement, live rather than cached — same fields, same words."""
    metadata = client.get("/tournaments/toy_open_2026/storybook").json()["metadata"]

    assert metadata["is_forecast"] is False
    assert metadata["classifier_limitation"] == CLASSIFIER_LIMITATION
    assert "not a forecast" in metadata["classifier_limitation"].lower()
    assert "20 of its 27" in metadata["classifier_limitation"]
    assert metadata["adapter"].endswith("stub_classifier_factory")
    assert metadata["data_through_year"] == 2026
    assert metadata["estimator_class"] == "StubClassifier"
    assert metadata["source"] == "toy_open.json"


def test_storybook_disclosure_is_the_same_shape_as_simulate(client):
    """Not a parallel format: both metadata blocks are ``ModelDisclosure``.

    The shared fields are inherited, so the two endpoints cannot drift into
    disclosing different things — and a consumer that learned to read one
    response's disclosure has learned to read the other's.

    **Inheritance is asserted directly, and the field-set check below is not a
    substitute for it.** A ``StorybookMetadata`` that copied all eight fields
    verbatim onto a plain ``BaseModel`` satisfies every field-name and every
    field-value assertion in this test while restoring exactly the drift risk
    the shared base exists to remove — verified by mutation, which the field-set
    check alone did not catch. So the structural claim is pinned structurally.
    """
    assert issubclass(StorybookMetadata, ModelDisclosure)
    assert issubclass(SimulationMetadata, ModelDisclosure)

    disclosure_fields = set(ModelDisclosure.model_fields)
    assert disclosure_fields <= set(StorybookMetadata.model_fields)

    story = client.get("/tournaments/toy_open_2026/storybook").json()["metadata"]
    cached = client.get("/tournaments/toy_open_2026/simulate").json()["metadata"]

    assert disclosure_fields <= set(story)
    assert disclosure_fields <= set(cached)
    for field in ("classifier_limitation", "is_forecast", "mode", "w"):
        assert story[field] == cached[field], field


def test_storybook_uses_the_cli_scoped_reconciliation_mode(client):
    """T3.3's choice, inherited: ``blend``, because the adapter is the same one.

    ``config.RECONCILE_MODE`` (``classifier_anchor``) would make the flattened
    ``P_clf`` the target every simulated match is solved to reproduce.
    """
    metadata = client.get("/tournaments/toy_open_2026/storybook").json()["metadata"]

    assert metadata["mode"] == config.SIM_CLI_RECONCILE_MODE == "blend"
    assert metadata["mode"] != config.RECONCILE_MODE
    assert metadata["w"] == pytest.approx(config.RECONCILE_BLEND_WEIGHT)


# --------------------------------------------------------------------------- #
# The adapter the shipped server actually resolves
# --------------------------------------------------------------------------- #
def test_config_default_resolves_to_the_same_adapter_precompute_uses():
    """The live endpoint and the offline cache publish the *same* model.

    ``api/`` may not import ``cli/``, so the default is late-bound through
    ``config.API_CLASSIFIER_ADAPTER``; this pins that indirection to the adapter
    ``scripts/precompute_sim.py`` names in its cache files, so ``/simulate`` and
    ``/storybook`` cannot end up disclosing different models.
    """
    factory = api_main.resolve_classifier_factory()
    assert api_main.adapter_name(factory) == precompute_sim.ADAPTER
    assert precompute_sim.make_classifier_prob is factory


@pytest.mark.parametrize("bad_spec", ["no_colon", ":make_classifier_prob", "mod:"])
def test_resolve_classifier_factory_rejects_a_malformed_spec(bad_spec):
    with pytest.raises(ValueError, match="module.:.attribute"):
        api_main.resolve_classifier_factory(bad_spec)


def test_resolve_classifier_factory_fails_loudly_on_a_missing_attribute():
    """A misconfigured adapter must break at startup, not on the first request."""
    with pytest.raises(ImportError, match="no attribute"):
        api_main.resolve_classifier_factory("cli.simulate_match:not_a_function")


# --------------------------------------------------------------------------- #
# Failures — the same gate /simulate uses
# --------------------------------------------------------------------------- #
def test_storybook_placeholder_draw_returns_409(client):
    """A draw with unfilled slots is refused cleanly, naming every offender."""
    response = client.get("/tournaments/qualifier_open_2026/storybook")
    assert response.status_code == 409

    detail = response.json()["detail"]
    assert detail["reason"] == "draw_not_simulatable"
    assert detail["tournament_id"] == "qualifier_open_2026"
    assert [slot["player"] for slot in detail["placeholder_slots"]] == [
        "Qualifier",
        "Qualifier/Lucky Loser",
    ]


def test_storybook_409_is_the_same_check_and_body_as_simulate(client):
    """Acceptance: T3.3's ``is_simulatable``, reused — not a second placeholder scan.

    Byte-for-byte the same detail body from both endpoints, because both go
    through ``_simulatable_entry`` and the verdict is
    ``TournamentEntry.is_simulatable``. A duplicated check could pass the test
    above and still drift.
    """
    story = client.get("/tournaments/qualifier_open_2026/storybook")
    cached = client.get("/tournaments/qualifier_open_2026/simulate")

    assert story.status_code == cached.status_code == 409
    assert story.json()["detail"] == cached.json()["detail"]


def test_storybook_409_refuses_exactly_what_storybook_run_refuses(
    client, skill_table, context
):
    """The 409 tracks T2.2's actual rule, checked against the simulator itself."""
    from sim.draw import load_draw

    refused = load_draw(FIXTURE_DRAWS / "qualifier_open.json", skill_table)
    with pytest.raises(ValueError, match="placeholder"):
        tournament.storybook_run(refused, skill_table, StubClassifier(), seed=1)

    accepted = load_draw(FIXTURE_DRAWS / "toy_open.json", skill_table)
    tournament.storybook_run(accepted, skill_table, StubClassifier(), seed=1)

    assert client.get("/tournaments/qualifier_open_2026/storybook").status_code == 409
    assert client.get("/tournaments/toy_open_2026/storybook").status_code == 200


def test_storybook_unknown_id_404_matches_simulate(client):
    story = client.get("/tournaments/no_such_event/storybook")
    cached = client.get("/tournaments/no_such_event/simulate")

    assert story.status_code == cached.status_code == 404
    assert story.json()["detail"] == cached.json()["detail"]
    assert "'toy_open_2026'" in story.json()["detail"]


def test_storybook_invalid_draw_file_422_matches_simulate(client):
    story = client.get("/tournaments/unresolvable_open/storybook")
    cached = client.get("/tournaments/unresolvable_open/simulate")

    assert story.status_code == cached.status_code == 422
    assert story.json()["detail"] == cached.json()["detail"]
    assert story.json()["detail"]["problems"]


# --------------------------------------------------------------------------- #
# The endpoint presents; it does not re-derive
# --------------------------------------------------------------------------- #
def test_storybook_response_reuses_storybookresult_fields(client, skill_table, context):
    """Every published fact is read off T2.4's structures, not recomputed.

    Same renderer/data separation ``scripts/precompute_sim.build_payload`` keeps
    for a Monte Carlo result: run the simulator directly with the same inputs and
    the response must agree with it field for field.
    """
    from sim.draw import load_draw

    draw = load_draw(FIXTURE_DRAWS / "toy_open.json", skill_table)
    expected = tournament.storybook_run(
        draw,
        skill_table,
        StubClassifier(),
        seed=7,
        mode=config.SIM_CLI_RECONCILE_MODE,
        w=config.RECONCILE_BLEND_WEIGHT,
    )

    body = client.get("/tournaments/toy_open_2026/storybook?seed=7").json()

    assert body["match_count"] == len(expected.matches)
    assert body["champion"]["player"] == expected.champion.player
    assert body["champion"]["seed"] == expected.champion_seed
    assert body["metadata"]["mode"] == expected.mode
    assert body["metadata"]["w"] == pytest.approx(expected.w)
    assert body["metadata"]["seed"] == expected.seed

    served_matches = [m for r in body["rounds"] for m in r["matches"]]
    for served, played in zip(served_matches, expected.matches):
        assert served["round_label"] == played.round_label
        assert served["match_index"] == played.match_index
        assert served["winner"] == played.winner
        assert served["loser"] == played.loser
        assert served["winner_seed"] == played.winner_seed
        assert served["loser_seed"] == played.loser_seed
        assert served["scoreline"] == played.scoreline
        assert served["line"] == played.line

    for served, run in zip(body["runs"], expected.runs):
        assert served["position"] == run.position
        assert served["player"] == run.player
        assert served["player_id"] == run.player_id
        assert served["seed"] == run.seed
        assert served["is_champion"] == run.is_champion
        assert served["furthest_round"] == run.furthest_round
        assert served["matches_won"] == run.matches_won
        assert served["beat"] == list(run.beat)
        assert served["eliminated_by"] == run.eliminated_by


def test_storybook_body_is_json_serialisable_and_stable(client):
    """A shareable link means the body survives a round trip unchanged."""
    raw = client.get("/tournaments/toy_open_2026/storybook?seed=7").content
    parsed = json.loads(raw)
    assert StorybookResponse.model_validate(parsed).metadata.seed == 7
