"""Tests for ``/tournaments/{id}/simulate`` and the precompute script (T3.3).

Nothing here touches the vendored CSVs or the shipped draws: the app is pointed
at ``tests/fixtures/draws/`` and a **fixture cache directory**, and the
precompute script's own pipeline loader is stubbed out. Two things this file
exists to prove, beyond the happy path:

* **The handler never simulates.** Structurally (the module imports no
  simulation entry point), behaviourally (raising traps on ``monte_carlo`` /
  ``simulate_bracket`` / the point engine, with a control proving the traps are
  live), and by construction — the injected ``ApiContext`` carries
  a stub one-column ``data`` frame and ``estimator=object()``, so any live
  reconciliation would blow up rather than quietly succeed (see the ``context``
  fixture for exactly where).
* **The seam-7 disclosure survives the round trip.** The reconciliation
  ``mode``/``w``, the ``is_forecast`` flag and the plain-language limitation are
  written by the script into the cache file *and* served in the API response,
  and a cache file missing them is rejected rather than published.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import config
import sim.match as match
import sim.reconcile as reconcile
import sim.tournament as tournament
from api.deps import ApiContext
from api.main import create_app
from api.registry import build_registry, cache_path_for
from api.schemas import SimulationMetadata, SimulationPlayer, SimulationResponse
from features.serve import PlayerSkill, SkillTable
from sim.draw import load_draw

# scripts/ is not on pyproject's ``pythonpath = ["src"]`` — same hack
# tests/test_refresh_data.py uses.
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
    """A fixture context, rigged so a handler that tried to simulate fails loudly.

    Both seams a live reconciliation would have to cross are broken on purpose,
    and the *first* one is what actually trips: ``data`` is a one-column stub, so
    ``cli/simulate_match.make_classifier_prob`` raises ``KeyError: 'p1_name'``
    while building its name-keyed feature row — **before** reaching the
    ``estimator.predict_proba`` call at ``cli/simulate_match.py:199``. ``estimator``
    is a bare ``object()`` (no ``predict_proba``) as the backstop behind that,
    for a caller that somehow arrived with a usable frame.
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


@pytest.fixture
def client(context: ApiContext):
    """TestClient over an app reading the *fixture* cache directory."""
    app = create_app(
        context_factory=lambda: context,
        draws_dir=FIXTURE_DRAWS,
        cache_dir=FIXTURE_CACHE,
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def empty_cache_client(context: ApiContext, tmp_path: Path):
    """TestClient over an app whose cache directory is empty."""
    app = create_app(
        context_factory=lambda: context,
        draws_dir=FIXTURE_DRAWS,
        cache_dir=tmp_path,
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def cached_payload() -> dict:
    """The fixture cache file, as raw JSON — the endpoint's expected output."""
    return json.loads((FIXTURE_CACHE / "toy_open_2026.json").read_text())


class StubClassifier:
    """A deterministic ``(a, b, surface) -> P_clf`` stand-in.

    Skews on name order so the simulated field is not a uniform coin flip, and
    counts calls so a test can show the whole-job cache is doing its work.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, player_a: str, player_b: str, surface: str) -> float:
        self.calls.append((player_a, player_b, surface))
        return 0.55 if player_a < player_b else 0.45


# --------------------------------------------------------------------------- #
# GET /tournaments/{id}/simulate — cached results
# --------------------------------------------------------------------------- #
def test_simulate_returns_the_cached_results(client, cached_payload):
    """Acceptance: with a cache file present, the parsed results are returned.

    Field by field against the file itself, not merely a 200: the endpoint's
    entire job is to hand back what the precompute script computed, so any
    divergence — a dropped round, a re-derived probability — is a defect.
    """
    response = client.get("/tournaments/toy_open_2026/simulate")
    assert response.status_code == 200

    body = response.json()
    for field in (
        "tournament_id",
        "name",
        "surface",
        "best_of",
        "final_set_tiebreak",
        "draw_size",
        "round_labels",
        "count",
    ):
        assert body[field] == cached_payload[field], field

    assert body["count"] == 8 == len(body["players"])
    assert body["players"] == cached_payload["players"]

    # Spot-check one entrant end to end rather than trusting the list compare.
    champion = body["players"][0]
    assert champion["player"] == "Bob Baseline"
    assert champion["player_id"] == "B2"
    assert champion["seed"] == 2
    assert champion["titles"] == 27
    assert champion["p_title"] == pytest.approx(0.27)
    assert champion["expected_rounds_won"] == pytest.approx(1.42)
    assert champion["reached"] == {"QF": 100, "SF": 70, "F": 45}
    assert champion["p_reach"]["F"] == pytest.approx(0.45)


def test_simulate_players_are_sorted_by_title_probability(client):
    probs = [p["p_title"] for p in client.get(
        "/tournaments/toy_open_2026/simulate"
    ).json()["players"]]
    assert probs == sorted(probs, reverse=True)


def test_simulate_metadata_makes_the_result_traceable(client, cached_payload):
    """Acceptance: runs, seed, data year — plus mode/w — travel with the numbers."""
    metadata = client.get("/tournaments/toy_open_2026/simulate").json()["metadata"]

    assert metadata["runs"] == 100
    assert metadata["seed"] == 7
    assert metadata["data_through_year"] == 2026
    assert metadata["mode"] == "blend"
    assert metadata["w"] == pytest.approx(0.5)
    assert metadata["workers"] == 1
    assert metadata["estimator_class"] == "StubClassifier"
    assert metadata["generated_at"].startswith("2026-08-03T09:15:00")
    assert metadata["source"] == "toy_open.json"
    assert set(metadata) == set(SimulationMetadata.model_fields)


def test_simulate_response_discloses_the_classifier_limitation(client):
    """The seam-7 requirement, checked on the wire and not only in the file.

    A consumer must be able to tell a demonstration from a forecast **without**
    reading the repository's docs, and without parsing prose: hence a boolean
    flag alongside the sentence, and the reconciliation mode that dilutes the
    defect next to both.
    """
    metadata = client.get("/tournaments/toy_open_2026/simulate").json()["metadata"]

    assert metadata["is_forecast"] is False
    assert metadata["adapter"] == "cli.simulate_match.make_classifier_prob"
    limitation = metadata["classifier_limitation"]
    assert "not a forecast" in limitation.lower()
    assert "20 of its 27" in limitation  # the actual, checkable defect
    assert metadata["mode"] in {"blend", "classifier_anchor"}


def test_simulate_rejects_a_cache_file_missing_its_disclosure(
    context, tmp_path, cached_payload
):
    """Disclosure is structural: strip it and the file stops being servable.

    This is what makes the requirement enforceable rather than a convention —
    ``SimulationMetadata`` gives no field a default, so an older or hand-edited
    cache without the limitation fails validation instead of publishing bare
    probabilities.
    """
    stripped = dict(cached_payload)
    stripped["metadata"] = {
        key: value
        for key, value in cached_payload["metadata"].items()
        if key not in {"classifier_limitation", "is_forecast"}
    }
    (tmp_path / "toy_open_2026.json").write_text(json.dumps(stripped))

    app = create_app(
        context_factory=lambda: context, draws_dir=FIXTURE_DRAWS, cache_dir=tmp_path
    )
    with TestClient(app) as client:
        response = client.get("/tournaments/toy_open_2026/simulate")

    assert response.status_code == 422
    assert response.json()["detail"]["reason"] == "cache_unreadable"


def test_simulate_response_is_schema_typed_and_exact(client):
    body = client.get("/tournaments/toy_open_2026/simulate").json()
    SimulationResponse.model_validate(body)

    assert set(body) == set(SimulationResponse.model_fields)
    for player in body["players"]:
        assert set(player) == set(SimulationPlayer.model_fields)


# --------------------------------------------------------------------------- #
# ?top=N
# --------------------------------------------------------------------------- #
def test_simulate_top_limits_rows_keeping_the_ranking(client, cached_payload):
    body = client.get("/tournaments/toy_open_2026/simulate?top=3").json()

    assert body["count"] == 3
    assert len(body["players"]) == 3
    assert body["players"] == cached_payload["players"][:3]
    # The full field size is still reported, so a client knows it was truncated.
    assert body["draw_size"] == 8


def test_simulate_top_beyond_the_field_returns_everyone(client):
    body = client.get("/tournaments/toy_open_2026/simulate?top=99").json()
    assert body["count"] == 8 == len(body["players"])


def test_simulate_top_must_be_at_least_one(client):
    """T2.5's ``--top 0`` "list everyone" sentinel does not transfer: over HTTP,
    *omitting* the parameter already means "everyone", so 0 is a malformed
    request rather than a second spelling of the default."""
    assert client.get("/tournaments/toy_open_2026/simulate?top=0").status_code == 422


# --------------------------------------------------------------------------- #
# Missing cache
# --------------------------------------------------------------------------- #
def test_simulate_without_a_cache_returns_the_documented_error(empty_cache_client):
    """Acceptance: missing cache → a clear, non-blocking error naming the fix."""
    response = empty_cache_client.get("/tournaments/toy_open_2026/simulate")
    assert response.status_code == 425

    detail = response.json()["detail"]
    assert detail["reason"] == "cache_missing"
    assert detail["tournament_id"] == "toy_open_2026"
    assert "scripts/precompute_sim.py" in detail["command"]
    assert "--draw toy_open_2026" in detail["command"]
    # The message must say why it is not simply computing the answer.
    assert "never run" in detail["message"]


def test_missing_cache_response_starts_no_work(empty_cache_client, monkeypatch):
    """Non-blocking is the point: the 425 path must not kick off a run either.

    A handler that "helpfully" triggered a background Monte Carlo would satisfy
    the status code and violate the rule it exists to enforce.
    """
    for module, name in SIM_TRAPS:
        monkeypatch.setattr(module, name, _trap(name))

    assert empty_cache_client.get(
        "/tournaments/toy_open_2026/simulate"
    ).status_code == 425
    # Still missing afterwards — nothing was generated on the side.
    assert empty_cache_client.get(
        "/tournaments/toy_open_2026/simulate"
    ).status_code == 425


# --------------------------------------------------------------------------- #
# Placeholder draws — T3.3's answer to T3.2's open question
# --------------------------------------------------------------------------- #
def test_simulate_placeholder_draw_returns_409(client):
    """A draw with unfilled slots is refused cleanly, naming every offender."""
    response = client.get("/tournaments/qualifier_open_2026/simulate")
    assert response.status_code == 409

    detail = response.json()["detail"]
    assert detail["reason"] == "draw_not_simulatable"
    assert detail["tournament_id"] == "qualifier_open_2026"
    assert detail["source"] == "qualifier_open.json"
    assert [slot["player"] for slot in detail["placeholder_slots"]] == [
        "Qualifier",
        "Qualifier/Lucky Loser",
    ]
    assert [slot["position"] for slot in detail["placeholder_slots"]] == [2, 6]
    assert "placeholder" in detail["message"]


def test_simulate_409_agrees_with_what_simulate_bracket_refuses(client, skill_table):
    """The 409 tracks T2.2's actual rule rather than a second opinion.

    The endpoint reads ``DrawSlot.is_placeholder`` (T2.1's flag) instead of
    importing ``sim.tournament``'s private refusal, so this test pins the two to
    the same verdict: the draw the API 409s on is exactly the draw the simulator
    itself will not run, and the draw it serves is one the simulator accepts.
    """
    refused = load_draw(FIXTURE_DRAWS / "qualifier_open.json", skill_table)
    with pytest.raises(ValueError, match="placeholder"):
        tournament.monte_carlo(refused, skill_table, StubClassifier(), n_runs=1)

    accepted = load_draw(FIXTURE_DRAWS / "toy_open.json", skill_table)
    tournament.monte_carlo(accepted, skill_table, StubClassifier(), n_runs=1)

    assert client.get("/tournaments/qualifier_open_2026/simulate").status_code == 409
    assert client.get("/tournaments/toy_open_2026/simulate").status_code == 200


def test_placeholder_409_precedes_the_cache_check(context, tmp_path, cached_payload):
    """Simulability is a property of the draw, not of what is on disk.

    Even with a cache file sitting there for it, a placeholder draw answers 409 —
    so the verdict for a given draw never depends on whether someone hand-wrote
    a file the precompute script would have refused to produce.
    """
    payload = dict(cached_payload)
    payload["tournament_id"] = "qualifier_open_2026"
    (tmp_path / "qualifier_open_2026.json").write_text(json.dumps(payload))

    app = create_app(
        context_factory=lambda: context, draws_dir=FIXTURE_DRAWS, cache_dir=tmp_path
    )
    with TestClient(app) as client:
        response = client.get("/tournaments/qualifier_open_2026/simulate")

    assert response.status_code == 409
    assert response.json()["detail"]["reason"] == "draw_not_simulatable"


def test_placeholder_draw_is_still_listed_and_its_bracket_still_served(client):
    """T3.3 answers simulability at ``/simulate`` and changes nothing in T3.2.

    The alternatives were a ``simulatable`` flag on ``/tournaments`` or filtering
    such draws out of the registry; both were rejected (see the endpoint
    docstring), so this pins that T3.2's behaviour — and its own test — still
    holds after T3.3.
    """
    listed = client.get("/tournaments").json()
    assert "qualifier_open_2026" in {t["tournament_id"] for t in listed["tournaments"]}
    assert client.get("/tournaments/qualifier_open_2026/bracket").status_code == 200
    # ...and no new field appeared on the listing to answer it a second way.
    assert set(listed["tournaments"][0]) == {
        "tournament_id",
        "name",
        "surface",
        "best_of",
        "draw_size",
    }


# --------------------------------------------------------------------------- #
# The other error paths
# --------------------------------------------------------------------------- #
def test_simulate_unknown_id_404_groups_ids_by_what_can_be_simulated(client):
    """Unknown id → 404, and the id list does not overstate what is simulatable.

    ``/bracket`` lists every registered id together, correctly — a file that
    failed validation is fetchable there as a 422. Simulation is different in
    *two* ways, so three groups are named apart: a valid placeholder-free draw,
    a valid draw still awaiting entrants, and a file that never loaded. The
    middle group is the one a flat list gets wrong — ``qualifier_open_2026``
    loads and lists perfectly well and still cannot be simulated.
    """
    response = client.get("/tournaments/no_such_event/simulate")
    assert response.status_code == 404

    detail = response.json()["detail"]
    assert "no_such_event" in detail

    simulatable = detail.split("Simulatable ids:")[1].split(".")[0]
    assert "'toy_open_2026'" in simulatable
    assert "qualifier_open_2026" not in simulatable

    awaiting = detail.split("Draws awaiting entrants")[1].split(".")[0]
    assert "'qualifier_open_2026'" in awaiting

    assert "'unresolvable_open'" in detail.split("failed validation")[1]


def test_simulate_invalid_draw_file_returns_422_with_the_problems(client):
    response = client.get("/tournaments/unresolvable_open/simulate")
    assert response.status_code == 422

    detail = response.json()["detail"]
    assert detail["source"] == "unresolvable_open.json"
    assert detail["problems"]
    # The same body /bracket returns for the same file — one shape, not two.
    assert detail == client.get(
        "/tournaments/unresolvable_open/bracket"
    ).json()["detail"]


def test_simulate_unreadable_cache_file_is_422_not_500(context, tmp_path):
    (tmp_path / "toy_open_2026.json").write_text("{ this is not json")

    app = create_app(
        context_factory=lambda: context, draws_dir=FIXTURE_DRAWS, cache_dir=tmp_path
    )
    with TestClient(app) as client:
        response = client.get("/tournaments/toy_open_2026/simulate")

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["reason"] == "cache_unreadable"
    assert "precompute" in detail["message"]


def test_simulate_stale_cache_file_is_rejected(context, tmp_path, cached_payload):
    """A cache describing a different draw than the registry holds is not served."""
    stale = dict(cached_payload, draw_size=16)
    (tmp_path / "toy_open_2026.json").write_text(json.dumps(stale))

    app = create_app(
        context_factory=lambda: context, draws_dir=FIXTURE_DRAWS, cache_dir=tmp_path
    )
    with TestClient(app) as client:
        response = client.get("/tournaments/toy_open_2026/simulate")

    assert response.status_code == 422
    assert response.json()["detail"]["reason"] == "cache_stale"


# --------------------------------------------------------------------------- #
# Proof: the handler never runs a live simulation
# --------------------------------------------------------------------------- #
def _trap(name):
    """A stand-in that fails loudly if a simulation entry point is entered."""

    def _raise(*args, **kwargs):
        raise AssertionError(f"{name} was called")

    return _raise


# Every simulation entry point, patched at the module that **holds the reference
# the caller resolves** — not necessarily the one that defines it. T2.2/T2.3 used
# the same pattern (and the same care) to prove ``outcome_only`` never reaches the
# point engine: ``sim/tournament.py`` imported ``simulate_reconciled_match`` by
# name, so patching ``sim.reconcile``'s copy would silently miss.
SIM_TRAPS = (
    (tournament, "monte_carlo"),
    (tournament, "simulate_bracket"),
    (tournament, "simulate_reconciled_match"),
    (reconcile, "simulate_match_bo3"),
    (reconcile, "simulate_match_bo5"),
    (match, "simulate_set"),
    (match, "simulate_game"),
)


def test_simulate_handler_never_runs_a_live_simulation(client, monkeypatch):
    """Acceptance, proved rather than timed: no simulation runs in the request.

    Fast is not the claim — *never* is. Every simulation entry point raises for
    the duration, and the endpoint still answers in full, repeatedly.
    """
    for module, name in SIM_TRAPS:
        monkeypatch.setattr(module, name, _trap(name))

    for _ in range(3):
        response = client.get("/tournaments/toy_open_2026/simulate")
        assert response.status_code == 200
        assert len(response.json()["players"]) == 8
    assert client.get("/tournaments/toy_open_2026/simulate?top=2").status_code == 200


@pytest.mark.parametrize(
    ("module", "name"), SIM_TRAPS, ids=[name for _, name in SIM_TRAPS]
)
def test_each_simulation_trap_is_live(module, name, monkeypatch, skill_table):
    """Control for the test above — a trap that silently misses would prove nothing.

    Each patched symbol must be the one the rest of the codebase actually
    resolves (patched at the holding module, not the defining one), so patching
    it and running a real simulation must fail. Storybook mode is used because it
    walks the whole stack: bracket → reconciled match → set → game.
    """
    monkeypatch.setattr(module, name, _trap(name))
    draw = load_draw(FIXTURE_DRAWS / "toy_open.json", skill_table)
    if name == "simulate_match_bo5":
        # The fixture draw is best-of-3 and would never reach the bo5 entry point.
        draw = dataclasses.replace(draw, best_of=5, final_set_tiebreak="10pt_at_6_6")

    with pytest.raises(AssertionError, match=f"{name} was called"):
        if name == "monte_carlo":
            tournament.monte_carlo(draw, skill_table, StubClassifier(), n_runs=1)
        else:
            tournament.storybook_run(
                draw, skill_table, StubClassifier(), seed=1, mode="blend"
            )


def test_api_main_imports_no_monte_carlo_entry_point():
    """The structural half: ``api/main.py`` cannot run a Monte Carlo.

    Traps prove the code path is not taken today; this proves there is no path —
    the module holds no reference to the aggregate simulator at all, which is why
    the "cached results only" rule cannot be broken by an accidental edit that
    resolves a name at import time.

    **Narrowed by T3.4, deliberately.** Until then the assertion was that
    ``api/main.py`` referenced *no* simulation entry point whatsoever;
    ``/storybook`` now runs one bracket live, so ``storybook_run`` is imported
    and called here on purpose (the Phase 3 rule bounds the *size* of a live run,
    not its existence — see the endpoint docstring). What must stay untrue is the
    thing the rule actually forbids: a 5,000-run aggregate, or the raw bracket
    loop it is built from, reachable from a request. The positive assertion below
    keeps this test honest — if ``storybook_run`` ever stops being called from
    here, this file should be tightened back, not left permissive.
    """
    source = Path("src/api/main.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    for forbidden in (
        "from sim.match",
        "import sim.match",
        # ...and no call to one, however it might have been imported. (The
        # endpoint docstrings name these functions in prose; a call carries the
        # opening parenthesis.)
        "monte_carlo(",
        "simulate_bracket(",
        "simulate_reconciled_match(",
    ):
        assert forbidden not in code, forbidden

    # The one live entry point T3.4 admits, and nothing broader from its module.
    assert "storybook_run(" in code
    assert "from sim.tournament import StorybookResult, storybook_run" in code


# --------------------------------------------------------------------------- #
# api.registry.cache_path_for
# --------------------------------------------------------------------------- #
def test_cache_path_for_maps_an_id_to_one_file(tmp_path):
    assert cache_path_for("toy_open_2026", tmp_path) == (
        tmp_path / "toy_open_2026.json"
    )


@pytest.mark.parametrize(
    "bad_id", ["../secrets", "a/b", "a\\b", "", ".", "..", "sub/../../x"]
)
def test_cache_path_for_refuses_ids_that_are_not_plain_filenames(bad_id, tmp_path):
    """The id arrives from a URL path, so it must never address another directory."""
    assert cache_path_for(bad_id, tmp_path) is None


# --------------------------------------------------------------------------- #
# TournamentEntry.is_simulatable — the one definition both consumers use
# --------------------------------------------------------------------------- #
def test_entry_is_simulatable_distinguishes_loadable_from_runnable(skill_table):
    """"Valid" and "simulatable" are three states, not two."""
    registry = build_registry(skill_table, FIXTURE_DRAWS)

    runnable = registry.get("toy_open_2026")
    awaiting = registry.get("qualifier_open_2026")
    broken = registry.get("unresolvable_open")

    assert runnable.is_valid and runnable.is_simulatable
    assert runnable.placeholder_slots == ()
    # Loads and lists, but cannot be run — the case a two-state view gets wrong.
    assert awaiting.is_valid and not awaiting.is_simulatable
    assert [slot.player for slot in awaiting.placeholder_slots] == [
        "Qualifier",
        "Qualifier/Lucky Loser",
    ]
    # Never loaded: no slots to judge, and its problems already say why.
    assert not broken.is_valid and not broken.is_simulatable
    assert broken.placeholder_slots == ()


# --------------------------------------------------------------------------- #
# scripts/precompute_sim.py
# --------------------------------------------------------------------------- #
@pytest.fixture
def stub_precompute(monkeypatch, context):
    """Run the script's ``main`` without the vendored pipeline or a real model.

    Only the two expensive/IO-bound seams are replaced — the startup context and
    the classifier adapter. Everything else (registry lookup, ``monte_carlo``,
    payload assembly, the file write) is the real thing.
    """
    classifier = StubClassifier()
    monkeypatch.setattr(precompute_sim, "build_api_context", lambda: context)
    monkeypatch.setattr(precompute_sim, "make_classifier_prob", lambda *a, **k: classifier)
    return classifier


def test_precompute_writes_a_cache_file_with_results_and_metadata(
    stub_precompute, tmp_path
):
    """Acceptance: the script produces a cache file with results + metadata."""
    exit_code = precompute_sim.main(
        [
            "--draw", "toy_open_2026",
            "--runs", "50",
            "--seed", "11",
            "--draws-dir", str(FIXTURE_DRAWS),
            "--cache-dir", str(tmp_path),
        ]
    )
    assert exit_code == 0

    path = tmp_path / "toy_open_2026.json"
    assert path.is_file()
    payload = SimulationResponse.model_validate_json(path.read_text())

    assert payload.tournament_id == "toy_open_2026"
    assert payload.draw_size == 8 == payload.count == len(payload.players)
    assert payload.round_labels == ["QF", "SF", "F"]
    assert sum(player.titles for player in payload.players) == 50
    assert all(player.reached["QF"] == 50 for player in payload.players)
    assert payload.metadata.runs == 50
    assert payload.metadata.seed == 11
    assert payload.metadata.data_through_year == 2026
    assert payload.metadata.estimator_class == "StubClassifier"
    assert payload.metadata.source == "toy_open.json"


def test_precompute_records_the_reconciliation_mode_and_the_limitation(
    stub_precompute, tmp_path
):
    """Seam-7 disclosure lands in the **file**, not only in the HTTP layer."""
    precompute_sim.main(
        [
            "--draw", "toy_open_2026",
            "--runs", "10",
            "--draws-dir", str(FIXTURE_DRAWS),
            "--cache-dir", str(tmp_path),
        ]
    )
    raw = json.loads((tmp_path / "toy_open_2026.json").read_text())["metadata"]

    # The mode defaults to the CLI-scoped mitigation, as the module documents.
    assert raw["mode"] == config.SIM_CLI_RECONCILE_MODE == "blend"
    assert raw["w"] == pytest.approx(config.RECONCILE_BLEND_WEIGHT)
    assert raw["is_forecast"] is False
    assert raw["adapter"] == "cli.simulate_match.make_classifier_prob"
    assert "not a forecast" in raw["classifier_limitation"].lower()


def test_precompute_reconcile_mode_override_is_recorded(stub_precompute, tmp_path):
    precompute_sim.main(
        [
            "--draw", "toy_open_2026",
            "--runs", "10",
            "--reconcile-mode", "classifier_anchor",
            "--draws-dir", str(FIXTURE_DRAWS),
            "--cache-dir", str(tmp_path),
        ]
    )
    raw = json.loads((tmp_path / "toy_open_2026.json").read_text())
    assert raw["metadata"]["mode"] == "classifier_anchor"


def test_precompute_output_is_servable_end_to_end(stub_precompute, tmp_path, context):
    """The script writes what the endpoint reads — one format, proved by use."""
    assert precompute_sim.main(
        [
            "--draw", "toy_open_2026",
            "--runs", "20",
            "--seed", "3",
            "--draws-dir", str(FIXTURE_DRAWS),
            "--cache-dir", str(tmp_path),
        ]
    ) == 0

    app = create_app(
        context_factory=lambda: context, draws_dir=FIXTURE_DRAWS, cache_dir=tmp_path
    )
    with TestClient(app) as client:
        response = client.get("/tournaments/toy_open_2026/simulate")

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["runs"] == 20
    assert body["metadata"]["seed"] == 3
    assert sum(player["titles"] for player in body["players"]) == 20


def test_precompute_is_deterministic_for_a_seed(stub_precompute, tmp_path):
    """CLAUDE.md's determinism rule: same seed → same file (bar the timestamp)."""
    args = [
        "--draw", "toy_open_2026",
        "--runs", "25",
        "--seed", "99",
        "--draws-dir", str(FIXTURE_DRAWS),
        "--cache-dir", str(tmp_path),
    ]
    precompute_sim.main(args)
    first = json.loads((tmp_path / "toy_open_2026.json").read_text())
    precompute_sim.main(args)
    second = json.loads((tmp_path / "toy_open_2026.json").read_text())

    first["metadata"].pop("generated_at")
    second["metadata"].pop("generated_at")
    assert first == second


def test_precompute_refuses_a_placeholder_draw_without_writing(
    stub_precompute, tmp_path, capsys
):
    """T2.2's refusal, surfaced as a message and a non-zero exit — no cache file.

    The same verdict the API gives as a 409, from the same underlying rule.
    """
    exit_code = precompute_sim.main(
        [
            "--draw", "qualifier_open_2026",
            "--runs", "5",
            "--draws-dir", str(FIXTURE_DRAWS),
            "--cache-dir", str(tmp_path),
        ]
    )

    assert exit_code == 1
    assert not list(tmp_path.glob("*.json"))
    output = capsys.readouterr().out
    assert "placeholder" in output
    assert "Qualifier" in output


def test_precompute_unknown_id_groups_ids_the_same_way_the_api_does(
    stub_precompute, tmp_path, capsys
):
    """The script offers as "available" only what it would actually run."""
    exit_code = precompute_sim.main(
        [
            "--draw", "nope",
            "--draws-dir", str(FIXTURE_DRAWS),
            "--cache-dir", str(tmp_path),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    available = output.split("Available draws:")[1].splitlines()[0]
    assert "toy_open_2026" in available
    assert "qualifier_open_2026" not in available
    assert "qualifier_open_2026" in output.split("Draws awaiting entrants")[1]
    assert "failed validation" in output


def test_precompute_invalid_draw_file_reports_the_problems(
    stub_precompute, tmp_path, capsys
):
    exit_code = precompute_sim.main(
        [
            "--draw", "unresolvable_open",
            "--draws-dir", str(FIXTURE_DRAWS),
            "--cache-dir", str(tmp_path),
        ]
    )

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "failed validation" in output
    assert "Nobody Atall" in output  # the validator's own accumulated problems
    assert not list(tmp_path.glob("*.json"))


def test_precompute_payload_reuses_montecarloresult_fields(skill_table):
    """``build_payload`` presents; it does not recompute.

    Every published number is read off ``MonteCarloResult``/``PlayerOutcome`` —
    the same renderer/data separation T2.4 and T2.5 keep — so the API cannot
    disagree with the simulator about its own results.
    """
    draw = load_draw(FIXTURE_DRAWS / "toy_open.json", skill_table)
    result = tournament.monte_carlo(
        draw, skill_table, StubClassifier(), n_runs=30, seed=5, mode="blend"
    )

    payload = precompute_sim.build_payload(
        result,
        data_through_year=2026,
        estimator_class="StubClassifier",
        source="toy_open.json",
    )

    assert payload.round_labels == list(result.round_labels)
    assert payload.metadata.mode == result.mode
    assert payload.metadata.w == result.w
    assert payload.metadata.workers == result.workers
    for served, outcome in zip(payload.players, result.players):
        assert served.position == outcome.position
        assert served.player == outcome.player
        assert served.player_id == outcome.player_id
        assert served.seed == outcome.seed
        assert served.titles == outcome.titles
        assert served.matches_won == outcome.matches_won
        assert served.p_title == outcome.p_title
        assert served.expected_rounds_won == outcome.expected_rounds_won
        assert served.reached == dict(outcome.reached)
        assert served.p_reach == {
            label: outcome.p_reach(label) for label in result.round_labels
        }


def test_precompute_write_cache_refuses_an_unsafe_id(skill_table, tmp_path):
    """A draw file claiming a path-shaped id cannot escape the cache directory."""
    draw = load_draw(FIXTURE_DRAWS / "toy_open.json", skill_table)
    result = tournament.monte_carlo(draw, skill_table, StubClassifier(), n_runs=1)
    payload = precompute_sim.build_payload(
        result, data_through_year=2026, estimator_class="X", source="toy_open.json"
    ).model_copy(update={"tournament_id": "../escaped"})

    with pytest.raises(ValueError, match="cache filename"):
        precompute_sim.write_cache(payload, tmp_path)
