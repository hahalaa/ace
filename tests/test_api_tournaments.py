"""Tests for the tournament endpoints and draw registry.

Everything runs against a **fixture draws directory** (``tests/fixtures/draws/``)
and a toy skill table, never ``data/draws/`` and never the vendored CSVs, so
the suite stays fast and does not move when a real draw file is added or a
season is re-vendered. The fixture directory holds one of each case that
matters:

  * ``toy_open.json``, valid, fully resolvable, seeded, no placeholders;
  * ``qualifier_open.json``, valid **with placeholder entrants**;
  * ``malformed_open.json``, not valid JSON at all;
  * ``unresolvable_open.json``, valid JSON, two unknown entrant names *and* a
    bad ``best_of``, so the whole accumulated problem list can be checked.

The placeholder case is pinned deliberately rather than left to happen by
accident: ``/bracket`` calls ``load_draw`` and never ``simulate_bracket``, so
the simulator's placeholder refusal does not apply here and such a draw is
legitimately listed and legitimately served. The simulability question is
answered at ``/simulate``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main as main
import config
from api.deps import ApiContext
from api.main import create_app
from api.registry import build_registry
from api.schemas import (
    BracketResponse,
    BracketSlot,
    InvalidDraw,
    TournamentListResponse,
    TournamentSummary,
)
from features.serve import PlayerSkill, SkillTable
from sim.draw import DrawValidationError, load_draw

FIXTURE_DRAWS = Path(__file__).parent / "fixtures" / "draws"

# The entrants the fixture draw files are written against.
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
    """A fixture context, the same shape ``tests/test_api.py`` injects."""
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
    """TestClient over an app pointed at the fixture draws directory."""
    app = create_app(context_factory=lambda: context, draws_dir=FIXTURE_DRAWS)
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# GET /tournaments
# --------------------------------------------------------------------------- #
def test_tournaments_lists_all_valid_draw_files(client):
    """Acceptance: every loadable draw file appears, with the five ticket fields."""
    response = client.get("/tournaments")
    assert response.status_code == 200

    body = response.json()
    assert body["count"] == 2
    assert {t["tournament_id"] for t in body["tournaments"]} == {
        "toy_open_2026",
        "qualifier_open_2026",
    }

    toy = next(t for t in body["tournaments"] if t["tournament_id"] == "toy_open_2026")
    assert toy == {
        "tournament_id": "toy_open_2026",
        "name": "Toy Open 2026",
        "surface": "Clay",
        "best_of": 3,
        "draw_size": 8,
    }


def test_tournaments_keys_on_the_files_tournament_id_not_its_filename(client):
    """The listed id is the draw's own ``tournament_id``, which differs from the stem.

    The fixture filenames (``toy_open.json``) are deliberately *not* the ids
    (``toy_open_2026``), so a registry that keyed on the filename would fail
    here instead of passing by coincidence.
    """
    ids = {t["tournament_id"] for t in client.get("/tournaments").json()["tournaments"]}
    stems = {path.stem for path in FIXTURE_DRAWS.glob("*.json")}
    assert ids.isdisjoint(stems)


def test_tournaments_lists_bad_files_instead_of_skipping_or_failing(client):
    """One malformed file degrades to an error row; the listing still succeeds.

    Both halves matter: the endpoint must not 500 because of an unrelated file,
    and the bad file must not silently vanish, a hand-entered draw that
    disappears with no signal is the failure mode this list exists to prevent.
    """
    body = client.get("/tournaments").json()

    # A bad file is not counted among the usable tournaments...
    assert body["count"] == 2
    assert len(body["tournaments"]) == 2
    # ...but it is still reported, addressably, with its problems.
    assert {i["tournament_id"] for i in body["invalid"]} == {
        "malformed_open",
        "unresolvable_open",
    }
    for invalid in body["invalid"]:
        assert invalid["problems"], "an invalid entry must say what is wrong"
        assert invalid["source"].endswith(".json")


def test_tournaments_invalid_entries_are_keyed_by_filename_stem(client):
    """A failed file is addressed by its stem, since its own id is untrustworthy.

    ``unresolvable_open.json`` *does* carry ``"tournament_id":
    "unresolvable_open_2026"``, but the file never validated, so that value is
    not used, the stem is, and it is what ``/bracket`` accepts.
    """
    body = client.get("/tournaments").json()
    invalid = next(
        i for i in body["invalid"] if i["source"] == "unresolvable_open.json"
    )
    raw_id = json.loads(
        (FIXTURE_DRAWS / "unresolvable_open.json").read_text()
    )["tournament_id"]

    assert raw_id == "unresolvable_open_2026"
    assert invalid["tournament_id"] == "unresolvable_open"
    assert client.get(f"/tournaments/{invalid['tournament_id']}/bracket").status_code == 422


def test_tournaments_malformed_json_is_reported_as_such(client):
    body = client.get("/tournaments").json()
    invalid = next(i for i in body["invalid"] if i["source"] == "malformed_open.json")
    assert len(invalid["problems"]) == 1
    assert "not valid JSON" in invalid["problems"][0]


def test_tournaments_surfaces_the_whole_validator_problem_list(client):
    """Not just the first problem, the validator accumulates them and so does the API."""
    body = client.get("/tournaments").json()
    invalid = next(
        i for i in body["invalid"] if i["source"] == "unresolvable_open.json"
    )

    assert len(invalid["problems"]) >= 2
    joined = " ".join(invalid["problems"])
    assert "did not resolve" in joined
    assert "Nobody Atall" in joined and "Ghost Player" in joined  # *all* bad names
    assert "best_of" in joined


def test_tournaments_response_is_schema_typed(client):
    parsed = TournamentListResponse.model_validate(client.get("/tournaments").json())
    assert parsed.tournaments and parsed.invalid


def test_tournaments_body_carries_exactly_the_declared_fields(client):
    """``extra="forbid"`` plus an exact key-set check, as the API established."""
    body = client.get("/tournaments").json()
    assert set(body) == set(TournamentListResponse.model_fields)
    for tournament in body["tournaments"]:
        assert set(tournament) == set(TournamentSummary.model_fields)
    for invalid in body["invalid"]:
        assert set(invalid) == set(InvalidDraw.model_fields)


# --------------------------------------------------------------------------- #
# GET /tournaments/{id}/bracket
# --------------------------------------------------------------------------- #
def test_bracket_returns_the_resolved_draw(client):
    """Acceptance: positions, players, resolved ids and seeds."""
    response = client.get("/tournaments/toy_open_2026/bracket")
    assert response.status_code == 200

    body = response.json()
    assert body["tournament_id"] == "toy_open_2026"
    assert body["name"] == "Toy Open 2026"
    assert body["surface"] == "Clay"
    assert body["best_of"] == 3
    assert body["final_set_tiebreak"] == "7pt_at_6_6"
    assert body["draw_size"] == 8

    slots = body["slots"]
    assert len(slots) == body["draw_size"]
    assert [slot["position"] for slot in slots] == list(range(1, 9))
    # Every entrant resolved to the canonical join key.
    assert all(slot["player_id"] == TOY_PLAYERS[slot["player"]] for slot in slots)
    assert not any(slot["is_placeholder"] for slot in slots)


def test_bracket_carries_seeds_per_slot(client):
    """Seeds ride on the slot they belong to; unseeded slots read ``None``."""
    slots = client.get("/tournaments/toy_open_2026/bracket").json()["slots"]
    seeds = {slot["player"]: slot["seed"] for slot in slots}

    assert seeds["Alice Ace"] == 1
    assert seeds["Bob Baseline"] == 2
    assert seeds["Cara Chip"] == 3
    assert seeds["Dan Drop"] is None
    # No seed is dropped in translation.
    assert sum(1 for seed in seeds.values() if seed is not None) == 3


def test_bracket_matches_load_draw_exactly(client, skill_table):
    """The response is ``load_draw``'s own output, not a second parse of the file.

    Reading the file through the draw loader directly and comparing field by field is what
    makes "reuse, don't reinvent" checkable: any divergence means the endpoint
    grew a parser of its own.
    """
    draw = load_draw(FIXTURE_DRAWS / "toy_open.json", skill_table)
    body = client.get("/tournaments/toy_open_2026/bracket").json()

    assert body["tournament_id"] == draw.tournament_id
    assert body["name"] == draw.name
    assert body["surface"] == draw.surface
    assert body["best_of"] == draw.best_of
    assert body["final_set_tiebreak"] == draw.final_set_tiebreak
    assert body["draw_size"] == draw.draw_size
    assert body["slots"] == [
        {
            "position": slot.position,
            "player": slot.player,
            "is_placeholder": slot.is_placeholder,
            "player_id": slot.player_id,
            "seed": draw.seed_of(slot),
        }
        for slot in draw.bracket
    ]


def test_bracket_unknown_id_returns_404_naming_known_ids(client):
    """Acceptance: unknown id → 404, with a message that helps."""
    response = client.get("/tournaments/no_such_event/bracket")
    assert response.status_code == 404

    detail = response.json()["detail"]
    assert "no_such_event" in detail
    assert "toy_open_2026" in detail  # the ids that do exist


def test_bracket_invalid_draw_file_returns_422_with_the_validator_problems(client):
    """Acceptance: an invalid draw file gives an informative error, not a 500."""
    response = client.get("/tournaments/unresolvable_open/bracket")
    assert response.status_code == 422

    detail = response.json()["detail"]
    assert detail["source"] == "unresolvable_open.json"
    assert detail["tournament_id"] == "unresolvable_open"
    assert len(detail["problems"]) == len(
        client.get("/tournaments").json()["invalid"][-1]["problems"]
    )
    assert str(len(detail["problems"])) in detail["message"]


def test_bracket_422_problems_are_the_validators_own(client, skill_table):
    """The published problems are ``DrawValidationError.problems``, verbatim."""
    with pytest.raises(DrawValidationError) as excinfo:
        load_draw(FIXTURE_DRAWS / "unresolvable_open.json", skill_table)

    detail = client.get("/tournaments/unresolvable_open/bracket").json()["detail"]
    assert detail["problems"] == list(excinfo.value.problems)


def test_bracket_response_is_schema_typed_and_exact(client):
    body = client.get("/tournaments/toy_open_2026/bracket").json()
    BracketResponse.model_validate(body)
    assert set(body) == set(BracketResponse.model_fields)
    for slot in body["slots"]:
        assert set(slot) == set(BracketSlot.model_fields)


# --------------------------------------------------------------------------- #
# Placeholder entrants, pinned behaviour, not an accident (see module docstring)
# --------------------------------------------------------------------------- #
def test_placeholder_draw_is_listed_and_its_bracket_succeeds(client):
    """A draw with ``Qualifier`` slots loads, so it is listed and served.

    ``/bracket`` calls ``load_draw`` and never ``simulate_bracket``, so the bracket sim's
    placeholder refusal is not in this code path at all. Placeholders resolve to
    the default skill profile exactly as everywhere else in the codebase. The
    registry deliberately does not filter such a draw out or pre-judge whether
    it is simulatable, that is `/simulate`'s decision, and this test is here so `/simulate`
    changes the behaviour on purpose rather than by surprise.
    """
    listed = client.get("/tournaments").json()
    assert "qualifier_open_2026" in {
        t["tournament_id"] for t in listed["tournaments"]
    }
    assert "qualifier_open_2026" not in {i["tournament_id"] for i in listed["invalid"]}

    response = client.get("/tournaments/qualifier_open_2026/bracket")
    assert response.status_code == 200

    slots = response.json()["slots"]
    placeholders = [slot for slot in slots if slot["is_placeholder"]]
    assert [slot["player"] for slot in placeholders] == [
        "Qualifier",
        "Qualifier/Lucky Loser",
    ]
    # Flagged and id-less, but present: the bracket keeps all 8 positions.
    assert len(slots) == 8
    assert all(slot["player_id"] is None for slot in placeholders)


# --------------------------------------------------------------------------- #
# Registry: startup scan, and the cases the endpoints cannot reach
# --------------------------------------------------------------------------- #
def test_registry_is_scanned_once_at_startup_not_per_request(context, monkeypatch):
    """The draws directory is scanned once per app, like the context."""
    calls: list[Path] = []
    real_build = main.build_registry

    def counting_build(skill_table, draws_dir):
        calls.append(Path(draws_dir))
        return real_build(skill_table, draws_dir)

    monkeypatch.setattr(main, "build_registry", counting_build)

    app = create_app(context_factory=lambda: context, draws_dir=FIXTURE_DRAWS)
    assert calls == []  # constructing the app scans nothing

    with TestClient(app) as client:
        first = app.state.registry
        for _ in range(5):
            assert client.get("/tournaments").status_code == 200
            assert (
                client.get("/tournaments/toy_open_2026/bracket").status_code == 200
            )
        assert app.state.registry is first  # same object throughout

    assert calls == [FIXTURE_DRAWS]


def test_importing_main_scans_no_draws():
    """The module-level ``app`` must not scan the draws directory at import time."""
    assert getattr(main.app.state, "registry", None) is None


def test_build_registry_tolerates_a_missing_directory(skill_table, tmp_path):
    """A missing draws directory is an empty catalogue, not a startup crash."""
    registry = build_registry(skill_table, tmp_path / "not_here")
    assert registry.entries == ()
    assert registry.valid == () and registry.invalid == ()
    assert registry.get("anything") is None


def test_build_registry_ignores_non_json_files(skill_table, tmp_path):
    shutil.copy(FIXTURE_DRAWS / "toy_open.json", tmp_path / "toy_open.json")
    (tmp_path / "README.md").write_text("not a draw", encoding="utf-8")

    registry = build_registry(skill_table, tmp_path)
    assert registry.ids() == ("toy_open_2026",)


def test_build_registry_flags_a_duplicate_tournament_id(skill_table, tmp_path):
    """Two files claiming one id: the second is an error, not a silent shadow.

    Letting the last file win would serve one of two indistinguishable draws
    with nothing in the response saying a choice was made.
    """
    shutil.copy(FIXTURE_DRAWS / "toy_open.json", tmp_path / "a_toy.json")
    shutil.copy(FIXTURE_DRAWS / "toy_open.json", tmp_path / "b_toy.json")

    registry = build_registry(skill_table, tmp_path)

    assert len(registry.valid) == 1
    assert registry.valid[0].source == "a_toy.json"  # first in filename order wins
    assert len(registry.invalid) == 1
    duplicate = registry.invalid[0]
    assert duplicate.tournament_id == "b_toy"  # still addressable
    assert "duplicate tournament id" in duplicate.problems[0]
    assert "a_toy.json" in duplicate.problems[0]


def test_build_registry_reports_an_unreadable_file(skill_table, tmp_path):
    """A path that cannot be read at all is an error entry, not an exception."""
    (tmp_path / "adirectory.json").mkdir()

    registry = build_registry(skill_table, tmp_path)
    assert len(registry.invalid) == 1
    assert "could not be read" in registry.invalid[0].problems[0]


def test_registry_entries_expose_valid_and_invalid_consistently(skill_table):
    registry = build_registry(skill_table, FIXTURE_DRAWS)
    assert len(registry.entries) == len(list(FIXTURE_DRAWS.glob("*.json")))
    # Partition, not merely subsets: every entry is in exactly one bucket.
    # (Entries hold a Draw, whose `seeds` dict makes them unhashable, hence
    # counts + membership rather than set algebra.)
    assert len(registry.valid) + len(registry.invalid) == len(registry.entries)
    assert all(
        (entry in registry.valid) != (entry in registry.invalid)
        for entry in registry.entries
    )
    assert all(entry.draw is not None and not entry.problems for entry in registry.valid)
    assert all(entry.draw is None and entry.problems for entry in registry.invalid)
    assert len(set(registry.ids())) == len(registry.entries)  # ids are unique


def test_registry_module_does_not_import_cli_or_fastapi():
    """Layering: the catalogue is a filesystem concern, not a transport one.

    ``api/`` may not import ``cli/`` (the project-wide rule), and keeping
    FastAPI out as well is what lets an offline script, `/simulate`'s precompute,
    resolve an id to a draw without importing the web app.
    """
    source = Path("src/api/registry.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    for forbidden in ("import cli", "from cli", "import fastapi", "from fastapi"):
        assert forbidden not in code
