"""Tests for src/sim/draw.py, the draw schema, loader and validator.

Most tests run against a hand-built :class:`SkillTable` (constructed directly,
so no data frame is needed) whose name→id map holds a handful of toy entrants.
The one exception is the placeholder-handling test, which builds a *real* skill
table from the vendored CSVs so it proves genuine player names resolve while
placeholder slots stay unresolved.
"""

import json

import pandas as pd
import pytest

import config
import data.loader as loader
import data.preprocess as preprocess
import features.serve as serve
from features.serve import PlayerSkill, SkillTable
from sim.draw import (
    Draw,
    DrawValidationError,
    is_placeholder,
    load_draw,
    parse_draw,
)

TOY_PLAYERS = {
    "Alice Ace": "A1",
    "Bob Baseline": "B2",
    "Cara Chip": "C3",
    "Dan Drop": "D4",
    "Eve Emphatic": "E5",
    "Frank Fault": "F6",
}


@pytest.fixture
def skill_table() -> SkillTable:
    """A toy id-keyed skill table covering TOY_PLAYERS on all three surfaces."""
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


def valid_draw_dict() -> dict:
    """A valid 8-slot toy draw, including a placeholder entrant."""
    names = [
        "Alice Ace",
        "Bob Baseline",
        "Cara Chip",
        "Dan Drop",
        "Eve Emphatic",
        "Qualifier",
        "Frank Fault",
        "Lucky Loser",
    ]
    return {
        "tournament_id": "toy_2026",
        "name": "Toy Open 2026",
        "surface": "Hard",
        "best_of": 5,
        "final_set_tiebreak": "10pt_at_6_6",
        "draw_size": 8,
        "seeds": {"Alice Ace": 1, "Frank Fault": 2},
        "bracket": [
            {"position": i + 1, "player": name} for i, name in enumerate(names)
        ],
    }


def load_toy(skill_table: SkillTable, **overrides) -> Draw:
    """Parse the valid toy draw with the given top-level fields replaced."""
    data = valid_draw_dict()
    data.update(overrides)
    return parse_draw(data, skill_table)


def problems_of(excinfo) -> tuple[str, ...]:
    return excinfo.value.problems


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_valid_toy_draw_loads(skill_table):
    draw = load_toy(skill_table)

    assert draw.tournament_id == "toy_2026"
    assert draw.name == "Toy Open 2026"
    assert draw.surface == "Hard"
    assert draw.best_of == 5
    assert draw.final_set_tiebreak == "10pt_at_6_6"
    assert draw.draw_size == 8
    assert draw.seeds == {"Alice Ace": 1, "Frank Fault": 2}
    assert len(draw.bracket) == 8


def test_valid_draw_resolves_player_ids(skill_table):
    draw = load_toy(skill_table)

    resolved = {slot.player: slot.player_id for slot in draw.bracket}
    assert resolved["Alice Ace"] == "A1"
    assert resolved["Frank Fault"] == "F6"
    assert all(
        slot.player_id == TOY_PLAYERS[slot.player]
        for slot in draw.bracket
        if not slot.is_placeholder
    )


def test_bracket_is_ordered_by_position(skill_table):
    data = valid_draw_dict()
    data["bracket"] = list(reversed(data["bracket"]))

    draw = parse_draw(data, skill_table)

    assert [slot.position for slot in draw.bracket] == list(range(1, 9))


def test_first_round_pairings_are_2k_minus_1_vs_2k(skill_table):
    draw = load_toy(skill_table)

    pairings = draw.first_round_pairings()

    assert len(pairings) == 4
    assert [(a.position, b.position) for a, b in pairings] == [
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
    ]


def test_seed_of_returns_seed_or_none(skill_table):
    draw = load_toy(skill_table)
    by_name = {slot.player: slot for slot in draw.bracket}

    assert draw.seed_of(by_name["Alice Ace"]) == 1
    assert draw.seed_of(by_name["Bob Baseline"]) is None


def test_load_draw_reads_a_file(skill_table, tmp_path):
    path = tmp_path / "toy.json"
    path.write_text(json.dumps(valid_draw_dict()), encoding="utf-8")

    draw = load_draw(path, skill_table)

    assert draw.draw_size == 8
    assert draw.bracket[0].player == "Alice Ace"


def test_load_draw_reports_invalid_json(skill_table, tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(DrawValidationError) as excinfo:
        load_draw(path, skill_table)

    assert any("not valid JSON" in p for p in problems_of(excinfo))


def test_unknown_keys_are_ignored(skill_table):
    data = valid_draw_dict()
    data["note"] = "provenance"
    data["bracket"][0]["country"] = "ITA"

    draw = parse_draw(data, skill_table)

    assert draw.bracket[0].player == "Alice Ace"


# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------


def test_placeholders_get_the_default_surface_profile(skill_table):
    draw = load_toy(skill_table)
    placeholders = [slot for slot in draw.bracket if slot.is_placeholder]

    assert [slot.player for slot in placeholders] == ["Qualifier", "Lucky Loser"]
    for slot in placeholders:
        assert slot.player_id is None
        assert draw.skill_for(slot, skill_table) == skill_table.default("Hard")


def test_placeholder_default_follows_the_draw_surface(skill_table):
    draw = load_toy(skill_table, surface="Clay")
    slot = next(s for s in draw.bracket if s.is_placeholder)

    assert draw.skill_for(slot, skill_table) == skill_table.default("Clay")
    assert draw.skill_for(slot, skill_table) != skill_table.default("Hard")


def test_real_entrants_do_not_get_the_default_profile(skill_table):
    draw = load_toy(skill_table)
    slot = next(s for s in draw.bracket if s.player == "Alice Ace")

    assert draw.skill_for(slot, skill_table) == skill_table.get("A1", "Hard")
    assert draw.skill_for(slot, skill_table) != skill_table.default("Hard")


@pytest.mark.parametrize(
    "entrant",
    ["Qualifier", "qualifier", " BYE ", "Lucky Loser", "Q", "Qualifier/Lucky Loser"],
)
def test_placeholder_tokens_are_recognised(entrant):
    assert is_placeholder(entrant)


@pytest.mark.parametrize("entrant", ["Alice Ace", "Qualifier/Alice Ace", "Wildcard"])
def test_real_names_are_not_placeholders(entrant):
    assert not is_placeholder(entrant)


# ---------------------------------------------------------------------------
# Validation rules, one test per failure mode
# ---------------------------------------------------------------------------


def test_bad_draw_size_is_rejected(skill_table):
    with pytest.raises(DrawValidationError) as excinfo:
        load_toy(skill_table, draw_size=12)

    assert any(
        p.startswith("draw_size must be one of [8, 16, 32, 64, 128] (a power of 2)")
        for p in problems_of(excinfo)
    )


def test_bracket_length_mismatch_is_rejected(skill_table):
    with pytest.raises(DrawValidationError) as excinfo:
        load_toy(skill_table, draw_size=16)

    assert any(
        p == "bracket has 8 entries but draw_size is 16" for p in problems_of(excinfo)
    )


def test_missing_position_is_rejected(skill_table):
    data = valid_draw_dict()
    del data["bracket"][3]["position"]

    with pytest.raises(DrawValidationError) as excinfo:
        parse_draw(data, skill_table)

    problems = problems_of(excinfo)
    assert any(
        p == "bracket[3]: 'position' must be an integer, got None" for p in problems
    )
    assert any("bracket positions must be exactly 1..8 (missing: [4])" in p for p in problems)


def test_non_contiguous_positions_are_rejected(skill_table):
    data = valid_draw_dict()
    data["bracket"][2]["position"] = 9

    with pytest.raises(DrawValidationError) as excinfo:
        parse_draw(data, skill_table)

    assert any(
        "bracket positions must be exactly 1..8 "
        "(missing: [3]; out of range: [9])" in p
        for p in problems_of(excinfo)
    )


def test_duplicate_positions_are_rejected(skill_table):
    data = valid_draw_dict()
    data["bracket"][2]["position"] = 4

    with pytest.raises(DrawValidationError) as excinfo:
        parse_draw(data, skill_table)

    assert any(
        "bracket positions must be exactly 1..8 "
        "(missing: [3]; duplicated: [4])" in p
        for p in problems_of(excinfo)
    )


def test_unknown_surface_is_rejected(skill_table):
    with pytest.raises(DrawValidationError) as excinfo:
        load_toy(skill_table, surface="Carpet")

    assert any(
        p == "surface must be one of ['Clay', 'Grass', 'Hard'], got 'Carpet'"
        for p in problems_of(excinfo)
    )


def test_bad_best_of_is_rejected(skill_table):
    with pytest.raises(DrawValidationError) as excinfo:
        load_toy(skill_table, best_of=4)

    assert any(p == "best_of must be one of [3, 5], got 4" for p in problems_of(excinfo))


def test_bad_final_set_tiebreak_is_rejected(skill_table):
    with pytest.raises(DrawValidationError) as excinfo:
        load_toy(skill_table, final_set_tiebreak="sudden_death")

    assert any(
        p.startswith("final_set_tiebreak must be one of")
        and "'sudden_death'" in p
        for p in problems_of(excinfo)
    )


@pytest.mark.parametrize("bad_value", [["Hard"], {"surface": "Hard"}])
def test_non_scalar_surface_is_rejected(skill_table, bad_value):
    """An unhashable value must be a reported problem, not a raw TypeError."""
    with pytest.raises(DrawValidationError) as excinfo:
        load_toy(skill_table, surface=bad_value)

    assert any(p.startswith("surface must be one of") for p in problems_of(excinfo))


@pytest.mark.parametrize("bad_value", [[5], {"best_of": 5}])
def test_non_scalar_best_of_is_rejected(skill_table, bad_value):
    with pytest.raises(DrawValidationError) as excinfo:
        load_toy(skill_table, best_of=bad_value)

    assert any(p.startswith("best_of must be one of") for p in problems_of(excinfo))


@pytest.mark.parametrize("bad_value", [["advantage"], {"rule": "advantage"}])
def test_non_scalar_final_set_tiebreak_is_rejected(skill_table, bad_value):
    with pytest.raises(DrawValidationError) as excinfo:
        load_toy(skill_table, final_set_tiebreak=bad_value)

    assert any(
        p.startswith("final_set_tiebreak must be one of") for p in problems_of(excinfo)
    )


@pytest.mark.parametrize("bad_value", [[8], {"draw_size": 8}])
def test_non_scalar_draw_size_is_rejected(skill_table, bad_value):
    with pytest.raises(DrawValidationError) as excinfo:
        load_toy(skill_table, draw_size=bad_value)

    assert any(p.startswith("draw_size must be one of") for p in problems_of(excinfo))


def test_unresolved_name_is_rejected(skill_table):
    data = valid_draw_dict()
    data["bracket"][1]["player"] = "Nobody Whatsoever"

    with pytest.raises(DrawValidationError) as excinfo:
        parse_draw(data, skill_table)

    assert any(
        "1 player name(s) did not resolve" in p and "'Nobody Whatsoever'" in p
        for p in problems_of(excinfo)
    )


def test_all_unresolved_names_are_reported_together(skill_table):
    data = valid_draw_dict()
    data["bracket"][1]["player"] = "Nobody Whatsoever"
    data["bracket"][2]["player"] = "Ghost Player"
    data["bracket"][3]["player"] = "Phantom Entrant"

    with pytest.raises(DrawValidationError) as excinfo:
        parse_draw(data, skill_table)

    resolution_problems = [
        p for p in problems_of(excinfo) if "did not resolve" in p
    ]
    assert len(resolution_problems) == 1
    message = resolution_problems[0]
    assert "3 player name(s) did not resolve" in message
    for name in ("Nobody Whatsoever", "Ghost Player", "Phantom Entrant"):
        assert repr(name) in message


def test_missing_required_field_is_rejected(skill_table):
    data = valid_draw_dict()
    del data["surface"]

    with pytest.raises(DrawValidationError) as excinfo:
        parse_draw(data, skill_table)

    assert any(p == "missing required field 'surface'" for p in problems_of(excinfo))


def test_bad_seed_value_is_rejected(skill_table):
    with pytest.raises(DrawValidationError) as excinfo:
        load_toy(skill_table, seeds={"Alice Ace": 0})

    assert any(
        p == "seeds['Alice Ace'] must be a positive integer, got 0"
        for p in problems_of(excinfo)
    )


def test_seed_for_a_non_entrant_is_rejected(skill_table):
    with pytest.raises(DrawValidationError) as excinfo:
        load_toy(skill_table, seeds={"Alice Ace": 1, "Cara Chipp": 2})

    assert any(
        p == "seeded name(s) not present in the bracket: 'Cara Chipp'"
        for p in problems_of(excinfo)
    )


def test_non_object_draw_is_rejected(skill_table):
    with pytest.raises(DrawValidationError) as excinfo:
        parse_draw([1, 2, 3], skill_table)

    assert problems_of(excinfo) == ("draw must be a JSON object, got list",)


def test_all_problems_are_reported_together(skill_table):
    """The validator collects every rule failure, not just the first."""
    data = valid_draw_dict()
    data["surface"] = "Ice"
    data["best_of"] = 4
    data["final_set_tiebreak"] = "sudden_death"
    data["draw_size"] = 12
    data["bracket"][1]["player"] = "Nobody Whatsoever"

    with pytest.raises(DrawValidationError) as excinfo:
        parse_draw(data, skill_table)

    problems = problems_of(excinfo)
    assert len(problems) >= 5
    for fragment in (
        "surface must be",
        "best_of must be",
        "final_set_tiebreak must be",
        "draw_size must be",
        "did not resolve",
    ):
        assert any(fragment in p for p in problems), fragment
    # The joined message carries every problem, so one read fixes the file.
    assert str(excinfo.value).count("  - ") == len(problems)


# ---------------------------------------------------------------------------
# Placeholder handling against a real skill table
# ---------------------------------------------------------------------------


def test_placeholder_draw_loads_against_the_real_skill_table():
    """Real entrant names resolve, placeholders stay unresolved (real skills).

    The one test that builds a *real* skill table from the vendored CSVs rather
    than the toy fixture: it proves that a draw mixing genuine player names with
    ``Qualifier``/``Qualifier/Lucky Loser`` placeholders both resolves the real
    names to ids and leaves the placeholders with ``player_id is None``.
    """
    df = loader.load_atp_data(config.TEST_YEAR, config.END_YEAR)
    processed = preprocess.preprocess_data(df)
    table = serve.build_skill_table(processed)

    payload = {
        "note": "In-test draw exercising placeholders against real skills.",
        "tournament_id": "placeholder_open_2026",
        "name": "Placeholder Open 2026",
        "surface": "Hard",
        "best_of": 5,
        "final_set_tiebreak": "10pt_at_6_6",
        "draw_size": 8,
        "seeds": {
            "Jannik Sinner": 1,
            "Carlos Alcaraz": 2,
            "Daniil Medvedev": 3,
            "Casper Ruud": 4,
        },
        "bracket": [
            {"position": 1, "player": "Jannik Sinner"},
            {"position": 2, "player": "Qualifier"},
            {"position": 3, "player": "Taylor Fritz"},
            {"position": 4, "player": "Casper Ruud"},
            {"position": 5, "player": "Daniil Medvedev"},
            {"position": 6, "player": "Qualifier/Lucky Loser"},
            {"position": 7, "player": "Andrey Rublev"},
            {"position": 8, "player": "Carlos Alcaraz"},
        ],
    }

    draw = parse_draw(payload, table)

    assert draw.draw_size == len(draw.bracket) == 8
    assert draw.surface in config.VALID_SURFACES
    real = [slot for slot in draw.bracket if not slot.is_placeholder]
    assert all(slot.player_id is not None for slot in real)
    assert [slot.player for slot in draw.bracket if slot.is_placeholder] == [
        "Qualifier",
        "Qualifier/Lucky Loser",
    ]
    assert all(slot.player_id is None for slot in draw.bracket if slot.is_placeholder)


def test_config_tiebreak_values_match_the_match_layer_enum():
    """config's allowed set is derived from the canonical enum, not re-listed."""
    from sim.match import FINAL_SET_TB_TARGET

    assert config.VALID_FINAL_SET_TIEBREAKS == frozenset(FINAL_SET_TB_TARGET)
    assert FINAL_SET_TB_TARGET is config.FINAL_SET_TB_TARGET
