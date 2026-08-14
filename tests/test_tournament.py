"""Tests for src/sim/tournament.py — T2.2's bracket sim and T2.3's Monte Carlo.

Everything runs against a hand-built :class:`SkillTable` and a toy 8-slot draw
(the T2.1 loader is exercised for real in ``test_draw.py``), plus a stub
classifier — the same dependency-inversion boundary ``sim/reconcile.py``
defines, so no model artefact or data frame is needed.

The load-bearing test here is
:func:`test_outcome_only_never_simulates_a_point`: it *proves* the Monte Carlo
path never reaches the point-by-point engine, rather than trusting that it is
true by construction.
"""

import dataclasses
import re

import numpy as np
import pandas as pd
import pytest

import config
import data.loader as loader
import data.preprocess as preprocess
import features.serve as serve
import sim.match as match
import sim.reconcile as reconcile
from features.serve import PlayerSkill, SkillTable
from sim.draw import load_draw, parse_draw
from sim.match import MatchResult, SetResult
from sim.tournament import (
    _run_chunk,
    format_scoreline,
    monte_carlo,
    reconciled_win_prob,
    render_storybook,
    round_label,
    round_labels,
    simulate_bracket,
    storybook_run,
)

# Eight toy entrants, strongest first — "Ace" has the best serve and return.
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


# A larger, placeholder-free field for the R32 structural test.
BIG_DRAW_SIZE = 32
BIG_PLAYERS = [f"Player{i:02d} Surname" for i in range(BIG_DRAW_SIZE)]


@pytest.fixture
def skill_table() -> SkillTable:
    """Toy id-keyed skills on all three surfaces, with a real spread."""
    skills = {}
    for index, player_id in enumerate(TOY_PLAYERS.values()):
        for surface in config.VALID_SURFACES:
            mu = config.SURFACE_MU[surface]
            # Descending strength down the list: index 0 serves and returns best.
            edge = 0.01 * (len(TOY_PLAYERS) - index)
            skills[(player_id, surface)] = PlayerSkill(
                spw=mu + edge,
                rpw=1.0 - mu + edge,
                n_serve_pts=1000.0,
                n_return_pts=1000.0,
            )
    return SkillTable(
        skills,
        dict(TOY_PLAYERS),
        dict(config.SURFACE_MU),
        cutoff=pd.Timestamp("2026-01-01"),
    )


def toy_draw_dict(**overrides) -> dict:
    """A valid 8-slot toy draw with no placeholders (see the T2.2 gap note)."""
    data = {
        "tournament_id": "toy_2026",
        "name": "Toy Open 2026",
        "surface": "Hard",
        "best_of": 5,
        "final_set_tiebreak": "10pt_at_6_6",
        "draw_size": 8,
        "seeds": {"Alice Ace": 1, "Bob Baseline": 2},
        "bracket": [
            {"position": i + 1, "player": name}
            for i, name in enumerate(TOY_PLAYERS)
        ],
    }
    data.update(overrides)
    return data


@pytest.fixture
def draw(skill_table):
    return parse_draw(toy_draw_dict(), skill_table)


class CountingClassifier:
    """A stub ``ClassifierProb`` that records every call it receives.

    Deliberately **not** memoised: T2.2's contract is that ``simulate_bracket``
    presents each matchup in one stable orientation and asks about it at most
    once per bracket, which only shows up if the stub answers every call.
    """

    def __init__(self, prob: float = 0.55) -> None:
        self.prob = prob
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, player_a: str, player_b: str, surface: str) -> float:
        self.calls.append((player_a, player_b, surface))
        return self.prob


@pytest.fixture
def classifier() -> CountingClassifier:
    return CountingClassifier()


# ---------------------------------------------------------------------------
# Round labels (pure helper)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("remaining", "expected"),
    [(128, "R128"), (64, "R64"), (32, "R32"), (16, "R16"), (8, "QF"), (4, "SF"), (2, "F")],
)
def test_round_label(remaining, expected):
    assert round_label(remaining) == expected


@pytest.mark.parametrize("bad", [0, 1, 3, 12, -8])
def test_round_label_rejects_non_powers_of_two(bad):
    with pytest.raises(ValueError):
        round_label(bad)


@pytest.mark.parametrize(
    ("draw_size", "expected"),
    [
        (8, ("QF", "SF", "F")),
        (16, ("R16", "QF", "SF", "F")),
        (32, ("R32", "R16", "QF", "SF", "F")),
        (128, ("R128", "R64", "R32", "R16", "QF", "SF", "F")),
    ],
)
def test_round_labels_for_each_draw_size(draw_size, expected):
    assert round_labels(draw_size) == expected


# ---------------------------------------------------------------------------
# Scoreline rendering (pure helper)
# ---------------------------------------------------------------------------


def test_format_scoreline_is_winner_first_with_tiebreak_parenthetical():
    result = MatchResult(
        winner=1,
        sets=[
            SetResult(winner=0, games_a=6, games_b=4),
            SetResult(winner=1, games_a=6, games_b=7, tb_score=(5, 7)),
            SetResult(winner=1, games_a=2, games_b=6),
        ],
        best_of=3,
    )

    # Player B won, so every set is written from B's side.
    assert format_scoreline(result) == "4-6 7-6(5) 6-2"


# ---------------------------------------------------------------------------
# Structure — both modes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("outcome_only", [False, True])
def test_bracket_plays_n_minus_1_matches_and_one_champion(
    draw, skill_table, classifier, outcome_only
):
    result = simulate_bracket(
        draw,
        skill_table,
        classifier,
        np.random.default_rng(7),
        outcome_only=outcome_only,
    )

    assert len(result.matches) == draw.draw_size - 1 == 7
    assert result.champion in [slot for slot in draw.bracket]
    # The champion is the winner of the one final.
    assert result.rounds[-1].matches[0].winner == result.champion
    assert result.outcome_only is outcome_only


@pytest.mark.parametrize("outcome_only", [False, True])
def test_round_structure_and_labels(draw, skill_table, classifier, outcome_only):
    result = simulate_bracket(
        draw,
        skill_table,
        classifier,
        np.random.default_rng(7),
        outcome_only=outcome_only,
    )

    assert [r.label for r in result.rounds] == ["QF", "SF", "F"]
    assert [len(r.matches) for r in result.rounds] == [4, 2, 1]
    assert [r.index for r in result.rounds] == [0, 1, 2]
    for r in result.rounds:
        assert all(m.round_label == r.label for m in r.matches)
        assert [m.match_index for m in r.matches] == list(range(len(r.matches)))


@pytest.fixture
def big_skill_table() -> SkillTable:
    """A 32-entrant version of ``skill_table`` — same descending-strength shape."""
    skills = {}
    for index in range(BIG_DRAW_SIZE):
        for surface in config.VALID_SURFACES:
            mu = config.SURFACE_MU[surface]
            edge = 0.002 * (BIG_DRAW_SIZE - index)
            skills[(f"ID{index:02d}", surface)] = PlayerSkill(
                spw=mu + edge,
                rpw=1.0 - mu + edge,
                n_serve_pts=1000.0,
                n_return_pts=1000.0,
            )
    return SkillTable(
        skills,
        {name: f"ID{i:02d}" for i, name in enumerate(BIG_PLAYERS)},
        dict(config.SURFACE_MU),
        cutoff=pd.Timestamp("2026-01-01"),
    )


@pytest.fixture
def big_draw(big_skill_table):
    return parse_draw(
        {
            "tournament_id": "toy32_2026",
            "name": "Toy Open 32",
            "surface": "Hard",
            "best_of": 5,
            "final_set_tiebreak": "10pt_at_6_6",
            "draw_size": BIG_DRAW_SIZE,
            "seeds": {BIG_PLAYERS[0]: 1, BIG_PLAYERS[1]: 2},
            "bracket": [
                {"position": i + 1, "player": name}
                for i, name in enumerate(BIG_PLAYERS)
            ],
        },
        big_skill_table,
    )


@pytest.mark.parametrize("outcome_only", [False, True])
def test_larger_draw_structure_is_correct(
    big_draw, big_skill_table, classifier, outcome_only
):
    """The 8-draw is the smallest possible bracket; check a real R32 too.

    Covers the rounds an 8-draw never exercises — the numbered ``R32``/``R16``
    labels ahead of the named QF/SF/F, and four halvings rather than two.
    """
    result = simulate_bracket(
        big_draw,
        big_skill_table,
        classifier,
        np.random.default_rng(7),
        outcome_only=outcome_only,
    )

    assert len(result.matches) == BIG_DRAW_SIZE - 1 == 31
    assert [r.label for r in result.rounds] == ["R32", "R16", "QF", "SF", "F"]
    assert [len(r.matches) for r in result.rounds] == [16, 8, 4, 2, 1]
    assert [r.index for r in result.rounds] == [0, 1, 2, 3, 4]

    # Exactly one champion, and they won the single final.
    assert len(result.rounds[-1].matches) == 1
    assert result.rounds[-1].matches[0].winner == result.champion

    # Every entrant plays exactly once in the first round.
    first_round = [
        slot.position
        for m in result.rounds[0].matches
        for slot in (m.slot_a, m.slot_b)
    ]
    assert sorted(first_round) == list(range(1, BIG_DRAW_SIZE + 1))

    # Orientation and mode invariants hold at every round.
    for m in result.matches:
        assert m.slot_a.position < m.slot_b.position
        assert (m.scoreline is None) is outcome_only
        assert (m.result is None) is outcome_only

    # The champion's path spans all five rounds and is a clean sweep.
    champion_path = result.path_of(result.champion.position)
    assert [m.round_label for m in champion_path] == ["R32", "R16", "QF", "SF", "F"]
    assert all(m.winner == result.champion for m in champion_path)
    # Everyone else's run ends in a defeat.
    eliminated = [
        pos
        for pos in range(1, BIG_DRAW_SIZE + 1)
        if result.path_of(pos)[-1].loser.position == pos
    ]
    assert len(eliminated) == BIG_DRAW_SIZE - 1 == 31


def test_first_round_pairs_slots_2k_minus_1_vs_2k(draw, skill_table, classifier):
    result = simulate_bracket(
        draw, skill_table, classifier, np.random.default_rng(1), outcome_only=True
    )

    pairs = [(m.slot_a.position, m.slot_b.position) for m in result.rounds[0].matches]
    assert pairs == [(1, 2), (3, 4), (5, 6), (7, 8)]


def test_winners_advance_in_bracket_order(draw, skill_table, classifier):
    result = simulate_bracket(
        draw, skill_table, classifier, np.random.default_rng(3), outcome_only=True
    )

    qf_winners = [m.winner.position for m in result.rounds[0].matches]
    sf_entrants = [
        p for m in result.rounds[1].matches for p in (m.slot_a.position, m.slot_b.position)
    ]
    assert sf_entrants == qf_winners
    # Player A is always the lower bracket position (the orientation convention).
    for m in result.matches:
        assert m.slot_a.position < m.slot_b.position


def test_path_of_reconstructs_a_players_run(draw, skill_table, classifier):
    result = simulate_bracket(
        draw, skill_table, classifier, np.random.default_rng(11), outcome_only=True
    )

    champion_path = result.path_of(result.champion.position)
    assert len(champion_path) == 3  # QF, SF, F
    assert [m.round_label for m in champion_path] == ["QF", "SF", "F"]
    assert all(m.winner == result.champion for m in champion_path)

    # Someone beaten in the first round played exactly one match, and lost it.
    first_loser = result.rounds[0].matches[0].loser
    loser_path = result.path_of(first_loser.position)
    assert len(loser_path) == 1
    assert loser_path[0].winner != first_loser
    assert loser_path[0].loser == first_loser


# ---------------------------------------------------------------------------
# outcome_only=False — real scorelines
# ---------------------------------------------------------------------------


def test_storybook_mode_gives_every_match_a_real_scoreline(
    draw, skill_table, classifier
):
    result = simulate_bracket(
        draw, skill_table, classifier, np.random.default_rng(5), outcome_only=False
    )

    assert len(result.matches) == 7
    for m in result.matches:
        assert m.result is not None
        assert m.scoreline is not None
        sets = m.scoreline.split(" ")
        # Best-of-5: the winner takes 3 sets, so 3-5 sets are played.
        assert 3 <= len(sets) <= 5
        assert len(m.result.sets) == len(sets)
        assert m.result.best_of == 5
        for line in sets:
            games = line.split("(")[0].split("-")
            assert len(games) == 2 and all(g.isdigit() for g in games)


def test_storybook_mode_serves_first_deterministically(draw, skill_table, classifier):
    """Player A (the lower bracket position) always serves the opening game.

    The convention is enforced by pinning ``first_server`` through
    ``simulate_reconciled_match``; here we capture what it actually passes.
    """
    seen = []
    real = reconcile.simulate_reconciled_match

    def spy(*args, **kwargs):
        seen.append(kwargs.get("first_server"))
        return real(*args, **kwargs)

    import sim.tournament as tournament

    original = tournament.simulate_reconciled_match
    tournament.simulate_reconciled_match = spy
    try:
        simulate_bracket(
            draw, skill_table, classifier, np.random.default_rng(2), outcome_only=False
        )
    finally:
        tournament.simulate_reconciled_match = original

    assert seen == [0] * 7


# ---------------------------------------------------------------------------
# outcome_only=True — winner only, and provably no point simulation
# ---------------------------------------------------------------------------


def test_outcome_only_records_no_scorelines(draw, skill_table, classifier):
    result = simulate_bracket(
        draw, skill_table, classifier, np.random.default_rng(5), outcome_only=True
    )

    assert len(result.matches) == 7
    assert all(m.scoreline is None for m in result.matches)
    assert all(m.result is None for m in result.matches)


def _trap(name):
    """A stand-in that fails loudly if the point-by-point engine is entered."""

    def _raise(*args, **kwargs):
        raise AssertionError(f"{name} was called")

    return _raise


# Every route from simulate_bracket into the point-by-point engine, patched at
# the module that *holds the reference* the caller resolves — sim/match.py for
# the game/set/tiebreak layer, sim/reconcile.py for the two match entry points
# it imported. Each is proved to be a live interception point by
# test_each_trap_fires_in_storybook_mode below, so a silent run in
# outcome_only=True means the path really was never taken.
POINT_SIM_TRAPS = (
    (match, "simulate_game"),
    (match, "simulate_tiebreak"),
    (match, "simulate_set"),
    (reconcile, "simulate_match_bo3"),
    (reconcile, "simulate_match_bo5"),
)


def test_outcome_only_never_simulates_a_point(
    draw, skill_table, classifier, monkeypatch
):
    """The acceptance criterion that must be *proved*, not assumed."""
    for module, name in POINT_SIM_TRAPS:
        monkeypatch.setattr(module, name, _trap(name))
    # Not part of POINT_SIM_TRAPS: match_win_prob_point_mc is a defensive no-op
    # guard, not a real check. sim/tournament.py never imports it, so patching
    # reconcile's copy cannot fire in *either* mode — it documents that the MC
    # estimator is off the hot path (see sim/reconcile.py's module docstring)
    # and would only catch a future edit that routed through reconcile itself.
    monkeypatch.setattr(
        reconcile, "match_win_prob_point_mc", _trap("match_win_prob_point_mc")
    )

    result = simulate_bracket(
        draw, skill_table, classifier, np.random.default_rng(5), outcome_only=True
    )

    assert len(result.matches) == 7
    assert result.champion is not None


@pytest.mark.parametrize(
    ("module", "name"), POINT_SIM_TRAPS, ids=[n for _, n in POINT_SIM_TRAPS]
)
def test_each_trap_fires_in_storybook_mode(
    draw, skill_table, classifier, monkeypatch, module, name
):
    """Control for the test above — patched *one at a time*.

    Without this, a trap that silently fails to intercept (wrong module, renamed
    symbol) would make ``test_outcome_only_never_simulates_a_point`` pass for the
    wrong reason. Each trap must demonstrably catch the mode that *does* simulate
    points. ``simulate_match_bo3`` is exercised against a best-of-3 draw, since
    the shared fixture is best-of-5 and would never reach it.
    """
    if name == "simulate_match_bo3":
        draw = dataclasses.replace(draw, best_of=3, final_set_tiebreak="7pt_at_6_6")

    monkeypatch.setattr(module, name, _trap(name))

    with pytest.raises(AssertionError, match=f"{name} was called"):
        simulate_bracket(
            draw, skill_table, classifier, np.random.default_rng(5), outcome_only=False
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def summarise(result):
    """A comparable digest of a whole bracket."""
    return [
        (
            m.round_label,
            m.match_index,
            m.slot_a.position,
            m.slot_b.position,
            m.winner.position,
            m.scoreline,
        )
        for m in result.matches
    ] + [("champion", result.champion.position)]


@pytest.mark.parametrize("outcome_only", [False, True])
def test_same_seed_gives_an_identical_bracket(
    draw, skill_table, outcome_only
):
    def run():
        return simulate_bracket(
            draw,
            skill_table,
            CountingClassifier(),
            np.random.default_rng(2026),
            outcome_only=outcome_only,
        )

    assert summarise(run()) == summarise(run())


@pytest.mark.parametrize("outcome_only", [False, True])
def test_different_seeds_generally_differ(draw, skill_table, outcome_only):
    """Sanity check that the seed is actually driving the simulation."""
    runs = {
        tuple(
            summarise(
                simulate_bracket(
                    draw,
                    skill_table,
                    CountingClassifier(),
                    np.random.default_rng(seed),
                    outcome_only=outcome_only,
                )
            )
        )
        for seed in range(12)
    }
    assert len(runs) > 1


# ---------------------------------------------------------------------------
# Reconciliation + the caching contract
# ---------------------------------------------------------------------------


def test_classifier_is_called_once_per_unique_matchup(draw, skill_table, classifier):
    """simulate_bracket must not defeat a caller-supplied classifier cache.

    The stub is uncached, so its call log shows exactly what the bracket asked
    for: one query per matchup actually played, in one stable orientation.
    """
    simulate_bracket(
        draw, skill_table, classifier, np.random.default_rng(9), outcome_only=True
    )

    assert len(classifier.calls) == 7
    assert len(set(classifier.calls)) == 7  # no repeated key
    # Always oriented lower-position-first, so a pairing yields one cache key.
    positions = {name: pos for pos, name in enumerate(TOY_PLAYERS)}
    for player_a, player_b, surface in classifier.calls:
        assert surface == "Hard"
        assert positions[player_a] < positions[player_b]


def test_prob_cache_is_reused_across_brackets(draw, skill_table):
    """The T2.3 contract: one shared cache spans the whole Monte Carlo job."""
    classifier = CountingClassifier()
    cache: dict = {}

    for seed in range(20):
        simulate_bracket(
            draw,
            skill_table,
            classifier,
            np.random.default_rng(seed),
            outcome_only=True,
            prob_cache=cache,
        )

    # 20 brackets × 7 matches = 140 evaluations, but an 8-draw admits at most
    # 4 + 2·(2·2) + 1·(4·4) = 28 distinct matchups, each solved exactly once.
    assert len(classifier.calls) == len(cache)
    assert len(cache) <= 28


def test_prob_cache_does_not_change_outcomes(draw, skill_table):
    def run(cache):
        return summarise(
            simulate_bracket(
                draw,
                skill_table,
                CountingClassifier(),
                np.random.default_rng(4),
                outcome_only=True,
                prob_cache=cache,
            )
        )

    warm: dict = {}
    run(warm)  # populate
    assert run(None) == run(warm)


def test_prob_cache_key_separates_reconciliation_modes(draw, skill_table):
    """Every key dimension must discriminate, or a shared job-wide cache lies.

    A cache keyed without ``mode``/``w`` would hand a ``"blend"`` probability to
    a ``"classifier_anchor"`` caller — silently, and only for the *second* mode
    used in the job.
    """
    slot_a, slot_b = draw.bracket[0], draw.bracket[1]
    cache: dict = {}

    anchor = reconciled_win_prob(
        draw, slot_a, slot_b, skill_table, CountingClassifier(0.8),
        mode="classifier_anchor", w=0.5, prob_cache=cache,
    )
    blend_5 = reconciled_win_prob(
        draw, slot_a, slot_b, skill_table, CountingClassifier(0.8),
        mode="blend", w=0.5, prob_cache=cache,
    )
    blend_8 = reconciled_win_prob(
        draw, slot_a, slot_b, skill_table, CountingClassifier(0.8),
        mode="blend", w=0.8, prob_cache=cache,
    )

    assert len(cache) == 3
    assert anchor != blend_5 and blend_5 != blend_8
    # A cached value equals the uncached one it stands in for.
    assert blend_5 == reconciled_win_prob(
        draw, slot_a, slot_b, skill_table, CountingClassifier(0.8),
        mode="blend", w=0.5,
    )


def test_prob_cache_key_separates_match_formats(draw, skill_table):
    """``best_of`` and ``final_set_tiebreak`` change the probability, so they key."""
    slot_a, slot_b = draw.bracket[0], draw.bracket[1]
    cache: dict = {}

    bo5 = reconciled_win_prob(
        draw, slot_a, slot_b, skill_table, CountingClassifier(),
        mode="blend", prob_cache=cache,
    )
    bo3 = reconciled_win_prob(
        dataclasses.replace(draw, best_of=3, final_set_tiebreak="7pt_at_6_6"),
        slot_a, slot_b, skill_table, CountingClassifier(),
        mode="blend", prob_cache=cache,
    )
    advantage = reconciled_win_prob(
        dataclasses.replace(draw, final_set_tiebreak="advantage"),
        slot_a, slot_b, skill_table, CountingClassifier(),
        mode="blend", prob_cache=cache,
    )

    assert len(cache) == 3
    assert bo5 != bo3 and bo5 != advantage


def test_prob_cache_key_separates_surfaces(draw, skill_table):
    """The toy table gives each surface a different mu, so all three differ."""
    slot_a, slot_b = draw.bracket[0], draw.bracket[1]
    cache: dict = {}

    values = {
        surface: reconciled_win_prob(
            dataclasses.replace(draw, surface=surface),
            slot_a, slot_b, skill_table, CountingClassifier(),
            mode="blend", prob_cache=cache,
        )
        for surface in config.VALID_SURFACES
    }

    assert len(cache) == len(config.VALID_SURFACES)
    assert len(set(values.values())) == len(config.VALID_SURFACES)


def test_reconciled_win_prob_tracks_the_classifier(draw, skill_table):
    """classifier_anchor: a stronger P_clf must give a stronger win probability."""
    slot_a, slot_b = draw.bracket[0], draw.bracket[1]

    low = reconciled_win_prob(
        draw, slot_a, slot_b, skill_table, CountingClassifier(0.30),
        mode="classifier_anchor",
    )
    high = reconciled_win_prob(
        draw, slot_a, slot_b, skill_table, CountingClassifier(0.80),
        mode="classifier_anchor",
    )

    assert 0.0 <= low < high <= 1.0
    # classifier_anchor reproduces P_clf through δ, within the solver tolerance.
    assert high == pytest.approx(0.80, abs=1e-2)


def test_reconciled_win_prob_is_symmetric(draw, skill_table):
    """P(A beats B) + P(B beats A) ≈ 1 for a symmetric classifier."""
    slot_a, slot_b = draw.bracket[0], draw.bracket[1]

    forward = reconciled_win_prob(
        draw, slot_a, slot_b, skill_table, CountingClassifier(0.65),
        mode="classifier_anchor",
    )
    reverse = reconciled_win_prob(
        draw, slot_b, slot_a, skill_table, CountingClassifier(0.35),
        mode="classifier_anchor",
    )

    assert forward + reverse == pytest.approx(1.0, abs=2e-2)


def test_outcome_only_win_rate_matches_the_reconciled_probability(draw, skill_table):
    """The Bernoulli draw uses the reconciled probability, not a coin flip."""
    classifier = CountingClassifier(0.90)
    rng = np.random.default_rng(0)

    # Alice (position 1) is both the stronger player and the classifier's pick,
    # so she should reach the final in the large majority of runs.
    finals = sum(
        1
        for _ in range(200)
        if any(
            m.slot_a.position == 1 or m.slot_b.position == 1
            for m in simulate_bracket(
                draw, skill_table, classifier, rng, outcome_only=True
            ).rounds[-1].matches
        )
    )
    assert finals > 150


# ---------------------------------------------------------------------------
# Placeholders (the documented T2.2 gap)
# ---------------------------------------------------------------------------


def test_placeholder_entrants_are_rejected_upfront(skill_table, classifier):
    data = toy_draw_dict()
    data["bracket"][5]["player"] = "Qualifier"
    data["bracket"][7]["player"] = "Lucky Loser"
    data["seeds"] = {"Alice Ace": 1}
    placeholder_draw = parse_draw(data, skill_table)

    with pytest.raises(ValueError) as excinfo:
        simulate_bracket(
            placeholder_draw, skill_table, classifier, np.random.default_rng(1)
        )

    message = str(excinfo.value)
    assert "2 placeholder entrant(s)" in message
    assert "6:'Qualifier'" in message and "8:'Lucky Loser'" in message
    # Refused before anything was simulated.
    assert classifier.calls == []


# ---------------------------------------------------------------------------
# The shipped placeholder-free example draw (addendum to T2.1/T2.2)
# ---------------------------------------------------------------------------


def test_full_example_draw_loads_and_simulates_end_to_end():
    """Smoke test for ``example_usopen_2024_full.json``.

    ``example_usopen_2026.json`` carries ``"Qualifier"``/``"Lucky Loser"`` slots
    and so is *correctly* refused by :func:`simulate_bracket`. This second
    example is a real, completed 128-draw in which every slot was filled by a
    named player, so it is the fixture T2.3 can actually measure against.

    Deliberately a smoke test: structure, labels and determinism are already
    covered above against toy draws. What is unique here is that the file
    resolves against skills built from the **real** vendored data and then runs
    to a champion in both modes. The classifier is the usual stub — the T2.2
    boundary — since this checks the draw, not the model.
    """
    df = loader.load_atp_data(config.START_YEAR, config.END_YEAR)
    table = serve.build_skill_table(preprocess.preprocess_data(df))

    draw = load_draw(config.DRAWS_DIR / "example_usopen_2024_full.json", table)

    assert draw.draw_size == len(draw.bracket) == 128
    # The whole point of this file: nothing for _reject_placeholders to catch.
    assert not any(slot.is_placeholder for slot in draw.bracket)
    assert all(slot.player_id is not None for slot in draw.bracket)

    for outcome_only in (False, True):
        result = simulate_bracket(
            draw,
            table,
            CountingClassifier(),
            np.random.default_rng(2026),
            outcome_only=outcome_only,
        )
        assert len(result.matches) == 127
        assert result.champion in draw.bracket
        assert result.rounds[-1].matches[0].winner == result.champion
        assert [r.label for r in result.rounds] == [
            "R128", "R64", "R32", "R16", "QF", "SF", "F",
        ]


def test_ausopen_2026_full_draw_loads_and_simulates_end_to_end():
    """Smoke test for ``ausopen_2026_atp_full.json`` (draw-addendum).

    The second real, placeholder-free 128-draw shipped: the 2026 Australian Open
    men's singles, reconstructed from the vendored round-by-round results the
    same way ``example_usopen_2024_full.json`` was. Proves the file resolves
    against skills built from real vendored data and runs to a champion in both
    modes. Like its sibling above, the classifier is the usual stub — this
    checks the draw, not the model.
    """
    df = loader.load_atp_data(config.START_YEAR, config.END_YEAR)
    table = serve.build_skill_table(preprocess.preprocess_data(df))

    draw = load_draw(config.DRAWS_DIR / "ausopen_2026_atp_full.json", table)

    assert draw.draw_size == len(draw.bracket) == 128
    assert draw.surface == "Hard" and draw.best_of == 5
    assert draw.final_set_tiebreak == "10pt_at_6_6"
    # A real, completed draw: no placeholder slots, every entrant resolved.
    assert not any(slot.is_placeholder for slot in draw.bracket)
    assert all(slot.player_id is not None for slot in draw.bracket)
    assert len({slot.player for slot in draw.bracket}) == 128
    # Canonical seed-doubling labelling: seed 1 -> position 1, seed 2 -> 65, etc.
    position_of = {slot.player: slot.position for slot in draw.bracket}
    seed_at = {seed: name for name, seed in draw.seeds.items()}
    assert position_of[seed_at[1]] == 1
    assert position_of[seed_at[2]] == 65

    for outcome_only in (False, True):
        result = simulate_bracket(
            draw,
            table,
            CountingClassifier(),
            np.random.default_rng(2026),
            outcome_only=outcome_only,
        )
        assert len(result.matches) == 127
        assert result.champion in draw.bracket
        assert [r.label for r in result.rounds] == [
            "R128", "R64", "R32", "R16", "QF", "SF", "F",
        ]


# ---------------------------------------------------------------------------
# T2.3 — Monte Carlo runner + aggregation
# ---------------------------------------------------------------------------
# The toy skill table gives "Alice Ace" the biggest serve/return edge and
# "Hank Half" the smallest, so with the point model in charge (w=0.0, i.e. the
# classifier is blended out) the field has a real, ordered strength gradient.
POINT_MODEL_ONLY = {"mode": "blend", "w": 0.0}


def test_monte_carlo_strongest_player_has_the_highest_title_probability(
    draw, skill_table, classifier
):
    """Toy 8-draw: the strongest entrant tops the board and probabilities sum to 1."""
    mc = monte_carlo(
        draw, skill_table, classifier, n_runs=400, seed=1, **POINT_MODEL_ONLY
    )

    assert mc.players[0].player == "Alice Ace"  # sorted by title probability
    assert mc.players[0].p_title > mc.by_position()[8].p_title  # vs the weakest
    assert sum(p.p_title for p in mc.players) == pytest.approx(1.0)
    assert sum(p.titles for p in mc.players) == 400  # exactly one champion a run

    # Sorted descending, and every probability is a probability.
    titles = [p.titles for p in mc.players]
    assert titles == sorted(titles, reverse=True)
    assert all(0.0 <= p.p_title <= 1.0 for p in mc.players)


def test_monte_carlo_round_survival_is_internally_consistent(
    draw, skill_table, classifier
):
    """Reach ⊇ reach-next-round ⊇ … ⊇ title, and each round's field is full."""
    mc = monte_carlo(
        draw, skill_table, classifier, n_runs=200, seed=2, **POINT_MODEL_ONLY
    )

    assert mc.round_labels == ("QF", "SF", "F")
    for player in mc.players:
        counts = [player.reached[label] for label in mc.round_labels]
        assert counts == sorted(counts, reverse=True)  # monotonically narrowing
        assert player.titles <= counts[-1]  # can't win a final you didn't play
        # Rounds won = one per round survived beyond the first, plus the title.
        assert player.matches_won == sum(counts[1:]) + player.titles
        assert player.expected_rounds_won == player.matches_won / 200

    # Every round is contested by exactly the right number of entrants.
    for index, label in enumerate(mc.round_labels):
        field_size = draw.draw_size // (2**index)
        assert sum(p.reached[label] for p in mc.players) == 200 * field_size


def test_monte_carlo_round_metrics_follow_the_draws_own_labels(
    big_draw, big_skill_table, classifier
):
    """Round-survival keys come from the draw, not a hardcoded QF/SF/F triple.

    An 8-draw's *first* round already is the QF; a 32-draw's is R32. The record
    must carry a ``p_reach_<label>`` per round the draw actually has, with the
    first-round probability trivially 1.0 (everyone plays it).
    """
    mc = monte_carlo(
        big_draw, big_skill_table, classifier, n_runs=40, seed=3, **POINT_MODEL_ONLY
    )

    assert mc.round_labels == ("R32", "R16", "QF", "SF", "F")
    record = mc.to_records()[0]
    assert [key for key in record if key.startswith("p_reach_")] == [
        "p_reach_R32", "p_reach_R16", "p_reach_QF", "p_reach_SF", "p_reach_F"
    ]
    assert all(p.p_reach("R32") == 1.0 for p in mc.players)
    assert record["p_reach_R32"] == 1.0
    # A label this draw does not have answers 0.0 rather than raising.
    assert mc.players[0].p_reach("R64") == 0.0
    # Expected rounds won averages to (draw_size - 1) matches shared out.
    assert sum(p.expected_rounds_won for p in mc.players) == pytest.approx(
        BIG_DRAW_SIZE - 1
    )


def test_monte_carlo_records_are_tidy_and_sorted(draw, skill_table, classifier):
    mc = monte_carlo(
        draw, skill_table, classifier, n_runs=50, seed=4, **POINT_MODEL_ONLY
    )

    records = mc.to_records()
    assert len(records) == draw.draw_size
    assert [r["p_title"] for r in records] == sorted(
        (r["p_title"] for r in records), reverse=True
    )
    alice = next(r for r in records if r["player"] == "Alice Ace")
    assert alice["seed"] == 1 and alice["player_id"] == "A1"
    assert alice["position"] == 1 and alice["n_runs"] == 50


def test_monte_carlo_is_reproducible_exactly(draw, skill_table):
    """Same base seed → identical counts, not merely similar probabilities."""
    def run(seed):
        return monte_carlo(
            draw, skill_table, CountingClassifier(), n_runs=120, seed=seed,
            **POINT_MODEL_ONLY,
        )

    first, second = run(2026), run(2026)
    assert first.players == second.players  # frozen dataclasses of ints
    assert first == second

    assert run(2027).players != first.players  # the seed really is driving it


def test_monte_carlo_seeds_each_run_from_its_own_spawned_child(draw, skill_table):
    """Run *i* is driven by ``SeedSequence(seed).spawn(n_runs)[i]`` specifically.

    ``test_monte_carlo_is_reproducible_exactly`` cannot establish this on its
    own: one base seed reused for *every* run also replays exactly, so it looks
    identical under a same-seed comparison. Its ``run(2027) != run(2026)`` line
    does catch a collapsed-seed job, but only by accident: it fires when the two
    arbitrary base seeds happen to produce the *same* whole bracket, and an audit
    that swept 20 alternative comparison seeds measured that at **2/20**. Swap
    ``2027`` for ``2029`` and a shared-seed regression ships green.
    Reconstructing the spawn by hand and demanding the counts match is
    unconditional — the same sweep catches it 20/20.
    """
    n_runs, seed = 12, 4242
    mc = monte_carlo(
        draw, skill_table, CountingClassifier(), n_runs=n_runs, seed=seed,
        **POINT_MODEL_ONLY,
    )

    titles = [0] * draw.draw_size
    for child in np.random.SeedSequence(seed).spawn(n_runs):
        result = simulate_bracket(
            draw, skill_table, CountingClassifier(), np.random.default_rng(child),
            outcome_only=True, **POINT_MODEL_ONLY,
        )
        titles[result.champion.position - 1] += 1

    by_position = sorted(mc.players, key=lambda outcome: outcome.position)
    assert [outcome.titles for outcome in by_position] == titles
    # Backstop that holds for any base seed: runs sharing one seed would put
    # every title on a single entrant, so a spread proves they were distinct.
    assert sum(1 for count in titles if count) > 1


def test_monte_carlo_always_passes_outcome_only_true(
    draw, skill_table, classifier, monkeypatch
):
    """Structural half of the criterion: every call carries outcome_only=True."""
    import sim.tournament as tournament

    seen = []
    real = tournament.simulate_bracket

    def spy(*args, **kwargs):
        seen.append(kwargs.get("outcome_only"))
        return real(*args, **kwargs)

    monkeypatch.setattr(tournament, "simulate_bracket", spy)
    monte_carlo(draw, skill_table, classifier, n_runs=25, seed=5, **POINT_MODEL_ONLY)

    assert len(seen) == 25
    assert set(seen) == {True}


def test_monte_carlo_never_simulates_a_point(
    draw, skill_table, classifier, monkeypatch
):
    """Behavioural half: the same raising traps T2.2 used, over a whole MC job.

    ``POINT_SIM_TRAPS`` is already proved to be a live interception point by
    ``test_each_trap_fires_in_storybook_mode``, so a clean run here means the
    point-by-point engine really was never entered — across every one of the
    runs, not just the first.
    """
    for module, name in POINT_SIM_TRAPS:
        monkeypatch.setattr(module, name, _trap(name))
    monkeypatch.setattr(
        reconcile, "match_win_prob_point_mc", _trap("match_win_prob_point_mc")
    )

    mc = monte_carlo(
        draw, skill_table, classifier, n_runs=30, seed=6, **POINT_MODEL_ONLY
    )

    assert mc.n_runs == 30
    assert sum(p.titles for p in mc.players) == 30


def test_monte_carlo_shares_one_prob_cache_across_every_run(draw, skill_table):
    """One cache for the whole single-threaded job — not one per bracket.

    The stub classifier is uncached, so its call log is a direct read-out of how
    many distinct matchups the job had to solve. Uncached, 200 runs × 7 matches
    would be 1,400 calls; with a job-wide cache each of the ≤28 matchups an
    8-draw admits is asked exactly once.
    """
    classifier = CountingClassifier()

    monte_carlo(
        draw, skill_table, classifier, n_runs=200, seed=7, **POINT_MODEL_ONLY
    )

    assert len(classifier.calls) == len(set(classifier.calls))  # never re-asked
    assert len(classifier.calls) <= 28
    assert len(classifier.calls) < 200 * 7


def test_each_worker_chunk_builds_its_own_prob_cache(draw, skill_table):
    """Multi-worker mode must give each worker a *fresh* cache, not a shared one.

    Run the worker body twice in-process over the same seeds with one shared
    classifier stub. If the cache were hoisted out of ``_run_chunk`` (a
    module-level dict, say), the second invocation would ask the classifier
    nothing. It asks for exactly the same matchups again, which is what "each
    worker starts cold" means — and is why the parallel speedup is sub-linear.
    """
    classifier = CountingClassifier()
    seeds = np.random.SeedSequence(8).spawn(20)
    payload = (draw, skill_table, classifier, seeds, "blend", 0.0)

    first_counts = _run_chunk(payload)
    after_first = len(classifier.calls)
    second_counts = _run_chunk(payload)

    assert after_first > 0
    assert len(classifier.calls) == 2 * after_first
    assert classifier.calls[:after_first] == classifier.calls[after_first:]
    # Same seeds → same counts, so the cold cache changed nothing but timing.
    assert first_counts == second_counts


def test_monte_carlo_multiprocessing_matches_single_threaded_exactly(
    draw, skill_table
):
    """workers>1 is a pure speed knob: identical counts, not approximate ones."""
    single = monte_carlo(
        draw, skill_table, CountingClassifier(), n_runs=64, seed=9, workers=1,
        **POINT_MODEL_ONLY,
    )
    parallel = monte_carlo(
        draw, skill_table, CountingClassifier(), n_runs=64, seed=9, workers=3,
        **POINT_MODEL_ONLY,
    )

    assert parallel.players == single.players
    assert parallel.workers == 3 and single.workers == 1  # provenance only


def test_monte_carlo_rejects_an_unpicklable_classifier_for_workers(draw, skill_table):
    """The realistic failure — the CLI's adapter is a closure — named up front."""
    def closure_classifier(a, b, surface):  # not picklable
        return 0.55

    with pytest.raises(ValueError, match="picklable classifier"):
        monte_carlo(draw, skill_table, closure_classifier, n_runs=4, seed=1, workers=2)


def test_monte_carlo_rejects_placeholders_before_spawning_anything(
    skill_table, classifier
):
    data = toy_draw_dict()
    data["bracket"][3]["player"] = "Qualifier"
    data["seeds"] = {"Alice Ace": 1}
    placeholder_draw = parse_draw(data, skill_table)

    with pytest.raises(ValueError, match="placeholder entrant"):
        monte_carlo(placeholder_draw, skill_table, classifier, n_runs=5, seed=1)
    assert classifier.calls == []


@pytest.mark.parametrize(("n_runs", "workers"), [(0, 1), (-1, 1), (5, 0), (5, -2)])
def test_monte_carlo_validates_its_counts(draw, skill_table, classifier, n_runs, workers):
    with pytest.raises(ValueError):
        monte_carlo(draw, skill_table, classifier, n_runs=n_runs, workers=workers)


def test_monte_carlo_defaults_to_the_system_reconciliation_mode(
    draw, skill_table, classifier
):
    """T2.3 decision: ``mode`` is a pass-through, with no opinion baked in here.

    The core must not inherit ``config.SIM_CLI_RECONCILE_MODE``, which is a
    CLI-scoped mitigation for one degraded adapter (seam 7), so the default has
    to agree with ``simulate_bracket``'s.
    """
    mc = monte_carlo(draw, skill_table, classifier, n_runs=5, seed=1)

    assert mc.mode == config.RECONCILE_MODE
    assert mc.w == config.RECONCILE_BLEND_WEIGHT


# ---------------------------------------------------------------------------
# T2.4 — Storybook single run
# ---------------------------------------------------------------------------
# A rendered match line: "<winner> def. <loser> 6-4 3-6 7-6(5) 6-2".
MATCH_LINE = re.compile(
    r"^(?P<winner>.+) def\. (?P<loser>.+?) "
    r"(?P<scoreline>\d+-\d+(?:\(\d+\))?(?: \d+-\d+(?:\(\d+\))?)*)$"
)


def test_storybook_calls_simulate_bracket_exactly_once(draw, skill_table, classifier):
    """The acceptance criterion, asserted by counting — not by inspection.

    One bracket per storybook run: not one per round, not one per match, and
    emphatically not the Monte Carlo path. The call must also carry
    ``outcome_only=False``, or there would be no scorelines to tell a story
    with.
    """
    import sim.tournament as tournament

    seen = []
    real = tournament.simulate_bracket

    def spy(*args, **kwargs):
        seen.append(kwargs.get("outcome_only"))
        return real(*args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(tournament, "simulate_bracket", spy)
    try:
        result = tournament.storybook_run(draw, skill_table, classifier, seed=11)
    finally:
        monkeypatch.undo()

    assert seen == [False]  # exactly one call, in storybook mode
    assert len(result.matches) == 7


def test_storybook_8_draw_has_seven_matches_across_three_rounds(
    draw, skill_table, classifier
):
    """The ticket's testing criterion: 7 matches, 3 rounds, valid scorelines."""
    result = storybook_run(draw, skill_table, classifier, seed=12)

    assert [r.label for r in result.rounds] == ["QF", "SF", "F"]
    assert [len(r.matches) for r in result.rounds] == [4, 2, 1]
    assert len(result.matches) == 7

    for storybook_match in result.matches:
        parsed = MATCH_LINE.match(storybook_match.line)
        assert parsed is not None, storybook_match.line
        assert parsed["winner"] == storybook_match.winner
        assert parsed["loser"] == storybook_match.loser
        assert parsed["scoreline"] == storybook_match.scoreline
        # Best-of-5, so 3-5 sets, and the underlying MatchResult is preserved.
        assert 3 <= len(storybook_match.scoreline.split(" ")) <= 5
        assert storybook_match.match.result is not None
        assert storybook_match.winner != storybook_match.loser

    # Exactly one champion, and it is whoever won the final.
    assert result.final is result.rounds[-1].matches[0]
    assert result.champion.player == result.final.winner
    assert result.champion == result.bracket.champion
    assert result.champion_seed == draw.seed_of(result.champion)
    assert sum(1 for run in result.runs if run.is_champion) == 1


def test_storybook_scorelines_come_from_the_bracket_untouched(
    draw, skill_table, classifier
):
    """No second scoreline formatter: the wrapper reuses what T2.2 produced."""
    result = storybook_run(draw, skill_table, classifier, seed=13)

    for storybook_match in result.matches:
        assert storybook_match.scoreline == storybook_match.match.scoreline
        assert storybook_match.scoreline == format_scoreline(
            storybook_match.match.result
        )


def test_storybook_run_summaries_describe_every_entrants_tournament(
    draw, skill_table, classifier
):
    """The documented ``PlayerRun`` shape, checked against the bracket itself."""
    result = storybook_run(draw, skill_table, classifier, seed=14)

    assert len(result.runs) == 8
    assert {run.position for run in result.runs} == set(range(1, 9))
    # Sorted best-finish-first, champion at the top — the documented key in
    # full, so the position tiebreak is pinned and not merely inherited from
    # Python's stable sort over a position-ordered draw.
    assert result.runs[0].is_champion
    keys = [(-run.matches_won, run.position) for run in result.runs]
    assert keys == sorted(keys)

    champion = result.runs[0]
    assert champion.player == result.champion.player
    assert champion.eliminated_by is None
    assert champion.furthest_round == "F"
    assert champion.matches_won == 3  # QF, SF, F
    assert champion.beat == tuple(
        m.loser for m in result.matches if m.winner == champion.player
    )
    assert champion.last_match is result.final

    for run in result.runs[1:]:
        assert not run.is_champion
        assert run.eliminated_by is not None
        # Their run ends on the match they lost, in the furthest round reached.
        assert run.last_match.loser == run.player
        assert run.last_match.winner == run.eliminated_by
        assert run.last_match.round_label == run.furthest_round
        assert run.matches_won == len(run.beat)
        assert run.player not in run.beat

    # Every match knocked exactly one entrant out; nobody plays after losing.
    assert sorted(run.furthest_round for run in result.runs) == sorted(
        ["QF"] * 4 + ["SF"] * 2 + ["F"] * 2
    )
    assert sum(run.matches_won for run in result.runs) == 7

    # run_of() is the same object, keyed by bracket position.
    assert result.run_of(champion.position) is champion
    with pytest.raises(KeyError):
        result.run_of(99)


def test_storybook_identity_fields_come_from_the_draw(draw, skill_table, classifier):
    """Seeds and ids are copied through, so a caller never re-resolves names."""
    result = storybook_run(draw, skill_table, classifier, seed=15)

    assert result.tournament_id == draw.tournament_id
    assert result.draw_size == draw.draw_size == 8
    assert result.surface == draw.surface
    assert result.best_of == draw.best_of

    by_position = {slot.position: slot for slot in draw.bracket}
    for run in result.runs:
        slot = by_position[run.position]
        assert (run.player, run.player_id) == (slot.player, slot.player_id)
        assert run.seed == draw.seed_of(slot)

    alice = result.run_of(1)
    assert (alice.player, alice.seed) == ("Alice Ace", 1)
    assert result.run_of(2).seed == 2
    assert result.run_of(3).seed is None  # unseeded


def test_storybook_is_deterministic_given_the_seed(draw, skill_table):
    """Same seed → identical result *and* identical text; a shared link replays."""
    first = storybook_run(draw, skill_table, CountingClassifier(), seed=99)
    second = storybook_run(draw, skill_table, CountingClassifier(), seed=99)

    assert first == second
    assert render_storybook(first) == render_storybook(second)


def test_storybook_different_seeds_tell_different_stories(draw, skill_table):
    """The seed reaches the *simulation*, not just the result's ``seed`` field.

    Compares the matches actually played, deliberately **not** the rendered
    text: the header prints the seed, so a ``storybook_run`` that ignored its
    argument and used a fixed RNG would still produce a different string per
    seed and a text-based assertion would pass vacuously. (It did — this test
    only fails on that mutation once it compares the bracket.)
    """
    played = {
        tuple(
            m.line
            for m in storybook_run(
                draw, skill_table, CountingClassifier(), seed=s
            ).matches
        )
        for s in range(6)
    }

    assert len(played) > 1


def test_render_storybook_is_round_by_round_and_ends_with_the_champion(
    draw, skill_table, classifier
):
    result = storybook_run(draw, skill_table, classifier, seed=16)
    text = render_storybook(result)
    lines = text.splitlines()

    # Both header lines pinned to result fields, exactly — a loose ``in`` check
    # would let the renderer swap in a fact it computed itself (the body check
    # in the separation test below only guards the round blocks).
    assert lines[0] == (
        f"{result.name} — {result.surface}, best of {result.best_of}, "
        f"{result.draw_size} entrants"
    )
    # (No ``(w=…)`` suffix: that branch is blend-only and is asserted in the
    # reconciliation-mode test, so this line assumes the non-blend default.)
    assert lines[1] == f"seed {result.seed} · reconciliation {result.mode}"

    # Every round appears as its own block, in order, with its matches under it.
    for storybook_round in result.rounds:
        header = lines.index(storybook_round.label)
        block = lines[header + 1 : header + 1 + len(storybook_round.matches)]
        assert block == [f"  {m.line}" for m in storybook_round.matches]
    assert lines.index("QF") < lines.index("SF") < lines.index("F")

    # Ends with the champion, and only that.
    assert lines[-1] == f"Champion: {result.champion.player}" + (
        f" [{result.champion_seed}]" if result.champion_seed is not None else ""
    )
    assert not text.endswith("\n")
    assert text.count("Champion:") == 1


def test_render_storybook_reads_only_the_result_structure(
    draw, skill_table, classifier
):
    """Data and renderer are separate: the text adds no fact the data lacks.

    Every non-blank body line is either a round label or one of the result's own
    match lines, so a frontend consuming :class:`StorybookResult` can reproduce
    this layout — and anything else — without re-deriving anything.
    """
    result = storybook_run(draw, skill_table, classifier, seed=17)
    lines = [line for line in render_storybook(result).splitlines() if line.strip()]

    labels = {r.label for r in result.rounds}
    match_lines = {f"  {m.line}" for m in result.matches}
    body = lines[2:-1]  # drop the two header lines and the champion line
    assert body
    assert all(line in labels or line in match_lines for line in body)


def test_storybook_defaults_to_the_system_reconciliation_mode(
    draw, skill_table, classifier
):
    """T2.4 decision: a pass-through, exactly like ``monte_carlo``'s.

    The core does not adopt ``config.SIM_CLI_RECONCILE_MODE`` even for the
    shareable mode — that mitigation belongs to the layer that builds the
    degraded adapter (seam 7). The result records what was used either way.
    """
    default = storybook_run(draw, skill_table, classifier, seed=18)
    assert default.mode == config.RECONCILE_MODE
    assert default.w == config.RECONCILE_BLEND_WEIGHT

    blended = storybook_run(
        draw, skill_table, classifier, seed=18, mode="blend", w=0.25
    )
    assert (blended.mode, blended.w) == ("blend", 0.25)
    assert "blend (w=0.25)" in render_storybook(blended)


def test_storybook_rejects_a_draw_with_placeholders(skill_table, classifier):
    """Inherited from ``simulate_bracket`` — refused before anything is played."""
    data = toy_draw_dict()
    data["bracket"][5]["player"] = "Qualifier"
    placeholder_draw = parse_draw(data, skill_table)

    with pytest.raises(ValueError, match="placeholder entrant"):
        storybook_run(placeholder_draw, skill_table, classifier, seed=19)
    assert classifier.calls == []


def test_full_example_draw_storybook_end_to_end():
    """The real 128-draw: a complete, readable 127-match narrative.

    The end-to-end smoke test for T2.4 — the placeholder-free
    ``example_usopen_2024_full.json`` bracket, real vendored skills, every match
    played out point by point. The classifier is the usual stub (the T2.2
    boundary), so "plausible champion" here means *a real entrant who actually
    won seven matches*, not a forecast — see seam 7 on why no stub-driven or
    CLI-adapter-driven bracket is one.
    """
    df = loader.load_atp_data(config.START_YEAR, config.END_YEAR)
    table = serve.build_skill_table(preprocess.preprocess_data(df))
    draw = load_draw(config.DRAWS_DIR / "example_usopen_2024_full.json", table)

    result = storybook_run(draw, table, CountingClassifier(), seed=2024)

    assert result.draw_size == 128
    assert [r.label for r in result.rounds] == [
        "R128", "R64", "R32", "R16", "QF", "SF", "F",
    ]
    assert [len(r.matches) for r in result.rounds] == [64, 32, 16, 8, 4, 2, 1]
    assert len(result.matches) == 127
    assert all(MATCH_LINE.match(m.line) for m in result.matches)

    # A plausible champion: a real entrant from the draw who won all seven.
    champion = result.runs[0]
    assert champion.is_champion
    assert result.champion in draw.bracket
    assert not result.champion.is_placeholder
    assert champion.matches_won == 7
    assert len(champion.beat) == 7 and len(set(champion.beat)) == 7
    assert champion.eliminated_by is None
    assert champion.last_match.round_label == "F"

    # 128 entrants, 127 of them beaten exactly once.
    assert len(result.runs) == 128
    assert sum(run.matches_won for run in result.runs) == 127
    assert sum(1 for run in result.runs if run.eliminated_by is None) == 1

    text = render_storybook(result)
    assert text.count(" def. ") == 127
    assert text.splitlines()[-1].startswith(f"Champion: {result.champion.player}")


# ---------------------------------------------------------------------------
# Layering
# ---------------------------------------------------------------------------


def test_tournament_module_does_not_import_cli():
    """The sim/ layering rule, asserted rather than assumed."""
    from pathlib import Path

    source = Path("src/sim/tournament.py").read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    assert "import cli" not in code
    assert "from cli" not in code
