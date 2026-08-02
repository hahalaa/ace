"""Tests for src/sim/tournament.py — T2.2's single bracket simulation.

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
    format_scoreline,
    reconciled_win_prob,
    round_label,
    round_labels,
    simulate_bracket,
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
