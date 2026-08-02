"""Tests for cli/simulate_tournament.py (T2.5) — the tournament simulation CLI.

Covers the ticket's acceptance + testing criteria:
  * Smoke tests for both modes that call the *underlying* functions
    (``run_monte_carlo`` / ``run_storybook``), not the argument-parsing entry
    point, against a fixture skill table + a stub classifier — no vendored
    dataset and no pickled model, the same standard as T1.10's tests.
  * Missing and malformed draw files produce a helpful, specific message rather
    than a raw exception — including a draw that validates but cannot be
    simulated (placeholder entrants, T2.2's refusal).
  * Storybook determinism: the same seed replays the same tournament.
  * The Monte Carlo table is sorted by title probability and carries the
    documented columns.

Plus the two decisions this CLI owns: it reconciles with
``config.SIM_CLI_RECONCILE_MODE`` (not the system-wide ``config.RECONCILE_MODE``)
and it never runs ``monte_carlo`` with ``workers > 1``.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import cli.simulate_match as SM
import cli.simulate_tournament as ST
import config
from features.serve import PlayerSkill, SkillTable
from sim.draw import parse_draw

# Eight toy entrants, strongest first — mirrors tests/test_tournament.py.
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
    """Toy id-keyed skills on all three surfaces, with a real spread."""
    skills = {}
    for index, player_id in enumerate(TOY_PLAYERS.values()):
        for surface in config.VALID_SURFACES:
            mu = config.SURFACE_MU[surface]
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
    """A valid, placeholder-free 8-slot toy draw."""
    data = {
        "tournament_id": "toy_2026",
        "name": "Toy Open 2026",
        "surface": "Hard",
        "best_of": 5,
        "final_set_tiebreak": "10pt_at_6_6",
        "draw_size": 8,
        "seeds": {"Alice Ace": 1, "Bob Baseline": 2},
        "bracket": [
            {"position": i + 1, "player": name} for i, name in enumerate(TOY_PLAYERS)
        ],
    }
    data.update(overrides)
    return data


@pytest.fixture
def draw(skill_table):
    return parse_draw(toy_draw_dict(), skill_table)


class StubClassifier:
    """A ``ClassifierProb`` stand-in: player A always favoured by ``prob``."""

    def __init__(self, prob: float = 0.55) -> None:
        self.prob = prob
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, player_a: str, player_b: str, surface: str) -> float:
        self.calls.append((player_a, player_b, surface))
        return self.prob


@pytest.fixture
def ctx(skill_table) -> SM.SimContext:
    """A SimContext wired to the fixture table + stub classifier (no pipeline run)."""
    return SM.SimContext(
        data=pd.DataFrame(),
        surface_history={},
        h2h_history={},
        skill_table=skill_table,
        classifier=StubClassifier(),
        estimator_class="StubClassifier",
        player_names=list(TOY_PLAYERS),
    )


def write_draw(tmp_path, data, name: str = "toy.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def table_rows(table: str) -> list[str]:
    """The table's body rows: everything between the rule and the blank line."""
    lines = table.splitlines()
    start = next(i for i, line in enumerate(lines) if set(line) == {"-"}) + 1
    end = next(i for i, line in enumerate(lines[start:], start) if not line.strip())
    return lines[start:end]


def row_cells(row: str, n_survival: int) -> tuple[str, str, str, list[str], str, str]:
    """Split one body row into its columns *by fixed offset*, not by whitespace.

    Splitting on whitespace cannot catch a miscolumned table (player names
    contain spaces, and every numeric cell looks alike), so this slices at the
    module's own widths and returns
    ``(rank, seed, player, survival_cells, title, e_w)``.
    """
    prefix = 3 + 2 + 4 + 2 + ST._PLAYER_WIDTH
    w = ST._PCT_WIDTH
    survival = [
        row[prefix + i * w : prefix + (i + 1) * w].strip() for i in range(n_survival)
    ]
    title = row[prefix + n_survival * w : prefix + (n_survival + 1) * w].strip()
    return (
        row[:3].strip(),
        row[3:9].strip(),
        row[11 : 11 + ST._PLAYER_WIDTH].strip(),
        survival,
        title,
        row[prefix + (n_survival + 1) * w :].strip(),
    )


# --------------------------------------------------------------------------- #
# Smoke: the underlying functions, both modes.
# --------------------------------------------------------------------------- #
def test_monte_carlo_smoke(draw, ctx):
    """A short MC job aggregates into a sorted, coherent title board."""
    result = ST.run_monte_carlo(draw, ctx, n_runs=40, seed=7)

    assert result.n_runs == 40
    assert result.draw_size == 8
    assert len(result.players) == 8
    # Sorted by title probability, descending.
    probs = [player.p_title for player in result.players]
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) == pytest.approx(1.0)
    # Every entrant contests the first round; the last round is the final.
    assert result.round_labels == ("QF", "SF", "F")
    assert all(player.p_reach("QF") == 1.0 for player in result.players)


def test_storybook_smoke(draw, ctx):
    """One bracket, played out: 7 matches over 3 rounds and one champion."""
    result = ST.run_storybook(draw, ctx, seed=7)

    assert len(result.rounds) == 3
    assert [len(r.matches) for r in result.rounds] == [4, 2, 1]
    assert all(m.scoreline for m in result.matches)

    text = ST.render_storybook(result)
    assert text.splitlines()[-1] == (
        f"Champion: {result.champion.player}"
        + (f" [{result.champion_seed}]" if result.champion_seed is not None else "")
    )
    assert " def. " in text


def test_monte_carlo_favours_the_stronger_half(draw, ctx):
    """Sanity: the strongest entrant wins the title more often than the weakest."""
    result = ST.run_monte_carlo(draw, ctx, n_runs=200, seed=3)
    by_name = {player.player: player for player in result.players}
    assert by_name["Alice Ace"].p_title > by_name["Hank Half"].p_title


# --------------------------------------------------------------------------- #
# The Monte Carlo table.
# --------------------------------------------------------------------------- #
def test_table_columns_and_ordering(draw, ctx):
    """Header carries every round after the first, plus Title and E[W]."""
    result = ST.run_monte_carlo(draw, ctx, n_runs=40, seed=7)
    table = ST.format_montecarlo_table(result, top_n=8)
    lines = table.splitlines()

    header = next(line for line in lines if line.lstrip().startswith("#"))
    assert header.split() == ["#", "Seed", "Player", "SF", "F", "Title", "E[W]"]
    # QF is the first round of an 8-draw — 100% for everyone, so it is omitted.
    assert "QF" not in header

    # Rows are the sorted players, rank-numbered, percentages to 1dp.
    body = table_rows(table)
    assert len(body) == 8
    assert [int(line.split()[0]) for line in body] == list(range(1, 9))
    assert result.players[0].player in body[0]
    assert f"{result.players[0].p_title:.1%}" in body[0]
    assert f"{result.players[0].expected_rounds_won:.2f}" in body[0]
    # The top seed's marker is rendered, and unseeded entrants have none.
    assert "[1]" in table and "[3]" not in table


def test_table_cells_transcribe_the_raw_outcome(draw, ctx):
    """Every cell equals the PlayerOutcome field its column claims to show.

    Checked cell by cell at fixed offsets, because the failure this guards
    against is a *miscolumn* — a table that renders plausible percentages in
    the wrong columns (survival columns showing the title probability, say)
    passes every header/ordering assertion while being wrong in every row.
    """
    result = ST.run_monte_carlo(draw, ctx, n_runs=40, seed=7)
    survival_labels = result.round_labels[1:]
    rows = table_rows(ST.format_montecarlo_table(result, top_n=8))

    # The first-round column is omitted precisely because it is redundant —
    # assert that redundancy rather than assuming it.
    first = result.round_labels[0]
    assert all(player.p_reach(first) == 1.0 for player in result.players)

    assert len(rows) == len(result.players)
    for rank, (row, player) in enumerate(zip(rows, result.players), start=1):
        cells = row_cells(row, len(survival_labels))
        assert cells[0] == str(rank)
        assert cells[1] == (f"[{player.seed}]" if player.seed is not None else "")
        assert cells[2] == player.player[: ST._PLAYER_WIDTH]
        assert cells[3] == [f"{player.p_reach(l):.1%}" for l in survival_labels]
        assert cells[4] == f"{player.p_title:.1%}"
        assert cells[5] == f"{player.expected_rounds_won:.2f}"

    # And the survival columns are genuinely distinct from the title column —
    # otherwise the assertions above could hold for the wrong reason.
    top = result.players[0]
    assert any(top.p_reach(label) != top.p_title for label in survival_labels)


def test_table_top_n_and_footer(draw, ctx):
    """--top clips the table and says so; 0 lists the whole field."""
    result = ST.run_monte_carlo(draw, ctx, n_runs=40, seed=7)

    clipped = ST.format_montecarlo_table(result, top_n=3)
    assert "Showing the top 3 of 8 entrants." in clipped
    assert len(table_rows(clipped)) == 3

    everyone = ST.format_montecarlo_table(result, top_n=0)
    assert "Showing the top" not in everyone
    assert len(table_rows(everyone)) == 8

    # The seam-7 caveat travels with any published probability table.
    assert ST.ADAPTER_CAVEAT in clipped and ST.ADAPTER_CAVEAT in everyone


# --------------------------------------------------------------------------- #
# Determinism (T2.4's guarantee, surfaced through the CLI).
# --------------------------------------------------------------------------- #
def test_storybook_same_seed_reproduces_the_tournament(draw, ctx):
    first = ST.render_storybook(ST.run_storybook(draw, ctx, seed=42))
    second = ST.render_storybook(ST.run_storybook(draw, ctx, seed=42))
    assert first == second


def test_storybook_different_seeds_diverge(draw, ctx):
    """Different seeds give different stories (checked across a few seeds)."""
    baseline = ST.render_storybook(ST.run_storybook(draw, ctx, seed=42))
    others = [
        ST.render_storybook(ST.run_storybook(draw, ctx, seed=seed))
        for seed in (1, 2, 3, 4)
    ]
    assert any(story != baseline for story in others)


def test_monte_carlo_same_seed_reproduces_counts(draw, ctx):
    """Identical counts, compared per player — not merely "it ran twice"."""
    first = ST.run_monte_carlo(draw, ctx, n_runs=50, seed=11)
    second = ST.run_monte_carlo(draw, ctx, n_runs=50, seed=11)

    # to_records() is keyed by player and carries titles, reached and E[W], so
    # this cannot pass on a coincidentally-equal multiset of counts.
    assert first.to_records() == second.to_records()
    assert [(p.player, p.titles, p.matches_won, p.reached) for p in first.players] == [
        (p.player, p.titles, p.matches_won, p.reached) for p in second.players
    ]
    # A different seed moves the counts, so the equality above has content.
    other = ST.run_monte_carlo(draw, ctx, n_runs=50, seed=12)
    assert [p.titles for p in other.players] != [p.titles for p in first.players]


# --------------------------------------------------------------------------- #
# Draw-file failure modes: helpful messages, not tracebacks.
# --------------------------------------------------------------------------- #
def test_missing_draw_file_message(tmp_path, skill_table):
    loaded = ST.load_tournament_draw(tmp_path / "nope.json", skill_table)
    assert loaded.draw is None
    assert "nope.json" in loaded.message
    assert "not found" in loaded.message
    assert "Traceback" not in loaded.message


def test_malformed_json_message(tmp_path, skill_table):
    path = tmp_path / "broken.json"
    path.write_text("{not json at all", encoding="utf-8")

    loaded = ST.load_tournament_draw(path, skill_table)
    assert loaded.draw is None
    assert "Invalid draw file" in loaded.message
    assert "not valid JSON" in loaded.message


def test_validation_problems_are_all_listed(tmp_path, skill_table):
    """The accumulated problem list is shown in full, one bullet per problem."""
    bad = toy_draw_dict(surface="Ice", best_of=4, seeds={})
    bad["bracket"][0]["player"] = "Nobody Whatsoever"
    path = write_draw(tmp_path, bad, "bad.json")

    loaded = ST.load_tournament_draw(path, skill_table)
    assert loaded.draw is None
    assert "3 problem(s) found" in loaded.message
    assert "surface must be one of" in loaded.message
    assert "best_of must be one of" in loaded.message
    assert "Nobody Whatsoever" in loaded.message
    assert loaded.message.count("     - ") == 3


def test_valid_file_loads(tmp_path, skill_table):
    path = write_draw(tmp_path, toy_draw_dict())
    loaded = ST.load_tournament_draw(path, skill_table)
    assert loaded.message == ""
    assert loaded.draw is not None and loaded.draw.draw_size == 8


def test_main_reports_a_missing_file_without_building_the_context(
    tmp_path, monkeypatch, capsys
):
    """A bad path fails before the (expensive) pipeline is ever run."""
    def explode(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("build_context must not run for a missing draw file")

    monkeypatch.setattr(ST, "build_context", explode)
    code = ST.main(["--draw", str(tmp_path / "absent.json")])

    assert code == 1
    assert "not found" in capsys.readouterr().out


def test_main_reports_placeholder_entrants(tmp_path, monkeypatch, capsys, ctx):
    """A draw that validates but cannot be simulated is refused readably."""
    data = toy_draw_dict()
    data["bracket"][7]["player"] = "Qualifier"
    path = write_draw(tmp_path, data, "placeholders.json")

    monkeypatch.setattr(ST, "build_context", lambda *a, **k: ctx)
    code = ST.main(["--draw", str(path), "--mode", "storybook"])

    out = capsys.readouterr().out
    assert code == 1
    assert "placeholder" in out
    assert "8:'Qualifier'" in out  # the offending slot is named


# --------------------------------------------------------------------------- #
# main(): both modes end to end (fixture context, no real pipeline).
# --------------------------------------------------------------------------- #
def test_main_montecarlo_prints_a_sorted_table(tmp_path, monkeypatch, capsys, ctx):
    path = write_draw(tmp_path, toy_draw_dict())
    monkeypatch.setattr(ST, "build_context", lambda *a, **k: ctx)

    code = ST.main(["--draw", str(path), "--mode", "montecarlo", "--runs", "40"])
    out = capsys.readouterr().out

    assert code == 0
    assert "Toy Open 2026" in out and "40 runs" in out
    titles = [float(line.split()[-2].rstrip("%")) for line in table_rows(out)]
    assert titles == sorted(titles, reverse=True)


def test_main_storybook_prints_the_rendered_run(tmp_path, monkeypatch, capsys, ctx):
    path = write_draw(tmp_path, toy_draw_dict())
    monkeypatch.setattr(ST, "build_context", lambda *a, **k: ctx)

    code = ST.main(["--draw", str(path), "--mode", "storybook", "--seed", "42"])
    out = capsys.readouterr().out

    assert code == 0
    assert out.strip() == ST.render_storybook(ST.run_storybook(
        ST.load_tournament_draw(path, ctx.skill_table).draw, ctx, seed=42
    ))


def test_main_storybook_notes_that_runs_is_ignored(tmp_path, monkeypatch, capsys, ctx):
    path = write_draw(tmp_path, toy_draw_dict())
    monkeypatch.setattr(ST, "build_context", lambda *a, **k: ctx)

    ST.main(["--draw", str(path), "--mode", "storybook", "--runs", "99"])
    assert "--runs is ignored in storybook mode" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# The two decisions this CLI owns.
# --------------------------------------------------------------------------- #
def test_uses_the_cli_reconcile_mode_not_the_global_default(draw, ctx, monkeypatch):
    """SIM_CLI_RECONCILE_MODE ('blend'), never config.RECONCILE_MODE."""
    assert config.SIM_CLI_RECONCILE_MODE != config.RECONCILE_MODE

    captured: dict = {}
    real_mc, real_sb = ST.monte_carlo, ST.storybook_run

    def mc_spy(*args, **kwargs):
        captured.update(kwargs)
        return real_mc(*args, **kwargs)

    def sb_spy(*args, **kwargs):
        captured.update(kwargs)
        return real_sb(*args, **kwargs)

    monkeypatch.setattr(ST, "monte_carlo", mc_spy)
    ST.run_monte_carlo(draw, ctx, n_runs=5, seed=1)
    assert captured["mode"] == config.SIM_CLI_RECONCILE_MODE == "blend"

    captured.clear()
    monkeypatch.setattr(ST, "storybook_run", sb_spy)
    ST.run_storybook(draw, ctx, seed=1)
    assert captured["mode"] == config.SIM_CLI_RECONCILE_MODE


def test_global_reconcile_mode_is_untouched():
    """This CLI mitigates its own default; it must not move the project's."""
    assert config.RECONCILE_MODE == "classifier_anchor"


def test_monte_carlo_always_runs_single_process(draw, ctx, monkeypatch):
    """workers=1 always, and no --workers flag exists to say otherwise (seam 8)."""
    captured: dict = {}
    real = ST.monte_carlo

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(ST, "monte_carlo", spy)
    ST.run_monte_carlo(draw, ctx, n_runs=5, seed=1)
    assert captured["workers"] == 1

    with pytest.raises(SystemExit):
        ST._parse_args(["--draw", "x.json", "--workers", "4"])
