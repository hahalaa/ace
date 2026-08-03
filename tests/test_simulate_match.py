"""Tests for cli/simulate_match.py (T1.10) — the single-match simulation CLI.

Covers the ticket's acceptance + testing criteria:
  * The underlying function (not the input loop) simulates a matchup from a
    *fixture* skill table + a *stub* classifier — no 35k-row real dataset and no
    real pickled model required.
  * The ClassifierProb adapter builds the same name-keyed feature row the REPL
    builds and calls predict_proba on it.
  * Unknown / ambiguous names produce a helpful message carrying the shared
    resolver's candidate list (identical text to the REPL's).
  * --seed reproducibility: the same seed replays the same scoreline AND the
    same MC estimate; a different seed generally does not.
  * The cli -> sim import direction only: sim/reconcile.py never imports cli/.

Plus the seam-7 mitigation follow-up: the CLI defaults to "blend"
(config.SIM_CLI_RECONCILE_MODE), --reconcile-mode classifier_anchor is honoured
and warned about, and config.RECONCILE_MODE is left alone.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import sys

import numpy as np
import pandas as pd
import pytest

import cli.interactive as interactive
import cli.simulate_match as SM
import config
from common.classifier_adapter import MatchupFeatures
from common.names import NameIndex
from features.serve import PlayerSkill, SkillTable
from sim.match import MatchResult, SetResult


# --------------------------------------------------------------------------- #
# Fixtures: a tiny skill table, a stub estimator, and a fake SimContext.
# --------------------------------------------------------------------------- #
def make_skill_table() -> SkillTable:
    """Two known players on Hard, A clearly the stronger server/returner."""
    skills = {
        ("A", "Hard"): PlayerSkill(spw=0.68, rpw=0.42, n_serve_pts=5000, n_return_pts=5000),
        ("B", "Hard"): PlayerSkill(spw=0.63, rpw=0.37, n_serve_pts=5000, n_return_pts=5000),
    }
    return SkillTable(
        skills,
        {"Player A": "A", "Player B": "B"},
        dict(config.SURFACE_MU),
        cutoff=None,
    )


def make_data() -> pd.DataFrame:
    """A two-row engineered-frame stand-in.

    Enough for ``get_latest`` (rank/age) *and*, since T3.5, for the adapter's
    as-of-now rolling-form snapshot — hence the outcome/games/sets columns, which
    are what ``features.rolling`` averages. A frame without them is not an
    engineered frame, and the adapter now says so instead of quietly filling the
    20 recent-form features with constants.

    Player A won the later match 2-0 (12 games to 8) and lost the earlier one
    1-2 (16 games to 20), so A's rolling form is decidedly not neutral.
    """
    return pd.DataFrame(
        {
            "p1_name": ["Player A", "Player A"],
            "p2_name": ["Player B", "Player B"],
            "p1_rank": [1.0, 2.0],
            "p2_rank": [3.0, 4.0],
            "p1_age": [22.0, 21.5],
            "p2_age": [24.0, 23.5],
            "tourney_date": pd.to_datetime(["2025-01-01", "2024-01-01"]),
            "target": [1, 0],
            "p1_games_won": [12, 16],
            "p1_games_lost": [8, 20],
            "p1_sets_won": [2, 1],
            "p1_sets_lost": [0, 2],
            "p2_games_won": [8, 20],
            "p2_games_lost": [12, 16],
            "p2_sets_won": [0, 2],
            "p2_sets_lost": [2, 1],
        }
    )


HISTORIES = (
    {"Player A": {"Hard": [8, 10]}, "Player B": {"Hard": [5, 10]}},  # surface_history
    {("Player A", "Player B"): [3, 1]},                              # h2h_history
)


class StubEstimator:
    """Minimal predict_proba stand-in; records the feature rows it was given."""

    def __init__(self, p_a: float = 0.70) -> None:
        self.p_a = p_a
        self.rows: list[pd.DataFrame] = []

    def predict_proba(self, X):
        self.rows.append(X)
        return np.array([[1.0 - self.p_a, self.p_a]])


def make_context(p_a: float = 0.70) -> tuple[SM.SimContext, StubEstimator]:
    """A SimContext wired to the fixture table + stub estimator (no pipeline run)."""
    data = make_data()
    surf_hist, h2h_hist = HISTORIES
    estimator = StubEstimator(p_a)
    ctx = SM.SimContext(
        data=data,
        surface_history=surf_hist,
        h2h_history=h2h_hist,
        skill_table=make_skill_table(),
        classifier=SM.make_classifier_prob(estimator, data, surf_hist, h2h_hist),
        estimator_class="StubEstimator",
        player_names=["Player A", "Player B"],
    )
    return ctx, estimator


# --------------------------------------------------------------------------- #
# Smoke test: the underlying function, fixture table + stub classifier.
# --------------------------------------------------------------------------- #
def test_simulate_named_match_smoke():
    """Two known players -> a real scoreline and a win % (no real data/model)."""
    ctx, _ = make_context()
    sim = SM.simulate_named_match(
        "Player A", "Player B", "Hard", ctx, best_of=3, n_sims=50, seed=7
    )

    assert isinstance(sim.result, MatchResult)
    assert sim.result.best_of == 3
    assert sim.winner in ("Player A", "Player B")
    assert 0.0 <= sim.win_prob_a <= 1.0
    assert sim.n_sims == 50
    # A scoreline like "6-4 3-6 7-6(5)" — one token per set played.
    tokens = sim.scoreline.split()
    assert len(tokens) == len(sim.result.sets) >= 2
    assert all("-" in t for t in tokens)


def test_simulate_named_match_bo5_and_stronger_player_favoured():
    """A is the stronger player and the stub classifier agrees -> A wins most MC runs."""
    ctx, _ = make_context(p_a=0.75)
    sim = SM.simulate_named_match(
        "Player A", "Player B", "Hard", ctx, best_of=5, n_sims=300, seed=1
    )
    assert sim.result.best_of == 5
    assert sim.win_prob_a > 0.6  # anchored on P_clf = 0.75
    assert sim.p_clf == pytest.approx(0.75)


def test_unresolvable_name_raises_from_the_sim_layer():
    """A name absent from the skill table fails loudly (T1.8's contract)."""
    ctx, _ = make_context()
    with pytest.raises(ValueError, match="resolve player name"):
        SM.simulate_named_match(
            "Nobody At All", "Player B", "Hard", ctx, n_sims=5, seed=0
        )


# --------------------------------------------------------------------------- #
# The ClassifierProb adapter.
# --------------------------------------------------------------------------- #
def test_adapter_builds_a_fully_real_feature_row_and_calls_predict_proba():
    """All 27 features are real state — the T3.5 fix, from the CLI's side.

    Before T3.5 this test asserted the row was byte-identical to
    ``interactive.build_feature_row``'s, i.e. that 20 of the 27 features were
    synthetic constants (``ace-04-current-state.md §7`` seam 7). The CLI now
    shares ``common.classifier_adapter`` with the API, so it gets the real
    rolling-form values too; what is pinned here is that the 7 history-derived
    features are unchanged from T1.10 **and** that the other 20 are no longer
    constants.
    """
    data = make_data()
    surf_hist, h2h_hist = HISTORIES
    estimator = StubEstimator(0.64)
    adapter = SM.make_classifier_prob(estimator, data, surf_hist, h2h_hist)

    prob = adapter("Player A", "Player B", "Hard")
    assert prob == pytest.approx(0.64)
    assert len(estimator.rows) == 1

    row = estimator.rows[0]
    assert list(row.columns) == config.MODEL_FEATURES
    assert not row.isna().any().any()

    # The seven T1.10 already built correctly, unchanged.
    assert row["p1_rank"].iloc[0] == 1.0          # A's latest rank (2025 row)
    assert row["p2_rank"].iloc[0] == 3.0
    assert row["p1_age"].iloc[0] == 22.0
    assert row["p2_age"].iloc[0] == 24.0
    assert row["p1_surface_win_pct"].iloc[0] == pytest.approx(0.8)
    assert row["p2_surface_win_pct"].iloc[0] == pytest.approx(0.5)
    assert row["h2h_diff"].iloc[0] == 2           # A leads 3-1

    # The twenty that used to be synthetic: A won one of two, B the other, and
    # the games/sets averages are the fixture's actual numbers.
    assert row["p1_recent_win_rate_5"].iloc[0] == pytest.approx(0.5)
    assert row["p1_recent_games_won_avg_5"].iloc[0] == pytest.approx(14.0)
    assert row["p1_recent_games_lost_avg_5"].iloc[0] == pytest.approx(14.0)
    assert row["p1_recent_sets_won_avg_5"].iloc[0] == pytest.approx(1.5)
    assert row["p2_recent_games_won_avg_5"].iloc[0] == pytest.approx(14.0)
    assert row["p2_recent_sets_won_avg_5"].iloc[0] == pytest.approx(1.0)

    # And they are *not* the constants the old row carried (0 for every
    # games/sets average) — the mutation this test exists to catch.
    old_row = interactive.build_feature_row(1.0, 3.0, 22.0, 24.0, 0.8, 0.5, 2)
    assert old_row["p1_recent_games_won_avg_5"].iloc[0] == 0
    assert row["p1_recent_games_won_avg_5"].iloc[0] != 0


def test_adapter_is_memoised_so_mc_runs_cost_one_predict_proba():
    """1 storybook + N MC runs must not mean N+1 predict_proba calls."""
    ctx, estimator = make_context()
    SM.simulate_named_match(
        "Player A", "Player B", "Hard", ctx, best_of=3, n_sims=40, seed=3
    )
    assert len(estimator.rows) == 1


def test_adapter_cache_key_includes_surface():
    """Surface is part of the memoisation key — a Clay request must not be
    served the cached Hard probability.

    The feature row is surface-dependent (``p1_surface_win_pct`` comes from the
    surface record), so dropping ``surface`` from the key would silently return
    a stale wrong-surface P_clf. Mutation-checked: with the key reduced to
    ``(player_a, player_b)`` this test fails and nothing else in the suite does.
    """
    data = make_data()
    surf_hist, h2h_hist = HISTORIES
    estimator = StubEstimator()
    adapter = SM.make_classifier_prob(estimator, data, surf_hist, h2h_hist)

    adapter("Player A", "Player B", "Hard")
    adapter("Player A", "Player B", "Clay")  # unplayed -> DEFAULT_WIN_PCT
    assert len(estimator.rows) == 2, "surface missing from the cache key"

    hard, clay = estimator.rows
    assert hard["p1_surface_win_pct"].iloc[0] == pytest.approx(0.8)
    assert clay["p1_surface_win_pct"].iloc[0] == pytest.approx(config.DEFAULT_WIN_PCT)

    # ...and the cache still works: a repeat of either key adds no call.
    adapter("Player A", "Player B", "Hard")
    adapter("Player A", "Player B", "Clay")
    assert len(estimator.rows) == 2


def test_adapter_raises_for_a_player_with_no_history():
    """No feature row is possible without match history — fail, don't default."""
    data = make_data()
    surf_hist, h2h_hist = HISTORIES
    adapter = SM.make_classifier_prob(StubEstimator(), data, surf_hist, h2h_hist)
    with pytest.raises(ValueError, match="No match history"):
        adapter("Ghost Player", "Player B", "Hard")


def test_matchup_stats_inherits_the_shared_feature_fields():
    """``MatchupStats`` must *subclass* the adapter's ``MatchupFeatures``.

    That is the stated reason the printed matchup table and the feature row the
    classifier scores cannot drift apart: they are the same fields, declared
    once. Re-declaring them here instead would compile, pass every other test,
    and reintroduce exactly the duplication the subclassing removes — so the
    relationship is asserted, not left implied.
    """
    assert issubclass(SM.MatchupStats, MatchupFeatures)

    inherited = [f.name for f in dataclasses.fields(MatchupFeatures)]
    declared = [f.name for f in dataclasses.fields(SM.MatchupStats)]
    # Every shared field comes from the base, in order, and the CLI adds only
    # its own rendered H2H line on the end.
    assert declared == inherited + ["h2h_msg"]
    assert set(SM.MatchupStats.__annotations__) == {"h2h_msg"}


def test_matchup_stats_reuses_history_helpers():
    """Surface %, H2H and rank/age come from the REPL helpers, not new logic."""
    data = make_data()
    surf_hist, h2h_hist = HISTORIES
    stats = SM.matchup_stats("Player A", "Player B", "Hard", data, surf_hist, h2h_hist)

    assert (stats.a_wins, stats.a_total) == (8, 10)
    assert (stats.b_wins, stats.b_total) == (5, 10)
    assert stats.a_pct == pytest.approx(0.8)
    assert stats.h2h_diff == 2
    assert "Player A leads 3-1" in stats.h2h_msg
    assert stats.a_rank == 1.0 and stats.b_rank == 3.0


def test_matchup_stats_unplayed_surface_falls_back_to_default_win_pct():
    data = make_data()
    surf_hist, h2h_hist = HISTORIES
    stats = SM.matchup_stats("Player A", "Player B", "Clay", data, surf_hist, h2h_hist)
    assert stats.a_pct == config.DEFAULT_WIN_PCT
    assert stats.b_pct == config.DEFAULT_WIN_PCT


# --------------------------------------------------------------------------- #
# Name resolution: helpful messages with the shared resolver's candidate list.
# --------------------------------------------------------------------------- #
def test_unknown_name_message_is_helpful():
    index = NameIndex.from_names(["Carlos Alcaraz", "Jannik Sinner"])
    res = SM.resolve_display_name("Zzzzqqq Xyzzy", index)
    assert res.name is None
    assert "not found" in res.message
    assert "Zzzzqqq Xyzzy" in res.message


def test_ambiguous_name_message_carries_the_resolver_candidate_list():
    """The suggestion mechanism is the REPL's, verbatim — not a reimplementation."""
    names = ["Alexander Zverev", "Mischa Zverev", "Carlos Alcaraz"]
    index = NameIndex.from_names(names)

    res = SM.resolve_display_name("Zverev", index)
    assert res.name is None
    assert "Alexander Zverev" in res.message and "Mischa Zverev" in res.message

    # Identical to what the REPL prints for the same query.
    from common.names import resolve_name as shared_resolve
    match = shared_resolve("Zverev", index)
    assert res.message == interactive.ambiguity_message("Zverev", match)


def test_resolve_display_name_deduces_and_reports():
    index = NameIndex.from_names(["Carlos Alcaraz", "Jannik Sinner"])

    exact = SM.resolve_display_name("Carlos Alcaraz", index)
    assert exact.name == "Carlos Alcaraz"
    assert exact.message == ""

    deduced = SM.resolve_display_name("alcaraz", index)
    assert deduced.name == "Carlos Alcaraz"
    assert "Deduced: Carlos Alcaraz" in deduced.message


def test_ambiguity_message_helper_still_drives_the_repl_print(capsys):
    """The extracted message helper is what interactive's printer emits."""
    index = NameIndex.from_names(["Alexander Zverev", "Mischa Zverev"])
    assert interactive.resolve_player_name("Zverev", index.names) is None
    printed = capsys.readouterr().out.strip()
    from common.names import resolve_name as shared_resolve
    assert printed == interactive.ambiguity_message(
        "Zverev", shared_resolve("Zverev", index)
    )


# --------------------------------------------------------------------------- #
# Determinism (--seed).
# --------------------------------------------------------------------------- #
def test_same_seed_reproduces_scoreline_and_mc_estimate():
    ctx_1, _ = make_context()
    ctx_2, _ = make_context()
    kwargs = dict(best_of=5, n_sims=200, seed=99)

    a = SM.simulate_named_match("Player A", "Player B", "Hard", ctx_1, **kwargs)
    b = SM.simulate_named_match("Player A", "Player B", "Hard", ctx_2, **kwargs)

    assert a.scoreline == b.scoreline
    assert a.result == b.result
    assert a.win_prob_a == b.win_prob_a
    assert a.winner == b.winner


def test_different_seeds_diverge():
    """Guards against a seed that is accepted but never actually threaded."""
    ctx, _ = make_context()
    runs = {
        SM.simulate_named_match(
            "Player A", "Player B", "Hard", ctx, best_of=5, n_sims=150, seed=s
        ).scoreline
        for s in range(6)
    }
    assert len(runs) > 1


# --------------------------------------------------------------------------- #
# Scoreline rendering.
# --------------------------------------------------------------------------- #
def test_format_scoreline_renders_tiebreaks():
    result = MatchResult(
        winner=0,
        sets=[
            SetResult(winner=0, games_a=6, games_b=4),
            SetResult(winner=1, games_a=3, games_b=6),
            SetResult(winner=0, games_a=7, games_b=6, tb_score=(7, 5)),
        ],
        best_of=3,
    )
    assert SM.format_scoreline(result) == "6-4 3-6 7-6(5)"


def test_format_scoreline_uses_the_losers_tiebreak_points():
    """A tiebreak lost by A still shows the loser's (A's) points in parens."""
    s = SetResult(winner=1, games_a=6, games_b=7, tb_score=(4, 7))
    assert SM.format_set(s) == "6-7(4)"


# --------------------------------------------------------------------------- #
# Reconciliation mode (seam-7 mitigation): CLI defaults to blend.
# --------------------------------------------------------------------------- #
def run_main(monkeypatch, argv: list[str]) -> tuple[int, dict]:
    """Drive main() against the fixture context, capturing simulate_named_match's kwargs."""
    ctx, _ = make_context()
    captured: dict = {}
    real = SM.simulate_named_match

    def spy(*args, **kwargs):
        captured.update(kwargs)
        sim = real(*args, **kwargs)
        captured["sim"] = sim
        return sim

    monkeypatch.setattr(SM, "build_context", lambda *a, **k: ctx)
    monkeypatch.setattr(SM, "simulate_named_match", spy)
    return SM.main(argv), captured


def test_cli_defaults_to_blend_mode(monkeypatch, capsys):
    """No --reconcile-mode flag -> blend, from config.SIM_CLI_RECONCILE_MODE.

    The CLI must NOT fall through to config.RECONCILE_MODE
    ("classifier_anchor"), which would anchor every scoreline on the adapter's
    form-blind P_clf (ace-04-current-state.md §7 seam 7).
    """
    assert config.SIM_CLI_RECONCILE_MODE == "blend"
    assert SM._parse_args(["Player A", "Player B"]).reconcile_mode == "blend"

    code, captured = run_main(
        monkeypatch, ["Player A", "Player B", "--sims", "5", "--seed", "1"]
    )
    assert code == 0
    assert captured["reconcile_mode"] == "blend"
    assert captured["sim"].reconcile_mode == "blend"

    out = capsys.readouterr().out
    assert "mode: blend" in out
    assert SM.ANCHOR_MODE_CAVEAT not in out  # no warning in blend mode


def test_reconcile_mode_flag_opts_into_classifier_anchor(monkeypatch, capsys):
    """--reconcile-mode classifier_anchor is respected and prints the caveat."""
    code, captured = run_main(
        monkeypatch,
        [
            "Player A", "Player B",
            "--sims", "5", "--seed", "1",
            "--reconcile-mode", "classifier_anchor",
        ],
    )
    assert code == 0
    assert captured["reconcile_mode"] == "classifier_anchor"
    assert captured["sim"].reconcile_mode == "classifier_anchor"

    out = capsys.readouterr().out
    assert "mode: classifier_anchor" in out
    assert SM.ANCHOR_MODE_CAVEAT in out
    # Post-T3.5 the note explains what anchoring *does* (the classifier alone
    # picks the winner) rather than warning that the row behind it is synthetic.
    assert "classifier alone decides the winner" in out
    assert "synthetic" not in out


def test_mode_reaches_the_sim_layer(monkeypatch):
    """The mode is threaded into sim.reconcile, not just stored on the result."""
    ctx, _ = make_context()
    seen: list[str] = []
    real = SM.simulate_reconciled_match

    def spy(*args, **kwargs):
        seen.append(kwargs["mode"])
        return real(*args, **kwargs)

    monkeypatch.setattr(SM, "simulate_reconciled_match", spy)

    SM.simulate_named_match("Player A", "Player B", "Hard", ctx, best_of=3, n_sims=3, seed=2)
    assert seen and set(seen) == {"blend"}

    seen.clear()
    SM.simulate_named_match(
        "Player A", "Player B", "Hard", ctx,
        best_of=3, n_sims=3, seed=2, reconcile_mode="classifier_anchor",
    )
    assert seen and set(seen) == {"classifier_anchor"}


def test_blend_lets_the_point_model_speak_when_p_clf_is_flat():
    """Why the default was changed: a 50/50 P_clf must not flatten a real gap.

    A is the clearly stronger player in the fixture skill table. Anchoring on a
    coin-flip P_clf reproduces the coin flip; blending keeps half the point
    model's edge.
    """
    ctx, _ = make_context(p_a=0.5)  # the flattened P_clf seam 7 measured
    kwargs = dict(best_of=5, n_sims=400, seed=11)

    anchored = SM.simulate_named_match(
        "Player A", "Player B", "Hard", ctx, reconcile_mode="classifier_anchor", **kwargs
    )
    blended = SM.simulate_named_match(
        "Player A", "Player B", "Hard", ctx, reconcile_mode="blend", **kwargs
    )

    assert anchored.win_prob_a == pytest.approx(0.5, abs=0.06)
    assert blended.win_prob_a > anchored.win_prob_a


def test_global_reconcile_mode_is_untouched():
    """This fix is CLI-scoped: T1.8's system-wide default must not have moved."""
    import sim.reconcile as reconcile

    assert config.RECONCILE_MODE == "classifier_anchor"
    assert config.SIM_CLI_RECONCILE_MODE != config.RECONCILE_MODE
    # sim/ still defaults to the global constant for non-CLI callers.
    for fn in (reconcile.reconciled_prob, reconcile.simulate_reconciled_match):
        assert inspect.signature(fn).parameters["mode"].default == config.RECONCILE_MODE


# --------------------------------------------------------------------------- #
# Layering: cli -> sim only, never the reverse.
# --------------------------------------------------------------------------- #
def test_sim_reconcile_does_not_import_cli():
    """sim/ receives the adapter as an argument; it must never reach into cli/."""
    for module in ("sim.reconcile", "sim.match", "sim.points"):
        sys.modules.pop(module, None)
    for name in [m for m in sys.modules if m == "cli" or m.startswith("cli.")]:
        sys.modules.pop(name, None)

    importlib.import_module("sim.reconcile")
    assert not [m for m in sys.modules if m == "cli" or m.startswith("cli.")]

    source = inspect.getsource(importlib.import_module("sim.reconcile"))
    assert "import cli" not in source and "from cli" not in source


def test_cli_constructs_the_adapter_and_passes_it_in():
    """The dependency inversion runs cli -> sim: reconcile only receives a callable."""
    sig = inspect.signature(
        importlib.import_module("sim.reconcile").simulate_reconciled_match
    )
    assert "classifier" in sig.parameters
    # The CLI is the side that builds it.
    assert callable(SM.make_classifier_prob)
    ctx, _ = make_context()
    assert callable(ctx.classifier)
