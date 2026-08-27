"""Tests for ``common/classifier_adapter.py``, the shared, real-feature
``ClassifierProb`` adapter that closes ``ace-04-current-state.md §7`` seam 7.

Four things this file exists to prove:

* **The rolling-form features are the pipeline's own, not constants.** Asserted
  by equivalence rather than by eyeball: the row the adapter assembles for
  ``(A, B, surface)`` is compared, column by column, against what
  ``features.engineering.add_features`` computes for a *real* match between A and
  B appended to the end of the same frame. If the snapshot accessor and the
  training pass ever disagree, this fails.
* **No ``cli/`` module is imported by ``api/`` or by the precompute script**,
  statically (an AST walk) or at runtime (a clean interpreter that resolves the
  configured adapter and reports every ``cli`` module in ``sys.modules``). The
  runtime half is what a static audit could not claim.
* **The real adapter works through a live endpoint.** The ``/storybook`` tests
  inject a stub ``ClassifierProb``, so nothing else exercises the shipped
  adapter over HTTP. Here
  ``/storybook`` runs on the factory ``config.API_CLASSIFIER_ADAPTER`` actually
  resolves, over a pipeline-built context and a real fitted estimator.
* **The behavioural contract survives the move**: memoisation
  per matchup, surface in the cache key, and a loud failure for a player with no
  history.

Nothing here touches the vendored CSVs or the pinned model: the frame is a small
synthetic tournament put through the *real* pipeline (``add_features``), and the
estimator is a logistic regression fitted on it.
"""

from __future__ import annotations

import ast
import dataclasses
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import common.classifier_adapter as adapter
import config
import features.engineering as engineering
import features.rolling as rolling
from api.deps import ApiContext
from api.main import (
    adapter_name,
    build_classifier,
    create_app,
    resolve_classifier_factory,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DRAWS = Path(__file__).parent / "fixtures" / "draws"
FIXTURE_CACHE = Path(__file__).parent / "fixtures" / "cache"

# The entrants of tests/fixtures/draws/toy_open.json, so the frame below can back
# a live /storybook run.
TOY_PLAYERS: dict[str, str] = {
    "Alice Ace": "A1",
    "Bob Baseline": "B2",
    "Cara Chip": "C3",
    "Dan Drop": "D4",
    "Eve Emphatic": "E5",
    "Frank Fault": "F6",
    "Gina Grip": "G7",
    "Hank Half": "H8",
}

# Constant per player across the whole frame, so "the player's latest rank/age"
# has an obvious independent value the assertions below can check against.
TOY_RANK: dict[str, float] = {
    name: float(index + 1) for index, name in enumerate(TOY_PLAYERS)
}
TOY_AGE: dict[str, float] = {
    name: 21.0 + index * 0.7 for index, name in enumerate(TOY_PLAYERS)
}


# --------------------------------------------------------------------------- #
# A small, structurally real match frame.
# --------------------------------------------------------------------------- #
def _match_row(
    date: pd.Timestamp,
    surface: str,
    winner: str,
    loser: str,
    p1_first: bool,
    games: tuple[int, int],
    sets: tuple[int, int],
    rank: dict[str, float],
    age: dict[str, float],
) -> dict:
    """One preprocessed-shaped row, already oriented p1/p2 with its target."""
    p1, p2 = (winner, loser) if p1_first else (loser, winner)
    p1_games, p2_games = games if p1_first else games[::-1]
    p1_sets, p2_sets = sets if p1_first else sets[::-1]
    row = {
        "tourney_date": date,
        "surface": surface,
        "score": "6-4 6-3",
        "has_serve_stats": True,
        "p1_name": p1,
        "p2_name": p2,
        "p1_id": TOY_PLAYERS[p1],
        "p2_id": TOY_PLAYERS[p2],
        "p1_rank": rank[p1],
        "p2_rank": rank[p2],
        "p1_age": age[p1],
        "p2_age": age[p2],
        "target": 1 if p1_first else 0,
        "p1_games_won": p1_games,
        "p1_games_lost": p2_games,
        "p2_games_won": p2_games,
        "p2_games_lost": p1_games,
        "p1_sets_won": p1_sets,
        "p1_sets_lost": p2_sets,
        "p2_sets_won": p2_sets,
        "p2_sets_lost": p1_sets,
    }
    # Serve lines, so features.serve.build_skill_table has real rows to pool.
    for side, won_games in (("p1", p1_games), ("p2", p2_games)):
        row[f"{side}_svpt"] = 40 + 4 * won_games
        row[f"{side}_1stWon"] = 18 + 2 * won_games
        row[f"{side}_2ndWon"] = 6 + won_games
    return row


def toy_processed_frame() -> pd.DataFrame:
    """A synthetic 8-player season in preprocessed shape.

    Deterministic and deliberately unbalanced: players earlier in ``TOY_PLAYERS``
    win more often and by wider margins, so every player's rolling form is
    distinct and no feature is accidentally neutral (a frame where they all came
    out at 0.5 would make the "real, not constant" assertions vacuous).
    """
    names = list(TOY_PLAYERS)
    strength = {name: len(names) - index for index, name in enumerate(names)}
    rank, age = TOY_RANK, TOY_AGE
    surfaces = ["Hard", "Clay", "Grass"]

    rows: list[dict] = []
    date = pd.Timestamp("2024-01-08")
    counter = 0
    # Four round-robin passes, so most players carry >5 matches and both the
    # 5- and 10-match windows are populated with different values.
    for cycle in range(4):
        for i, first in enumerate(names):
            for second in names[i + 1:]:
                counter += 1
                stronger, weaker = (
                    (first, second)
                    if strength[first] >= strength[second]
                    else (second, first)
                )
                # The stronger player wins 3 of every 4 meetings.
                upset = (counter + cycle) % 4 == 0
                winner, loser = (weaker, stronger) if upset else (stronger, weaker)
                margin = 1 + (strength[winner] - strength[loser]) % 4
                rows.append(
                    _match_row(
                        date=date + pd.Timedelta(days=counter),
                        surface=surfaces[counter % 3],
                        winner=winner,
                        loser=loser,
                        p1_first=counter % 2 == 0,
                        games=(12, 12 - margin),
                        sets=(2, 0 if margin > 2 else 1),
                        rank=rank,
                        age=age,
                    )
                )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def pipeline():
    """The real pipeline over the toy frame: ``(data, surf_hist, h2h_hist, skills)``."""
    return engineering.add_features(toy_processed_frame())


@pytest.fixture(scope="module")
def estimator(pipeline):
    """A real fitted classifier over the engineered toy frame.

    A logistic regression rather than a stub: the point of the end-to-end tests
    is that a genuine ``predict_proba`` accepts the assembled row, which a stub
    that ignores its input cannot demonstrate.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = pipeline[0]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(data[config.MODEL_FEATURES], data["target"])
    return model


@pytest.fixture
def real_adapter(pipeline, estimator):
    data, surface_history, h2h_history, _ = pipeline
    return adapter.make_classifier_prob(estimator, data, surface_history, h2h_history)


# --------------------------------------------------------------------------- #
# Acceptance: the rolling-form features are real
# --------------------------------------------------------------------------- #
def expected_next_match_row(
    processed: pd.DataFrame, player_a: str, player_b: str, surface: str
) -> pd.Series:
    """What the *training pipeline* would compute for A vs B played next.

    Appends one more match (A as p1, B as p2, latest date) to the preprocessed
    frame and re-runs ``add_features``. Every feature on that row is what the
    pipeline itself considers correct for this matchup at this point in time,
    which is exactly what the adapter claims to reproduce without a match row.
    """
    extra = processed.iloc[[0]].copy()
    extra["tourney_date"] = processed["tourney_date"].max() + pd.Timedelta(days=1)
    extra["surface"] = surface
    extra["p1_name"], extra["p2_name"] = player_a, player_b
    extra["p1_id"], extra["p2_id"] = TOY_PLAYERS[player_a], TOY_PLAYERS[player_b]
    # Rank/age are per-row inputs, not pipeline-derived: give the appended match
    # each player's own values, which is what "their latest" resolves to here.
    extra["p1_rank"], extra["p2_rank"] = TOY_RANK[player_a], TOY_RANK[player_b]
    extra["p1_age"], extra["p2_age"] = TOY_AGE[player_a], TOY_AGE[player_b]
    extended = pd.concat([processed, extra], ignore_index=True)
    engineered, *_ = engineering.add_features(extended)
    return engineered.sort_values("tourney_date").iloc[-1]


@pytest.mark.parametrize(
    ("player_a", "player_b", "surface"),
    [
        ("Alice Ace", "Hank Half", "Clay"),
        ("Cara Chip", "Dan Drop", "Hard"),
        ("Eve Emphatic", "Bob Baseline", "Grass"),
    ],
)
def test_adapter_row_equals_the_pipelines_own_row_for_the_same_matchup(
    pipeline, estimator, player_a, player_b, surface
):
    """Acceptance criterion: real rolling-form values, not synthetic constants.

    The comparison is against ``add_features`` itself, for the same players at
    the same point in time, so it cannot be satisfied by plausible-looking
    numbers, only by the pipeline's own.
    """
    data, surface_history, h2h_history, _ = pipeline
    built = adapter.ClassifierProbAdapter(
        estimator, data, surface_history, h2h_history
    ).feature_row(player_a, player_b, surface)

    expected = expected_next_match_row(
        toy_processed_frame(), player_a, player_b, surface
    )

    for column in config.MODEL_FEATURES:
        assert built[column].iloc[0] == pytest.approx(float(expected[column])), column

    # And the 20 rolling features are not the constants the old row carried.
    rolling_columns = [
        f"{side}_{name}"
        for side in ("p1", "p2")
        for name in rolling.rolling_feature_names()
    ]
    assert len(rolling_columns) == 20
    assert all(column in config.MODEL_FEATURES for column in rolling_columns)
    averages = [c for c in rolling_columns if "win_rate" not in c]
    assert all(built[column].iloc[0] != 0.0 for column in averages)


def test_adapter_row_differs_from_the_old_synthetic_row(pipeline, estimator):
    """The fix changes the prediction, not just the row.

    A row whose 20 recent-form features are ``0.5``/``0`` is the seam-7 row; if
    the estimator scored both identically the fix would be cosmetic.
    """
    import cli.interactive as interactive

    data, surface_history, h2h_history, _ = pipeline
    built = adapter.ClassifierProbAdapter(
        estimator, data, surface_history, h2h_history
    )
    real_row = built.feature_row("Alice Ace", "Hank Half", "Clay")
    features = adapter.matchup_features(
        "Alice Ace", "Hank Half", "Clay", data, surface_history, h2h_history
    )
    synthetic_row = interactive.build_feature_row(
        features.a_rank,
        features.b_rank,
        features.a_age,
        features.b_age,
        features.a_pct,
        features.b_pct,
        features.h2h_diff,
    )

    real_p = float(estimator.predict_proba(real_row)[0][1])
    synthetic_p = float(estimator.predict_proba(synthetic_row.astype(float))[0][1])
    assert real_p != pytest.approx(synthetic_p, abs=1e-6)


def test_every_model_feature_is_populated_and_finite(real_adapter):
    """All 27, from real state, no NaN, no silent default."""
    row = real_adapter.feature_row("Alice Ace", "Bob Baseline", "Hard")

    assert list(row.columns) == config.MODEL_FEATURES
    assert len(row) == 1
    assert np.isfinite(row.to_numpy(dtype=float)).all()


def test_feature_row_build_fails_loudly_if_model_features_gains_a_column(
    pipeline, estimator, monkeypatch
):
    """A new feature must break the adapter, not be quietly defaulted.

    This is the guard against seam 7 reappearing: the failure mode was a row
    builder that filled whatever it did not know how to compute.
    """
    monkeypatch.setattr(
        config, "MODEL_FEATURES", config.MODEL_FEATURES + ["p1_something_new"]
    )
    data, surface_history, h2h_history, _ = pipeline
    built = adapter.ClassifierProbAdapter(estimator, data, surface_history, h2h_history)

    with pytest.raises(KeyError, match="p1_something_new"):
        built.feature_row("Alice Ace", "Bob Baseline", "Hard")


# --------------------------------------------------------------------------- #
# Behaviour preserved by the move
# --------------------------------------------------------------------------- #
def test_adapter_is_memoised_per_matchup(pipeline, estimator):
    calls: list[pd.DataFrame] = []

    class Counting:
        def predict_proba(self, X):
            calls.append(X)
            return estimator.predict_proba(X)

    data, surface_history, h2h_history, _ = pipeline
    built = adapter.make_classifier_prob(Counting(), data, surface_history, h2h_history)

    for _ in range(50):
        built("Alice Ace", "Bob Baseline", "Hard")
    assert len(calls) == 1


def test_surface_is_part_of_the_cache_key(pipeline, estimator):
    """A Clay request must not be served the cached Hard probability."""
    data, surface_history, h2h_history, _ = pipeline
    built = adapter.make_classifier_prob(estimator, data, surface_history, h2h_history)

    hard = built("Alice Ace", "Bob Baseline", "Hard")
    clay = built("Alice Ace", "Bob Baseline", "Clay")
    assert hard != clay

    hard_row = built.feature_row("Alice Ace", "Bob Baseline", "Hard")
    clay_row = built.feature_row("Alice Ace", "Bob Baseline", "Clay")
    assert (
        hard_row["p1_surface_win_pct"].iloc[0] != clay_row["p1_surface_win_pct"].iloc[0]
    )


def test_unknown_player_raises_rather_than_defaulting(real_adapter):
    with pytest.raises(ValueError, match="No match history"):
        real_adapter("Ghost Player", "Bob Baseline", "Hard")


def test_probability_is_orientation_sensitive(real_adapter):
    """``P(A beats B)`` and ``P(B beats A)`` are different questions."""
    forward = real_adapter("Alice Ace", "Hank Half", "Clay")
    reverse = real_adapter("Hank Half", "Alice Ace", "Clay")
    assert forward != pytest.approx(reverse)


def test_adapter_is_picklable(real_adapter):
    """A module-level class, not a closure.

    Not a requirement of this ticket, ``§7`` seam 8 (``workers > 1``) is still
    open and still owns exposing a ``--workers`` flag and re-verifying it. But
    the barrier that seam names, "the only adapter is an unpicklable closure",
    is gone, and a test keeps it gone.
    """
    revived = pickle.loads(pickle.dumps(real_adapter))
    assert revived("Alice Ace", "Bob Baseline", "Hard") == real_adapter(
        "Alice Ace", "Bob Baseline", "Hard"
    )


def test_history_helpers_match_the_cli_wrappers(pipeline):
    """``cli/interactive``'s helpers are wrappers now, same answers, one impl."""
    import cli.interactive as interactive

    data, surface_history, h2h_history, _ = pipeline

    assert interactive.get_latest("Alice Ace", data) == adapter.latest_rank_age(
        "Alice Ace", data
    )
    assert interactive.get_surf_record(
        "Alice Ace", "Clay", surface_history
    ) == adapter.surface_record("Alice Ace", "Clay", surface_history)

    diff, message = interactive.compute_h2h("Alice Ace", "Bob Baseline", h2h_history)
    a_wins, b_wins = adapter.h2h_record("Alice Ace", "Bob Baseline", h2h_history)
    assert diff == a_wins - b_wins
    assert message  # the CLI keeps the rendering


# --------------------------------------------------------------------------- #
# Layering: nothing under api/ or scripts/ imports cli/, statically or at runtime
# --------------------------------------------------------------------------- #
AUDITED_SOURCES = [
    *sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "src/api").glob("*.py")),
    "scripts/precompute_sim.py",
]


@pytest.mark.parametrize("relative_path", AUDITED_SOURCES)
def test_no_static_import_of_cli(relative_path):
    """AST walk, not a substring scan: every import node, including inside
    functions and ``try`` blocks, which a text search over comments and
    docstrings would both miss and false-positive on.
    """
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))

    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    offenders = [m for m in imported if m == "cli" or m.startswith("cli.")]
    assert offenders == [], f"{relative_path} imports {offenders}"


def _clean_interpreter(code: str) -> list[str]:
    """Run ``code`` in a fresh interpreter with ``src/`` and ``scripts/`` importable."""
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")]
        ),
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.splitlines()


CLI_MODULES_SNIPPET = (
    "cli_modules = sorted(m for m in sys.modules if m == 'cli' or m.startswith('cli.'))"
)


def test_resolving_the_configured_adapter_imports_no_cli_module():
    """The runtime half of the layering claim, in a clean interpreter.

    A ``sys.modules`` diff *inside* pytest proves little: ``cli.simulate_match``
    is already imported by other test modules, so nothing new would appear
    either way. A fresh interpreter that imports only what the running server
    imports is the honest version of that check, and this is the exact import
    that pulled ``cli/`` in, via ``resolve_classifier_factory``.
    """
    name, cli_modules = _clean_interpreter(
        "import sys\n"
        "import config\n"
        "from api.main import adapter_name, resolve_classifier_factory\n"
        "factory = resolve_classifier_factory()\n"
        f"{CLI_MODULES_SNIPPET}\n"
        "print(adapter_name(factory))\n"
        "print(','.join(cli_modules))\n"
    )

    assert name == "common.classifier_adapter.make_classifier_prob"
    assert cli_modules == "", f"resolving the adapter imported {cli_modules}"


def test_importing_the_api_app_imports_no_cli_module():
    """``import api.main``, what ``uvicorn api.main:app`` does, stays cli-free."""
    (cli_modules,) = _clean_interpreter(
        "import sys\n"
        "import api.main\n"
        "api.main.create_app()\n"
        f"{CLI_MODULES_SNIPPET}\n"
        "print(','.join(cli_modules))\n"
    )
    assert cli_modules == "", f"importing api.main imported {cli_modules}"


def test_importing_the_precompute_script_imports_no_cli_module():
    """The offline producer, too, it used to import ``cli.simulate_match``."""
    name, cli_modules = _clean_interpreter(
        "import sys\n"
        "import precompute_sim\n"
        f"{CLI_MODULES_SNIPPET}\n"
        "print(precompute_sim.ADAPTER)\n"
        "print(','.join(cli_modules))\n"
    )

    assert name == "common.classifier_adapter.make_classifier_prob"
    assert cli_modules == "", f"importing precompute_sim imported {cli_modules}"


def test_the_shared_adapter_module_imports_neither_cli_nor_api():
    """``common/`` is below both callers, the layering rule, applied to this module."""
    tree = ast.parse(
        (REPO_ROOT / "src/common/classifier_adapter.py").read_text(encoding="utf-8")
    )
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)

    forbidden = [
        m
        for m in imported
        if m.split(".")[0] in {"cli", "api", "sim", "scripts"}
    ]
    assert forbidden == []


# --------------------------------------------------------------------------- #
# End-to-end: the real adapter through a live endpoint
# --------------------------------------------------------------------------- #
@pytest.fixture
def live_client(pipeline, estimator):
    """An app running on the *configured* adapter, no injected stub.

    ``classifier_factory=None`` means ``create_app`` resolves
    ``config.API_CLASSIFIER_ADAPTER`` exactly as the shipped server does, so what
    serves these requests is the real code path, over a real (small) pipeline and
    a real fitted estimator.
    """
    data, surface_history, h2h_history, skill_table = pipeline
    context = ApiContext(
        data=data,
        surface_history=surface_history,
        h2h_history=h2h_history,
        skill_table=skill_table,
        estimator=estimator,
        estimator_class=type(estimator).__name__,
        data_through_year=int(data["tourney_date"].dt.year.max()),
    )
    app = create_app(
        context_factory=lambda: context,
        draws_dir=FIXTURE_DRAWS,
        cache_dir=FIXTURE_CACHE,
    )
    with TestClient(app) as client:
        yield client


def test_storybook_runs_end_to_end_on_the_real_adapter(live_client):
    """Acceptance criterion: the fixed adapter, exercised through /storybook."""
    response = live_client.get("/tournaments/toy_open_2026/storybook?seed=7")
    assert response.status_code == 200, response.text

    body = response.json()
    assert body["match_count"] == 7  # an 8-draw: 4 + 2 + 1
    assert body["champion"]["player"] in TOY_PLAYERS
    assert all(
        match["scoreline"]
        for storybook_round in body["rounds"]
        for match in storybook_round["matches"]
    )
    assert body["metadata"]["adapter"] == "common.classifier_adapter.make_classifier_prob"
    assert body["metadata"]["mode"] == config.SIM_CLI_RECONCILE_MODE


def test_storybook_on_the_real_adapter_is_deterministic(live_client):
    """Same seed, same story, with a real classifier in the loop."""
    first = live_client.get("/tournaments/toy_open_2026/storybook?seed=11")
    second = live_client.get("/tournaments/toy_open_2026/storybook?seed=11")
    assert first.content == second.content

    other = live_client.get("/tournaments/toy_open_2026/storybook?seed=12")
    assert other.status_code == 200


def test_live_storybook_reflects_the_classifier_it_was_given(live_client, pipeline):
    """The endpoint's numbers move with ``P_clf``, so the adapter is really in play.

    A storybook whose outcomes were independent of the classifier would pass
    every structural assertion above. Here the *same* draw and seed is served by
    a second app whose estimator always answers "player A wins", and the story
    must change.
    """
    data, surface_history, h2h_history, skill_table = pipeline

    class AlwaysA:
        def predict_proba(self, X):
            return np.array([[0.02, 0.98]])

    context = ApiContext(
        data=data,
        surface_history=surface_history,
        h2h_history=h2h_history,
        skill_table=skill_table,
        estimator=AlwaysA(),
        estimator_class="AlwaysA",
        data_through_year=2024,
    )
    app = create_app(
        context_factory=lambda: context,
        draws_dir=FIXTURE_DRAWS,
        cache_dir=FIXTURE_CACHE,
    )
    with TestClient(app) as skewed_client:
        skewed = skewed_client.get("/tournaments/toy_open_2026/storybook?seed=7").json()

    baseline = live_client.get("/tournaments/toy_open_2026/storybook?seed=7").json()

    # Under "A always wins", every match goes to the lower bracket position, so
    # the champion is position 1, and that differs from the real model's story
    # only if the classifier is genuinely driving the simulation.
    assert skewed["champion"]["position"] == 1
    assert skewed != baseline


# --------------------------------------------------------------------------- #
# Startup warm-up: the rolling-form table is built eagerly, not on first request
# --------------------------------------------------------------------------- #
def _context_from(pipeline, estimator) -> ApiContext:
    data, surface_history, h2h_history, skill_table = pipeline
    return ApiContext(
        data=data,
        surface_history=surface_history,
        h2h_history=h2h_history,
        skill_table=skill_table,
        estimator=estimator,
        estimator_class=type(estimator).__name__,
        data_through_year=int(data["tourney_date"].dt.year.max()),
    )


def test_build_classifier_warms_the_rolling_form_table_eagerly(pipeline, estimator):
    """The deferred scan is charged to startup, not to the first simulate request.

    ``build_classifier`` moves the adapter's one lazy artefact, the as-of-now
    rolling-form table, off the first-request path. After it returns, the table
    is already built and memoised, so the first ``/match`` or ``/storybook``
    does not pay the ~0.08 s scan on a cold instance.
    """
    adapter_wrapper = build_classifier(_context_from(pipeline, estimator), adapter.make_classifier_prob)
    # Reaching into the private memo is deliberate: the whole point of the fix
    # is that this is populated *before* any prediction has been asked for.
    assert adapter_wrapper.call._form_table is not None


def test_build_classifier_warm_up_never_breaks_startup_on_a_bad_frame(
    pipeline, estimator
):
    """Warming is best-effort: an unbuildable frame falls back to lazy, not a crash.

    A minimal frame that lacks the columns ``build_rolling_form_table`` needs
    (the shape every small test fixture has) must not abort app startup. The
    adapter is still returned, and the table simply builds, or fails, on first
    use exactly as it did before the warm-up existed.
    """
    context = _context_from(pipeline, estimator)
    broken = dataclasses.replace(context, data=context.data[["tourney_date"]].copy())
    # Must not raise despite the frame being unusable for a form table.
    adapter_wrapper = build_classifier(broken, adapter.make_classifier_prob)
    assert adapter_wrapper.call._form_table is None


def test_config_default_and_precompute_still_name_the_same_adapter():
    """The live-endpoint invariant, re-pinned on the new module.

    ``/simulate`` (cached) and ``/storybook`` (live) must not publish different
    models while claiming the same one.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import precompute_sim  # noqa: E402

    factory = resolve_classifier_factory()
    assert adapter_name(factory) == precompute_sim.ADAPTER
    assert factory is adapter.make_classifier_prob
    assert precompute_sim.make_classifier_prob is adapter.make_classifier_prob
