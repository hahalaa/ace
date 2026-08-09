"""Tests for scripts/benchmark_vs_market.py — the T6.1 market benchmark.

Three things are under test, in descending order of how badly a regression would
hurt:

1. **The hard boundary.** ``tennis-data.co.uk`` data must never reach
   ``preprocess.py``, ``features/``, ``sim/`` or any training/fitting code. That
   is checked by *static analysis* over every module in the repo — an AST walk
   for imports plus a source scan for the vendor and its constants — not by
   asserting it in prose. See :class:`TestNoReverseDependency`.
2. **The de-vigging arithmetic**, against the ticket's worked example with the
   expected numbers computed by hand *in the test*, as exact fractions, so the
   test would still fail if the implementation and the assertion were both wrong
   in the same way.
3. **The name join**, on a fixture carrying both a resolvable and an
   unresolvable name, asserting each takes its own path — matched vs.
   logged-and-skipped. The unhappy path is the one that matters: silently
   dropping rows is exactly how a benchmark comes to compare two different match
   sets and report it as one.

Nothing here loads the vendored CSVs, trains anything, or reads the committed
odds snapshot's contents — every frame is hand-built.

``scripts/`` is not on pyproject's ``pythonpath = ["src"]``, so the sys.path hack
from ``tests/test_update_and_cache.py`` is repeated here.
"""

from __future__ import annotations

import ast
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../scripts")))

import benchmark_vs_market as bench  # noqa: E402
import config  # noqa: E402
from common.names import NameIndex  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Fixtures — a two-match market snapshot and the model rows it joins to.
# --------------------------------------------------------------------------- #
RESOLVABLE = "Carlos Alcaraz"
ALSO_RESOLVABLE = "Jannik Sinner"
# Not in the model frame under any spelling, and not close enough for difflib's
# 0.6 cutoff to reach either name above.
UNRESOLVABLE_RAW = "Zzyzx Q."


# A pair the market fixture never mentions. Its row exists so the model frame is
# strictly LARGER than the matched set — without it `model_rows` and
# `model_rows.loc[matched_index]` are the same object and the same-match-set
# guarantee cannot be tested at all (see
# TestRunBenchmark::test_model_is_scored_on_exactly_the_matched_rows).
UNJOINED_P1 = "Novak Djokovic"
UNJOINED_P2 = "Daniil Medvedev"


@pytest.fixture
def model_rows() -> pd.DataFrame:
    """Three engineered held-out-season rows; only two of them are joinable.

    The unjoinable row sits in the **middle** on purpose. With it at either end,
    a positional slice (``model_rows.iloc[:n_matched]``) would still happen to
    select exactly the matched rows, and a wrong-rows bug of that shape would go
    undetected; from the middle, no contiguous slice reproduces the matched set.
    """
    return pd.DataFrame(
        {
            "tourney_date": pd.to_datetime(["2025-05-25", "2025-08-25", "2025-06-30"]),
            "p1_name": [RESOLVABLE, UNJOINED_P1, ALSO_RESOLVABLE],
            "p2_name": [ALSO_RESOLVABLE, UNJOINED_P2, RESOLVABLE],
            "target": [1, 1, 0],  # p1 won row 101; p2 (Alcaraz) won row 202
        },
        index=[101, 303, 202],
    )


@pytest.fixture
def market_rows() -> pd.DataFrame:
    """Three market rows: two joinable, one carrying an unresolvable player."""
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-06-08", "2025-07-13", "2025-07-13"]),
            "Tournament": ["French Open", "Wimbledon", "Wimbledon"],
            "Round": ["The Final", "The Final", "1st Round"],
            "Comment": ["Completed", "Completed", "Completed"],
            "Winner": ["Alcaraz C.", "Alcaraz C.", "Sinner J."],
            "Loser": ["Sinner J.", "Sinner J.", UNRESOLVABLE_RAW],
            "B365W": [1.50, 1.50, 1.20],
            "B365L": [2.80, 2.80, 4.50],
        }
    )


# --------------------------------------------------------------------------- #
# 1. De-vigging — hand-computed expectations
# --------------------------------------------------------------------------- #
class TestDevig:
    def test_ticket_worked_example_1_50_vs_2_80(self):
        """odds 1.50 / 2.80 -> 28/43 and 15/43, computed here as exact fractions.

        By hand: 1/1.50 = 2/3 and 1/2.80 = 5/14 are the raw implied
        probabilities. They sum to 2/3 + 5/14 = 28/42 + 15/42 = 43/42 > 1 — the
        overround is 1/42 = 0.0238, i.e. 2.38%. Removing it proportionally:

            p_winner = (2/3) / (43/42) = (2/3) * (42/43) = 28/43 = 0.6511627907
            p_loser  = (5/14) / (43/42) = (5/14) * (42/43) = 15/43 = 0.3488372093

        Both are written as ``Fraction`` so the expected values come from the
        arithmetic above and not from whatever the implementation returned.
        """
        expected_winner = Fraction(28, 43)
        expected_loser = Fraction(15, 43)
        assert expected_winner + expected_loser == 1

        p_winner = bench.devig(1.50, 2.80)

        assert p_winner == pytest.approx(float(expected_winner), abs=1e-12)
        assert 1.0 - p_winner == pytest.approx(float(expected_loser), abs=1e-12)
        # Spelled out, so a reader does not have to evaluate the fraction.
        assert p_winner == pytest.approx(0.6511627907, abs=1e-9)

    def test_overround_of_the_worked_example(self):
        """1/1.5 + 1/2.8 - 1 = 43/42 - 1 = 1/42."""
        assert bench.overround(1.50, 2.80) == pytest.approx(float(Fraction(1, 42)), abs=1e-12)

    def test_devigged_pair_always_sums_to_one(self):
        """The defining property: the normalised book has no margin left in it."""
        for o_w, o_l in ((1.01, 51.0), (2.0, 2.0), (1.75, 2.15), (1.30, 3.60)):
            assert bench.devig(o_w, o_l) + bench.devig(o_l, o_w) == pytest.approx(1.0)

    def test_fair_market_is_a_coin_flip(self):
        assert bench.devig(2.0, 2.0) == pytest.approx(0.5)

    def test_shorter_price_gets_the_higher_probability(self):
        assert bench.devig(1.20, 4.50) > 0.5

    def test_vig_is_actually_removed(self):
        """The raw reciprocal overstates the favourite; de-vigging pulls it back."""
        raw = 1.0 / 1.50
        assert bench.devig(1.50, 2.80) < raw

    @pytest.mark.parametrize("bad", [1.0, 0.5, 0.0, -2.0])
    def test_rejects_non_prices(self, bad):
        with pytest.raises(ValueError):
            bench.devig(bad, 2.0)
        with pytest.raises(ValueError):
            bench.devig(2.0, bad)

    def test_rejects_nan(self):
        with pytest.raises(ValueError):
            bench.devig(float("nan"), 2.0)


# --------------------------------------------------------------------------- #
# 2. Query-form rewriting — formatting only, all matching stays in common.names
# --------------------------------------------------------------------------- #
class TestQueryForms:
    def test_surname_initial_is_transposed_for_the_initials_strategy(self):
        assert bench.market_name_query_forms("Alcaraz C.")[0] == "C Alcaraz"

    def test_surname_alone_is_offered_as_a_fallback(self):
        assert "Alcaraz" in bench.market_name_query_forms("Alcaraz C.")

    def test_compound_initials_keep_only_the_first(self):
        forms = bench.market_name_query_forms("Struff J.L.")
        assert forms[0] == "J Struff"
        assert "Struff" in forms

    def test_compound_surname_is_kept_whole(self):
        forms = bench.market_name_query_forms("Carreno Busta P.")
        assert forms[0] == "P Carreno Busta"

    def test_already_canonical_name_passes_through(self):
        assert bench.market_name_query_forms("Carlos Alcaraz") == ["Carlos Alcaraz"]

    def test_forms_are_deduplicated(self):
        forms = bench.market_name_query_forms("Alcaraz C.")
        assert len(forms) == len(set(forms))


# --------------------------------------------------------------------------- #
# 3. Name resolution — both paths, on the same index
# --------------------------------------------------------------------------- #
class TestResolveMarketName:
    @pytest.fixture
    def index(self) -> NameIndex:
        return NameIndex.from_names([RESOLVABLE, ALSO_RESOLVABLE, "Jan-Lennard Struff"])

    def test_known_name_resolves(self, index):
        res = bench.resolve_market_name("Alcaraz C.", index)
        assert res.name == RESOLVABLE
        assert res.reason is None

    def test_compound_given_name_resolves_via_the_surname_form(self, index):
        assert bench.resolve_market_name("Struff J.L.", index).name == "Jan-Lennard Struff"

    def test_unknown_name_reports_a_reason_and_never_raises(self, index):
        res = bench.resolve_market_name(UNRESOLVABLE_RAW, index)
        assert res.name is None
        assert res.reason == "no_match"
        assert res.raw == UNRESOLVABLE_RAW

    def test_ambiguity_is_reported_with_candidates_not_guessed(self):
        """Two Cerundolos: the resolver must hand back both, not pick one."""
        index = NameIndex.from_names(["Juan Manuel Cerundolo", "Francisco Cerundolo"])
        res = bench.resolve_market_name("Cerundolo J.M.", index)
        assert res.name is None
        assert res.reason == "ambiguous"
        assert set(res.candidates) == {"Juan Manuel Cerundolo", "Francisco Cerundolo"}

    def test_cache_is_used(self, index):
        cache: dict[str, bench.NameResolution] = {}
        first = bench.resolve_market_name("Alcaraz C.", index, cache)
        assert "Alcaraz C." in cache
        assert bench.resolve_market_name("Alcaraz C.", index, cache) is first


# --------------------------------------------------------------------------- #
# 4. The join — matched vs. logged-and-skipped
# --------------------------------------------------------------------------- #
class TestJoin:
    def test_resolvable_row_is_matched_and_unresolvable_row_is_logged(
        self, market_rows, model_rows
    ):
        report = bench.join_market_to_model(market_rows, model_rows)

        assert report.n_matched == 2, "both joinable rows should match"
        assert report.n_market_rows == 3

        # The unresolvable row is skipped *with its reason and identity*, never
        # dropped in silence.
        assert report.skipped_by_reason() == {"unresolved_name": 1}
        skipped = report.skipped[0]
        assert skipped["reason"] == "unresolved_name"
        assert skipped["date"] == "2025-07-13"
        assert UNRESOLVABLE_RAW in skipped["players"]

        assert UNRESOLVABLE_RAW in report.unresolved_names
        assert report.unresolved_names[UNRESOLVABLE_RAW].reason == "no_match"

    def test_resolution_rate_is_reported(self, market_rows, model_rows):
        report = bench.join_market_to_model(market_rows, model_rows)
        assert report.resolution_rate == pytest.approx(2 / 3)

    def test_market_probability_is_oriented_onto_p1_not_onto_the_winner(
        self, market_rows, model_rows
    ):
        """Row 202 has p1 = Sinner, who *lost*; his market prob must be flipped."""
        report = bench.join_market_to_model(market_rows, model_rows)
        by_index = report.matched.set_index("model_index")

        # Row 101: p1 = Alcaraz = the market Winner, priced 1.50 -> 28/43.
        assert by_index.loc[101, "p_market"] == pytest.approx(float(Fraction(28, 43)))
        assert by_index.loc[101, "outcome"] == 1
        # Row 202: p1 = Sinner = the market Loser -> 15/43.
        assert by_index.loc[202, "p_market"] == pytest.approx(float(Fraction(15, 43)))
        assert by_index.loc[202, "outcome"] == 0

    def test_each_model_row_is_consumed_at_most_once(self, market_rows, model_rows):
        """Rows 0 and 1 are the same pair; they must take different model rows."""
        report = bench.join_market_to_model(market_rows, model_rows)
        assert sorted(report.matched["model_index"]) == [101, 202]

    def test_row_outside_the_date_window_is_skipped_not_stretched(
        self, market_rows, model_rows
    ):
        far = market_rows.iloc[[0]].copy()
        far["Date"] = pd.to_datetime(["2025-11-30"])
        report = bench.join_market_to_model(far, model_rows)
        assert report.n_matched == 0
        assert report.skipped_by_reason() == {"no_model_match_in_window": 1}

    def test_pair_the_model_never_recorded_is_skipped(self, model_rows):
        market = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2025-06-08"]),
                "Tournament": ["French Open"],
                "Comment": ["Completed"],
                "Winner": ["Alcaraz C."],
                "Loser": ["Alcaraz C."],
                "B365W": [1.5],
                "B365L": [2.8],
            }
        )
        report = bench.join_market_to_model(market, model_rows)
        assert report.skipped_by_reason() == {"no_model_match": 1}

    def test_missing_odds_are_skipped_with_their_own_reason(self, market_rows, model_rows):
        market = market_rows.iloc[[0]].copy()
        market["B365W"] = [np.nan]
        report = bench.join_market_to_model(market, model_rows)
        assert report.n_matched == 0
        assert report.skipped_by_reason() == {"missing_odds": 1}

    def test_unknown_book_is_refused(self, market_rows, model_rows):
        with pytest.raises(KeyError):
            bench.join_market_to_model(market_rows, model_rows, book="Nope")

    def test_book_absent_from_the_snapshot_is_refused(self, market_rows, model_rows):
        with pytest.raises(KeyError, match="PSW"):
            bench.join_market_to_model(market_rows, model_rows, book="PS")


# --------------------------------------------------------------------------- #
# 5. Snapshot loading — never fetches, fails with instructions
# --------------------------------------------------------------------------- #
class TestSnapshotLoading:
    def test_missing_snapshot_names_the_manual_step(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="MANUAL"):
            bench.load_market_snapshot(tmp_path / "nope.csv")

    def test_snapshot_missing_a_required_column_is_rejected(self, tmp_path):
        path = tmp_path / "odds.csv"
        path.write_text("Date,Winner,B365W,B365L\n2025-01-01,Alcaraz C.,1.5,2.8\n")
        with pytest.raises(ValueError, match="Loser"):
            bench.load_market_snapshot(path)

    def test_committed_snapshot_exists_and_carries_the_documented_columns(self):
        """The acceptance criterion's artefact, checked as an artefact."""
        assert config.BENCHMARK_ODDS_SNAPSHOT.exists(), (
            "The static odds snapshot is missing; it is a committed, one-time "
            "manual download — see scripts/benchmark_vs_market.py."
        )
        header = config.BENCHMARK_ODDS_SNAPSHOT.read_text().splitlines()[0].split(",")
        for column in ("Date", "Winner", "Loser", "B365W", "B365L", "PSW", "PSL"):
            assert column in header


# --------------------------------------------------------------------------- #
# 6. End-to-end over the fixture, with a stub estimator
# --------------------------------------------------------------------------- #
class _StubEstimator:
    """Returns a fixed P(p1 wins) for every row; MODEL_FEATURES never read.

    Records the frame it was handed. That record is what lets a test assert the
    model was scored on *exactly* the matched rows rather than inferring it from
    a downstream shape error.
    """

    def __init__(self, p: float = 0.6):
        self.p = p
        self.seen_index: list = []
        self.n_calls = 0

    def predict_proba(self, X):
        self.n_calls += 1
        self.seen_index = list(X.index)
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


class TestRunBenchmark:
    def test_reports_both_briers_over_the_same_match_set(
        self, tmp_path, market_rows, model_rows
    ):
        snapshot = tmp_path / "odds.csv"
        market_rows.to_csv(snapshot, index=False)

        # MODEL_FEATURES are absent from the fixture frame; the stub ignores them,
        # but the subset selection must not, so add them as zeros.
        rows = model_rows.copy()
        for feature in config.MODEL_FEATURES:
            rows[feature] = 0.0

        join, model_brier, market_brier, text = bench.run_benchmark(
            snapshot_path=snapshot,
            classifier=_StubEstimator(0.6),
            model_rows=rows,
            save_plot=False,
        )

        assert join.n_matched == 2
        # Hand-computed: predictions 0.6/0.6 against outcomes 1/0.
        # Brier = ((0.6-1)^2 + (0.6-0)^2) / 2 = (0.16 + 0.36) / 2 = 0.26
        assert model_brier == pytest.approx(0.26)
        # Market: 28/43 against outcome 1, 15/43 against outcome 0.
        p_hi, p_lo = float(Fraction(28, 43)), float(Fraction(15, 43))
        assert market_brier == pytest.approx(((p_hi - 1) ** 2 + p_lo**2) / 2)

        assert "Reconciled model" in text
        assert "Market" in text
        assert f"{model_brier:.4f}" in text and f"{market_brier:.4f}" in text
        # The join rate is part of the output, not an afterthought.
        assert "66.7%" in text

    def test_model_is_scored_on_exactly_the_matched_rows(
        self, tmp_path, market_rows, model_rows
    ):
        """The same-match-set guarantee, asserted directly rather than inferred.

        This is the whole premise of the ticket: the two Brier scores are only
        comparable because they are computed over one identical set of matches.
        The fixture deliberately carries a third model row (Djokovic–Medvedev)
        that no market row mentions, so ``model_rows`` is strictly larger than
        the matched set and "scored the model on everything" is a *distinguishable*
        mistake. Asserting on the estimator's own input catches that directly —
        relying on ``brier_score``'s length check would only catch it by accident,
        and not at all when the two sets happen to be the same size.
        """
        snapshot = tmp_path / "odds.csv"
        market_rows.to_csv(snapshot, index=False)
        rows = model_rows.copy()
        for feature in config.MODEL_FEATURES:
            rows[feature] = 0.0

        stub = _StubEstimator(0.6)
        join, model_brier, market_brier, _ = bench.run_benchmark(
            snapshot_path=snapshot, classifier=stub, model_rows=rows, save_plot=False
        )

        # The premise: the model frame really is bigger than the matched set.
        assert len(rows) == 3
        assert join.n_matched == 2
        assert len(rows) > join.n_matched

        # The guarantee: the estimator saw exactly the matched rows, in order.
        assert stub.n_calls == 1
        assert len(stub.seen_index) == join.n_matched
        assert stub.seen_index == list(join.matched["model_index"])
        assert 303 not in stub.seen_index, "the unjoined row must never be scored"

        # And the market side is the same set, so the two Briers are comparable.
        assert len(join.matched["p_market"]) == join.n_matched
        assert model_brier == pytest.approx(0.26)

    def test_scoring_a_different_set_than_the_join_is_refused(
        self, tmp_path, market_rows, model_rows
    ):
        """The runtime guard behind the guarantee, exercised on its own.

        ``run_benchmark`` cross-checks the three vectors before scoring. A
        classifier that returns the wrong number of rows is the one way the
        subset selection can go wrong without the frame lengths disagreeing.
        """
        snapshot = tmp_path / "odds.csv"
        market_rows.to_csv(snapshot, index=False)
        rows = model_rows.copy()
        for feature in config.MODEL_FEATURES:
            rows[feature] = 0.0

        class _WrongLengthEstimator:
            def predict_proba(self, X):
                return np.column_stack([np.full(5, 0.4), np.full(5, 0.6)])

        with pytest.raises(RuntimeError, match="same match set"):
            bench.run_benchmark(
                snapshot_path=snapshot,
                classifier=_WrongLengthEstimator(),
                model_rows=rows,
                save_plot=False,
            )

    def test_empty_join_refuses_to_report_a_comparison(self, tmp_path, model_rows):
        market = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2025-06-08"]),
                "Comment": ["Completed"],
                "Winner": [UNRESOLVABLE_RAW],
                "Loser": ["Zqqx W."],
                "B365W": [1.5],
                "B365L": [2.8],
            }
        )
        snapshot = tmp_path / "odds.csv"
        market.to_csv(snapshot, index=False)
        rows = model_rows.copy()
        for feature in config.MODEL_FEATURES:
            rows[feature] = 0.0

        with pytest.raises(RuntimeError, match="No market rows joined"):
            bench.run_benchmark(
                snapshot_path=snapshot,
                classifier=_StubEstimator(),
                model_rows=rows,
                save_plot=False,
            )

    def test_main_reports_a_missing_snapshot_without_a_traceback(self, tmp_path, capsys):
        assert bench.main(["--snapshot", str(tmp_path / "nope.csv")]) == 1
        assert "MANUAL" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 7. The hard boundary — verified statically, not asserted
# --------------------------------------------------------------------------- #
BENCHMARK_MODULE = "benchmark_vs_market"

# Everything the ticket forbids market data from reaching, plus model/ (the
# training/fitting half) and api/ (which would publish it).
TRAINING_PATH_DIRS = ("data", "features", "sim", "model", "api", "common", "cli")


def _python_sources() -> list[Path]:
    """Every tracked .py under src/ and scripts/, excluding the benchmark itself."""
    files: list[Path] = []
    for root in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            if path.stem == BENCHMARK_MODULE:
                continue
            files.append(path)
    return files


def _imported_modules(path: Path) -> set[str]:
    """Top-level module names imported by ``path``, via AST (not regex)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
    return names


class TestNoReverseDependency:
    def test_the_scan_actually_sees_the_repo(self):
        """Guard the guard: an empty file list would make every check below vacuous."""
        sources = _python_sources()
        assert len(sources) > 20
        stems = {p.stem for p in sources}
        for expected in ("preprocess", "engineering", "tournament", "train", "calibrate"):
            assert expected in stems

    def test_nothing_imports_the_benchmark_script(self):
        offenders = [
            str(path.relative_to(REPO_ROOT))
            for path in _python_sources()
            if any(m.split(".")[0] == BENCHMARK_MODULE for m in _imported_modules(path))
        ]
        assert offenders == [], (
            f"{offenders} import {BENCHMARK_MODULE}. T6.1 is evaluation-only: the "
            "dependency must stay one-way."
        )

    def test_no_module_mentions_the_vendor_or_the_snapshot(self):
        """No training-path module may name tennis-data.co.uk or the snapshot file.

        Docs and comments are allowed to *discuss* the source — this checks code
        and paths, so the needles are the vendor host and the on-disk artefacts.
        """
        needles = ("tennis-data.co.uk", "tennis_data_atp", "data/benchmarks")
        offenders: list[str] = []
        for path in _python_sources():
            if path.parts[-2] not in TRAINING_PATH_DIRS and path.parent.name != "src":
                continue
            text = path.read_text()
            for needle in needles:
                if needle in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {needle}")
        # config.py is the one deliberate exception: it holds the *path* constant
        # (per the project's "no magic paths in modules" rule) and a comment
        # naming the doc. It holds no odds and reads no file.
        offenders = [o for o in offenders if not o.startswith("src/config.py")]
        assert offenders == [], f"market-data references in the training path: {offenders}"

    def test_benchmark_config_constants_are_read_only_by_the_benchmark(self):
        """The sharpest version of the boundary: nothing else even names them.

        ``config.py`` is imported by ``preprocess.py``, so the BENCHMARK_* names
        are technically *reachable* from the training path. This asserts nothing
        there reaches for them — which is what "no market data in training" means
        operationally once the constants live in config by convention.
        """
        constants = [n for n in dir(config) if n.startswith("BENCHMARK_")]
        assert constants, "expected the T6.1 constants to exist in config"

        offenders: list[str] = []
        for path in _python_sources():
            if path.name == "config.py":
                continue
            text = path.read_text()
            for name in constants:
                if name in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)}: {name}")
        assert offenders == [], f"BENCHMARK_* referenced outside the benchmark: {offenders}"

    def test_refresh_and_scheduled_paths_do_not_touch_the_benchmark(self):
        """T6.1 must not have been wired into T0.1's refresh or T5.3's cron."""
        for relative in (
            "scripts/refresh_data.py",
            "scripts/update_and_cache.py",
            ".github/workflows/refresh-and-simulate.yml",
            ".github/workflows/ci.yml",
        ):
            text = (REPO_ROOT / relative).read_text()
            for needle in ("benchmark", "tennis-data", "tennis_data_atp"):
                assert needle not in text.lower(), f"{relative} references {needle}"

    def test_benchmark_imports_only_flow_inwards(self):
        """The script may read the pipeline; that direction is fine and expected."""
        imported = _imported_modules(REPO_ROOT / "scripts" / "benchmark_vs_market.py")
        tops = {m.split(".")[0] for m in imported}
        assert "config" in tops
        assert {"model", "common"} <= tops
        # It must not reach for the raw loader / preprocessor itself: the
        # held-out rows come from T1.8's harness, one definition of the split.
        assert "data" not in tops
        assert "features" not in tops
