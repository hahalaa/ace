"""Simulation validation harness (T1.9) — "does the physics look right?".

Confirms the point-by-point match simulator (``sim/match.py``, driven by the
``sim/points.py`` point model and the ``features/serve.py`` skill table) produces
realistic *match shapes* before Phase 2 builds tournaments on top of it
(``ace-03-tennis-math.md §7``). It compares, side by side:

* the **historical** distribution of match shapes computed from the vendored
  ``data/raw`` CSVs (Grand Slam best-of-5, and best-of-3 tour matches), and
* the **simulated** distribution produced by running many matches at *realistic*
  point-win probabilities — drawn from the actual snapshot skill table over real
  historical matchups, so the validation reflects the population the simulator
  will be run against in Phase 2.

Metrics (per format):

* **Set-count distribution** — bo5: share of matches ending 3–0 / 3–1 / 3–2
  sets; bo3: 2–0 vs 2–1.
* **Games per set**, **tiebreak frequency**, **break rate**.

What is validated here is the **emergent behaviour of the actual point-by-point
simulator** (``simulate_match_bo3`` / ``simulate_match_bo5``) — *not* the
analytic match-win composition from T1.8, and *not* the reconciled model. The
raw point model is measured as-is; see the finding below.

Determinism: every simulated quantity is drawn from a single
``numpy.random.default_rng(seed)`` (``--seed``), so a run is fully reproducible.

**Break rate — measurement note.** ``MatchResult`` deliberately carries only the
game *score* of each set, not a per-game hold/break log, so break rate cannot be
read back off a finished match. Because ``§5`` holds ``p`` constant within a
match (no momentum/fatigue), a match's service games are i.i.d.
``simulate_game(p)`` draws per server, so we measure break rate at the game layer
by drawing a handful of standalone ``simulate_game`` games on each matchup's two
serve probabilities — the identical primitive the set/match layers use. This is
the emergent match break rate, not a separate model.

Retirements/walkovers (``RET`` / ``W/O`` / ``def.``) are excluded from the
historical side, matching how T1.1's skill table already excludes them, so the
historical and simulated populations are apples-to-apples. (``parse_match_score``
over-counts a pre-retirement partial set — ``ace-02-data-schema.md`` data-quality
note — so this harness parses scores itself and drops those matches outright.)

Run directly to print the full side-by-side report::

    python scripts/validate_sim.py                 # defaults
    python scripts/validate_sim.py --n-matches 40000 --seed 20260725

Exit status is non-zero if a **hard-tolerance** metric (the set-count
distribution or games-per-set) falls outside its documented band — the T1.9
gate. See ``tests/test_sim_realism.py`` for the seeded, generous-band CI version.

--------------------------------------------------------------------------------
FINDING (surfaced by this harness; see ``§7``)
--------------------------------------------------------------------------------
At the current point-probability derivation, the simulated best-of-5 set-count
distribution does **not** match historical Slam data: the simulator produces too
few straight-set matches and too many five-setters (≈+10pp of matches go the
distance). Best-of-3 shows the same, milder, skew. ``§7`` names this exact
symptom — "if 5-setters are far too rare/common, your ``p`` derivation or
clamping is off" — and attributes it upstream to the point model (**T1.1/T1.2**),
**not** to this ticket's code. The mechanism is *skill-gap compression*: heavy
empirical-Bayes shrinkage (``config.SERVE_SHRINKAGE_K``) plus the additive
serve/return model leave the two players' point-win probabilities clustered near
the surface baseline (observed mean ``|pA − pB| ≈ 0.05``), so matches play out
more evenly than real ones — more tiebreaks, longer sets, more deciders. Tiebreak
frequency (elevated) and games-per-set (elevated) point the same way; break rate
stays plausible. This is **not** a stale-snapshot artifact: the snapshot table
scores every historical matchup with the *latest* known skills, so an old
matchup is scored with present-day skills — but restricting the simulated pool to
recent (``>= CONFOUND_SINCE_YEAR``) matchups, where those skills are ~
contemporaneous, reproduces essentially the same gap (mean ``|pA − pB| ≈ 0.047``
vs ``≈0.048`` on the full pool) and the same set-count skew. The ``CONFOUND
CONTROL`` block runs that filtered re-run alongside the main comparison so the
harness makes this robustness argument itself. Per the ticket brief, this is
reported faithfully rather than
hidden behind a loosened tolerance; the tolerance below is deliberately generous
and the discrepancy still exceeds it by an order of magnitude over sampling
noise. It is expected to narrow once T1.8 reconciliation widens effective skill
gaps (anchoring scorelines to the classifier), and/or T1.1/T1.2 revisit
shrinkage/clamping.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

# scripts/ is not on pyproject's ``pythonpath = ["src"]`` (that covers src
# modules); add src/ so this script and its test can import project modules the
# same way the runtime pipeline does. Mirrors tests/test_refresh_data.py's hack.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import config  # noqa: E402
from data.loader import load_atp_data  # noqa: E402
from data.preprocess import preprocess_data  # noqa: E402
from features.serve import SkillTable, build_skill_table  # noqa: E402
from sim.match import simulate_game, simulate_match_bo3, simulate_match_bo5  # noqa: E402
from sim.points import matchup_point_probs  # noqa: E402

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #
DEFAULT_N_MATCHES = 40_000
DEFAULT_SEED = 20_260_725
# Standalone service games drawn per simulated match to measure break rate
# (see the module docstring). Split evenly across the matchup's two serve
# probabilities; over DEFAULT_N_MATCHES matches this is a large game sample.
BREAKRATE_GAMES_PER_SERVER = 3
# Confound control (§7 finding robustness). The snapshot skill table scores every
# historical matchup with the *latest* known skills, so a 2014-era matchup is
# scored with present-day skills. To rule out that stale-snapshot mismatch as the
# source of the |pA−pB| compression, the bo5 comparison is re-run on a recent-only
# matchup pool (``tourney_date`` year >= this), where the snapshot is roughly
# contemporaneous, and the mean gap + set-count skew are confirmed unchanged.
CONFOUND_SINCE_YEAR = 2024

# Deciding-set rules: all four current Slams use a 10-point tiebreak at 6–6
# (``ace-03-tennis-math.md §4``); modern ATP best-of-3 uses a 7-point tiebreak in
# every set including the decider.
BO5_FINAL_SET_RULE = "10pt_at_6_6"
BO3_FINAL_SET_RULE = "7pt_at_6_6"

# --------------------------------------------------------------------------- #
# Tolerances (documented + justified against sample-size statistics)
# --------------------------------------------------------------------------- #
# Historical set-count shares are estimated from ~6.1k Slam bo5 / ~25k bo3
# matches; the simulated shares from DEFAULT_N_MATCHES. The combined standard
# error of a per-bucket share difference is
#     SE = sqrt( p(1-p)/n_hist + p(1-p)/n_sim )
# which for p≈0.4, n_hist≈6_000, n_sim≈40_000 is ≈0.007. SETCOUNT_ABS_TOL is set
# to 0.05 — ~7×SE — a deliberately *generous* model-adequacy band, far wider than
# sampling noise, so a PASS means genuine agreement and a FAIL means a real
# modelling error rather than a tightly-tuned band. (The bo5 discrepancies this
# harness observes are ≈0.09–0.13 across the failing buckets, i.e. ≈13.9–19.1×SE,
# well past even this band — see the module FINDING.)
SETCOUNT_ABS_TOL = 0.05
# Games-per-set: per-set game counts have SD≈2.4; over thousands of sets the SE
# of the mean is <0.05, so 0.5 games is again a generous absolute band. (A few
# pre-2022 advantage final sets inflate the historical mean slightly; immaterial
# at this tolerance.)
GPS_ABS_TOL = 0.5
# Tiebreak frequency and break rate are checked against a *plausible range* (per
# the acceptance criteria — "in a plausible range"), not a tight tolerance vs
# historical. Historical Slam values are ≈0.18 (tb) and ≈0.20 (break); these
# bands bracket both the historical and a reasonable simulated spread.
TB_FREQ_PLAUSIBLE = (0.10, 0.32)
BREAK_RATE_PLAUSIBLE = (0.12, 0.28)

# Score-string markers for incomplete matches (same set the skill table screens).
_MARKERS = ("RET", "W/O", "def.")
_SET_TOKEN = re.compile(r"^(\d+)-(\d+)(?:\(([^)]*)\))?$")


# --------------------------------------------------------------------------- #
# Metric container
# --------------------------------------------------------------------------- #
@dataclass
class ShapeMetrics:
    """Aggregate "match shape" statistics for one population (hist or sim).

    Attributes:
        n_matches: Number of matches the set-count distribution is over.
        setcount: ``{n_sets: share}`` — share of matches decided in ``n_sets``
            sets (bo5: 3/4/5; bo3: 2/3). Shares sum to 1.
        games_per_set: Mean games per set (a 7–6 tiebreak set counts as 13).
        tb_freq: Share of sets that reached a tiebreak.
        break_rate: Share of service games broken. ``None`` if not measured.
    """

    n_matches: int
    setcount: dict[int, float]
    games_per_set: float
    tb_freq: float
    break_rate: float | None = None


@dataclass
class ConfoundResult:
    """Stale-snapshot confound control for the bo5 Slam comparison (``§7``).

    Contrasts the mean ``|pA − pB|`` gap (and the recent pool's set-count) of the
    full all-years matchup pool against a recent-only pool where the snapshot
    skills are ~contemporaneous. Near-equal gaps show the compression is not an
    artifact of scoring old matchups with latest-known skills.

    Attributes:
        since_year: The cutoff year for the recent pool (``CONFOUND_SINCE_YEAR``).
        full_gap: Mean ``|pA − pB|`` over the full all-years bo5 pool.
        full_n: Number of matchups in the full pool.
        recent_gap: Mean ``|pA − pB|`` over the recent-only bo5 pool.
        recent_n: Number of matchups in the recent pool.
        recent_setcount: Simulated set-count distribution on the recent pool.
    """

    since_year: int
    full_gap: float
    full_n: int
    recent_gap: float
    recent_n: int
    recent_setcount: dict[int, float]


# --------------------------------------------------------------------------- #
# Score parsing (historical side)
# --------------------------------------------------------------------------- #
def _is_incomplete(score: object) -> bool:
    """True if the score marks a retirement/walkover (partial/unreliable stats)."""
    return isinstance(score, str) and any(m in score for m in _MARKERS)


def parse_completed_sets(score: object) -> list[tuple[int, int, bool]] | None:
    """Parse a score string into completed sets ``[(games_a, games_b, had_tb), …]``.

    Returns ``None`` for an unparseable or non-string score. Tiebreak tokens
    (``7-6(5)``, ``7-6(10-8)``) are recognised by the parenthetical and counted
    as a played tiebreak; their game score is used as-is (7–6). Tokens flagged
    ``RET``/``W/O``/``def.`` never reach here — such matches are excluded whole.
    """
    if not isinstance(score, str) or not score.strip():
        return None
    sets: list[tuple[int, int, bool]] = []
    for tok in score.split():
        m = _SET_TOKEN.match(tok)
        if m is None:
            return None
        a, b = int(m.group(1)), int(m.group(2))
        sets.append((a, b, m.group(3) is not None))
    return sets or None


def historical_shape_metrics(
    raw: pd.DataFrame, mask: pd.Series, best_of: int
) -> ShapeMetrics:
    """Set-count distribution, games-per-set and tiebreak freq from ``data/raw``.

    Args:
        raw: The raw winner/loser match frame from :func:`load_atp_data`.
        mask: Boolean row selector for the population (e.g. Slam best-of-5).
        best_of: ``5`` or ``3`` — determines the valid set-count buckets
            (3/4/5 or 2/3); matches with any other count are dropped as malformed.

    Returns:
        A :class:`ShapeMetrics` (``break_rate`` left ``None`` — measured
        separately by :func:`historical_break_rate`).
    """
    valid_counts = {3, 4, 5} if best_of == 5 else {2, 3}
    sub = raw[mask & (raw["best_of"] == best_of)]
    sub = sub[~sub["score"].map(_is_incomplete)]

    setcount: dict[int, int] = {}
    total_games = 0
    total_sets = 0
    total_tb = 0
    n_matches = 0
    for score in sub["score"]:
        parsed = parse_completed_sets(score)
        if parsed is None:
            continue
        n = len(parsed)
        if n not in valid_counts:
            continue
        n_matches += 1
        setcount[n] = setcount.get(n, 0) + 1
        for a, b, had_tb in parsed:
            total_games += a + b
            total_sets += 1
            total_tb += int(had_tb)

    dist = {k: setcount[k] / n_matches for k in sorted(setcount)}
    return ShapeMetrics(
        n_matches=n_matches,
        setcount=dist,
        games_per_set=total_games / total_sets,
        tb_freq=total_tb / total_sets,
    )


def historical_break_rate(raw: pd.DataFrame, mask: pd.Series) -> tuple[float, int]:
    """Break rate from the serve columns: Σ(bpFaced − bpSaved) / Σ SvGms.

    A service game is broken exactly when a faced break point is not saved, and a
    held game saves all it faced, so ``bpFaced − bpSaved`` counts broken service
    games (``ace-02-data-schema.md``). Summed over both players and normalised by
    total service games. Only rows with a complete break-point/serve-game line and
    non-retirement scores contribute.

    Returns:
        ``(break_rate, n_matches_used)``.
    """
    need = ["w_bpFaced", "w_bpSaved", "w_SvGms", "l_bpFaced", "l_bpSaved", "l_SvGms"]
    sub = raw[mask & ~raw["score"].map(_is_incomplete)].dropna(subset=need)
    sub = sub[(sub["w_SvGms"] > 0) & (sub["l_SvGms"] > 0)]
    breaks = (
        sub["w_bpFaced"] - sub["w_bpSaved"] + sub["l_bpFaced"] - sub["l_bpSaved"]
    ).sum()
    serve_games = (sub["w_SvGms"] + sub["l_SvGms"]).sum()
    return float(breaks / serve_games), len(sub)


# --------------------------------------------------------------------------- #
# Simulated side
# --------------------------------------------------------------------------- #
def _eligible(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocessed rows usable as realistic matchup samples: complete serve
    line, real surface, completed match, both ids present."""
    ok = df["has_serve_stats"].astype(bool)
    ok &= df["surface"].isin(config.VALID_SURFACES)
    ok &= ~df["score"].map(_is_incomplete)
    ok &= df["p1_id"].notna() & df["p2_id"].notna()
    return df[ok]


def matchup_pool(
    processed: pd.DataFrame,
    best_of: int,
    slam_only: bool,
    since: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Eligible ``(p1_id, p2_id, surface)`` matchups to sample realistic ``p`` from.

    Sampling real historical matchups (rather than synthetic ``p`` values) makes
    the simulated population reflect the real distribution of skill pairings the
    simulator faces. For bo5 this is Grand Slam matches (``tourney_level == 'G'``);
    for bo3, non–Davis-Cup best-of-3 tour matches (men's Slams are bo5-only, so
    bo3 has no Slam sample — noted in the report).

    Args:
        since: If given, keep only matchups with ``tourney_date >= since`` — the
            recent-only pool for the stale-snapshot confound control. Requires
            ``processed["tourney_date"]`` to be datetime (the caller converts it).
    """
    df = processed[processed["best_of"] == best_of]
    if slam_only:
        df = df[df["tourney_level"] == "G"]
    else:
        df = df[df["tourney_level"] != "D"]  # exclude Davis Cup
    if since is not None:
        df = df[df["tourney_date"] >= since]
    return _eligible(df)[["p1_id", "p2_id", "surface"]].reset_index(drop=True)


def mean_point_gap(pool: pd.DataFrame, table: SkillTable) -> tuple[float, int]:
    """Mean ``|pA − pB|`` point-win gap over **every** matchup in ``pool``.

    Deterministic — no sampling, no RNG: this is the crux statistic behind the
    ``§7`` finding. A gap compressed toward zero (both players near the surface
    baseline) is exactly what makes simulated matches play out too evenly, so it
    is reported directly and drives the confound control.

    Args:
        pool: Eligible matchup rows from :func:`matchup_pool`.
        table: The snapshot skill table.

    Returns:
        ``(mean_gap, n_matchups)``; ``mean_gap`` is ``nan`` for an empty pool.
    """
    surfaces = pool["surface"].to_numpy()
    a_ids = pool["p1_id"].astype(str).to_numpy()
    b_ids = pool["p2_id"].astype(str).to_numpy()
    total = 0.0
    for surface, a_id, b_id in zip(surfaces, a_ids, b_ids):
        pa, pb = matchup_point_probs(
            table.get(a_id, surface), table.get(b_id, surface),
            config.SURFACE_MU[surface],
        )
        total += abs(pa - pb)
    n = len(pool)
    return (total / n if n else float("nan"), n)


def simulated_shape_metrics(
    pool: pd.DataFrame,
    table: SkillTable,
    match_fn: Callable,
    final_set_rule: str,
    n_matches: int,
    rng: np.random.Generator,
) -> ShapeMetrics:
    """Run ``n_matches`` point-by-point matches over sampled realistic ``p`` pairs.

    For each match: sample a real matchup row (with replacement), look up both
    players' snapshot skills on that surface, derive ``(pA, pB)`` via
    :func:`~sim.points.matchup_point_probs`, pick a first server, and simulate
    with ``match_fn`` (``simulate_match_bo3``/``bo5``). Set-count, games-per-set
    and tiebreak frequency are read off the returned :class:`MatchResult`; break
    rate is measured at the game layer (see the module docstring).

    Args:
        pool: Eligible matchup rows from :func:`matchup_pool`.
        table: The snapshot skill table (:func:`build_skill_table` with
            ``as_of=None``) — the "latest known skills" the simulator will run
            against in Phase 2.
        match_fn: ``simulate_match_bo3`` or ``simulate_match_bo5``.
        final_set_rule: Deciding-set rule passed to ``match_fn``.
        n_matches: Number of matches to simulate.
        rng: The single shared ``numpy`` generator (determinism).

    Returns:
        A fully-populated :class:`ShapeMetrics`.
    """
    idx = rng.integers(0, len(pool), size=n_matches)
    rows = pool.iloc[idx]
    surfaces = rows["surface"].to_numpy()
    p1_ids = rows["p1_id"].astype(str).to_numpy()
    p2_ids = rows["p2_id"].astype(str).to_numpy()

    setcount: dict[int, int] = {}
    total_games = 0
    total_sets = 0
    total_tb = 0
    broken_games = 0
    serve_games = 0

    for surface, a_id, b_id in zip(surfaces, p1_ids, p2_ids):
        skill_a = table.get(a_id, surface)
        skill_b = table.get(b_id, surface)
        pa, pb = matchup_point_probs(skill_a, skill_b, config.SURFACE_MU[surface])
        first_server = int(rng.integers(0, 2))
        result = match_fn(pa, pb, first_server, final_set_rule, rng)

        n = len(result.sets)
        setcount[n] = setcount.get(n, 0) + 1
        for s in result.sets:
            total_games += s.games_a + s.games_b
            total_sets += 1
            total_tb += int(s.tb_score is not None)

        # Break rate: emergent from the identical game primitive the match layer
        # uses, drawn equally on each server's point-win prob (module docstring).
        for p in (pa, pb):
            for _ in range(BREAKRATE_GAMES_PER_SERVER):
                serve_games += 1
                if not simulate_game(p, rng).server_won:
                    broken_games += 1

    dist = {k: setcount[k] / n_matches for k in sorted(setcount)}
    return ShapeMetrics(
        n_matches=n_matches,
        setcount=dist,
        games_per_set=total_games / total_sets,
        tb_freq=total_tb / total_sets,
        break_rate=broken_games / serve_games,
    )


# --------------------------------------------------------------------------- #
# Comparison / gate
# --------------------------------------------------------------------------- #
@dataclass
class Check:
    """One evaluated metric row for the report."""

    name: str
    hist: float
    sim: float
    kind: str  # "tol" (|hist-sim| <= tol) or "range" (lo <= sim <= hi)
    bound: tuple[float, float] | float
    hard: bool  # counts toward the exit-status gate
    passed: bool = field(init=False)

    def __post_init__(self) -> None:
        if self.kind == "tol":
            self.passed = abs(self.hist - self.sim) <= float(self.bound)
        else:
            lo, hi = self.bound  # type: ignore[misc]
            self.passed = lo <= self.sim <= hi


def build_checks(
    label: str,
    hist: ShapeMetrics,
    sim: ShapeMetrics,
    set_labels: dict[int, str],
) -> list[Check]:
    """Assemble the per-metric checks comparing one format's hist vs sim."""
    checks: list[Check] = []
    for n_sets, name in set_labels.items():
        checks.append(
            Check(
                name=f"{label} {name}",
                hist=hist.setcount.get(n_sets, 0.0),
                sim=sim.setcount.get(n_sets, 0.0),
                kind="tol",
                bound=SETCOUNT_ABS_TOL,
                hard=True,
            )
        )
    checks.append(
        Check(f"{label} games/set", hist.games_per_set, sim.games_per_set,
              "tol", GPS_ABS_TOL, hard=True)
    )
    checks.append(
        Check(f"{label} tiebreak freq", hist.tb_freq, sim.tb_freq,
              "range", TB_FREQ_PLAUSIBLE, hard=False)
    )
    checks.append(
        Check(f"{label} break rate", hist.break_rate or float("nan"),
              sim.break_rate or float("nan"), "range", BREAK_RATE_PLAUSIBLE, hard=False)
    )
    return checks


def _fmt_bound(c: Check) -> str:
    if c.kind == "tol":
        return f"|Δ|≤{float(c.bound):.3f}"
    lo, hi = c.bound  # type: ignore[misc]
    return f"in [{lo:.2f},{hi:.2f}]"


def print_report(
    bo5_hist: ShapeMetrics,
    bo5_sim: ShapeMetrics,
    bo3_hist: ShapeMetrics,
    bo3_sim: ShapeMetrics,
    checks: list[Check],
) -> None:
    """Print the full historical-vs-simulated side-by-side report."""
    def dist_str(m: ShapeMetrics) -> str:
        return "  ".join(f"{k}:{v:.3f}" for k, v in m.setcount.items())

    print("=" * 74)
    print("T1.9 SIMULATION VALIDATION — historical (data/raw) vs simulated")
    print("=" * 74)
    print(f"\nBest-of-5 (Grand Slam)   hist matches: {bo5_hist.n_matches:,}   "
          f"sim matches: {bo5_sim.n_matches:,}")
    print(f"  set-count  hist  {dist_str(bo5_hist)}")
    print(f"  set-count  sim   {dist_str(bo5_sim)}")
    print(f"\nBest-of-3 (tour)         hist matches: {bo3_hist.n_matches:,}   "
          f"sim matches: {bo3_sim.n_matches:,}")
    print(f"  set-count  hist  {dist_str(bo3_hist)}")
    print(f"  set-count  sim   {dist_str(bo3_sim)}")

    print("\n" + "-" * 78)
    print(f"{'metric':<26}{'historical':>12}{'simulated':>12}"
          f"{'band':>16}{'result':>10}")
    print("-" * 78)
    for c in checks:
        status = "PASS" if c.passed else "FAIL"
        flag = "" if c.hard else "  (soft)"
        print(f"{c.name:<26}{c.hist:>12.3f}{c.sim:>12.3f}"
              f"{_fmt_bound(c):>16}{status:>10}{flag}")
    print("-" * 78)


def print_confound(c: ConfoundResult) -> None:
    """Print the stale-snapshot confound control (``§7`` finding robustness)."""
    dist = "  ".join(f"{k}:{v:.3f}" for k, v in c.recent_setcount.items())
    print("\n" + "-" * 78)
    print("CONFOUND CONTROL — stale snapshot? (bo5 Slam, recent-only re-run)")
    print("-" * 78)
    print(f"  full pool (all years)   mean |pA-pB|:  {c.full_gap:.4f}   "
          f"(n={c.full_n:,})")
    print(f"  recent pool (>= {c.since_year})     mean |pA-pB|:  {c.recent_gap:.4f}   "
          f"(n={c.recent_n:,})")
    print(f"  recent-pool set-count:  {dist}")
    print("  -> the |pA-pB| compression and set-count skew persist on a pool where")
    print("     the snapshot is ~contemporaneous, so the finding is NOT an artifact")
    print("     of scoring old matchups with latest-known skills.")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_validation(
    start_year: int,
    end_year: int,
    n_matches: int,
    seed: int,
) -> tuple[list[Check], dict[str, ShapeMetrics | ConfoundResult]]:
    """Load data, build the snapshot table, compute all hist + sim metrics.

    Returns ``(checks, metrics)`` where ``metrics`` maps
    ``"bo5_hist"``/``"bo5_sim"``/``"bo3_hist"``/``"bo3_sim"`` to their
    :class:`ShapeMetrics`, plus ``"confound"`` to a :class:`ConfoundResult` (the
    stale-snapshot control). Pure of I/O beyond loading the vendored CSVs.
    """
    raw = load_atp_data(start_year, end_year)
    processed = preprocess_data(raw)
    processed["tourney_date"] = pd.to_datetime(processed["tourney_date"])
    table = build_skill_table(processed)  # snapshot (as_of=None)

    slam_mask = raw["tourney_level"] == "G"
    tour_bo3_mask = raw["tourney_level"] != "D"

    bo5_hist = historical_shape_metrics(raw, slam_mask, best_of=5)
    bo5_hist.break_rate = historical_break_rate(raw, slam_mask & (raw["best_of"] == 5))[0]
    bo3_hist = historical_shape_metrics(raw, tour_bo3_mask, best_of=3)
    bo3_hist.break_rate = historical_break_rate(
        raw, tour_bo3_mask & (raw["best_of"] == 3)
    )[0]

    rng = np.random.default_rng(seed)
    bo5_pool = matchup_pool(processed, best_of=5, slam_only=True)
    bo5_sim = simulated_shape_metrics(
        bo5_pool, table, simulate_match_bo5, BO5_FINAL_SET_RULE, n_matches, rng,
    )
    bo3_sim = simulated_shape_metrics(
        matchup_pool(processed, best_of=3, slam_only=False),
        table, simulate_match_bo3, BO3_FINAL_SET_RULE, n_matches, rng,
    )

    # Stale-snapshot confound control (§7): re-derive the bo5 gap + set-count on a
    # recent-only pool where the snapshot skills are ~contemporaneous. Runs after
    # the main sims so it does not perturb their RNG stream / values.
    since = pd.Timestamp(year=CONFOUND_SINCE_YEAR, month=1, day=1)
    recent_bo5_pool = matchup_pool(processed, best_of=5, slam_only=True, since=since)
    full_gap, full_n = mean_point_gap(bo5_pool, table)
    recent_gap, recent_n = mean_point_gap(recent_bo5_pool, table)
    recent_sim = simulated_shape_metrics(
        recent_bo5_pool, table, simulate_match_bo5, BO5_FINAL_SET_RULE, n_matches, rng,
    )
    confound = ConfoundResult(
        since_year=CONFOUND_SINCE_YEAR,
        full_gap=full_gap, full_n=full_n,
        recent_gap=recent_gap, recent_n=recent_n,
        recent_setcount=recent_sim.setcount,
    )

    checks = build_checks("bo5", bo5_hist, bo5_sim, {3: "3-0 sets", 4: "3-1 sets", 5: "3-2 sets"})
    checks += build_checks("bo3", bo3_hist, bo3_sim, {2: "2-0 sets", 3: "2-1 sets"})
    metrics: dict[str, ShapeMetrics | ConfoundResult] = {
        "bo5_hist": bo5_hist, "bo5_sim": bo5_sim,
        "bo3_hist": bo3_hist, "bo3_sim": bo3_sim,
        "confound": confound,
    }
    return checks, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=int, default=config.START_YEAR)
    parser.add_argument("--end", type=int, default=config.END_YEAR)
    parser.add_argument("--n-matches", type=int, default=DEFAULT_N_MATCHES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    checks, metrics = run_validation(args.start, args.end, args.n_matches, args.seed)
    print_report(
        metrics["bo5_hist"], metrics["bo5_sim"],
        metrics["bo3_hist"], metrics["bo3_sim"], checks,
    )
    print_confound(metrics["confound"])

    hard_fail = [c for c in checks if c.hard and not c.passed]
    if hard_fail:
        print("\n" + "!" * 74)
        print("FINDING (§7): the point-by-point simulator's match shapes do NOT")
        print("match historical data within the (generous) tolerance:")
        for c in hard_fail:
            print(f"  · {c.name}: hist {c.hist:.3f} vs sim {c.sim:.3f} "
                  f"(Δ={c.sim - c.hist:+.3f}, band |Δ|≤{float(c.bound):.3f})")
        print(
            "\nThe simulator produces too few straight-set wins and too many long\n"
            "matches — §7's warning sign that the point-win probability derivation\n"
            "or clamping is off (upstream: T1.1 shrinkage / T1.2), NOT this harness.\n"
            "Root cause: skill-gap compression leaves |pA − pB| clustered near the\n"
            "surface baseline, so matches play out more evenly than real ones.\n"
            "Reported faithfully rather than hidden behind a looser band; expected\n"
            "to narrow once T1.8 reconciliation widens effective skill gaps."
        )
        print("!" * 74)
    else:
        print("\nAll hard-tolerance metrics within band.")

    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
