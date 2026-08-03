"""UI-free ``ClassifierProb`` adapter with **real** rolling-form features (T3.5).

This module closes ``ace-04-current-state.md §7`` seam 7, open since T1.10. It
does two things that were previously impossible together:

1. **It lives where neither ``cli/`` nor ``api/`` owns it** — the same precedent
   T0.6 set when it lifted the fuzzy name resolver out of ``cli/interactive.py``
   into ``common/names.py``. Before this, the only ``ClassifierProb`` adapter was
   ``cli.simulate_match.make_classifier_prob``, built on
   ``cli/interactive.build_feature_row``; the layering rule forbids ``api/`` from
   importing ``cli/``, so T3.4 had to reach it by name through
   ``config.API_CLASSIFIER_ADAPTER`` and disclose the runtime coupling. With the
   adapter here, that coupling is gone rather than disclosed: ``api/`` and
   ``scripts/`` import ``common``, which imports nothing but ``config`` and
   ``features``.

2. **It assembles all 27 of ``config.MODEL_FEATURES`` from real state.** The
   7 the CLI already built correctly (both ranks, both ages, both surface
   win %, ``h2h_diff``) come from the same leakage-safe histories
   ``features.engineering.add_features`` returns. The other 20 — every
   recent-form rolling column, previously filled with ``0.5``/``0`` constants —
   come from :class:`features.rolling.RollingFormTable`, the as-of-now snapshot
   accessor added alongside this module. Those values are the ones the training
   pipeline would compute for a match played immediately after the last row of
   the data, so a row this adapter builds is the same *kind* of row the
   classifier was trained on rather than one that is out-of-distribution on 74%
   of its features.

**What "as of now" means, and why it is leakage-safe.** Every input here is a
snapshot of the *end* of the loaded data: latest rank/age, full surface record,
full H2H, last-N form. That is the correct state for simulating a match played
now (which is what the simulator does), and it never reaches backwards into a
row's own future the way a leaky training feature would — nothing here is used
to fit anything. It does mean simulating an **already-played** draw is
retrospective rather than a forecast: the model knows what happened after the
event. That caveat is what ``api.schemas.CLASSIFIER_LIMITATION`` now describes.

**Shape.** :func:`make_classifier_prob` keeps the exact factory signature T1.10
established and T3.3/T3.4 build against — ``(estimator, data, surface_history,
h2h_history) -> ClassifierProb`` — so ``sim/reconcile.py``'s expected interface,
``api.main.ClassifierFactory`` and ``scripts/precompute_sim.py`` are unchanged.
The returned :class:`ClassifierProbAdapter` is a **module-level class with a
``__call__``**, not a closure: same behaviour, same memoisation, but it pickles,
which is the one thing ``sim/tournament.py``'s ``workers > 1`` path needs from an
adapter (``§7`` seam 8 — unblocked here as a side effect; actually exposing a
``--workers`` flag, and re-verifying it, remains that seam's own work).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

import config
import features.rolling as rolling


# --------------------------------------------------------------------------- #
# Name-keyed history lookups.
#
# These are the helpers ``cli/interactive.py`` grew for the REPL (``get_latest``,
# ``get_surf_record``, ``compute_h2h``). They are the *classifier's* view of the
# pipeline's history structures, not UI, so they belong here; the CLI's versions
# are now thin wrappers that keep their printable extras (T0.6's pattern for the
# resolver, applied to the same file's other reusable helpers).
# --------------------------------------------------------------------------- #
def latest_rank_age(name: str, data: pd.DataFrame) -> tuple[float | None, float | None]:
    """A player's rank and age as of their most recent match in ``data``.

    Args:
        name: Display name, as it appears in ``p1_name``/``p2_name``.
        data: The engineered match frame.

    Returns:
        ``(rank, age)``, or ``(None, None)`` if the player has no match in
        ``data``.
    """
    player_mask = (data['p1_name'] == name) | (data['p2_name'] == name)
    player_matches = data.loc[player_mask]

    if player_matches.empty:
        return None, None

    latest_match = player_matches.sort_values('tourney_date', ascending=False).iloc[0]

    if latest_match['p1_name'] == name:
        return latest_match['p1_rank'], latest_match['p1_age']
    return latest_match['p2_rank'], latest_match['p2_age']


def surface_record(player: str, surface: str, surface_history: dict) -> tuple[int, int]:
    """A player's ``(wins, total)`` on ``surface`` from the leakage-safe history.

    Returns ``(0, 0)`` when the player has never played the surface.
    """
    if player in surface_history and surface in surface_history[player]:
        return surface_history[player][surface]
    return 0, 0


def surface_win_pct(wins: int, total: int) -> float:
    """Win rate from a surface record, falling back to ``config.DEFAULT_WIN_PCT``."""
    return wins / total if total > 0 else config.DEFAULT_WIN_PCT


def h2h_record(player_a: str, player_b: str, h2h_history: dict) -> tuple[int, int]:
    """Head-to-head wins as ``(a_wins, b_wins)``.

    ``h2h_history`` is keyed by the *sorted* name pair (see
    ``features/engineering.py``), so the stored pair is re-oriented to the
    caller's A/B order here.
    """
    key = tuple(sorted([player_a, player_b]))
    if key not in h2h_history:
        return 0, 0
    first_wins, second_wins = h2h_history[key]
    if player_a == key[0]:
        return first_wins, second_wins
    return second_wins, first_wins


# --------------------------------------------------------------------------- #
# The assembled feature row.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MatchupFeatures:
    """The non-rolling half of a feature row: the 7 features built from history.

    Carries the raw surface records alongside the percentages because callers
    that *display* a matchup want "80% (8-2)", not just the rate — that is what
    ``cli/simulate_match.MatchupStats`` wraps for its table.
    """

    a_rank: float
    b_rank: float
    a_age: float
    b_age: float
    a_pct: float
    b_pct: float
    a_wins: int
    a_total: int
    b_wins: int
    b_total: int
    h2h_diff: int


def matchup_features(
    player_a: str,
    player_b: str,
    surface: str,
    data: pd.DataFrame,
    surface_history: dict,
    h2h_history: dict,
) -> MatchupFeatures:
    """Gather the classifier's name-keyed inputs for one matchup.

    Args:
        player_a, player_b: Canonical display names (already resolved).
        surface: ``"Hard"``/``"Clay"``/``"Grass"``.
        data: The engineered match frame (latest rank/age).
        surface_history, h2h_history: The leakage-safe history dicts from
            ``features.engineering.add_features``.

    Returns:
        A :class:`MatchupFeatures`.

    Raises:
        ValueError: If either player has no match history in ``data``. Failing
            here is deliberate — a feature row built from defaults for an unknown
            player is a silently wrong prediction, which is the class of problem
            this module exists to remove.
    """
    a_rank, a_age = latest_rank_age(player_a, data)
    b_rank, b_age = latest_rank_age(player_b, data)
    if a_rank is None or b_rank is None:
        missing = player_a if a_rank is None else player_b
        raise ValueError(
            f"No match history for {missing!r}; cannot build a classifier feature row."
        )

    a_wins, a_total = surface_record(player_a, surface, surface_history)
    b_wins, b_total = surface_record(player_b, surface, surface_history)
    a_h2h, b_h2h = h2h_record(player_a, player_b, h2h_history)

    return MatchupFeatures(
        a_rank=a_rank,
        b_rank=b_rank,
        a_age=a_age,
        b_age=b_age,
        a_pct=surface_win_pct(a_wins, a_total),
        b_pct=surface_win_pct(b_wins, b_total),
        a_wins=a_wins,
        a_total=a_total,
        b_wins=b_wins,
        b_total=b_total,
        h2h_diff=a_h2h - b_h2h,
    )


def build_feature_row(
    features: MatchupFeatures,
    rolling_a: dict[str, float],
    rolling_b: dict[str, float],
) -> pd.DataFrame:
    """Assemble the single-row ``config.MODEL_FEATURES`` frame ``predict_proba`` takes.

    Every one of the 27 columns is populated from real state: the 7 in
    ``features`` and the 20 in ``rolling_a``/``rolling_b`` (whose keys are
    ``features.rolling``'s unprefixed ``recent_*`` names, prefixed ``p1_``/``p2_``
    here).

    Args:
        features: The history-derived half, from :func:`matchup_features`.
        rolling_a, rolling_b: ``RollingFormTable.latest(player)`` for A and B.

    Returns:
        A one-row float ``DataFrame`` whose columns are exactly
        ``config.MODEL_FEATURES``, in order.

    Raises:
        KeyError: If a model feature has no value — i.e. if ``MODEL_FEATURES``
            has gained a column neither half supplies. Loud by design: quietly
            defaulting an unknown feature is precisely the seam-7 defect.
    """
    values: dict[str, float] = {
        'p1_rank': features.a_rank,
        'p2_rank': features.b_rank,
        'p1_age': features.a_age,
        'p2_age': features.b_age,
        'p1_surface_win_pct': features.a_pct,
        'p2_surface_win_pct': features.b_pct,
        'h2h_diff': features.h2h_diff,
    }
    for column, value in rolling_a.items():
        values[f'p1_{column}'] = value
    for column, value in rolling_b.items():
        values[f'p2_{column}'] = value

    missing = [column for column in config.MODEL_FEATURES if column not in values]
    if missing:
        raise KeyError(
            f"no value assembled for model feature(s) {missing}; the adapter and "
            f"config.MODEL_FEATURES have drifted apart."
        )

    return pd.DataFrame(
        [[float(values[column]) for column in config.MODEL_FEATURES]],
        columns=config.MODEL_FEATURES,
        dtype=float,
    )


# --------------------------------------------------------------------------- #
# The adapter.
# --------------------------------------------------------------------------- #
class ClassifierProbAdapter:
    """``(player_a, player_b, surface) -> P(A beats B)`` over the pinned estimator.

    Implements ``sim.reconcile.ClassifierProb``. Constructing one does **no**
    work — the contract every existing caller was built against
    (``api.main.build_classifier``, ``scripts/precompute_sim.py``). The as-of-now
    rolling-form table is built on first use and kept (0.08 s over the full
    frame, once per adapter), the first call for a matchup builds its row and
    calls ``predict_proba``, and every repeat is served from the memo table. A
    5,000-run Monte Carlo over a 128 draw therefore costs at most 8,128
    ``predict_proba`` calls, not 635,000 — the property T2.2's ``prob_cache``
    contract depends on.

    A class rather than a closure so the adapter is picklable (``§7`` seam 8) and
    so its assembled row is inspectable via :meth:`feature_row` — which is how
    ``tests/test_classifier_adapter.py`` proves the rolling features are the
    pipeline's real ones rather than constants.
    """

    def __init__(
        self,
        estimator: Any,
        data: pd.DataFrame,
        surface_history: dict,
        h2h_history: dict,
        form_table: rolling.RollingFormTable | None = None,
    ) -> None:
        """
        Args:
            estimator: The pinned classifier — treated opaquely, only
                ``predict_proba`` is assumed (``ace-04-current-state.md §4``).
            data: The engineered match frame.
            surface_history, h2h_history: Name-keyed histories from
                ``features.engineering.add_features``.
            form_table: A prebuilt :class:`~features.rolling.RollingFormTable`;
                derived from ``data`` on first use when omitted. Passing one lets
                a caller that already has it (or a test) skip the scan.
        """
        self.estimator = estimator
        self.data = data
        self.surface_history = surface_history
        self.h2h_history = h2h_history
        self._form_table = form_table
        self._cache: dict[tuple[str, str, str], float] = {}

    @property
    def form_table(self) -> rolling.RollingFormTable:
        """The as-of-now rolling-form snapshot, built once on first use.

        Deferred rather than built in ``__init__`` so that *constructing* an
        adapter stays free: startup wiring (and every app fixture that never
        simulates) pays nothing, and the scan is charged to the first simulation
        that needs it.
        """
        if self._form_table is None:
            self._form_table = rolling.build_rolling_form_table(self.data)
        return self._form_table

    def feature_row(self, player_a: str, player_b: str, surface: str) -> pd.DataFrame:
        """The full 27-feature row for one matchup, exactly as scored.

        Raises:
            ValueError: If either player has no history (see
                :func:`matchup_features`) — including the rolling-form lookup,
                whose ``KeyError`` is translated here so callers see one
                exception type for "unknown player".
        """
        features = matchup_features(
            player_a, player_b, surface, self.data, self.surface_history, self.h2h_history
        )
        form_table = self.form_table
        for name in (player_a, player_b):
            if name not in form_table:
                # matchup_features already rejects a player absent from `data`,
                # so reaching here means `data` and the form table disagree — a
                # wiring error, not a user error. Say so rather than defaulting.
                raise ValueError(
                    f"No rolling-form history for {name!r}; the form table and the "
                    f"match frame were built from different data."
                )
        return build_feature_row(
            features, form_table.latest(player_a), form_table.latest(player_b)
        )

    def __call__(self, player_a: str, player_b: str, surface: str) -> float:
        """P(A beats B) on ``surface``, memoised per matchup.

        ``surface`` is part of the key because the row is surface-dependent
        (``p*_surface_win_pct``); dropping it would serve a Clay request the
        cached Hard answer.
        """
        key = (player_a, player_b, surface)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        row = self.feature_row(player_a, player_b, surface)
        prob = float(self.estimator.predict_proba(row)[0][1])  # P(player A wins)
        self._cache[key] = prob
        return prob


def make_classifier_prob(
    estimator: Any,
    data: pd.DataFrame,
    surface_history: dict,
    h2h_history: dict,
) -> ClassifierProbAdapter:
    """Build the ``ClassifierProb`` adapter ``sim/reconcile.py`` takes.

    The factory signature T1.10 established, kept verbatim so every existing
    caller (``api.main.build_classifier``, ``scripts/precompute_sim.py``,
    ``cli/simulate_match.build_context``) is unchanged by the move.

    Args:
        estimator: The pinned ``predict_proba`` estimator.
        data: The engineered match frame.
        surface_history, h2h_history: Name-keyed histories from ``add_features``.

    Returns:
        A :class:`ClassifierProbAdapter`.
    """
    return ClassifierProbAdapter(estimator, data, surface_history, h2h_history)
