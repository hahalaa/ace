"""Elo ratings, a standalone *display* feature (elo-ratings branch).

⚠️ **DISPLAY ONLY. This module is walled off from the model.** The ratings it
computes are shown on a Rankings screen and nowhere else. They must never enter
``config.MODEL_FEATURES``, the classifier, the point model, reconciliation, or
any simulation path. Nothing under ``data/preprocess.py``, ``features/`` (except
this file), ``model/`` or ``sim/`` may import it or name one of its
``config.ELO_*`` knobs. That wall is not a convention here,
``tests/test_elo_isolation.py`` walks the import graph and the source for both
and fails if either is breached, the same way
``tests/test_benchmark_vs_market.py`` guards the ``BENCHMARK_*`` constants that
also live in ``config.py``. Adding Elo as "a 28th feature" would need the same multi-tier
validation-year process the four prior model-accuracy experiments went through,
which is explicitly out of scope for a display feature.

What it is
----------
A per-player Elo rating computed from match results, in four parallel tracks:

  * **overall**, updated by *every* match, whatever the surface.
  * **Hard / Clay / Grass**, each updated *only* by matches on that surface.

Each track carries its own rating and its own match count (so a player's clay K
shrinks on their clay history, independent of their hard-court history). Carpet
and surface-unknown matches move only the overall track, mirroring the rest of
the project's three-surface discipline (``config.VALID_SURFACES``).

The formula is textbook Elo (``docs/ace-03-tennis-math.md`` uses the same
logistic elsewhere)::

    E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))
    R_A' = R_A + K * (S_A - E_A)          # S_A = 1 for a win, 0 for a loss

with a **dynamic K** that shrinks as a player accumulates matches, the standard
stabiliser used by public tennis-Elo work (Kovalchik 2016; FiveThirtyEight)::

    K(n) = ELO_K_FACTOR / (n + ELO_K_OFFSET) ** ELO_K_SHAPE

where ``n`` is the number of matches the player has *already* played on that
track. A newcomer's rating moves fast; an established player's barely twitches
on any single result, which is what keeps a veteran's number from swinging on
one upset. ``k_factor`` is injectable so a test can pin a constant K and
hand-verify the arithmetic.

Chronological correctness
-------------------------
Elo is path-dependent: process the same matches in a different order and you get
different ratings. Feeding an unordered frame to a rating loop is therefore a
real correctness bug, not a style point. :func:`compute_elo` sorts by date (then
by a within-tournament round order) before it iterates, and :func:`_elo_pass`
**asserts** the sequence it is handed is non-decreasing in date and raises if it
is not, the ordering requirement is enforced in code, never merely assumed of
the input.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

import config

# --------------------------------------------------------------------------- #
# Constants.
# --------------------------------------------------------------------------- #
# The tunable knobs live in config.py under the "ELO — DISPLAY ONLY" block, next
# to BENCHMARK_* and for the same reason: the project keeps magic numbers in one
# place, and the real wall (nothing in the model/simulation path importing this
# module or naming a config.ELO_* constant) is enforced by static analysis in
# tests/test_elo_isolation.py, not by hiding the values here. Re-exported at
# module scope so callers and tests can read them from features.elo directly.
ELO_INITIAL_RATING = config.ELO_INITIAL_RATING
ELO_K_FACTOR = config.ELO_K_FACTOR
ELO_K_OFFSET = config.ELO_K_OFFSET
ELO_K_SHAPE = config.ELO_K_SHAPE
ELO_ACTIVE_WINDOW_DAYS = config.ELO_ACTIVE_WINDOW_DAYS
ELO_SURFACES: tuple[str, ...] = config.ELO_SURFACES
ELO_LEADERBOARD_LIMIT = config.ELO_LEADERBOARD_LIMIT
ELO_OVERALL = "overall"
ELO_TRACKS: tuple[str, ...] = (ELO_OVERALL, *ELO_SURFACES)

# Where the precompute writes the ratings and the API reads them back, the one
# definition of that path, shared by writer and reader (mirrors
# api.registry.cache_path_for for the Monte Carlo cache).
ELO_CACHE_FILENAME = "elo_ratings.json"

# Within a shared tournament date, matches should still be applied in play order.
# tourney_date is the tournament's *start* date, so every match in one event
# shares it; this orders the rounds within that event. Round-robin first, then
# the knockout from the widest round inward, with the bronze-medal match sitting
# just before the final. Unknown rounds fall in the middle and never reorder
# across dates (date is always the primary key).
ROUND_ORDER: dict[str, int] = {
    "RR": 0,
    "R128": 1,
    "R64": 2,
    "R32": 3,
    "R16": 4,
    "QF": 5,
    "SF": 6,
    "BR": 7,
    "F": 8,
}
_ROUND_ORDER_DEFAULT = 4


def dynamic_k(matches_played: int) -> float:
    """The default, match-count-shrinking K (see the module docstring).

    Args:
        matches_played: Matches this player has already played on the track being
            updated (0 for their first). Larger ``n`` → smaller K → a steadier
            rating.
    """
    return ELO_K_FACTOR / (matches_played + ELO_K_OFFSET) ** ELO_K_SHAPE


# --------------------------------------------------------------------------- #
# Result types.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PlayerRating:
    """One player's standing in one track.

    Attributes:
        player_id: Canonical id (the join key everywhere in ``sim``/``api``).
        player_name: Latest display name seen for the id.
        rating: Elo rating after the player's last match on this track.
        matches: Matches that moved this track's rating for the player.
        last_played: Date of the player's last match *on this track*.
        is_active: Whether the player is active tour-wide (their overall last
            match is within :data:`ELO_ACTIVE_WINDOW_DAYS` of the data's latest
            match). A whole-player property, identical across every track, so a
            clay board does not drop a player merely between clay swings.
    """

    player_id: str
    player_name: str
    rating: float
    matches: int
    last_played: date
    is_active: bool


@dataclass(frozen=True)
class Leaderboard:
    """One track's players, sorted by rating descending."""

    track: str
    ratings: tuple[PlayerRating, ...]


@dataclass(frozen=True)
class EloResult:
    """Everything the precompute needs to serialise a full ratings run.

    Attributes:
        leaderboards: Track name → :class:`Leaderboard`. Keys are
            :data:`ELO_TRACKS`.
        as_of: The latest match date in the data, the reference the active
            window is measured back from.
        active_cutoff: ``as_of - ELO_ACTIVE_WINDOW_DAYS``; a player whose overall
            last match is on or after this date is active.
        n_matches: Matches that contributed to the overall track.
        data_through_year: Latest season present in the data.
    """

    leaderboards: dict[str, Leaderboard]
    as_of: date
    active_cutoff: date
    n_matches: int
    data_through_year: int


# --------------------------------------------------------------------------- #
# Match preparation.
# --------------------------------------------------------------------------- #
# The columns a rating pass needs from the raw loader frame (data/loader.py),
# which carries winner/loser directly, before preprocess.py randomises them into
# p1/p2. Using the raw frame keeps "who won" explicit rather than reconstructing
# it from the target label.
_REQUIRED_COLUMNS = (
    "tourney_date",
    "winner_id",
    "winner_name",
    "loser_id",
    "loser_name",
)


def _clean_id(value: object) -> str | None:
    """Normalise a raw id to a non-empty string, or ``None`` if unusable.

    Vendor ids can be missing or NaN (see ace-02-data-schema.md); a match with no
    id on either side has no player to attach a rating to and is dropped.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def prepare_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Select, clean and **chronologically sort** the match frame for rating.

    Drops rows missing a date or either player id, then sorts by
    ``(tourney_date, round order, match_num)`` so matches apply in play order.
    The returned frame is guaranteed non-decreasing in ``tourney_date``, the
    property :func:`_elo_pass` asserts.

    Args:
        df: The raw loader frame (``data.loader.load_atp_data``), with
            ``winner_*``/``loser_*`` columns.

    Returns:
        A new, sorted frame with a clean ``surface`` string and stringified ids.

    Raises:
        KeyError: If a required column is absent (a vendor schema change).
    """
    missing = [column for column in _REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(
            f"match frame is missing column(s) required for Elo: {missing}"
        )

    work = df.copy()
    work["winner_id"] = work["winner_id"].map(_clean_id)
    work["loser_id"] = work["loser_id"].map(_clean_id)
    work = work.dropna(subset=["tourney_date", "winner_id", "loser_id"])

    round_column = work["round"] if "round" in work.columns else pd.Series(index=work.index, dtype=object)
    work = work.assign(
        _round_order=round_column.map(
            lambda value: ROUND_ORDER.get(value, _ROUND_ORDER_DEFAULT)
        ),
        _match_num=work["match_num"] if "match_num" in work.columns else 0,
    )
    work = work.sort_values(
        ["tourney_date", "_round_order", "_match_num"], kind="stable"
    ).reset_index(drop=True)
    return work


# --------------------------------------------------------------------------- #
# The rating pass.
# --------------------------------------------------------------------------- #
def _expected(rating_a: float, rating_b: float) -> float:
    """Logistic expected score for A vs B (``E_A`` in the module docstring)."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


class _Track:
    """Mutable per-track state during a rating pass: rating, count, last date."""

    def __init__(self, initial_rating: float, k_factor: Callable[[int], float]):
        self._initial = initial_rating
        self._k = k_factor
        self.rating: dict[str, float] = {}
        self.matches: dict[str, int] = {}
        self.last_played: dict[str, date] = {}

    def update(self, winner: str, loser: str, when: date) -> None:
        """Apply one result, moving each player by their own dynamic K."""
        r_w = self.rating.get(winner, self._initial)
        r_l = self.rating.get(loser, self._initial)
        e_w = _expected(r_w, r_l)
        k_w = self._k(self.matches.get(winner, 0))
        k_l = self._k(self.matches.get(loser, 0))
        # S_winner = 1, S_loser = 0; E_loser = 1 - E_winner.
        self.rating[winner] = r_w + k_w * (1.0 - e_w)
        self.rating[loser] = r_l + k_l * (0.0 - (1.0 - e_w))
        self.matches[winner] = self.matches.get(winner, 0) + 1
        self.matches[loser] = self.matches.get(loser, 0) + 1
        self.last_played[winner] = when
        self.last_played[loser] = when


def _elo_pass(
    matches: pd.DataFrame,
    *,
    initial_rating: float,
    k_factor: Callable[[int], float],
) -> tuple[dict[str, _Track], dict[str, str], dict[str, date]]:
    """Run the rating loop over an **already-sorted** frame.

    Asserts the frame is chronologically ordered and raises otherwise, this is
    the load-bearing correctness check the ordering requirement calls for, kept
    separate from :func:`compute_elo`'s convenience sort so it can be exercised
    directly with out-of-order input.

    Returns:
        ``(tracks, latest_name, overall_last_played)``, the per-track state, the
        latest display name seen per id, and each id's tour-wide last-match date.

    Raises:
        AssertionError: If ``matches`` is not non-decreasing in ``tourney_date``.
    """
    dates = matches["tourney_date"]
    assert dates.is_monotonic_increasing, (
        "Elo must process matches in chronological order, but the frame handed to "
        "_elo_pass is not sorted by tourney_date. Ratings are path-dependent, so "
        "an unordered pass would compute wrong numbers. Call compute_elo (which "
        "sorts) or sort by tourney_date first."
    )

    tracks: dict[str, _Track] = {
        name: _Track(initial_rating, k_factor) for name in ELO_TRACKS
    }
    latest_name: dict[str, str] = {}
    overall_last: dict[str, date] = {}

    for row in matches.itertuples(index=False):
        winner = row.winner_id
        loser = row.loser_id
        when = row.tourney_date.date() if hasattr(row.tourney_date, "date") else row.tourney_date
        surface = getattr(row, "surface", None)

        latest_name[winner] = str(row.winner_name)
        latest_name[loser] = str(row.loser_name)

        tracks[ELO_OVERALL].update(winner, loser, when)
        if surface in ELO_SURFACES:
            tracks[surface].update(winner, loser, when)

        overall_last[winner] = when
        overall_last[loser] = when

    return tracks, latest_name, overall_last


def compute_elo(
    df: pd.DataFrame,
    *,
    initial_rating: float = ELO_INITIAL_RATING,
    k_factor: Callable[[int], float] = dynamic_k,
    active_window_days: int = ELO_ACTIVE_WINDOW_DAYS,
    leaderboard_limit: int = ELO_LEADERBOARD_LIMIT,
    assume_sorted: bool = False,
) -> EloResult:
    """Compute overall + per-surface Elo leaderboards from a match frame.

    Args:
        df: Raw loader frame with ``winner_*``/``loser_*``/``surface`` columns.
        initial_rating: Rating for a player's first appearance (default 1500).
        k_factor: ``n_matches_so_far -> K``. Defaults to the dynamic curve; pass a
            constant (e.g. ``lambda n: 32.0``) to hand-verify the arithmetic.
        active_window_days: Recency window for the active flag.
        leaderboard_limit: Max players published per track, top-rated first.
        assume_sorted: If True, skip the convenience sort and trust ``df`` is
            already chronological, the assert in :func:`_elo_pass` still fires if
            it is not, so a false claim is rejected rather than silently miscomputed.

    Returns:
        An :class:`EloResult` with one :class:`Leaderboard` per track.

    Raises:
        ValueError: If no usable matches remain after cleaning.
        AssertionError: If ``assume_sorted`` is True but ``df`` is not sorted.
    """
    matches = df if assume_sorted else prepare_matches(df)
    if matches.empty:
        raise ValueError("no usable matches to compute Elo from (all rows dropped)")

    tracks, latest_name, overall_last = _elo_pass(
        matches, initial_rating=initial_rating, k_factor=k_factor
    )

    as_of = max(overall_last.values())
    active_cutoff = as_of - timedelta(days=active_window_days)

    leaderboards: dict[str, Leaderboard] = {}
    for track_name in ELO_TRACKS:
        track = tracks[track_name]
        ratings = [
            PlayerRating(
                player_id=player_id,
                player_name=latest_name.get(player_id, player_id),
                rating=round(rating, 2),
                matches=track.matches[player_id],
                last_played=track.last_played[player_id],
                # Active is a tour-wide property: use the player's OVERALL last
                # match, not their last match on this surface.
                is_active=overall_last[player_id] >= active_cutoff,
            )
            for player_id, rating in track.rating.items()
        ]
        # Highest rating first; player_id breaks ties for a stable order.
        ratings.sort(key=lambda entry: (-entry.rating, entry.player_id))
        leaderboards[track_name] = Leaderboard(
            track=track_name,
            ratings=tuple(ratings[:leaderboard_limit]),
        )

    return EloResult(
        leaderboards=leaderboards,
        as_of=as_of,
        active_cutoff=active_cutoff,
        n_matches=len(matches),
        data_through_year=int(as_of.year),
    )


def elo_cache_path(cache_dir: str | Path | None = None) -> Path:
    """Path the ratings cache lives at, the one shared writer/reader definition.

    Args:
        cache_dir: Directory holding the cache; defaults to ``config.CACHE_DIR``.

    Returns:
        ``<cache_dir>/elo_ratings.json`` as a ``pathlib.Path``.
    """
    base = Path(cache_dir) if cache_dir is not None else config.CACHE_DIR
    return base / ELO_CACHE_FILENAME
