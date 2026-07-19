"""Per-player serve/return skill table (surface-adjusted, recency-weighted).

Builds the id-keyed skill table the point-based simulator (Phase 1+) reads:
serve-points-won (``spw``) and return-points-won (``rpw``) rates, **per surface**,
**recency- and volume-weighted**, computed leakage-safely. This is the analogue
of ``engineering.py``'s incremental ``surface_history``/``h2h_history`` builders,
but keyed on **``player_id``** (the id side of the name/id seam — see
``ace-04-current-state.md §7``), not on player name.

Feeds ``ace-03-tennis-math.md §1``'s point-win model (T1.2): that formula needs
``spw_server``, ``rpw_returner`` and the surface baseline ``μ`` — all produced
here. This module only *computes and stores* those rates; it does not itself turn
them into a point-win probability (that is T1.2).

Leakage-safety — two modes over one entry point:
  * **snapshot** (``as_of=None``, the primary deliverable): aggregate *all* rows
    in ``df``, with recency measured back from the latest match date. This is the
    "latest known skills" table used to simulate a *future* tournament — every
    row it uses predates the thing being simulated, so it carries no look-ahead.
  * **as-of** (``as_of=<date>``, for backtesting): use only matches strictly
    **before** ``as_of``, with recency measured back from ``as_of``. Lets a past
    matchup be simulated using only what was known at the time.

Design decisions worth stating:
  * The surface baseline ``μ`` used for shrinkage and the unknown-player default
    is the fixed, data-derived tour average in ``config.SURFACE_MU`` — a global
    constant, not a player-specific quantity, so reading it introduces no
    per-matchup leakage. ``compute_surface_mu`` is how those config numbers were
    derived; re-run it if the data range changes.
  * Only ``Hard``/``Clay``/``Grass`` rows are aggregated. ``Carpet`` and
    ``Unknown``/NaN surfaces are excluded outright — they never leak into a real
    surface's totals (``ace-02-data-schema.md`` "Surface handling").
  * Rows without a usable serve line (``has_serve_stats`` False) and
    retirements/walkovers (``RET``/``W/O``/``def.`` in ``score``) are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from common.names import NameIndex, resolve_name

# Score-string markers for matches with unreliable/partial serve stats. Same set
# parse_match_score screens on (ace-04-current-state.md §2 / ace-02 data-quality).
_INCOMPLETE_MARKERS = ("RET", "W/O", "def.")


@dataclass(frozen=True)
class PlayerSkill:
    """A player's serve/return profile on one surface.

    Attributes:
        spw: Serve-points-won rate, recency+volume weighted then shrunk toward
            the surface baseline ``μ`` (empirical Bayes). In ``[0, 1]``.
        rpw: Return-points-won rate, same weighting/shrinkage. In ``[0, 1]``.
        n_serve_pts: Total (raw, unweighted) serve points in the sample — the
            sample size the shrinkage uses. ``0`` for a default profile.
        n_return_pts: Total (raw, unweighted) return points in the sample.
    """

    spw: float
    rpw: float
    n_serve_pts: float
    n_return_pts: float


class SkillTable:
    """Id-keyed serve/return skills with an unknown-player default.

    Construct via :func:`build_skill_table`. Lookups key on ``player_id``; names
    resolve to ids through the shared resolver (``common/names.py``, T0.6).
    """

    def __init__(
        self,
        skills: dict[tuple[str, str], PlayerSkill],
        name_to_id: dict[str, str],
        mu: dict[str, float],
        cutoff: pd.Timestamp | None,
    ) -> None:
        self._skills = skills
        self._name_index = NameIndex.from_mapping(name_to_id)
        self._mu = mu
        # The latest match date included in this snapshot; documents the "as of"
        # boundary of the skills (None only for an empty table).
        self.cutoff = cutoff

    def get(self, player_id: str | None, surface: str) -> PlayerSkill:
        """Return ``player_id``'s skill on ``surface``.

        Falls back to :meth:`default` for an unknown/null id or a player with no
        aggregated data on that surface.
        """
        self._check_surface(surface)
        if player_id is None:
            return self.default(surface)
        skill = self._skills.get((player_id, surface))
        return skill if skill is not None else self.default(surface)

    def default(self, surface: str) -> PlayerSkill:
        """The unknown-entrant profile on ``surface``: exactly the baseline ``μ``.

        ``spw = μ`` and ``rpw = 1 − μ`` (an average returner), with zero sample.
        """
        self._check_surface(surface)
        mu = self._mu[surface]
        return PlayerSkill(spw=mu, rpw=1.0 - mu, n_serve_pts=0.0, n_return_pts=0.0)

    def resolve_name(self, name: str) -> str | None:
        """Resolve a display name to a ``player_id`` (thin adapter over T0.6).

        Returns ``None`` when nothing matches or the query is ambiguous (the
        caller cannot disambiguate through this adapter).
        """
        match = resolve_name(name, self._name_index)
        if match is None:
            return None
        return match.player_id

    def _check_surface(self, surface: str) -> None:
        if surface not in self._mu:
            raise ValueError(
                f"Unknown surface {surface!r}; skill table covers {sorted(self._mu)}"
            )


def _is_incomplete(score: object) -> bool:
    """True if the score marks a retirement/walkover (partial serve stats)."""
    if not isinstance(score, str):
        return False
    return any(marker in score for marker in _INCOMPLETE_MARKERS)


def _eligible_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for serve aggregation: complete serve line, real surface,
    completed match. Leaves the caller to apply any ``as_of`` date filter."""
    has_stats = df["has_serve_stats"].astype(bool)
    surface_ok = df["surface"].isin(config.VALID_SURFACES)
    complete = ~df["score"].map(_is_incomplete).astype(bool)
    return df[has_stats & surface_ok & complete]


def compute_surface_mu(df: pd.DataFrame) -> dict[str, float]:
    """Data-derived tour-average serve-points-won (``μ``) per surface.

    Pools both players' serve points over all eligible rows on each surface:
    ``μ = Σ(1stWon + 2ndWon) / Σ svpt``. This is how the canonical
    ``config.SURFACE_MU`` numbers (and the table in ``ace-02-data-schema.md``)
    were derived — re-run it if the vendored data range changes.
    """
    rows = _eligible_rows(df)
    mu: dict[str, float] = {}
    for surface in sorted(config.VALID_SURFACES):
        sub = rows[rows["surface"] == surface]
        won = (
            sub["p1_1stWon"] + sub["p1_2ndWon"] + sub["p2_1stWon"] + sub["p2_2ndWon"]
        ).sum()
        played = (sub["p1_svpt"] + sub["p2_svpt"]).sum()
        mu[surface] = float(won / played) if played > 0 else float("nan")
    return mu


def _recency_weights(dates: pd.Series, reference: pd.Timestamp) -> np.ndarray:
    """Exponential-decay weights: ``0.5 ** (age_days / half_life)``.

    A match on ``reference`` weighs 1.0; one a half-life older weighs 0.5. Recent
    matches dominate the rate (``config.SERVE_RECENCY_HALFLIFE_DAYS``).
    """
    age_days = (reference - dates).dt.total_seconds().to_numpy() / 86400.0
    half_life = config.SERVE_RECENCY_HALFLIFE_DAYS
    return np.power(0.5, age_days / half_life)


def _long_perspective(
    rows: pd.DataFrame, weights: np.ndarray, subject: str, opponent: str
) -> pd.DataFrame:
    """One row per match from ``subject``'s perspective (serve + return).

    ``subject``/``opponent`` are ``"p1"``/``"p2"``. Return points won against the
    opponent = ``opp_svpt − (opp_1stWon + opp_2ndWon)`` over ``opp_svpt`` played
    (``ace-02-data-schema.md`` "Deriving the quantities the simulator needs").
    """
    s_won = rows[f"{subject}_1stWon"] + rows[f"{subject}_2ndWon"]
    s_pts = rows[f"{subject}_svpt"]
    opp_won = rows[f"{opponent}_1stWon"] + rows[f"{opponent}_2ndWon"]
    opp_pts = rows[f"{opponent}_svpt"]
    r_won = opp_pts - opp_won
    r_pts = opp_pts
    return pd.DataFrame(
        {
            "player_id": rows[f"{subject}_id"],
            "surface": rows["surface"],
            # Recency-weighted numerators/denominators for the rate estimate.
            "w_s_won": weights * s_won.to_numpy(),
            "w_s_pts": weights * s_pts.to_numpy(),
            "w_r_won": weights * r_won.to_numpy(),
            "w_r_pts": weights * r_pts.to_numpy(),
            # Raw (unweighted) point totals — the sample size shrinkage keys on.
            "n_serve_pts": s_pts.to_numpy(),
            "n_return_pts": r_pts.to_numpy(),
        }
    )


def _build_name_to_id(df: pd.DataFrame) -> dict[str, str]:
    """Map each display name to its most-frequent non-null ``player_id``.

    Built from every row (both p1/p2 sides) so resolution stays broad even for
    players with no usable serve line — they resolve to an id and then fall back
    to the default profile. On the rare name collision, the id that appears most
    often wins (deterministic tie-break: first in that order)."""
    frames = []
    for side in ("p1", "p2"):
        pair = df[[f"{side}_name", f"{side}_id"]].rename(
            columns={f"{side}_name": "name", f"{side}_id": "player_id"}
        )
        frames.append(pair)
    pairs = pd.concat(frames, ignore_index=True).dropna()
    if pairs.empty:
        return {}
    pairs["player_id"] = pairs["player_id"].astype(str)
    pairs["name"] = pairs["name"].astype(str)
    counts = pairs.groupby(["name", "player_id"]).size().reset_index(name="n")
    counts = counts.sort_values(["name", "n"], ascending=[True, False])
    top = counts.drop_duplicates("name", keep="first")
    return dict(zip(top["name"], top["player_id"]))


def build_skill_table(
    df: pd.DataFrame, as_of: pd.Timestamp | str | None = None
) -> SkillTable:
    """Build the serve/return skill table from a preprocessed p1/p2 frame.

    Consumes the output of ``preprocess.preprocess_data`` (needs ``p1_id``/
    ``p2_id``, ``p1_name``/``p2_name``, ``surface``, ``tourney_date`` (datetime),
    ``score``, ``has_serve_stats`` and the per-player serve columns carried by
    T0.4). Aggregates per ``(player_id, surface)`` with recency + volume weighting
    and empirical-Bayes shrinkage toward ``config.SURFACE_MU``.

    Args:
        df: Preprocessed match frame (see above).
        as_of: ``None`` → snapshot over all rows, recency measured from the
            latest match date (primary deliverable). A date → use only matches
            strictly before it, recency measured from it (backtest/as-of mode).

    Returns:
        A :class:`SkillTable`. ``cutoff`` records the latest included match date.
    """
    df = df.copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])

    if as_of is not None:
        as_of = pd.Timestamp(as_of)
        df = df[df["tourney_date"] < as_of]

    name_to_id = _build_name_to_id(df)

    rows = _eligible_rows(df).copy()
    rows = rows[rows["p1_id"].notna() & rows["p2_id"].notna()]

    mu = dict(config.SURFACE_MU)

    if rows.empty:
        return SkillTable({}, name_to_id, mu, cutoff=None)

    rows["p1_id"] = rows["p1_id"].astype(str)
    rows["p2_id"] = rows["p2_id"].astype(str)

    reference = as_of if as_of is not None else rows["tourney_date"].max()
    weights = _recency_weights(rows["tourney_date"], reference)

    long = pd.concat(
        [
            _long_perspective(rows, weights, "p1", "p2"),
            _long_perspective(rows, weights, "p2", "p1"),
        ],
        ignore_index=True,
    )

    agg = long.groupby(["player_id", "surface"], sort=False).sum()

    # Recency+volume-weighted raw rates, then empirical-Bayes shrinkage toward μ:
    #   rate = (n·rate_raw + k·μ) / (n + k)   with k = config.SERVE_SHRINKAGE_K
    # (k in serve/return-point units). Low-sample players are pulled to baseline.
    k = config.SERVE_SHRINKAGE_K
    skills: dict[tuple[str, str], PlayerSkill] = {}
    for (player_id, surface), r in agg.iterrows():
        base = mu[surface]
        spw_raw = r["w_s_won"] / r["w_s_pts"]
        rpw_raw = r["w_r_won"] / r["w_r_pts"]
        n_s = r["n_serve_pts"]
        n_r = r["n_return_pts"]
        spw = (n_s * spw_raw + k * base) / (n_s + k)
        rpw = (n_r * rpw_raw + k * (1.0 - base)) / (n_r + k)
        skills[(player_id, surface)] = PlayerSkill(
            spw=float(spw),
            rpw=float(rpw),
            n_serve_pts=float(n_s),
            n_return_pts=float(n_r),
        )

    cutoff = rows["tourney_date"].max()
    return SkillTable(skills, name_to_id, mu, cutoff=cutoff)
