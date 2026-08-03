"""Tournament draw registry — what ``data/draws/`` actually contains (T3.2).

Scans a draws directory once and turns it into an addressable catalogue: which
tournament ids exist, which files back them, and — for the ones that do not
load — exactly why. ``/tournaments`` renders the catalogue and
``/tournaments/{id}/bracket`` looks an entry up in it.

**It reuses T2.1 wholesale.** Every file goes through
:func:`sim.draw.load_draw`; this module does not open, decode or validate a
draw itself, so there is exactly one draw parser in the codebase and the
problems reported over HTTP are the validator's own accumulated list.

**A bad file never takes the listing down.** ``load_draw``'s two failure modes
(unreadable file, :class:`~sim.draw.DrawValidationError`) are caught per file
and recorded *on the entry* rather than propagated, so one mistyped draw
degrades to a single error row instead of a 500 for every other event. The
listing endpoint reports valid and invalid entries in separate lists, so a
client that only wants usable draws never has to filter, and an operator who
mistyped a file can still see it.

**Addressing.** A valid draw is keyed by its own ``tournament_id`` (the ticket's
lookup key). An *invalid* one has no trustworthy id — the field may be missing,
or be the very thing that failed validation — so it is keyed by its **filename
stem** instead. Whatever ``/tournaments`` reports as an entry's
``tournament_id`` is always the id ``/tournaments/{id}/bracket`` accepts, valid
or not; that is what makes "invalid draw file → informative error" reachable
over HTTP at all rather than collapsing to a 404.

Deliberately free of FastAPI imports: this is a filesystem catalogue, not a
transport concern, and keeping it framework-free means an offline script (T3.3's
precompute) can resolve an id → draw without importing the web app. T3.3 takes
that up: ``scripts/precompute_sim.py`` builds a registry to turn its ``--draw
<id>`` into a :class:`~sim.draw.Draw`, and :func:`cache_path_for` — the *other*
id → file mapping in the system — lives here too, so the script that writes a
cache file and the endpoint that reads one cannot disagree about where it is.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import config
from features.serve import SkillTable
from sim.draw import Draw, DrawValidationError, load_draw

logger = logging.getLogger(__name__)

# Draw files are JSON, one event per file (ace-02-data-schema.md).
DRAW_GLOB = "*.json"


@dataclass(frozen=True)
class TournamentEntry:
    """One draw file in the registry — loaded, or loaded-and-failed.

    Attributes:
        tournament_id: The id this entry is addressed by. ``Draw.tournament_id``
            when the file loaded; the filename stem when it did not (see the
            module docstring on addressing).
        source: The file's **basename**, not its path — enough to name the file
            an operator must fix, without publishing server paths over HTTP.
        draw: The validated :class:`~sim.draw.Draw`, or ``None`` if the file
            failed to load.
        problems: Every problem found, in the validator's own order. Empty
            **iff** ``draw`` is not ``None``.
    """

    tournament_id: str
    source: str
    draw: Draw | None
    problems: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        """True when the file loaded and ``draw`` is populated."""
        return self.draw is not None

    @property
    def placeholder_slots(self) -> tuple:
        """Slots naming an unfilled position (``Qualifier``, ``Bye``, …).

        Empty for an invalid entry: a file that did not load has no slots to
        judge, and its problem list already says so.
        """
        if self.draw is None:
            return ()
        return tuple(slot for slot in self.draw.bracket if slot.is_placeholder)

    @property
    def is_simulatable(self) -> bool:
        """True when this draw can actually be simulated (T3.3).

        **Loadable is not the same as simulatable**, and both consumers of this
        module need the distinction: ``/tournaments/{id}/simulate`` answers 409
        for a draw that fails it, and ``scripts/precompute_sim.py`` must not
        offer such a draw as an available target. Expressed once, here, rather
        than as the same comprehension in three places.

        The rule is T2.2's: ``simulate_bracket`` refuses a bracket containing
        placeholder entrants, because a placeholder has no ``player_id`` and no
        classifier history, so no match-win probability exists for it. This
        property reads T2.1's ``is_placeholder`` flag to reach the same verdict
        without importing the simulator (``tests/test_api_simulate.py`` pins the
        two together).
        """
        return self.is_valid and not self.placeholder_slots


@dataclass(frozen=True)
class TournamentRegistry:
    """The catalogue of a draws directory, built once by :func:`build_registry`.

    Attributes:
        draws_dir: The directory scanned (kept for diagnostics/logging).
        entries: Every ``*.json`` found, in sorted-filename order, valid and
            invalid alike. Ids are unique across entries by construction.
    """

    draws_dir: Path
    entries: tuple[TournamentEntry, ...] = ()

    @property
    def valid(self) -> tuple[TournamentEntry, ...]:
        """Entries whose file loaded, in filename order."""
        return tuple(entry for entry in self.entries if entry.is_valid)

    @property
    def invalid(self) -> tuple[TournamentEntry, ...]:
        """Entries whose file failed to load, in filename order."""
        return tuple(entry for entry in self.entries if not entry.is_valid)

    def get(self, tournament_id: str) -> TournamentEntry | None:
        """The entry addressed by ``tournament_id``, or ``None`` if unknown.

        A linear scan on purpose: a draws directory holds a handful of
        hand-entered files, so an index would be more machinery than lookup.
        :func:`build_registry` guarantees ids are unique, so the first match is
        the only match.
        """
        for entry in self.entries:
            if entry.tournament_id == tournament_id:
                return entry
        return None

    def ids(self) -> tuple[str, ...]:
        """Every addressable id, in filename order — for "did you mean" messages."""
        return tuple(entry.tournament_id for entry in self.entries)


def cache_path_for(
    tournament_id: str, cache_dir: Path | str = config.CACHE_DIR
) -> Path | None:
    """Where ``tournament_id``'s precomputed simulation lives (T3.3).

    The single definition of that mapping: ``scripts/precompute_sim.py`` calls it
    to decide where to write, and ``GET /tournaments/{id}/simulate`` calls it to
    decide where to read, so the two cannot drift.

    **It also refuses ids that are not plain filenames**, which matters because
    the read side takes its id straight from a URL path. Registered ids come from
    draw files' own ``tournament_id`` fields, and a hand-entered file could
    perfectly well carry ``"../../etc/passwd"``; the endpoint's registry lookup
    would then hand that id here. Rejecting anything containing a path separator
    (and ``.``/``..``), then asserting the assembled path's parent really is
    ``cache_dir``, keeps this from ever addressing a file outside the cache.

    Args:
        tournament_id: The registry id to look up.
        cache_dir: Directory of cache files; defaults to ``config.CACHE_DIR``.

    Returns:
        The path the cache file would occupy — **whether or not it exists** (the
        caller decides what a missing file means) — or ``None`` if the id cannot
        safely be a filename.
    """
    if not tournament_id or tournament_id in {".", ".."}:
        return None
    if os.sep in tournament_id or (os.altsep and os.altsep in tournament_id):
        return None
    if "/" in tournament_id or "\\" in tournament_id or "\x00" in tournament_id:
        return None

    cache_dir = Path(cache_dir)
    candidate = cache_dir / f"{tournament_id}{config.CACHE_SUFFIX}"
    # Belt and braces: even with the checks above, the assembled path must sit
    # directly in cache_dir. No filesystem access — an unwritten cache directory
    # is normal, and resolve(strict=False) tolerates it.
    if candidate.parent.resolve() != cache_dir.resolve():
        return None
    return candidate


def _load_one(
    path: Path, skill_table: SkillTable
) -> tuple[Draw | None, tuple[str, ...]]:
    """Load one draw file, converting both failure modes into a problem list.

    ``load_draw`` already folds bad JSON into a :class:`DrawValidationError`, so
    the only other thing that can go wrong is reading the bytes.
    """
    try:
        return load_draw(path, skill_table), ()
    except DrawValidationError as exc:
        return None, exc.problems
    except (OSError, UnicodeDecodeError) as exc:
        return None, (f"file could not be read: {exc}",)


def _free_id(base: str, taken: set[str]) -> str:
    """``base``, or the first ``base#n`` variant not already claimed."""
    if base not in taken:
        return base
    suffix = 2
    while f"{base}#{suffix}" in taken:
        suffix += 1
    return f"{base}#{suffix}"


def build_registry(
    skill_table: SkillTable, draws_dir: Path | str = config.DRAWS_DIR
) -> TournamentRegistry:
    """Scan ``draws_dir`` and load every draw file it contains.

    Called once at app startup (see ``api.main.create_app``), never per request.

    Args:
        skill_table: T1.1 skill table, used by ``load_draw`` to resolve entrant
            names to ``player_id``s.
        draws_dir: Directory of ``*.json`` draw files. Defaults to
            ``config.DRAWS_DIR``; tests point it at a fixture directory.

    Returns:
        The populated :class:`TournamentRegistry`. A missing directory is not an
        error — it yields an empty registry and a logged warning, so the app
        still starts and ``/tournaments`` answers an honest empty list.
    """
    draws_dir = Path(draws_dir)
    if not draws_dir.is_dir():
        logger.warning(
            "draws directory %s does not exist; no tournaments registered", draws_dir
        )
        return TournamentRegistry(draws_dir=draws_dir)

    entries: list[TournamentEntry] = []
    taken: set[str] = set()
    for path in sorted(draws_dir.glob(DRAW_GLOB)):
        draw, problems = _load_one(path, skill_table)
        base = draw.tournament_id if draw is not None else path.stem
        if base in taken:
            # Two files claiming one id is a configuration error, not something
            # to resolve silently by letting the last writer win: downgrade this
            # one to an error entry (addressable under its own filename) so the
            # collision is visible instead of deciding which file to shadow.
            first = next(entry.source for entry in entries if entry.tournament_id == base)
            problems = (
                f"duplicate tournament id {base!r}: already registered from {first}",
                *problems,
            )
            draw = None
            base = path.stem
        tournament_id = _free_id(base, taken)
        taken.add(tournament_id)
        entries.append(
            TournamentEntry(
                tournament_id=tournament_id,
                source=path.name,
                draw=draw,
                problems=problems,
            )
        )

    registry = TournamentRegistry(draws_dir=draws_dir, entries=tuple(entries))
    logger.info(
        "Tournament registry: %d draw(s) in %s (%d valid, %d invalid)",
        len(registry.entries),
        draws_dir,
        len(registry.valid),
        len(registry.invalid),
    )
    return registry
