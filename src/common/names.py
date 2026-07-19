"""UI-free player-name resolver shared by the CLI, simulator, and API (T0.6).

The matching logic used to live in ``cli/interactive.py`` as
``resolve_player_name``. It returned a *name* and *printed* on ambiguity, which
would have forced ``sim/``, ``sim/draw.py``, and ``api/`` to import from
``cli/`` — the layering inversion ``ace-01-architecture.md`` (principle #2)
forbids. This module holds the pure matching core: it **never prints**, has no
``cli``/``sim``/``api`` imports, and returns ambiguity as data (candidates on a
``NameMatch``) so each caller decides how to surface it.

The four matching strategies and their order are preserved exactly from the
original CLI resolver:

  1. exact (case-insensitive)
  2. initials — ``"F. Lastname"`` / ``"F Lastname"``
  3. substring / prefix (input contained in a known name)
  4. ``difflib`` fuzzy fallback (cutoff ``config.FUZZY_MATCH_CUTOFF``)

The resolver is **id-agnostic**: a ``NameIndex`` wraps either a plain list of
names (the CLI baseline) or a name→id mapping (the T1.1 skill table / T2.1 draw
loader). ``player_id`` on the result is ``None`` when the index carries no ids.
See the contract in ``ace-04-current-state.md §3``.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping

import config


class MatchStrategy(Enum):
    """Which of the four strategies produced a match.

    Callers (e.g. the CLI) use this to render strategy-specific messages
    without the resolver having to know about any UI.
    """

    EXACT = "exact"
    INITIALS = "initials"
    SUBSTRING = "substring"
    FUZZY = "fuzzy"


@dataclass(frozen=True)
class NameMatch:
    """Result of resolving a query against a :class:`NameIndex`.

    Attributes:
        player_id: Canonical id when the index carries one for ``name``;
            ``None`` for a name-only index (pre-T1.1) or when ambiguous.
        name: The resolved canonical display name; empty string when ambiguous.
        candidates: Populated (``len > 1``) only when the query was ambiguous;
            empty on a unique match.
        strategy: The strategy that produced this match.
    """

    name: str
    player_id: str | None = None
    strategy: MatchStrategy = MatchStrategy.EXACT
    candidates: list[str] = field(default_factory=list)

    @property
    def is_ambiguous(self) -> bool:
        return len(self.candidates) > 1


@dataclass(frozen=True)
class NameIndex:
    """A resolvable set of player names, optionally keyed to ids.

    Build with :meth:`from_names` (CLI baseline — ids resolve to ``None``) or
    :meth:`from_mapping` (name→id, for the skill table / draw loader). The
    original name ordering is preserved so matching iterates in the same order
    the old list-based resolver did.
    """

    names: list[str]
    ids: dict[str, str | None]

    @classmethod
    def from_names(cls, names: Iterable[str]) -> "NameIndex":
        name_list = list(names)
        return cls(names=name_list, ids={n: None for n in name_list})

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str | None]) -> "NameIndex":
        return cls(names=list(mapping.keys()), ids=dict(mapping))

    def id_for(self, name: str) -> str | None:
        return self.ids.get(name)


def resolve_name(query: str, index: NameIndex) -> NameMatch | None:
    """Resolve ``query`` to a canonical name (and id) against ``index``.

    Pure: never prints. Tries the four strategies in order (exact → initials →
    substring → fuzzy). A unique match returns a :class:`NameMatch` with empty
    ``candidates``; an ambiguous match returns one with ``candidates`` set and
    an empty ``name`` (the caller decides how to disambiguate). No match at all
    returns ``None``.

    Args:
        query: The raw user/query string.
        index: The names (and optional ids) to resolve against.

    Returns:
        A :class:`NameMatch`, or ``None`` when nothing matches.
    """
    if not query:
        return None

    all_names = index.names
    norm_input = query.lower().strip()

    # 1. Exact match (case-insensitive)
    for name in all_names:
        if name.lower() == norm_input:
            return _matched(index, name, MatchStrategy.EXACT)

    # 2. Initials check (e.g. "C Alcaraz" -> "Carlos Alcaraz")
    #    Gather ALL initial matches to detect ambiguity.
    initial_matches: list[str] = []
    parts = norm_input.split()
    if len(parts) >= 2 and len(parts[0]) == 1:
        first_initial = parts[0]
        lastname = " ".join(parts[1:])
        for name in all_names:
            name_parts = name.lower().split()
            if len(name_parts) >= 2:
                if name_parts[0].startswith(first_initial):
                    if " ".join(name_parts[1:]) == lastname:
                        initial_matches.append(name)

    if len(initial_matches) == 1:
        return _matched(index, initial_matches[0], MatchStrategy.INITIALS)
    elif len(initial_matches) > 1:
        return _ambiguous(initial_matches, MatchStrategy.INITIALS)

    # 3. Substring/Prefix match — every known name containing the input.
    substring_matches = [name for name in all_names if norm_input in name.lower()]

    if len(substring_matches) == 1:
        return _matched(index, substring_matches[0], MatchStrategy.SUBSTRING)
    elif len(substring_matches) > 1:
        return _ambiguous(substring_matches, MatchStrategy.SUBSTRING)

    # 4. Fuzzy match using difflib (uses the original, un-normalised query, as
    #    the CLI resolver did). cutoff lives in config.FUZZY_MATCH_CUTOFF.
    matches = difflib.get_close_matches(
        query, all_names, n=3, cutoff=config.FUZZY_MATCH_CUTOFF
    )

    if matches:
        if len(matches) > 1:
            return _ambiguous(matches, MatchStrategy.FUZZY)
        return _matched(index, matches[0], MatchStrategy.FUZZY)

    return None


def _matched(index: NameIndex, name: str, strategy: MatchStrategy) -> NameMatch:
    return NameMatch(name=name, player_id=index.id_for(name), strategy=strategy)


def _ambiguous(candidates: list[str], strategy: MatchStrategy) -> NameMatch:
    return NameMatch(name="", strategy=strategy, candidates=list(candidates))
