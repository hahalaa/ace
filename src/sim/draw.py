"""Tournament draw schema, loader and validator (T2.1).

Defines the on-disk draw format (``ace-02-data-schema.md`` "Tournament draw JSON
schema") and turns one of those files into a validated :class:`Draw` whose
entrants are resolved to skill-table ``player_id``s. Draws are entered by hand,
one JSON file per event under ``config.DRAWS_DIR``.

This module is **structure only**: it says who is in the draw, where they sit,
and under what match format the event is played. It deliberately imports
*nothing* from ``sim/match.py``, ``sim/points.py`` or ``sim/reconcile.py`` —
simulating the bracket is T2.2's job. Its only project imports are ``config``
and ``features.serve`` (for name → id resolution and the default profile), and,
like the rest of ``sim/``, it never imports from ``cli/``.

**Validation is fail-loudly-but-complete.** Every problem found in a file is
collected and reported together in a single :class:`DrawValidationError`, rather
than raising on the first one — including *all* unresolved player names, so a
hand-entered 128-draw can be fixed in one pass instead of 128.

**Placeholder entrants.** Strings that name a slot rather than a person
(``"Qualifier"``, ``"Lucky Loser"``, ``"Bye"``, … — see
``config.DRAW_PLACEHOLDER_ENTRANTS``) are not resolved. Their
:attr:`DrawSlot.player_id` stays ``None`` and :meth:`Draw.skill_for` hands back
``SkillTable.default(surface)``, T1.1's surface-baseline profile.

**Unknown keys are ignored** — both at the top level and inside a bracket slot —
so a file can carry provenance/notes alongside the schema fields.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import config
from features.serve import PlayerSkill, SkillTable

# Top-level fields every draw file must carry.
REQUIRED_FIELDS = (
    "tournament_id",
    "name",
    "surface",
    "best_of",
    "final_set_tiebreak",
    "draw_size",
    "seeds",
    "bracket",
)

# A composite entrant like "Qualifier/Lucky Loser" is a placeholder iff every
# part is; this is what the entrant string is split on before comparison.
_ENTRANT_SEPARATOR = "/"


class DrawValidationError(ValueError):
    """Raised when a draw file is malformed, listing **all** problems found.

    Attributes:
        source: Where the draw came from (file path, or a label for in-memory
            data) — prefixed to the message so the error names the bad file.
        problems: Every validation problem found, in the order they were
            detected. Exposed separately so callers/tests can assert on
            individual rules without parsing the joined message.
    """

    def __init__(self, source: str, problems: list[str]) -> None:
        self.source = source
        self.problems = tuple(problems)
        detail = "\n".join(f"  - {problem}" for problem in self.problems)
        super().__init__(
            f"{source}: {len(self.problems)} draw validation problem(s):\n{detail}"
        )


@dataclass(frozen=True)
class DrawSlot:
    """One entrant position in the bracket.

    Attributes:
        position: 1-based slot number. Slots ``2k-1`` and ``2k`` meet in round 1.
        player: The entrant string exactly as written in the draw file (display
            name, or a placeholder token like ``"Qualifier"``).
        is_placeholder: True when ``player`` names a slot rather than a person.
        player_id: Resolved skill-table id. ``None`` for placeholders (which
            take the default profile instead).
    """

    position: int
    player: str
    is_placeholder: bool
    player_id: str | None = None


@dataclass(frozen=True)
class Draw:
    """A validated tournament draw with entrants resolved to ``player_id``s.

    Attributes:
        tournament_id: Stable machine id for the event (e.g. ``"usopen_2026_atp"``).
        name: Human-readable event name.
        surface: One of ``config.VALID_SURFACES``.
        best_of: 3 or 5.
        final_set_tiebreak: Deciding-set rule; a key of
            ``config.FINAL_SET_TB_TARGET`` (the same enum ``sim/match.py`` calls
            ``final_set_rule``).
        draw_size: Number of slots; one of ``config.VALID_DRAW_SIZES``.
        seeds: Entrant name → seed number, for the seeded players only.
        bracket: The slots, ordered by ``position`` (1..``draw_size``).
    """

    tournament_id: str
    name: str
    surface: str
    best_of: int
    final_set_tiebreak: str
    draw_size: int
    seeds: Mapping[str, int]
    bracket: tuple[DrawSlot, ...]

    def first_round_pairings(self) -> list[tuple[DrawSlot, DrawSlot]]:
        """Round-1 matchups: slots ``2k-1`` vs ``2k``, in bracket order."""
        return [
            (self.bracket[i], self.bracket[i + 1])
            for i in range(0, self.draw_size, 2)
        ]

    def seed_of(self, slot: DrawSlot) -> int | None:
        """The slot's seed number, or ``None`` if the entrant is unseeded."""
        return self.seeds.get(slot.player)

    def skill_for(self, slot: DrawSlot, skill_table: SkillTable) -> PlayerSkill:
        """The slot's serve/return profile on this draw's surface.

        Placeholder entrants (and any id the table doesn't know) fall back to
        ``SkillTable.default(surface)`` — T1.1's existing default mechanism.
        """
        return skill_table.get(slot.player_id, self.surface)


def is_placeholder(player: str) -> bool:
    """True when ``player`` names an unfilled slot rather than a real entrant.

    Case- and whitespace-insensitive against ``config.DRAW_PLACEHOLDER_ENTRANTS``.
    A ``"/"``-joined entrant (``"Qualifier/Lucky Loser"``) counts as a
    placeholder only when every part does.
    """
    parts = [part.strip().lower() for part in player.split(_ENTRANT_SEPARATOR)]
    parts = [part for part in parts if part]
    if not parts:
        return False
    return all(part in config.DRAW_PLACEHOLDER_ENTRANTS for part in parts)


def _is_int(value: Any) -> bool:
    """True for a genuine int — bools are ints in Python, and are not wanted."""
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_string(data: Mapping[str, Any], key: str, problems: list[str]) -> str:
    """Return ``data[key]`` if it is a non-empty string, else record a problem."""
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        problems.append(f"{key} must be a non-empty string, got {value!r}")
        return ""
    return value


def _validate_positions(
    positions: list[int], expected: int, problems: list[str]
) -> None:
    """Check the parsed positions are exactly the contiguous range 1..expected."""
    counts = Counter(positions)
    duplicated = sorted(p for p, n in counts.items() if n > 1)
    out_of_range = sorted({p for p in positions if not 1 <= p <= expected})
    missing = sorted(set(range(1, expected + 1)) - set(positions))
    parts = []
    if missing:
        parts.append(f"missing: {missing}")
    if duplicated:
        parts.append(f"duplicated: {duplicated}")
    if out_of_range:
        parts.append(f"out of range: {out_of_range}")
    if parts:
        problems.append(
            f"bracket positions must be exactly 1..{expected} "
            f"({'; '.join(parts)})"
        )


def _parse_bracket(
    raw_bracket: Any, draw_size: int | None, problems: list[str]
) -> list[tuple[int, str]]:
    """Validate the raw bracket list; return the usable ``(position, player)`` pairs.

    ``draw_size`` is the validated size, or ``None`` when it was itself invalid —
    in which case position contiguity is checked against the bracket's own
    length so the caller still gets a useful second error.
    """
    if not isinstance(raw_bracket, list):
        problems.append(
            f"bracket must be a list of slot objects, got {type(raw_bracket).__name__}"
        )
        return []

    if draw_size is not None and len(raw_bracket) != draw_size:
        problems.append(
            f"bracket has {len(raw_bracket)} entries but draw_size is {draw_size}"
        )

    entries: list[tuple[int, str]] = []
    positions: list[int] = []
    for index, entry in enumerate(raw_bracket):
        if not isinstance(entry, Mapping):
            problems.append(
                f"bracket[{index}] must be an object with 'position' and "
                f"'player', got {type(entry).__name__}"
            )
            continue
        position = entry.get("position")
        player = entry.get("player")
        position_ok = _is_int(position)
        if not position_ok:
            problems.append(
                f"bracket[{index}]: 'position' must be an integer, got {position!r}"
            )
        player_ok = isinstance(player, str) and bool(player.strip())
        if not player_ok:
            problems.append(
                f"bracket[{index}]: 'player' must be a non-empty string, "
                f"got {player!r}"
            )
        if position_ok:
            positions.append(position)
        if position_ok and player_ok:
            entries.append((position, player.strip()))

    _validate_positions(
        positions, draw_size if draw_size is not None else len(raw_bracket), problems
    )
    return entries


def _resolve_entrants(
    entries: list[tuple[int, str]], skill_table: SkillTable, problems: list[str]
) -> list[DrawSlot]:
    """Resolve every non-placeholder entrant, reporting all failures together."""
    slots: list[DrawSlot] = []
    unresolved: list[str] = []
    for position, player in entries:
        if is_placeholder(player):
            slots.append(DrawSlot(position, player, is_placeholder=True))
            continue
        # Reuse T1.1's adapter over the shared T0.6 resolver — never a new one.
        player_id = skill_table.resolve_name(player)
        if player_id is None and player not in unresolved:
            unresolved.append(player)
        slots.append(
            DrawSlot(position, player, is_placeholder=False, player_id=player_id)
        )
    if unresolved:
        names = ", ".join(repr(name) for name in unresolved)
        problems.append(
            f"{len(unresolved)} player name(s) did not resolve in the skill "
            f"table (unknown or ambiguous): {names}"
        )
    return slots


def _validate_seeds(
    raw_seeds: Any, entrants: set[str], problems: list[str]
) -> dict[str, int]:
    """Validate the seeds mapping against the entrants actually in the bracket."""
    if not isinstance(raw_seeds, Mapping):
        problems.append(
            f"seeds must be an object mapping player name -> seed number, "
            f"got {type(raw_seeds).__name__}"
        )
        return {}
    seeds: dict[str, int] = {}
    for name, seed in raw_seeds.items():
        if not _is_int(seed) or seed < 1:
            problems.append(f"seeds[{name!r}] must be a positive integer, got {seed!r}")
            continue
        seeds[str(name)] = seed
    unknown = [name for name in raw_seeds if name not in entrants]
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        problems.append(f"seeded name(s) not present in the bracket: {names}")
    return seeds


def parse_draw(
    data: Any, skill_table: SkillTable, source: str = "<draw>"
) -> Draw:
    """Validate already-parsed draw data and resolve its entrants.

    The in-memory half of :func:`load_draw` — same rules, no file I/O.

    Args:
        data: The decoded JSON object.
        skill_table: T1.1 skill table used to resolve entrant names to ids.
        source: Label used in error messages (a path, when called by
            :func:`load_draw`).

    Returns:
        A validated :class:`Draw` with ``bracket`` ordered by position.

    Raises:
        DrawValidationError: If anything is wrong; the message lists every
            problem found, not just the first.
    """
    if not isinstance(data, Mapping):
        raise DrawValidationError(
            source,
            [f"draw must be a JSON object, got {type(data).__name__}"],
        )

    problems: list[str] = []
    for field_name in REQUIRED_FIELDS:
        if field_name not in data:
            problems.append(f"missing required field {field_name!r}")

    tournament_id = _validate_string(data, "tournament_id", problems)
    name = _validate_string(data, "name", problems)

    # Every enum check below type-checks *before* testing membership: an
    # unhashable JSON value (a list/object where a scalar belongs) would
    # otherwise raise TypeError out of `in <frozenset>` and lose the collected
    # problem list. Same pattern as draw_size.
    surface = data.get("surface")
    if not isinstance(surface, str) or surface not in config.VALID_SURFACES:
        problems.append(
            f"surface must be one of {sorted(config.VALID_SURFACES)}, got {surface!r}"
        )

    best_of = data.get("best_of")
    if not _is_int(best_of) or best_of not in config.VALID_BEST_OF:
        problems.append(
            f"best_of must be one of {sorted(config.VALID_BEST_OF)}, got {best_of!r}"
        )

    final_set_tiebreak = data.get("final_set_tiebreak")
    if (
        not isinstance(final_set_tiebreak, str)
        or final_set_tiebreak not in config.VALID_FINAL_SET_TIEBREAKS
    ):
        problems.append(
            f"final_set_tiebreak must be one of "
            f"{sorted(config.VALID_FINAL_SET_TIEBREAKS)}, got {final_set_tiebreak!r}"
        )

    raw_draw_size = data.get("draw_size")
    draw_size_ok = _is_int(raw_draw_size) and raw_draw_size in config.VALID_DRAW_SIZES
    if not draw_size_ok:
        problems.append(
            f"draw_size must be one of {sorted(config.VALID_DRAW_SIZES)} "
            f"(a power of 2), got {raw_draw_size!r}"
        )
    draw_size = raw_draw_size if draw_size_ok else None

    entries = _parse_bracket(data.get("bracket"), draw_size, problems)
    slots = _resolve_entrants(entries, skill_table, problems)
    seeds = _validate_seeds(
        data.get("seeds", {}), {player for _, player in entries}, problems
    )

    if problems:
        raise DrawValidationError(source, problems)

    return Draw(
        tournament_id=tournament_id,
        name=name,
        surface=surface,
        best_of=best_of,
        final_set_tiebreak=final_set_tiebreak,
        draw_size=draw_size,
        seeds=seeds,
        bracket=tuple(sorted(slots, key=lambda slot: slot.position)),
    )


def load_draw(path: str | Path, skill_table: SkillTable) -> Draw:
    """Load, validate and resolve a draw JSON file.

    Args:
        path: Path to the draw file (conventionally under ``config.DRAWS_DIR``).
        skill_table: T1.1 skill table used to resolve entrant names to ids.

    Returns:
        A validated :class:`Draw`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        DrawValidationError: If the file is not valid JSON, or fails any
            validation rule — every problem is reported together.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DrawValidationError(str(path), [f"file is not valid JSON: {exc}"]) from exc
    return parse_draw(data, skill_table, source=str(path))
