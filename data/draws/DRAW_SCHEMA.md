# Draw JSON schema reference

The thing to reach for when **hand-authoring or uploading your own tournament
draw**. Every constraint below is enforced by the real validator in
`src/sim/draw.py` (`parse_draw` / `load_draw`); the allowed value sets are
owned by `src/config.py`, so this page names them but `config.py` is the source
of truth. A test (`tests/test_draw_schema_doc.py`) fails if this document and
`config` ever disagree.

Two paths consume a draw file, and they validate it **identically**, through the
same `parse_draw`:

- **Curated draw**: a file committed under `data/draws/`, loaded by
  `load_draw(path, skill_table)`. This is how the two example draws ship.
- **Upload**: the same JSON `POST`ed to the draw-upload endpoint, held only in
  memory. The upload path reads **one extra field** (`event_date`, below) that
  `parse_draw` itself ignores.

Validation is **fail-loudly-but-complete**: every problem in a file is collected
and reported together in a single `DrawValidationError`, not one-at-a-time, so a
128-draw with ten unresolved names is fixed in one pass, not ten.

## Top-level fields

A draw file is a single JSON object. These eight fields are **required**
(`sim.draw.REQUIRED_FIELDS`); anything else at the top level is ignored (see
[Optional & ignored fields](#optional--ignored-fields)).

| Field | Type | Required | Constraint (enforced by `parse_draw`) |
|---|---|:---:|---|
| `tournament_id` | string | Yes | Non-empty (after trimming). Stable machine id for the event, e.g. `"ausopen_2026_atp_full"`. |
| `name` | string | Yes | Non-empty. Human-readable event name. |
| `surface` | string | Yes | One of `config.VALID_SURFACES`: **`"Clay"`, `"Grass"`, `"Hard"`**. Exact case. |
| `best_of` | integer | Yes | One of `config.VALID_BEST_OF`: **`3` or `5`**. A JSON `true`/`false` is *not* accepted (bools are rejected even though Python treats them as ints). |
| `final_set_tiebreak` | string | Yes | One of `config.VALID_FINAL_SET_TIEBREAKS`: **`"7pt_at_6_6"`, `"10pt_at_6_6"`, `"advantage"`**. Derived from the match layer's `FINAL_SET_TB_TARGET`, so it can't drift from what `sim/match.py` actually plays. |
| `draw_size` | integer | Yes | One of `config.VALID_DRAW_SIZES`: **`8`, `16`, `32`, `64`, `128`** (a power of two, so the bracket halves cleanly each round). |
| `seeds` | object | Yes | Map of entrant **name → seed number**. Each value a **positive integer** (`≥ 1`). Every seeded name **must appear in `bracket`**. May be empty (`{}`). Only seeded players are listed. |
| `bracket` | array | Yes | List of **slot objects** (next section). Its length **must equal `draw_size`**, and its `position`s must be exactly the contiguous range `1..draw_size`: no gaps, no duplicates, none out of range. |

### `final_set_tiebreak` values, in plain terms

| Value | Deciding set is won by… | `FINAL_SET_TB_TARGET` |
|---|---|---|
| `"7pt_at_6_6"` | first to 7 points (win by 2) in a tiebreak at 6–6 | `7` |
| `"10pt_at_6_6"` | first to 10 points (win by 2) in a tiebreak at 6–6, current Slam standard | `10` |
| `"advantage"` | no tiebreak; keep playing until a player leads by two games | `None` |

## Slot fields (`bracket[i]`)

Each entry in `bracket` is a JSON object. You author **two** fields; the loader
**derives** two more onto the in-memory `DrawSlot`; those derived fields are
**not** written in the file.

### Fields you author

| Field | Type | Required | Constraint |
|---|---|:---:|---|
| `position` | integer | Yes | 1-based slot number. Slots `2k-1` and `2k` meet in round 1 (positions 1 v 2, 3 v 4, …). Across the bracket the set of positions must be exactly `1..draw_size`. |
| `player` | string | Yes | Non-empty (after trimming). A real player's **display name** (resolved to a skill-table `player_id`) **or** a placeholder token (see below). Unknown keys inside a slot are ignored. |

### Fields the loader derives

Read-only, on the `DrawSlot` dataclass, never written in JSON.

| Field | Type | Meaning |
|---|---|---|
| `is_placeholder` | bool | `True` when `player` names a slot rather than a person (see below). |
| `player_id` | string \| null | The resolved skill-table id. `null` for placeholders (and a real name that fails to resolve is a **validation error**, not a `null`). |

### Placeholder entrants

A `player` string that names an **unfilled slot** rather than a person is a
placeholder: it is not resolved against the skill table, and its `player_id`
stays `null` (it takes `SkillTable.default(surface)` when a skill profile is
needed). The recognised tokens are `config.DRAW_PLACEHOLDER_ENTRANTS`, compared
**case-insensitively and whitespace-trimmed**:

> `alternate`, `bye`, `ll`, `lucky loser`, `q`, `qualifier`, `tbd`

A `/`-joined entrant (`"Qualifier/Lucky Loser"`) counts as a placeholder **only
when every part is** a placeholder token.

> **A draw containing any placeholder cannot be simulated.**
> `simulate_bracket` refuses it, because a placeholder slot has no
> `player_id` and no classifier-visible history, so no reconciled match-win
> probability exists for it. A fully-simulable draw, like
> `ausopen_2026_atp_full.json`, has **zero** placeholders. This is deliberate,
> not a bug to work around.

### Name resolution

Non-placeholder `player` names are resolved to a `player_id` through the shared
fuzzy resolver over the skill table. A name that is **unknown or ambiguous**
fails resolution and is reported as a validation problem (**all** unresolved
names in the file are listed together). Skills are built from the vendored
seasons, so an entrant only resolves if that player appears in the data your
skill table was built from.

## Optional & ignored fields

`parse_draw` **ignores any key it doesn't recognise**, at the top level and
inside each slot. That is what lets a file carry provenance without breaking
validation: the shipped draws use `note` and `source` this way.

One ignored-by-`parse_draw` key is meaningful to the upload path only:

| Field | Type | Read by | Meaning |
|---|---|---|---|
| `event_date` | string `"YYYY-MM-DD"` | the upload endpoint (`config.UPLOAD_EVENT_DATE_FIELD`) | The event's date. A **future** date marks the simulation a genuine **forecast** (`is_forecast = true`); a past date (or no `event_date` at all) is treated as a historical retrospective. If present it must be a valid `YYYY-MM-DD` string, or the upload is rejected `422`. `parse_draw` does not read it, so it is optional for curated files and has no effect there. |

Other conventional-but-ignored keys the example files use: `note` (what the draw
is) and `source` (where it was reconstructed from).

## A complete example draw

For a real, full-size, placeholder-free draw that exercises every field above,
see [`ausopen_2026_atp_full.json`](https://github.com/hahalaa/ace/blob/main/data/draws/ausopen_2026_atp_full.json)
(or `example_usopen_2024_full.json`), both committed under `data/draws/`.
