# Tournament draws

Draw files are validated and loaded by `sim/draw.py` (`load_draw`, T2.1) and
simulated by `sim/tournament.py` (`simulate_bracket`, T2.2). Each file's own
`note` field is authoritative about what it is; this is the index.

| File | Size | Placeholders | Simulable |
|---|---|---|---|
| `example_usopen_2026.json` | 8 | yes (`Qualifier`, `Qualifier/Lucky Loser`) | **no** |
| `example_usopen_2024_full.json` | 128 | none | yes |

**Which one to use.** `example_usopen_2026.json` is the schema illustration —
a toy 8-slot bracket with invented structure, kept because it is the only
example that exercises placeholder handling. It **cannot be simulated**:
`simulate_bracket` refuses any draw containing placeholders, because a
`"Qualifier"` slot has no `player_id`, no skill profile and no
classifier-visible history, so no reconciled match-win probability exists for
it. That refusal is deliberate and is not a bug to work around.

`example_usopen_2024_full.json` is the one to simulate. It is the **real**
128-player men's singles bracket of the 2024 US Open, reconstructed from the
vendored round-by-round results in `data/raw/` and verified by replaying all
127 matches through the reconstructed pairings. Every entrant is a genuine
named player — qualifiers and wildcards included — so no slot needs a
placeholder. It exists so T2.3 has a full-size, fully-resolvable draw to
measure Monte Carlo performance against.

**Two caveats on the real draw**, both spelled out in its `note`:

1. *It is not a forecast.* The default snapshot skill table is built from every
   vendored season, including this event and everything after it, so simulating
   the file re-plays a known draw with hindsight-informed skills. Use
   `build_skill_table(df, as_of=...)` for a genuine backtest.
2. *Positions are labels, not the official draw sheet.* The bracket structure —
   who can meet whom, and in which round — is exact. The position numbering is a
   canonical labelling that puts each seed at the top of its block (seed 1 → 1,
   seed 2 → 65, seed 3 → 33, seed 4 → 97), whereas a real draw sheet places
   seed 2 at position 128.

Resolving this file's entrants needs vendored data from 2024 onward; the two
players who last appear in 2024 do not resolve against a 2025+ skill table.
