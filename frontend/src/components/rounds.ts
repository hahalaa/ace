/**
 * Bracket structure, derived from a `/bracket` response, pure, no React.
 *
 * `BracketResponse` carries **entrants, not matches**: `draw_size` slots in
 * position order and nothing about who meets whom after round one. Everything a
 * column layout needs is a function of those two facts, and this module is the
 * one place that function lives:
 *
 *   * **Round labels mirror `sim.tournament.round_labels`**, `log2(draw_size)`
 *     rounds, halving the field each time, with `8/4/2` named `QF/SF/F`. They
 *     are *derived*, never assumed: an 8 draw's first round **is** the QF, so a
 *     layout hardcoded to `R128 … F` is wrong for every draw size but one.
 *   * **Only round one has known participants.** Later rounds are real slots in
 *     the layout with `null` entrants, because the shape of the draw is known
 *     long before its results are. Later overlays fill them in; nothing here
 *     invents a name for them.
 *
 * Kept out of `Bracket.tsx` so the pure part is importable and testable without
 * rendering, and so the component file exports only components.
 */

import type { BracketSlot } from '../api/types';

/** Rounds whose label is a name rather than a field size. */
const NAMED_ROUNDS: Record<number, string> = { 8: 'QF', 4: 'SF', 2: 'F' };

/** One match's two sides. `null` = a participant this round has not produced yet. */
export interface BracketMatchModel {
  /** 0-based, first round first. */
  roundIndex: number;
  /** 0-based within the round, top of the draw first. */
  matchIndex: number;
  /** The lower-numbered bracket position, `sim.tournament`'s player A. */
  top: BracketSlot | null;
  bottom: BracketSlot | null;
}

/** One column of the bracket. */
export interface BracketRoundModel {
  index: number;
  label: string;
  matches: BracketMatchModel[];
}

/** A draw size the bracket layout can halve cleanly: a power of two ≥ 2. */
export function isValidDrawSize(drawSize: number): boolean {
  return Number.isInteger(drawSize) && drawSize >= 2 && (drawSize & (drawSize - 1)) === 0;
}

/**
 * Label for the round that `playersRemaining` players start.
 *
 * `128 → "R128"`, `16 → "R16"`, `8 → "QF"`, `4 → "SF"`, `2 → "F"`, the same
 * mapping as `sim.tournament.round_label`.
 */
export function roundLabel(playersRemaining: number): string {
  return NAMED_ROUNDS[playersRemaining] ?? `R${playersRemaining}`;
}

/**
 * Every round label for a `drawSize` bracket, first round first.
 *
 * An 8 draw gives `["QF", "SF", "F"]`; a 128 draw gives
 * `["R128", "R64", "R32", "R16", "QF", "SF", "F"]`.
 *
 * @returns The labels, or `[]` for a draw size that cannot halve cleanly, the
 *   caller renders that as an error rather than looping forever.
 */
export function roundLabels(drawSize: number): string[] {
  if (!isValidDrawSize(drawSize)) return [];
  const labels: string[] = [];
  for (let remaining = drawSize; remaining >= 2; remaining /= 2) {
    labels.push(roundLabel(remaining));
  }
  return labels;
}

/**
 * Lay the draw out as one column per round.
 *
 * Round one pairs slots `2k-1` and `2k` (`sim.draw.first_round_pairings`), read
 * off `slots` sorted by `position` rather than by array index, so the pairing
 * rule is the API's and not this file's assumption about ordering. A slot the
 * response did not supply becomes `null`, a short `slots` array must not
 * render `undefined`.
 *
 * @returns One `BracketRoundModel` per round, or `[]` for an invalid draw size.
 */
export function buildRounds(drawSize: number, slots: BracketSlot[]): BracketRoundModel[] {
  const labels = roundLabels(drawSize);
  if (labels.length === 0) return [];

  const ordered = [...slots].sort((a, b) => a.position - b.position);

  return labels.map((label, roundIndex) => {
    const matchCount = drawSize / 2 ** (roundIndex + 1);
    const matches: BracketMatchModel[] = [];
    for (let matchIndex = 0; matchIndex < matchCount; matchIndex += 1) {
      const firstRound = roundIndex === 0;
      matches.push({
        roundIndex,
        matchIndex,
        top: firstRound ? (ordered[matchIndex * 2] ?? null) : null,
        bottom: firstRound ? (ordered[matchIndex * 2 + 1] ?? null) : null,
      });
    }
    return { index: roundIndex, label, matches };
  });
}
