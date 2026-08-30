// Bracket column structure derived from a /bracket response (entrants, not matches), pure, no React. Round labels mirror sim.tournament.round_labels and are derived from draw_size, never hardcoded. Only round one has known participants; later rounds are real slots with null entrants.

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

/** Label for the round that `playersRemaining` players start (same mapping as sim.tournament.round_label). */
export function roundLabel(playersRemaining: number): string {
  return NAMED_ROUNDS[playersRemaining] ?? `R${playersRemaining}`;
}

/** Every round label for a `drawSize` bracket, first round first; `[]` for a draw size that cannot halve cleanly. */
export function roundLabels(drawSize: number): string[] {
  if (!isValidDrawSize(drawSize)) return [];
  const labels: string[] = [];
  for (let remaining = drawSize; remaining >= 2; remaining /= 2) {
    labels.push(roundLabel(remaining));
  }
  return labels;
}

/** Lay the draw out as one column per round; round one pairs slots sorted by `position`. `[]` for an invalid draw size. */
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
