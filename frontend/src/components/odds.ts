/**
 * Title-odds derivation — pure, no React.
 *
 * Kept out of `TitleOdds.tsx` for the reason `rounds.ts` is kept out of
 * `Bracket.tsx`: the interesting logic is testable without rendering, and a
 * component file exports only components (which is also what keeps React Fast
 * Refresh working).
 */

import type { SimulationPlayer } from '../api/types';

/** A probability in `[0, 1]` as a percentage to one decimal place. */
export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * Rows in the order the table shows them: title probability descending, ties
 * broken by bracket position so the order is total and reproducible.
 *
 * The server already sorts by this key; re-applying it here rather than
 * trusting array order is the same discipline `rounds.ts` applies to
 * `position`, and it costs one comparison per row. Returns a new array — the
 * response is not mutated.
 */
export function rankPlayers(players: SimulationPlayer[]): SimulationPlayer[] {
  return [...players].sort((a, b) => b.p_title - a.p_title || a.position - b.position);
}

/**
 * The survival columns to render: every round label **after the first**.
 *
 * Round one is contested by every entrant by construction, so a column of
 * `100.0%` carries no information — T2.5's CLI table drops it for the same
 * reason. An 8 draw's labels are `["QF","SF","F"]`, so its first round *is* the
 * QF and the QF column is the one dropped.
 *
 * @returns The columns, `[]` for a single-round draw (nothing sits between
 *   round one and the title).
 */
export function survivalColumns(roundLabels: string[]): string[] {
  return roundLabels.slice(1);
}
