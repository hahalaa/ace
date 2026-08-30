// Title-odds derivation, pure, no React.

import type { SimulationPlayer } from '../api/types';

/** A probability in `[0, 1]` as a percentage to one decimal place. */
export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/** Rows in table order: title probability descending, ties broken by bracket position. Returns a new array. */
export function rankPlayers(players: SimulationPlayer[]): SimulationPlayer[] {
  return [...players].sort((a, b) => b.p_title - a.p_title || a.position - b.position);
}

/** The survival columns to render: every round label after the first (round one is 100% by construction). */
export function survivalColumns(roundLabels: string[]): string[] {
  return roundLabels.slice(1);
}
