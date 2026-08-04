/**
 * One matchup: two entrant rows, each with a seed, a name and a reserved
 * probability cell.
 *
 * **The prop shape anticipates T4.3/T4.4 without pre-building them.** A win
 * probability belongs to an *entrant*, not to a match — a card needs two of
 * them, they must stay attached to the side they describe, and a later ticket
 * also wants per-side result state (winner, scoreline). So each side is a
 * {@link MatchEntrant} record rather than a bare `BracketSlot`, and adding
 * `winProbability` (already declared, deliberately never passed here) or a
 * future `isWinner`/`scoreline` is a **field on that record**, not a second
 * parallel prop and not a restructure of the component's children. The layout
 * reserves the column those numbers land in — see `.prob` in the stylesheet —
 * so populating it later shifts no name and rewraps no card.
 *
 * **Three entrant states, three renderings, none of them `undefined`:**
 *
 *   * a resolved player — seed chip (when seeded) and name;
 *   * a **placeholder** (`is_placeholder: true`, `player_id: null`) — the draw
 *     file's own wording (`Qualifier`, `Qualifier/Lucky Loser`) rendered
 *     verbatim but styled as unfilled, because the API deliberately serves such
 *     draws (`example_usopen_2026.json`) even though they cannot be simulated;
 *   * an **unknown** side (`slot: null`) — every round after the first, whose
 *     participants the `/bracket` response does not know. Reads `TBD`.
 *
 * The middle two are visually and programmatically distinct (`data-state`): a
 * slot nobody has filled is not the same fact as a slot this round has not
 * produced yet.
 */

import type { BracketSlot } from '../api/types';
import styles from './bracket.module.css';

/**
 * One side of a match.
 *
 * Everything except `slot` is optional and unpopulated in T4.2 — the extension
 * points described in the module docstring.
 */
export interface MatchEntrant {
  /** The draw slot, or `null` when this round has not produced a participant. */
  slot: BracketSlot | null;
  /**
   * P(this entrant wins this match), in `[0, 1]`.
   *
   * Reserved for T4.3/T4.4; **nothing passes it today**, and the cell it
   * renders into is laid out whether or not it arrives.
   */
  winProbability?: number;
}

export interface MatchCardProps {
  /** The round this card sits in, e.g. `R128` or `SF` — used for its label. */
  roundLabel: string;
  /** 0-based position within the round, top of the draw first. */
  matchIndex: number;
  /** The lower-numbered bracket position. */
  top: MatchEntrant;
  bottom: MatchEntrant;
}

type EntrantState = 'player' | 'placeholder' | 'unknown';

function entrantState(slot: BracketSlot | null): EntrantState {
  if (slot === null) return 'unknown';
  return slot.is_placeholder ? 'placeholder' : 'player';
}

/** Percentage text for the reserved cell, or `''` when there is no probability. */
function formatProbability(value: number | undefined): string {
  if (value === undefined || !Number.isFinite(value)) return '';
  return `${Math.round(value * 100)}%`;
}

function EntrantRow({ entrant }: { entrant: MatchEntrant }) {
  const { slot } = entrant;
  const state = entrantState(slot);

  // `slot.player` is a required string on the wire; the fallback is here so a
  // malformed body degrades to a readable word instead of "undefined".
  const name = state === 'unknown' ? 'TBD' : slot?.player || 'Unknown entrant';
  const seed = slot?.seed ?? null;

  return (
    <div className={styles.entrant} data-state={state}>
      <span className={styles.seed} data-cell="seed" aria-hidden={seed === null}>
        {seed === null ? '' : seed}
      </span>
      <span
        className={styles.name}
        data-cell="name"
        title={state === 'placeholder' ? 'Entrant not yet decided' : undefined}
      >
        {name}
      </span>
      {/* Reserved for the T4.3/T4.4 win-probability overlay: laid out and sized
          here so arriving numbers displace nothing. Empty until then. */}
      <span className={styles.prob} data-cell="prob">
        {formatProbability(entrant.winProbability)}
      </span>
    </div>
  );
}

/** One matchup, as a list item inside its round's `<ol>`. */
export default function MatchCard({ roundLabel, matchIndex, top, bottom }: MatchCardProps) {
  return (
    <li className={styles.card} aria-label={`${roundLabel} match ${matchIndex + 1}`}>
      <EntrantRow entrant={top} />
      <EntrantRow entrant={bottom} />
    </li>
  );
}
