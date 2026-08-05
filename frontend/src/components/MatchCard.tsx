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
 * **T4.4 took that extension point, and split it exactly where the data
 * splits.** `result` — did *this side* win — is per entrant, so it is the field
 * on {@link MatchEntrant} the docstring above reserved. The **scoreline is not**:
 * the API sends one winner-first string per match (`6-4 3-6 7-6(5) 6-2`), not a
 * pair of per-side values, so it is a match-level prop rendered once beneath the
 * two rows. Putting half a string on each entrant would have been the parallel
 * prop this file warns against, wearing the opposite costume.
 *
 * **Both are optional, and absent means absent.** With neither passed — every
 * T4.2/T4.3 caller — the emitted markup is byte-identical to what this component
 * shipped with: no `data-result` attribute, no extra class, no scoreline node.
 * That is pinned by the snapshot differential in `Bracket.test.tsx`.
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

/** Which side of a played match this entrant was — T4.4's storybook result. */
export type MatchResult = 'won' | 'lost';

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
  /**
   * Whether this entrant won the match, once a storybook run has played it out.
   *
   * `undefined` — every caller outside T4.4's storybook — renders exactly as
   * before: an unplayed match, with no result claimed either way.
   */
  result?: MatchResult;
}

export interface MatchCardProps {
  /** The round this card sits in, e.g. `R128` or `SF` — used for its label. */
  roundLabel: string;
  /** 0-based position within the round, top of the draw first. */
  matchIndex: number;
  /** The lower-numbered bracket position. */
  top: MatchEntrant;
  bottom: MatchEntrant;
  /**
   * The match's scoreline, **winner-first**, e.g. `6-4 3-6 7-6(5) 6-2`.
   *
   * A match-level fact rather than an entrant one — see the module docstring.
   * Omitted for a match nothing has played.
   */
  scoreline?: string;
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
  const { slot, result } = entrant;
  const state = entrantState(slot);

  // `slot.player` is a required string on the wire; the fallback is here so a
  // malformed body degrades to a readable word instead of "undefined".
  const name = state === 'unknown' ? 'TBD' : slot?.player || 'Unknown entrant';
  const seed = slot?.seed ?? null;

  return (
    // `result === undefined` (every non-storybook caller) leaves both the class
    // list and the attribute set exactly as T4.2 emitted them: React omits an
    // `undefined` attribute, and the winner class is only ever appended.
    <div
      className={result === 'won' ? `${styles.entrant} ${styles.winner}` : styles.entrant}
      data-state={state}
      data-result={result}
    >
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
export default function MatchCard({
  roundLabel,
  matchIndex,
  top,
  bottom,
  scoreline,
}: MatchCardProps) {
  return (
    <li className={styles.card} aria-label={`${roundLabel} match ${matchIndex + 1}`}>
      <EntrantRow entrant={top} />
      <EntrantRow entrant={bottom} />
      {scoreline !== undefined && (
        <p className={styles.scoreline} data-cell="scoreline">
          {scoreline}
        </p>
      )}
    </li>
  );
}
