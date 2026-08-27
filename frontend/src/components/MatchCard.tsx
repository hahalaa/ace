/**
 * One matchup: two entrant rows, each with a seed, a name and a reserved
 * probability cell.
 *
 * **The prop shape carries optional result state without a restructure.** A win
 * probability belongs to an *entrant*, not to a match, a card needs two of
 * them, they must stay attached to the side they describe, and per-side result
 * state (winner, scoreline) works the same way. So each side is a
 * {@link MatchEntrant} record rather than a bare `BracketSlot`, and adding
 * `winProbability` (declared, never passed here) or `result` is a **field on
 * that record**, not a second parallel prop and not a restructure of the
 * component's children. The layout reserves the column those numbers land in,
 * see `.prob` in the stylesheet, so populating it later shifts no name and
 * rewraps no card.
 *
 * The storybook overlay splits its data exactly where the data splits: `result`
 * did *this side* win, is per entrant, so it is a field on
 * {@link MatchEntrant}. The **scoreline is not**: the API sends one winner-first
 * string per match (`6-4 3-6 7-6(5) 6-2`), not a pair of per-side values, so it
 * is a match-level prop rendered once beneath the two rows. Putting half a
 * string on each entrant would have been the parallel prop this file warns
 * against, wearing the opposite costume.
 *
 * **Both are optional, and absent means absent.** With neither passed the
 * emitted markup is byte-identical to a bare bracket render: no `data-result`
 * attribute, no extra class, no scoreline node. That is pinned by the snapshot
 * differential in `Bracket.test.tsx`.
 *
 * **Three entrant states, three renderings, none of them `undefined`:**
 *
 *   * a resolved player, seed chip (when seeded) and name;
 *   * a **placeholder** (`is_placeholder: true`, `player_id: null`), the draw
 *     file's own wording (`Qualifier`, `Qualifier/Lucky Loser`) rendered
 *     verbatim but styled as unfilled, because the API deliberately serves
 *     draws holding placeholder slots even though they cannot be simulated;
 *   * an **unknown** side (`slot: null`), every round after the first, whose
 *     participants the `/bracket` response does not know. Reads `TBD`.
 *
 * The middle two are visually and programmatically distinct (`data-state`): a
 * slot nobody has filled is not the same fact as a slot this round has not
 * produced yet.
 */

import type { BracketSlot } from '../api/types';
import styles from './bracket.module.css';

/** Which side of a played match this entrant was, the storybook result. */
export type MatchResult = 'won' | 'lost';

/**
 * One side of a match.
 *
 * Everything except `slot` is optional and unpopulated in a bare bracket render,
 * the extension points described in the module docstring.
 */
export interface MatchEntrant {
  /** The draw slot, or `null` when this round has not produced a participant. */
  slot: BracketSlot | null;
  /**
   * P(this entrant wins this match), in `[0, 1]`.
   *
   * Reserved for a future win-probability overlay; **nothing passes it today**, and the cell it
   * renders into is laid out whether or not it arrives.
   */
  winProbability?: number;
  /**
   * Whether this entrant won the match, once a storybook run has played it out.
   *
   * `undefined`, every caller outside the storybook overlay, renders exactly as
   * before: an unplayed match, with no result claimed either way.
   */
  result?: MatchResult;
}

export interface MatchCardProps {
  /** The round this card sits in, e.g. `R128` or `SF`, used for its label. */
  roundLabel: string;
  /** 0-based position within the round, top of the draw first. */
  matchIndex: number;
  /** The lower-numbered bracket position. */
  top: MatchEntrant;
  bottom: MatchEntrant;
  /**
   * The match's scoreline, **winner-first**, e.g. `6-4 3-6 7-6(5) 6-2`.
   *
   * A match-level fact rather than an entrant one, see the module docstring.
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
    // list and the attribute set exactly as a bare render emits them: React omits an
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
      {/* Reserved for the win-probability overlay: laid out and sized
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
