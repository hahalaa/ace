// One matchup: two entrant rows. Per-side state (result) is a field on the MatchEntrant it describes; the scoreline is match-level. Three entrant states, distinct via data-state: player, placeholder, unknown.

import type { BracketSlot } from '../api/types';
import styles from './bracket.module.css';

/** Which side of a played match this entrant was, the storybook result. */
export type MatchResult = 'won' | 'lost';

/** One side of a match; everything except `slot` is optional and unpopulated in a bare bracket render. */
export interface MatchEntrant {
  /** The draw slot, or `null` when this round has not produced a participant. */
  slot: BracketSlot | null;
  /** P(this entrant wins this match). Reserved for a future overlay; nothing passes it today. */
  winProbability?: number;
  /** Whether this entrant won, once a storybook run has played the match out; `undefined` renders as an unplayed match. */
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
  /** The match's winner-first scoreline (a match-level fact); omitted for a match nothing has played. */
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

  // Fallback so a malformed body degrades to a readable word, not "undefined".
  const name = state === 'unknown' ? 'TBD' : slot?.player || 'Unknown entrant';
  const seed = slot?.seed ?? null;

  return (
    // result === undefined leaves the class list and attributes as a bare render emits them.
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
      {/* Reserved for the win-probability overlay: sized here so arriving numbers displace nothing. */}
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
