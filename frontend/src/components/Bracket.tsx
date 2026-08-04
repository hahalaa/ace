/**
 * The bracket: one column per round, matchups as {@link MatchCard}s.
 *
 * Fetches through **T4.1's `getBracket`** — there is no second fetch wrapper
 * here, and no second copy of the wire types. Everything about the layout is
 * derived from the response (`./rounds`), so an 8 draw renders three columns
 * labelled `QF/SF/F` and a 128 draw renders seven labelled `R128 … F` from the
 * same code path.
 *
 * **Navigating a large draw: horizontal scroll *and* a round selector.** They
 * solve different problems and neither covers the other. Seven columns of
 * legible cards is ~1,700 px, so the all-rounds view scrolls horizontally
 * rather than shrinking cards to fit. But round one of a 128 draw is 64 cards —
 * ~4,000 px of vertical page — so "scroll to the QF" is a poor way to reach the
 * QF; the selector filters the board to a single column instead. Small draws
 * need neither, and get both anyway, because a size-dependent affordance is a
 * hardcoded draw-size assumption in another costume.
 *
 * **Every failure the client can throw renders as itself.** T4.1's typed errors
 * are branched on directly — an unknown id, a broken draw file and an
 * unreachable server are visibly different panels rather than one "something
 * went wrong". That branching lives in {@link ErrorPanel}, shared with T4.3's
 * screen; only the one failure unique to laying out a bracket (a draw size that
 * cannot halve) is rendered here.
 */

import { useEffect, useMemo, useState } from 'react';

import { getBracket } from '../api/client';
import type { BracketResponse } from '../api/types';
import ErrorPanel from './ErrorPanel';
import MatchCard from './MatchCard';
import { buildRounds } from './rounds';
import styles from './bracket.module.css';
import panelStyles from './panel.module.css';

export interface BracketProps {
  /** Registry id, as listed by `/tournaments`. */
  tournamentId: string;
}

type LoadState =
  | { status: 'loading' }
  | { status: 'ready'; bracket: BracketResponse }
  | { status: 'error'; error: unknown };

/** `null` = show every round; otherwise the index of the only round shown. */
type RoundFilter = number | null;

// --------------------------------------------------------------------------
// The bracket.
// --------------------------------------------------------------------------

export default function Bracket({ tournamentId }: BracketProps) {
  const [state, setState] = useState<LoadState>({ status: 'loading' });
  const [selectedRound, setSelectedRound] = useState<RoundFilter>(null);

  useEffect(() => {
    const controller = new AbortController();
    setState({ status: 'loading' });
    setSelectedRound(null);

    getBracket(tournamentId, { signal: controller.signal }).then(
      (bracket) => {
        if (!controller.signal.aborted) setState({ status: 'ready', bracket });
      },
      (error: unknown) => {
        // An abort surfaces as an ApiNetworkError; it is this effect tearing
        // down, not a failure worth showing.
        if (!controller.signal.aborted) setState({ status: 'error', error });
      },
    );

    return () => controller.abort();
  }, [tournamentId]);

  const bracket = state.status === 'ready' ? state.bracket : null;
  const rounds = useMemo(
    () => (bracket === null ? [] : buildRounds(bracket.draw_size, bracket.slots)),
    [bracket],
  );

  if (state.status === 'loading') {
    return (
      <p className={panelStyles.panel} role="status">
        Loading the {tournamentId} bracket…
      </p>
    );
  }

  if (state.status === 'error') {
    return <ErrorPanel error={state.error} tournamentId={tournamentId} />;
  }

  if (rounds.length === 0) {
    return (
      <div className={panelStyles.panel} role="alert" data-error-kind="draw-size">
        <h2 className={panelStyles.panelHeading}>This draw cannot be laid out</h2>
        <p className={panelStyles.panelBody}>
          A bracket halves each round, so its draw size must be a power of two — the
          server reported {state.bracket.draw_size}.
        </p>
      </div>
    );
  }

  const visibleRounds = selectedRound === null ? rounds : [rounds[selectedRound]];

  return (
    <section className={styles.bracket} aria-label={`${state.bracket.name} bracket`}>
      <header className={styles.toolbar}>
        <div className={styles.meta}>
          <h2 className={styles.title}>{state.bracket.name}</h2>
          <p className={styles.subtitle}>
            {state.bracket.draw_size}-player draw · {state.bracket.surface} · best of{' '}
            {state.bracket.best_of}
          </p>
        </div>
        <nav className={styles.rail} aria-label="Round selector">
          <button
            type="button"
            className={styles.railButton}
            aria-pressed={selectedRound === null}
            onClick={() => setSelectedRound(null)}
          >
            All rounds
          </button>
          {rounds.map((round) => (
            <button
              key={round.label}
              type="button"
              className={styles.railButton}
              aria-pressed={selectedRound === round.index}
              onClick={() => setSelectedRound(round.index)}
            >
              {round.label}
            </button>
          ))}
        </nav>
      </header>

      <div className={styles.board}>
        {visibleRounds.map((round) => (
          <div className={styles.round} key={round.label}>
            <h3 className={styles.roundHeading}>
              {round.label}
              <span className={styles.roundCount}>
                {round.matches.length} {round.matches.length === 1 ? 'match' : 'matches'}
              </span>
            </h3>
            <ol className={styles.matches} aria-label={`${round.label} matches`}>
              {round.matches.map((match) => (
                <MatchCard
                  key={match.matchIndex}
                  roundLabel={round.label}
                  matchIndex={match.matchIndex}
                  top={{ slot: match.top }}
                  bottom={{ slot: match.bottom }}
                />
              ))}
            </ol>
          </div>
        ))}
      </div>
    </section>
  );
}
