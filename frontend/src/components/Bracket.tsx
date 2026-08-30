// The bracket: one column per round, layout derived from the /bracket response by ./rounds. The optional `storybook` prop overlays a played-out run (keyed on by ./overlay) and changes nothing when absent.

import { useEffect, useMemo, useState } from 'react';

import { getBracket } from '../api/client';
import type { BracketResponse, StorybookResponse } from '../api/types';
import ErrorPanel from './ErrorPanel';
import MatchCard from './MatchCard';
import { buildStorybookOverlay, describesBracket, overlayKey } from './overlay';
import { buildRounds } from './rounds';
import styles from './bracket.module.css';
import panelStyles from './panel.module.css';

export interface BracketProps {
  /** Registry id, as listed by `/tournaments`. */
  tournamentId: string;
  /** A played-out run to draw over the draw; passed only by {@link Storybook}. */
  storybook?: StorybookResponse | null;
}

type LoadState =
  | { status: 'loading' }
  | { status: 'ready'; bracket: BracketResponse }
  | { status: 'error'; error: unknown };

/** `null` = show every round; otherwise the index of the only round shown. */
type RoundFilter = number | null;

export default function Bracket({ tournamentId, storybook = null }: BracketProps) {
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
        // An abort surfaces as an ApiNetworkError; it is this effect tearing down.
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
  const overlay = useMemo(() => {
    if (bracket === null || storybook === null || storybook === undefined) return null;
    // A run for another draw is not drawn at all, see `./overlay`.
    if (!describesBracket(storybook, bracket.tournament_id, bracket.draw_size)) return null;
    return buildStorybookOverlay(rounds, storybook);
  }, [bracket, rounds, storybook]);

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
          A bracket halves each round, so its draw size must be a power of two. The
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
              {round.matches.map((match) => {
                // `played === undefined` for every non-storybook render.
                const played = overlay?.get(overlayKey(round.index, match.matchIndex));
                return (
                  <MatchCard
                    key={match.matchIndex}
                    roundLabel={round.label}
                    matchIndex={match.matchIndex}
                    top={{ slot: played ? played.top : match.top, result: played?.topResult }}
                    bottom={{
                      slot: played ? played.bottom : match.bottom,
                      result: played?.bottomResult,
                    }}
                    scoreline={played?.scoreline}
                  />
                );
              })}
            </ol>
          </div>
        ))}
      </div>
    </section>
  );
}
