/**
 * The landing screen: two ways in, and nothing loaded until the user picks one.
 *
 * The app used to open straight onto a 128-player bracket. It now opens here, so
 * the first choice is the user's — simulate a single match (the featured path),
 * or work with a full tournament (an example draw, or upload your own). No data
 * is fetched on this screen; each card is a link that carries the URL state the
 * destination reads (`./App` dispatches on `?view=`).
 */

import type { BestOf, Surface } from '../api/types';
import styles from './dashboard.module.css';

export interface DashboardProps {
  /** The example tournament the "explore a tournament" card opens. */
  tournamentId: string;
}

/** A curated matchup that makes the single-match card a one-click demo. */
const EXAMPLE_MATCH: { a: string; b: string; surface: Surface; bestOf: BestOf } = {
  a: 'Carlos Alcaraz',
  b: 'Jannik Sinner',
  surface: 'Clay',
  bestOf: 5,
};

function exampleMatchHref(): string {
  const params = new URLSearchParams({
    view: 'match',
    a: EXAMPLE_MATCH.a,
    b: EXAMPLE_MATCH.b,
    surface: EXAMPLE_MATCH.surface,
    bo: String(EXAMPLE_MATCH.bestOf),
  });
  return `?${params.toString()}`;
}

export default function Dashboard({ tournamentId }: DashboardProps) {
  const tournamentHref = `?tournament=${encodeURIComponent(tournamentId)}&view=bracket`;

  return (
    <section className={styles.dashboard} aria-label="Start">
      <header className={styles.hero}>
        <h1 className={styles.heroTitle}>Simulate the tennis you want to see.</h1>
        <p className={styles.heroLede}>
          A point-by-point model of the ATP tour. Play a single match out for its
          odds and likely scoreline, or run a whole Grand Slam draw.
        </p>
      </header>

      <div className={styles.cards}>
        <a className={`${styles.card} ${styles.featured}`} href="?view=match" data-card="match">
          <span className={styles.cardKicker}>Featured</span>
          <h2 className={styles.cardTitle}>Simulate a single match</h2>
          <p className={styles.cardBody}>
            Pick any two players, a surface and a format. Get each player's win
            chance, the likely set scores, and how many games to expect.
          </p>
          <span className={styles.cardCta}>Pick two players</span>
        </a>

        <div className={styles.card} data-card="tournament">
          <span className={styles.cardKicker}>Tournament</span>
          <h2 className={styles.cardTitle}>Run a full draw</h2>
          <p className={styles.cardBody}>
            Explore the 2026 Australian Open bracket and its title odds, play it
            out match by match, or upload your own draw.
          </p>
          <div className={styles.cardLinks}>
            <a className={styles.cardCta} href={tournamentHref} data-link="example">
              Australian Open 2026
            </a>
            <a className={styles.cardCtaSoft} href="?view=upload" data-link="upload">
              Upload a draw
            </a>
          </div>
        </div>
      </div>

      <p className={styles.demoLink}>
        Or try an example matchup:{' '}
        <a href={exampleMatchHref()} data-link="example-match">
          {EXAMPLE_MATCH.a} vs {EXAMPLE_MATCH.b} on {EXAMPLE_MATCH.surface}
        </a>
      </p>
    </section>
  );
}
