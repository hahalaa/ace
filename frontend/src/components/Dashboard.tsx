/**
 * The landing screen: two ways in, and nothing loaded until the user picks one.
 *
 * The app used to open straight onto a 128-player bracket. It now opens here, so
 * the first choice is the user's — simulate a single match (the featured path),
 * or work with a full tournament (an example draw, or upload your own). No data
 * is fetched on this screen; each card is a link that carries the URL state the
 * destination reads (`./App` dispatches on `?view=`).
 *
 * **The two cards are a toolbar, keyboard-operable as a unit.** Both cards are
 * activatable, not just the featured one: each is a single navigational control
 * with one primary destination. `role="toolbar"` (not `tablist`/`radiogroup`) is
 * the right container role because activating a card *navigates away* rather than
 * selecting a persistent option or switching a same-page panel, and a toolbar is
 * exactly a set of controls a user arrows between. Roving tabindex: one card is
 * tabbable at a time, Left/Right (and Home/End) move focus between them, and
 * Enter/Space activates the focused card's primary action — mirroring a click.
 *
 * **"Run a full draw" activates to the Australian Open example.** That card holds
 * two inner links (the example bracket and the upload flow), so its own primary
 * action is the example bracket — the same destination its most prominent inner
 * link carries, treated as the card's headline choice. The two inner links stay
 * independently clickable and focusable; a click or Enter on one of them does its
 * own thing and does not also trigger the card.
 */

import { useRef, useState, type KeyboardEvent } from 'react';

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

/** How many roving items the toolbar holds. */
const CARD_COUNT = 2;

export default function Dashboard({ tournamentId }: DashboardProps) {
  const tournamentHref = `?tournament=${encodeURIComponent(tournamentId)}&view=bracket`;
  const matchHref = '?view=match';

  // Roving tabindex: exactly one card is in the tab order at a time; arrow keys
  // move focus (and the tabbable index) between the two. `active` starts on the
  // featured card so Tab lands there first.
  const cardRefs = useRef<Array<HTMLElement | null>>([]);
  const [active, setActive] = useState(0);

  const navigate = (href: string) => {
    window.location.href = href;
  };

  const focusCard = (index: number) => {
    const next = (index + CARD_COUNT) % CARD_COUNT;
    setActive(next);
    cardRefs.current[next]?.focus();
  };

  // Arrow/Home/End move between cards. Only act when focus is on a card itself,
  // so arrowing while focus sits on an inner link is left alone.
  const onToolbarKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (!cardRefs.current.includes(event.target as HTMLElement)) return;
    switch (event.key) {
      case 'ArrowRight':
        event.preventDefault();
        focusCard(active + 1);
        break;
      case 'ArrowLeft':
        event.preventDefault();
        focusCard(active - 1);
        break;
      case 'Home':
        event.preventDefault();
        focusCard(0);
        break;
      case 'End':
        event.preventDefault();
        focusCard(CARD_COUNT - 1);
        break;
    }
  };

  // Enter/Space on a focused card activates its primary destination — but only
  // when the key event originates on the card, not on a bubbled-up inner link.
  const activateOn = (href: string) => (event: KeyboardEvent<HTMLElement>) => {
    if (event.target !== event.currentTarget) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      navigate(href);
    }
  };

  const tabIndexFor = (index: number) => (active === index ? 0 : -1);

  return (
    <section className={styles.dashboard} aria-label="Start">
      <header className={styles.hero}>
        <h1 className={styles.heroTitle}>Simulate the tennis you want to see.</h1>
        <p className={styles.heroLede}>
          A point-by-point model of the ATP tour. Play a single match out for its
          odds and likely scoreline, or run a whole Grand Slam draw.
        </p>
      </header>

      <div
        className={styles.cards}
        role="toolbar"
        aria-label="Ways to start"
        aria-orientation="horizontal"
        onKeyDown={onToolbarKeyDown}
      >
        <a
          ref={(el) => {
            cardRefs.current[0] = el;
          }}
          className={styles.card}
          href={matchHref}
          data-card="match"
          aria-label="Simulate a single match"
          tabIndex={tabIndexFor(0)}
          onFocus={(event) => {
            if (event.target === event.currentTarget) setActive(0);
          }}
          onKeyDown={activateOn(matchHref)}
        >
          <span className={styles.cardKicker}>Featured</span>
          <h2 className={styles.cardTitle}>Simulate a single match</h2>
          <p className={styles.cardBody}>
            Pick any two players, a surface and a format. Get each player's win
            chance, the likely set scores, and how many games to expect.
          </p>
          <span className={styles.cardCta}>Pick two players</span>
        </a>

        <div
          ref={(el) => {
            cardRefs.current[1] = el;
          }}
          className={styles.card}
          data-card="tournament"
          role="link"
          aria-label="Run a full draw: the 2026 Australian Open bracket"
          tabIndex={tabIndexFor(1)}
          onFocus={(event) => {
            if (event.target === event.currentTarget) setActive(1);
          }}
          onClick={() => navigate(tournamentHref)}
          onKeyDown={activateOn(tournamentHref)}
        >
          <span className={styles.cardKicker}>Tournament</span>
          <h2 className={styles.cardTitle}>Run a full draw</h2>
          <p className={styles.cardBody}>
            Explore the 2026 Australian Open bracket and its title odds, play it
            out match by match, or upload your own draw.
          </p>
          <div className={styles.cardLinks}>
            <a
              className={styles.cardCta}
              href={tournamentHref}
              data-link="example"
              onClick={(event) => event.stopPropagation()}
            >
              Australian Open 2026
            </a>
            <a
              className={styles.cardCtaSoft}
              href="?view=upload"
              data-link="upload"
              onClick={(event) => event.stopPropagation()}
            >
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
