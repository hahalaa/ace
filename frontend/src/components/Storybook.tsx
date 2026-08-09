/**
 * One tournament, played out: a seeded storybook run drawn onto the bracket,
 * with a champion at the end and a URL that reproduces it exactly.
 *
 * **The seed is the feature, not a parameter.** `/storybook` is the one endpoint
 * that simulates live (T3.4), and the same id + the same seed returns a
 * byte-identical body — that is what makes a link shareable. So the seed lives
 * in the **URL** (`./seed` reads and writes it), never in a `useState`
 * initialiser that rolls a fresh number, and every run goes through `runSeed`:
 * pick a seed, write it to the address bar, fetch it. Reloading the page
 * re-reads the same seed and replays the same tournament; **re-run** picks a
 * *different* one (and says so in the URL); **share** copies the address bar.
 * Nothing here re-rolls on a render, so a React re-render cannot silently change
 * the story on screen.
 *
 * **The bracket is not reimplemented.** T4.2's {@link Bracket} fetches and lays
 * out the draw exactly as it does on its own screen; this passes it the run to
 * draw over, and `./storybook` does the keying. Before a run, and while one is
 * in flight, the same component renders the same empty draw — no second layout,
 * no storybook-only bracket.
 *
 * **What is shown comes from the response, not from re-derivation.** The
 * champion is `StorybookResponse.champion` (never "whoever won the last match"),
 * scorelines are copied strings, and the disclosure is the shared
 * {@link Disclosure} — `StorybookMetadata` carries the same required
 * `ModelDisclosure` fields as the odds board (T3.4), so a story cannot be
 * published without the caveat any more than a probability can.
 *
 * **No staggered reveal.** The ticket offers it as optional; it would mean a
 * timer-driven round cursor, an interruption path for re-run, and a class of
 * test flake, for no acceptance criterion. The run appears when it arrives.
 */

import { useCallback, useEffect, useState } from 'react';

import { getStorybook } from '../api/client';
import type { StorybookResponse } from '../api/types';
import Bracket from './Bracket';
import Disclosure from './Disclosure';
import ErrorPanel from './ErrorPanel';
import { buildShareUrl, nextSeed, readSeed } from './seed';
import styles from './storybook.module.css';
import panelStyles from './panel.module.css';

export interface StorybookProps {
  /** Registry id, as listed by `/tournaments`. */
  tournamentId: string;
}

type LoadState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'ready'; story: StorybookResponse }
  | { status: 'error'; error: unknown };

type CopyState = 'idle' | 'copied' | 'failed';

// --------------------------------------------------------------------------
// Champion.
// --------------------------------------------------------------------------

function Champion({ story }: { story: StorybookResponse }) {
  // Identity is the response's own `champion` field. The run row is looked up
  // for the *summary* numbers only, and the final's line for its scoreline —
  // neither decides who won.
  const { champion } = story;
  const run = story.runs.find((entrant) => entrant.is_champion) ?? null;
  const final = story.rounds[story.rounds.length - 1]?.matches[0] ?? null;

  return (
    <section className={styles.champion} aria-label="Champion" data-champion={champion.player}>
      <p className={styles.championLabel}>Champion</p>
      <p className={styles.championName}>
        {champion.seed !== null && <span className={styles.championSeed}>[{champion.seed}]</span>}
        {champion.player}
      </p>
      {run !== null && (
        <p className={styles.championMeta}>
          {run.matches_won} {run.matches_won === 1 ? 'match' : 'matches'} won · from position{' '}
          {champion.position}
        </p>
      )}
      {final !== null && <p className={styles.championFinal}>{final.line}</p>}
    </section>
  );
}

// --------------------------------------------------------------------------
// The screen.
// --------------------------------------------------------------------------

export default function Storybook({ tournamentId }: StorybookProps) {
  // Read once, from the URL a page load actually arrived with.
  const [seed, setSeed] = useState<number | null>(() => readSeed(window.location.search));
  const [state, setState] = useState<LoadState>({ status: 'idle' });
  const [copied, setCopied] = useState<CopyState>('idle');

  useEffect(() => {
    if (seed === null) {
      setState({ status: 'idle' });
      return;
    }

    const controller = new AbortController();
    setState({ status: 'loading' });

    getStorybook(tournamentId, seed, { signal: controller.signal }).then(
      (story) => {
        if (!controller.signal.aborted) setState({ status: 'ready', story });
      },
      (error: unknown) => {
        // An abort surfaces as an ApiNetworkError; it is this effect tearing
        // down, not a failure worth showing.
        if (!controller.signal.aborted) setState({ status: 'error', error });
      },
    );

    return () => controller.abort();
  }, [tournamentId, seed]);

  /**
   * Run one seed: address bar first, then state.
   *
   * `replaceState`, not `pushState`: this component reads the seed at mount and
   * does not listen for `popstate`, so stacking history entries would give the
   * back button a URL the screen ignores. One entry, always matching what is on
   * screen.
   */
  const runSeed = useCallback(
    (value: number) => {
      window.history.replaceState({}, '', buildShareUrl(tournamentId, value));
      setCopied('idle');
      setSeed(value);
    },
    [tournamentId],
  );

  const share = useCallback(async () => {
    if (seed === null) return;
    const url = buildShareUrl(tournamentId, seed);
    try {
      await navigator.clipboard.writeText(url);
      setCopied('copied');
    } catch {
      // No clipboard permission, or no clipboard at all (http, older browsers).
      // The URL is shown instead, so sharing still works by hand.
      setCopied('failed');
    }
  }, [seed, tournamentId]);

  const story = state.status === 'ready' ? state.story : null;
  const hasRun = seed !== null;

  return (
    <section className={styles.storybook} aria-label="Storybook run">
      <header className={styles.toolbar}>
        <div className={styles.actions}>
          <button
            type="button"
            className={styles.primary}
            data-action={hasRun ? 'rerun' : 'simulate'}
            disabled={state.status === 'loading'}
            onClick={() => runSeed(nextSeed(seed))}
          >
            {hasRun ? 'Re-run with a new seed' : 'Simulate this tournament'}
          </button>
          <button
            type="button"
            className={styles.secondary}
            data-action="share"
            disabled={!hasRun}
            onClick={() => void share()}
          >
            Share this run
          </button>
        </div>
        {/* The seed the address bar carries — i.e. the one a link shared right
            now would replay. The footnote reports `metadata.seed`, the one the
            server says it ran; they are the same value from two sources, and
            keeping them separately labelled is what would make a disagreement
            visible rather than plausible. */}
        {hasRun && (
          <p className={styles.seed} data-meta="url_seed">
            seed {seed}
          </p>
        )}
      </header>

      {copied !== 'idle' && seed !== null && (
        <p className={styles.copy} data-copy={copied}>
          {copied === 'copied' ? (
            <>Link copied. It replays this exact tournament.</>
          ) : (
            <>
              Copy this link to replay this exact tournament:{' '}
              <code className={styles.copyUrl}>{buildShareUrl(tournamentId, seed)}</code>
            </>
          )}
        </p>
      )}

      {state.status === 'idle' && (
        <p className={panelStyles.panel} data-state="idle">
          Play this draw out match by match. Every run gets a seed, and the seed is in the
          address bar, so the link you share replays the same tournament, point for point.
        </p>
      )}

      {state.status === 'loading' && (
        <p className={panelStyles.panel} role="status">
          Simulating the {tournamentId} draw…
        </p>
      )}

      {state.status === 'error' && <ErrorPanel error={state.error} tournamentId={tournamentId} />}

      {story !== null && (
        <>
          <Champion story={story} />
          <Disclosure metadata={story.metadata} />
        </>
      )}

      <Bracket tournamentId={tournamentId} storybook={story} />

      {story !== null && (
        <p className={styles.footnote}>
          <span data-meta="match_count">{story.match_count} matches</span> ·{' '}
          <span data-meta="seed">seed {story.metadata.seed}</span> ·{' '}
          <span data-meta="data_through_year">data through {story.metadata.data_through_year}</span>{' '}
          ·{' '}
          <span data-meta="mode">
            {story.metadata.mode}
            {story.metadata.mode === 'blend' && ` (w=${story.metadata.w})`}
          </span>{' '}
          · <span data-meta="estimator">{story.metadata.estimator_class}</span> ·{' '}
          <span data-meta="source">{story.metadata.source}</span>
        </p>
      )}
    </section>
  );
}
