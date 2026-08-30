// One tournament played out: a seeded /storybook run drawn onto the bracket (Bracket lays it out, this passes it the run). The seed lives in the URL, never a useState initialiser, so a re-render cannot change the story.

import { useCallback, useEffect, useState } from 'react';

import { getStorybook } from '../api/client';
import type { StorybookResponse } from '../api/types';
import { recordStorybook } from '../history/store';
import Bracket from './Bracket';
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

function Champion({ story }: { story: StorybookResponse }) {
  // Identity is the response's own `champion` field; the run row and final's line are for display only.
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
        if (!controller.signal.aborted) {
          setState({ status: 'ready', story });
          recordStorybook(story); // log to browser-local history the dashboard reads back
        }
      },
      (error: unknown) => {
        // An abort surfaces as an ApiNetworkError; it is this effect tearing down.
        if (!controller.signal.aborted) setState({ status: 'error', error });
      },
    );

    return () => controller.abort();
  }, [tournamentId, seed]);

  // Run one seed: address bar first (replaceState, not pushState), then state.
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
      // No clipboard: the URL is shown instead so sharing still works by hand.
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
        {/* The seed the address bar carries: what a link shared now would replay. */}
        {hasRun && (
          <p className={styles.seed} data-meta="url_seed">
            seed {seed}
          </p>
        )}
      </header>

      {copied !== 'idle' && seed !== null && (
        <p className={styles.copy} data-copy={copied}>
          {copied === 'copied' ? (
            <>Link copied.</>
          ) : (
            <>
              Copy this link to replay this exact tournament:{' '}
              <code className={styles.copyUrl}>{buildShareUrl(tournamentId, seed)}</code>
            </>
          )}
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
          {/* Provenance of the draw itself: an uploaded draw is user-submitted and unverified. */}
          {story.metadata.content_source === 'user_upload' && (
            <p className={styles.contentNote} data-content-source="user_upload">
              {story.metadata.content_note}
            </p>
          )}
        </>
      )}

      <Bracket tournamentId={tournamentId} storybook={story} />
    </section>
  );
}
