/**
 * The "Recent simulations" panel on the dashboard: a short, scannable list of the
 * user's own past results, each a deep link back to the exact reproduced run.
 *
 * **It reads history, it does not make it.** Entries are logged by the match and
 * storybook views as their results complete (see `../history/store`); this only
 * renders what is stored and offers a way to clear it. Every item is a plain link
 * carrying the existing `?view=…&seed=…` URL state, so following one lands on the
 * unchanged view with its own disclosures intact — nothing here is duplicated or
 * altered.
 *
 * **Empty means empty.** With no history there is nothing to render, so a
 * first-time visitor's landing page is not cluttered with an empty-state
 * placeholder: the component returns `null`.
 *
 * **Honest about ephemeral uploads.** An entry flagged `ephemeral` (an
 * uploaded-draw storybook) is labelled as possibly no longer playable, because the
 * server holds uploaded draws in memory only. Clicking through degrades via the
 * storybook view's existing not-found panel; no new error UI lives here.
 */

import { useCallback, useId, useState } from 'react';

import { clearHistory, readHistory, type HistoryEntry } from '../history/store';
import { relativeTime } from '../history/time';
import styles from './recent.module.css';

/** A single logged result as a list item linking back to its reproduced run. */
function Item({ entry }: { entry: HistoryEntry }) {
  const when = relativeTime(entry.timestamp);
  const iso = new Date(entry.timestamp).toISOString();
  // One accessible name that carries everything the eye gets from the row: what
  // it was, when it ran, and (for an upload) that it may not open any more.
  const label = entry.ephemeral
    ? `${entry.summary}, ${when}, may no longer be playable`
    : `${entry.summary}, ${when}`;

  return (
    <li className={styles.item}>
      <a className={styles.link} href={entry.url} aria-label={label}>
        <span className={styles.summary}>{entry.summary}</span>
        <span className={styles.meta}>
          <time dateTime={iso}>{when}</time>
          {entry.ephemeral && (
            <span className={styles.ephemeral} data-ephemeral="true">
              may no longer be playable
            </span>
          )}
        </span>
      </a>
    </li>
  );
}

export default function RecentSimulations() {
  const [entries, setEntries] = useState<HistoryEntry[]>(() => readHistory());
  const headingId = useId();

  const onClear = useCallback(() => {
    clearHistory();
    setEntries([]);
  }, []);

  // Nothing logged yet: render nothing at all, not an empty-state placeholder.
  if (entries.length === 0) return null;

  return (
    <section className={styles.recent} aria-labelledby={headingId}>
      <div className={styles.head}>
        <h2 id={headingId} className={styles.heading}>
          Recent simulations
        </h2>
        <button
          type="button"
          className={styles.clear}
          data-action="clear-history"
          onClick={onClear}
        >
          Clear history
        </button>
      </div>

      <ul className={styles.list}>
        {entries.map((entry) => (
          <Item key={entry.id} entry={entry} />
        ))}
      </ul>

      <p className={styles.note}>
        Kept in this browser only, never sent anywhere.
      </p>
    </section>
  );
}
