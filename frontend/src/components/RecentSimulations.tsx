// The "Recent simulations" dashboard panel: renders what ../history/store logged as deep links; returns null with no history.

import { useCallback, useId, useState } from 'react';

import { clearHistory, readHistory, type HistoryEntry } from '../history/store';
import { relativeTime } from '../history/time';
import styles from './recent.module.css';

/** A single logged result as a list item linking back to its reproduced run. */
function Item({ entry }: { entry: HistoryEntry }) {
  const when = relativeTime(entry.timestamp);
  const iso = new Date(entry.timestamp).toISOString();
  // One accessible name carrying what the eye gets from the row.
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
    </section>
  );
}
