/**
 * One panel per distinguishable failure — the single place T4.1's typed errors
 * become something a person can read.
 *
 * Extracted from `Bracket.tsx` by T4.3, which needs the same branching plus one
 * more case (`425 cache_missing`). A second copy would have meant a second set
 * of headings drifting out of step with the first, which is the same argument
 * that keeps one fetch wrapper and one set of wire types in `src/api/`.
 *
 * **The branches are the API's, not invented here.** `ApiConfigError` (the app
 * is misconfigured, no request was attempted), `ApiNetworkError` (the request
 * never landed), and `ApiError` split by `status` and by the machine-readable
 * `detail.reason` the server attaches precisely so a client need not parse
 * prose. Anything unrecognised still renders, with its status and message,
 * rather than collapsing to "something went wrong".
 *
 * Every panel carries `data-error-kind`, so a test names the case it means
 * instead of matching on wording that is free to change.
 */

import { describeError } from './errors';
import styles from './panel.module.css';

export interface ErrorPanelProps {
  /** Whatever the client threw. Narrowed here, not by the caller. */
  error: unknown;
  /** The id being loaded — names the thing that 404'd. */
  tournamentId: string;
}

export default function ErrorPanel({ error, tournamentId }: ErrorPanelProps) {
  const { kind, heading, message, problems, command } = describeError(error, tournamentId);

  return (
    <div className={styles.panel} role="alert" data-error-kind={kind}>
      <h2 className={styles.panelHeading}>{heading}</h2>
      <p className={styles.panelBody}>{message}</p>
      {problems !== undefined && (
        <ul className={styles.problems}>
          {problems.map((problem) => (
            <li key={problem}>{problem}</li>
          ))}
        </ul>
      )}
      {command !== undefined && <code className={styles.command}>{command}</code>}
    </div>
  );
}
