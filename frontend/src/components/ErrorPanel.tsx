// One panel per distinguishable failure. Branching lives in ./errors and follows the API's own status/detail.reason; every panel carries data-error-kind so a test names the case, not the wording.

import { describeError } from './errors';
import styles from './panel.module.css';

export interface ErrorPanelProps {
  /** Whatever the client threw. Narrowed here, not by the caller. */
  error: unknown;
  /** The id being loaded, names the thing that 404'd; omitted by tournament-free views. */
  tournamentId?: string;
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
