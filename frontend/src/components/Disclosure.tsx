// The model disclosure that travels with anything this app publishes; the prop type is the shared `ModelDisclosure` base. Two variants: `full` keeps the collapsed mechanics detail, `compact` renders only the one-line summary.

import type { ModelDisclosure } from '../api/types';
import styles from './disclosure.module.css';

export interface DisclosureProps {
  /** Either endpoint's metadata block; only the shared base is read. */
  metadata: ModelDisclosure;
  /** @default 'full' */
  variant?: 'full' | 'compact';
  /** Overrides the rendered summary line without touching the wire field (single-match wants distinct wording). */
  text?: string;
}

export default function Disclosure({ metadata, variant = 'full', text }: DisclosureProps) {
  // The always-visible one-liner. The is_forecast branch is unreached today (every shipped draw is historical).
  const summary =
    text ??
    (metadata.is_forecast
      ? 'These are model projections for a draw that has not yet been played.'
      : metadata.classifier_limitation);

  if (variant === 'compact') {
    return (
      <p className={styles.compactLine} data-forecast={metadata.is_forecast}>
        {summary}
      </p>
    );
  }

  return (
    <section className={styles.disclosure} data-forecast={metadata.is_forecast}>
      <p className={styles.kicker}>On the model</p>
      <p className={styles.disclosureHead}>{summary}</p>
      <details className={styles.limitation}>
        <summary>What the model does and does not know</summary>
        <p className={styles.limitationBody}>{metadata.classifier_limitation_detail}</p>
      </details>
    </section>
  );
}
