/**
 * The model disclosure that has to travel with anything this app publishes.
 *
 * `metadata.is_forecast`, `metadata.classifier_limitation` and
 * `metadata.classifier_limitation_detail` are **required** fields on the wire —
 * `api.schemas.ModelDisclosure`, the base of both `SimulationMetadata` (T3.3)
 * and `StorybookMetadata` (T3.4) — precisely so probabilities and scorelines
 * cannot be published without their caveat. The point of the caveat is that it
 * reaches a *person*, not just a response body, so this renders above the thing
 * it qualifies.
 *
 * **Progressive disclosure.** The server splits the caveat in two: a one-line
 * `classifier_limitation` summary always on screen, and the full
 * `classifier_limitation_detail` account tucked behind a `<details>` that is
 * **collapsed by default** — the summary is the sentence a reader must not
 * miss, the detail is there for anyone who wants the mechanics. Native
 * `<details>`/`<summary>` carries its own expanded/collapsed state to assistive
 * tech, matching the disclosure pattern used elsewhere in the app.
 *
 * **Extracted from `TitleOdds.tsx` by T4.4**, on the same reasoning that pulled
 * `ErrorPanel` out of `Bracket.tsx` in T4.3: two screens now publish model
 * output, the server deliberately gives them **one** disclosure to publish, and
 * a second copy of this component is how two screens end up disclosing two
 * different things while claiming the same source. `ModelDisclosure` — the
 * shared base, not either subclass — is the prop type, so neither screen can
 * make the disclosure depend on a field the other one lacks.
 *
 * **Two variants, one source of truth.** `variant="full"` (the default, used by
 * the storybook screen) keeps the kicker and the collapsed mechanics detail.
 * `variant="compact"` (single match, title odds) renders only the one-line
 * summary, small and muted — the mechanics behind it are not something a user
 * doing the thing needs to read.
 */

import type { ModelDisclosure } from '../api/types';
import styles from './disclosure.module.css';

export interface DisclosureProps {
  /** Either endpoint's metadata block; only the shared base is read. */
  metadata: ModelDisclosure;
  /** @default 'full' */
  variant?: 'full' | 'compact';
  /**
   * Overrides the rendered summary line without touching the wire field.
   * Single-match wants wording distinct from `metadata.classifier_limitation`
   * (see MatchSim.tsx) while the response still carries the server's text for
   * any non-browser consumer.
   */
  text?: string;
}

export default function Disclosure({ metadata, variant = 'full', text }: DisclosureProps) {
  // The always-visible one-liner. Every draw shipped today is historical
  // (`is_forecast: false`), so the server's own summary is the honest lead.
  // When a forecast-capable draw first ships, `is_forecast` flips true and this
  // branch is where the wording diverges — kept as a conditional now, with a
  // placeholder, so adding that copy is a one-line change rather than a rewrite.
  const summary =
    text ??
    (metadata.is_forecast
      ? // TODO: real "these are projections" copy once a not-yet-played draw
        // exists. None ships today, so this branch is currently unreached.
        'These are model projections for a draw that has not yet been played.'
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
