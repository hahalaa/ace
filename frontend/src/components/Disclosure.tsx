/**
 * The model disclosure that has to travel with anything this app publishes.
 *
 * `metadata.is_forecast` and `metadata.classifier_limitation` are **required**
 * fields on the wire — `api.schemas.ModelDisclosure`, the base of both
 * `SimulationMetadata` (T3.3) and `StorybookMetadata` (T3.4) — precisely so
 * probabilities and scorelines cannot be published without their caveat. The
 * point of `ace-04-current-state.md` §7 seam 7 is that the caveat reaches a
 * *person*, not just a response body, so this renders above the thing it
 * qualifies and carries the limitation prose in an open-by-default `<details>`
 * rather than a tooltip that can be missed.
 *
 * **Extracted from `TitleOdds.tsx` by T4.4**, on the same reasoning that pulled
 * `ErrorPanel` out of `Bracket.tsx` in T4.3: two screens now publish model
 * output, the server deliberately gives them **one** disclosure to publish, and
 * a second copy of this component is how two screens end up disclosing two
 * different things while claiming the same source. `ModelDisclosure` — the
 * shared base, not either subclass — is the prop type, so neither screen can
 * make the disclosure depend on a field the other one lacks.
 */

import type { ModelDisclosure } from '../api/types';
import styles from './disclosure.module.css';

export interface DisclosureProps {
  /** Either endpoint's metadata block; only the shared base is read. */
  metadata: ModelDisclosure;
}

export default function Disclosure({ metadata }: DisclosureProps) {
  return (
    <section className={styles.disclosure} data-forecast={metadata.is_forecast}>
      <p className={styles.disclosureHead}>
        {metadata.is_forecast ? (
          <>These are model projections for a draw that has not been played.</>
        ) : (
          <>
            <strong>Not a forecast.</strong> Read these as a retrospective: the model's knowledge
            runs to {metadata.data_through_year}, which for an event already played includes the
            event itself.
          </>
        )}
      </p>
      <details className={styles.limitation} open>
        <summary>What the model does and does not know</summary>
        <p className={styles.limitationBody}>{metadata.classifier_limitation}</p>
      </details>
    </section>
  );
}
