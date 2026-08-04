/**
 * The Monte Carlo board: every entrant ranked by title probability, with their
 * round-by-round survival, and the disclosure that has to travel with it.
 *
 * **Columns are the draw's, not this file's.** `SimulationResponse.round_labels`
 * is `["R128", … "F"]` for a 128 draw and `["QF","SF","F"]` for an 8 draw, so a
 * table with fixed reach-QF/SF/F columns silently drops six columns on the one
 * and renders empty cells on the other. The columns rendered are
 * `round_labels.slice(1)` — **the first round is skipped**, following T2.5's CLI
 * precedent, because every entrant contests round one by construction and a
 * column of `100.0%` carries no information. `p_reach` is keyed by those same
 * labels, so cell lookup is by label and never by position.
 *
 * **The disclosure is not optional and not a footnote detail.**
 * `metadata.is_forecast` and `metadata.classifier_limitation` are *required*
 * fields on the wire (`api.schemas.ModelDisclosure`) precisely so probabilities
 * cannot be published without their caveat — the point of seam 7 is that the
 * caveat reaches a person, not just a response body. When `is_forecast` is
 * `false` — true for every draw shipped today, whose events have already been
 * played — this says so **above the numbers**, where they are read, and carries
 * the limitation prose in an open-by-default `<details>` rather than a tooltip
 * that can be missed.
 *
 * **Sorting is re-applied, not assumed.** The server already sorts by title
 * probability (ties by bracket position) and this sorts by the same key rather
 * than trusting array order — the same discipline `rounds.ts` applies to
 * `position`, and it costs one comparison per row.
 */

import { useEffect, useMemo, useState } from 'react';

import { getSimulate } from '../api/client';
import type { SimulationResponse } from '../api/types';
import ErrorPanel from './ErrorPanel';
import PlayerCard from './PlayerCard';
import { formatPercent, rankPlayers, survivalColumns } from './odds';
import styles from './titleOdds.module.css';
import panelStyles from './panel.module.css';

export interface TitleOddsProps {
  /** Registry id, as listed by `/tournaments`. */
  tournamentId: string;
}

type LoadState =
  | { status: 'loading' }
  | { status: 'ready'; simulation: SimulationResponse }
  | { status: 'error'; error: unknown };

// --------------------------------------------------------------------------
// Disclosure — required, rendered, not merely received.
// --------------------------------------------------------------------------

function Disclosure({ metadata }: { metadata: SimulationResponse['metadata'] }) {
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

function Footnote({ metadata, drawSize }: { metadata: SimulationResponse['metadata']; drawSize: number }) {
  return (
    <p className={styles.footnote}>
      <span data-meta="runs">{metadata.runs.toLocaleString()} simulations</span> ·{' '}
      <span data-meta="seed">seed {metadata.seed}</span> ·{' '}
      <span data-meta="data_through_year">data through {metadata.data_through_year}</span> ·{' '}
      <span data-meta="draw_size">{drawSize}-player draw</span> ·{' '}
      <span data-meta="mode">
        {metadata.mode}
        {metadata.mode === 'blend' && ` (w=${metadata.w})`}
      </span>{' '}
      · <span data-meta="estimator">{metadata.estimator_class}</span> ·{' '}
      <span data-meta="source">{metadata.source}</span>
    </p>
  );
}

// --------------------------------------------------------------------------
// The table.
// --------------------------------------------------------------------------

export default function TitleOdds({ tournamentId }: TitleOddsProps) {
  const [state, setState] = useState<LoadState>({ status: 'loading' });
  const [expanded, setExpanded] = useState<number | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setState({ status: 'loading' });
    setExpanded(null);

    getSimulate(tournamentId, undefined, { signal: controller.signal }).then(
      (simulation) => {
        if (!controller.signal.aborted) setState({ status: 'ready', simulation });
      },
      (error: unknown) => {
        // An abort surfaces as an ApiNetworkError; it is this effect tearing
        // down, not a failure worth showing.
        if (!controller.signal.aborted) setState({ status: 'error', error });
      },
    );

    return () => controller.abort();
  }, [tournamentId]);

  const simulation = state.status === 'ready' ? state.simulation : null;

  const rows = useMemo(
    () => (simulation === null ? [] : rankPlayers(simulation.players)),
    [simulation],
  );
  const columns = useMemo(
    () => (simulation === null ? [] : survivalColumns(simulation.round_labels)),
    [simulation],
  );

  if (state.status === 'loading') {
    return (
      <p className={panelStyles.panel} role="status">
        Loading the {tournamentId} simulation…
      </p>
    );
  }

  if (state.status === 'error') {
    return <ErrorPanel error={state.error} tournamentId={tournamentId} />;
  }

  const { simulation: sim } = state;

  return (
    <section className={styles.odds} aria-label={`${sim.name} title probabilities`}>
      <header className={styles.head}>
        <h2 className={styles.title}>{sim.name}</h2>
        <p className={styles.subtitle}>
          {sim.draw_size}-player draw · {sim.surface} · best of {sim.best_of}
        </p>
      </header>

      <Disclosure metadata={sim.metadata} />

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <caption className={styles.caption}>
            Title and round-survival probabilities, most likely champion first.
            {sim.count < sim.draw_size && ` Showing ${sim.count} of ${sim.draw_size} entrants.`}
          </caption>
          <thead>
            <tr>
              <th scope="col" className={styles.numeric}>
                #
              </th>
              <th scope="col">Seed</th>
              <th scope="col">Player</th>
              {columns.map((label) => (
                <th scope="col" key={label} className={styles.numeric} data-column={label}>
                  {label}
                </th>
              ))}
              <th scope="col" className={styles.numeric} data-column="title">
                Title
              </th>
              <th scope="col" className={styles.numeric} data-column="expected_rounds_won">
                E[W]
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((player, index) => {
              const isOpen = expanded === player.position;
              return [
                <tr key={player.position} data-position={player.position}>
                  <td className={styles.numeric}>{index + 1}</td>
                  <td data-cell="seed">{player.seed === null ? '' : player.seed}</td>
                  <td>
                    <button
                      type="button"
                      className={styles.rowButton}
                      aria-expanded={isOpen}
                      onClick={() => setExpanded(isOpen ? null : player.position)}
                    >
                      {player.player}
                    </button>
                  </td>
                  {columns.map((label) => (
                    <td key={label} className={styles.numeric} data-cell={`reach_${label}`}>
                      {/* A label the response does not carry for this player is
                          a server bug, not a 0 — but render something readable. */}
                      {player.p_reach[label] === undefined
                        ? '—'
                        : formatPercent(player.p_reach[label])}
                    </td>
                  ))}
                  <td className={styles.numeric} data-cell="p_title">
                    {formatPercent(player.p_title)}
                  </td>
                  <td className={styles.numeric} data-cell="expected_rounds_won">
                    {player.expected_rounds_won.toFixed(2)}
                  </td>
                </tr>,
                isOpen ? (
                  <tr key={`${player.position}-detail`} className={styles.detailRow}>
                    <td colSpan={columns.length + 5}>
                      <PlayerCard
                        player={player}
                        surface={sim.surface}
                        roundLabels={columns}
                        tournamentId={tournamentId}
                      />
                    </td>
                  </tr>
                ) : null,
              ];
            })}
          </tbody>
        </table>
      </div>

      <Footnote metadata={sim.metadata} drawSize={sim.draw_size} />
    </section>
  );
}
