// The Monte Carlo board: entrants ranked by title probability. Columns are the draw's own round_labels.slice(1); p_reach is keyed by those labels, so cell lookup is by label, never position.

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
        // An abort surfaces as an ApiNetworkError; it is this effect tearing down.
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

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <caption className={styles.caption}>
            Sorted by most likely champion first.
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
                      {/* A missing label is a server bug, not a 0. */}
                      {player.p_reach[label] === undefined
                        ? 'n/a'
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
    </section>
  );
}
