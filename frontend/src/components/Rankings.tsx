// The rankings board: a display-only Elo leaderboard (GET /rankings). A tab picks the track; a toggle shows long-inactive players, hidden by default.

import { useEffect, useMemo, useState } from 'react';

import { getRankings } from '../api/client';
import type { RankedPlayer, RankingsResponse } from '../api/types';
import ErrorPanel from './ErrorPanel';
import styles from './rankings.module.css';
import panelStyles from './panel.module.css';

type LoadState =
  | { status: 'loading' }
  | { status: 'ready'; rankings: RankingsResponse }
  | { status: 'error'; error: unknown };

/** Tab order, overall first, then surfaces in the order players think of them. */
const TRACK_ORDER = ['overall', 'Hard', 'Clay', 'Grass'] as const;
const TRACK_LABELS: Record<string, string> = {
  overall: 'Overall',
  Hard: 'Hard',
  Clay: 'Clay',
  Grass: 'Grass',
};

/** The season part of a `YYYY-MM-DD` date, for a compact "last seen" column. */
function seasonOf(isoDate: string): string {
  return isoDate.slice(0, 4);
}

export default function Rankings() {
  const [state, setState] = useState<LoadState>({ status: 'loading' });
  const [track, setTrack] = useState<string>('overall');
  const [showInactive, setShowInactive] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    setState({ status: 'loading' });

    getRankings({ signal: controller.signal }).then(
      (rankings) => {
        if (!controller.signal.aborted) setState({ status: 'ready', rankings });
      },
      (error: unknown) => {
        if (!controller.signal.aborted) setState({ status: 'error', error });
      },
    );

    return () => controller.abort();
  }, []);

  const rankings = state.status === 'ready' ? state.rankings : null;

  // The tracks that are actually present, in display order.
  const tabs = useMemo(() => {
    if (rankings === null) return [];
    const present = new Set(rankings.tracks.map((entry) => entry.track));
    return TRACK_ORDER.filter((name) => present.has(name));
  }, [rankings]);

  const activeTrack = useMemo(
    () => rankings?.tracks.find((entry) => entry.track === track) ?? null,
    [rankings, track],
  );

  const rows = useMemo<RankedPlayer[]>(() => {
    if (activeTrack === null) return [];
    return showInactive
      ? activeTrack.players
      : activeTrack.players.filter((player) => player.is_active);
  }, [activeTrack, showInactive]);

  if (state.status === 'loading') {
    return (
      <p className={panelStyles.panel} role="status">
        Loading the rankings…
      </p>
    );
  }

  if (state.status === 'error') {
    return <ErrorPanel error={state.error} />;
  }

  const inactiveCount = activeTrack
    ? activeTrack.players.filter((player) => !player.is_active).length
    : 0;

  return (
    <section className={styles.rankings} aria-label="Elo rankings">
      <header className={styles.head}>
        <h2 className={styles.title}>Rankings</h2>
        <p className={styles.note}>{state.rankings.metadata.note}</p>
      </header>

      <div className={styles.controls}>
        <div className={styles.tabs} role="tablist" aria-label="Ranking surface">
          {tabs.map((name) => (
            <button
              key={name}
              type="button"
              role="tab"
              aria-selected={name === track}
              className={styles.tab}
              onClick={() => setTrack(name)}
            >
              {TRACK_LABELS[name]}
            </button>
          ))}
        </div>

        {inactiveCount > 0 && (
          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={showInactive}
              onChange={(event) => setShowInactive(event.target.checked)}
            />
            Include inactive players
          </label>
        )}
      </div>

      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col" className={styles.numeric}>
                #
              </th>
              <th scope="col">Player</th>
              <th scope="col" className={styles.numeric}>
                Rating
              </th>
              <th scope="col" className={styles.numeric}>
                Matches
              </th>
              <th scope="col" className={styles.numeric}>
                Last seen
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((player, index) => (
              <tr
                key={player.player_id}
                data-active={player.is_active}
                data-player={player.player_id}
              >
                <td className={styles.numeric}>{index + 1}</td>
                <td>
                  {player.player_name}
                  {!player.is_active && <span className={styles.tag}>inactive</span>}
                </td>
                <td className={styles.numeric} data-cell="rating">
                  {Math.round(player.rating)}
                </td>
                <td className={styles.numeric}>{player.matches}</td>
                <td className={styles.numeric}>{seasonOf(player.last_played)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
