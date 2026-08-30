// One entrant's detail, expanded from a TitleOdds row: skill from /players (fetched on expand), probabilities from the /simulate row the caller holds. n_serve_pts === 0 is the "surface baseline, not measured" signal.

import { useEffect, useState } from 'react';

import { searchPlayers } from '../api/client';
import type { PlayerSummary, SimulationPlayer, Surface } from '../api/types';
import ErrorPanel from './ErrorPanel';
import styles from './titleOdds.module.css';

export interface PlayerCardProps {
  /** The simulate row this card expands, the source of every probability shown. */
  player: SimulationPlayer;
  /** The tournament's surface, highlighted among the three. */
  surface: Surface;
  /** Round labels after the first, in draw order, the columns `p_reach` is keyed by. */
  roundLabels: string[];
  /** For the error panel's 404 wording, should the lookup fail outright. */
  tournamentId: string;
}

type SkillState =
  | { status: 'loading' }
  | { status: 'ready'; summary: PlayerSummary }
  | { status: 'ambiguous'; candidates: PlayerSummary[] }
  | { status: 'missing' }
  | { status: 'error'; error: unknown };

const SURFACES: Surface[] = ['Hard', 'Clay', 'Grass'];

/** A rate in `[0, 1]` as a percentage to one decimal place. */
function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/** Pick the entry that really is this player out of a multi-candidate answer. */
function exactMatch(candidates: PlayerSummary[], name: string): PlayerSummary | null {
  return candidates.find((candidate) => candidate.name === name) ?? null;
}

export default function PlayerCard({
  player,
  surface,
  roundLabels,
  tournamentId,
}: PlayerCardProps) {
  const [state, setState] = useState<SkillState>({ status: 'loading' });

  useEffect(() => {
    const controller = new AbortController();
    setState({ status: 'loading' });

    searchPlayers(player.player, { signal: controller.signal }).then(
      (response) => {
        if (controller.signal.aborted) return;
        const exact = exactMatch(response.players, player.player);
        if (exact !== null) {
          setState({ status: 'ready', summary: exact });
        } else if (response.count === 0) {
          setState({ status: 'missing' });
        } else {
          setState({ status: 'ambiguous', candidates: response.players });
        }
      },
      (error: unknown) => {
        if (!controller.signal.aborted) setState({ status: 'error', error });
      },
    );

    return () => controller.abort();
  }, [player.player]);

  return (
    <div className={styles.card} data-testid="player-card">
      <div className={styles.cardHead}>
        <h4 className={styles.cardName}>
          {player.player}
          {player.seed !== null && <span className={styles.cardSeed}>[{player.seed}]</span>}
        </h4>
        <p className={styles.cardMeta}>
          Bracket position {player.position}
          {player.player_id !== null && ` · ${player.player_id}`}
        </p>
      </div>

      <div className={styles.cardSection}>
        <h5 className={styles.cardSectionHeading}>Serve and return, by surface</h5>
        {state.status === 'loading' && (
          <p className={styles.cardNote} role="status">
            Loading skill…
          </p>
        )}
        {state.status === 'error' && (
          <ErrorPanel error={state.error} tournamentId={tournamentId} />
        )}
        {state.status === 'missing' && (
          <p className={styles.cardNote}>
            No skill record for this name. The draw resolved it, but `/players` did not.
          </p>
        )}
        {state.status === 'ambiguous' && (
          <p className={styles.cardNote}>
            The name "{player.player}" matched {state.candidates.length} players, none of them
            exactly. Skill is not shown rather than shown for the wrong person.
          </p>
        )}
        {state.status === 'ready' && (
          <table className={styles.skillTable}>
            <thead>
              <tr>
                <th scope="col">Surface</th>
                <th scope="col">Serve pts won</th>
                <th scope="col">Return pts won</th>
              </tr>
            </thead>
            <tbody>
              {SURFACES.map((name) => {
                const skill = state.summary.skills[name];
                // 0 raw points = the surface baseline, not a measurement of them.
                const measured = skill.n_serve_pts > 0 || skill.n_return_pts > 0;
                return (
                  <tr
                    key={name}
                    data-surface={name}
                    data-measured={measured}
                    aria-current={name === surface ? 'true' : undefined}
                  >
                    <th scope="row">
                      {name}
                      {name === surface && <span className={styles.thisSurface}> · this event</span>}
                    </th>
                    <td>{percent(skill.spw)}</td>
                    <td>{percent(skill.rpw)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        {state.status === 'ready' &&
          SURFACES.every((name) => state.summary.skills[name].n_serve_pts === 0) && (
            <p className={styles.cardNote}>
              No measured serve points on any surface, so every rate above is the surface baseline.
            </p>
          )}
      </div>

      <div className={styles.cardSection}>
        <h5 className={styles.cardSectionHeading}>This simulation</h5>
        <dl className={styles.stats}>
          <div>
            <dt>Title</dt>
            <dd data-stat="p_title">{percent(player.p_title)}</dd>
          </div>
          <div>
            <dt>E[rounds won]</dt>
            <dd data-stat="expected_rounds_won">{player.expected_rounds_won.toFixed(2)}</dd>
          </div>
          {roundLabels.map((label) => (
            <div key={label}>
              <dt>Reach {label}</dt>
              <dd data-stat={`p_reach_${label}`}>{percent(player.p_reach[label] ?? 0)}</dd>
            </div>
          ))}
        </dl>
      </div>
    </div>
  );
}
