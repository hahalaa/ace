/**
 * The single-match view: pick two players, a surface and a format, and get a
 * live-simulated win probability with the betting-relevant breakdowns a
 * tournament board cannot show, the set-score distribution and a total-games
 * (over/under) summary.
 *
 * **This is the featured path, and it simulates live.** `/match/simulate` runs a
 * Monte Carlo for one matchup per request, a deliberate, bounded exception to the
 * cache-only rule for the 128-draw board (a single match is a different cost
 * class). Same players, surface, format and seed return a byte-identical body, so
 * the run is reproducible, which is why the seed lives in the URL (`./match`
 * reads and writes it), never in a `useState` that could re-roll on a render.
 * Reloading a shared link replays the same numbers; "Simulate again" picks a new
 * seed and says so in the address bar.
 *
 * **Players are chosen, not typed blind.** Each picker searches `/players` (the
 * same resolver the server uses), so a user selects a player the model knows.
 * The server still refuses an unknown or ambiguous name with a 422; the UI just
 * makes that hard to hit.
 *
 * **The disclosure component is shared; its wording here is not.**
 * `MatchMetadata` carries the same required `ModelDisclosure` fields as every
 * other published number, so the same {@link Disclosure} renders a caveat
 * here too, a probability cannot be shown without it. The compact line's
 * text is overridden with {@link SINGLE_MATCH_DISCLAIMER} rather than the
 * server's `metadata.classifier_limitation`: the wire field
 * (`MATCH_CLASSIFIER_LIMITATION` in `api/schemas.py`) is already
 * single-match-specific, not shared with `/simulate` or `/storybook`, so this
 * override only changes what the browser renders, the response body, and
 * any non-browser consumer reading it, are untouched.
 */

import { useCallback, useEffect, useId, useRef, useState } from 'react';

import { isApiError, searchPlayers, simulateMatch } from '../api/client';
import { recordMatch } from '../history/store';
import type {
  BestOf,
  MatchSimulateResponse,
  PlayerSummary,
  Surface,
} from '../api/types';
import Disclosure from './Disclosure';
import type { MatchForm, MatchQuery } from './match';
import {
  BEST_OF_OPTIONS,
  SURFACES,
  buildMatchUrl,
  nextMatchSeed,
  readMatchForm,
  readMatchQuery,
} from './match';
import styles from './matchSim.module.css';
import panelStyles from './panel.module.css';

const SINGLE_MATCH_DISCLAIMER =
  'Disclaimer: this is a model estimate for a hypothetical matchup, not a betting tip.';

// --------------------------------------------------------------------------
// Player picker, an accessible combobox over /players.
// --------------------------------------------------------------------------

interface PlayerPickerProps {
  label: string;
  value: string;
  /** Called with the chosen (or typed) name whenever it changes. */
  onChange: (name: string) => void;
}

function PlayerPicker({ label, value, onChange }: PlayerPickerProps) {
  const listId = useId();
  const [text, setText] = useState(value);
  const [results, setResults] = useState<PlayerSummary[]>([]);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  const [loading, setLoading] = useState(false);
  const blurTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Keep the input text in step when the parent resets it (e.g. a shared URL
  // pre-fills both names).
  useEffect(() => {
    setText(value);
  }, [value]);

  // Debounced search. A query under two characters searches nothing (the server
  // would return ~hundreds of players for a single letter).
  useEffect(() => {
    const query = text.trim();
    if (query.length < 2) {
      setResults([]);
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    const timer = setTimeout(() => {
      setLoading(true);
      searchPlayers(query, { signal: controller.signal }).then(
        (response) => {
          if (!controller.signal.aborted) {
            setResults(response.players.slice(0, 8));
            setActive(-1);
            setLoading(false);
          }
        },
        () => {
          if (!controller.signal.aborted) {
            setResults([]);
            setLoading(false);
          }
        },
      );
    }, 200);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [text]);

  const choose = useCallback(
    (player: PlayerSummary) => {
      setText(player.name);
      onChange(player.name);
      setOpen(false);
      setActive(-1);
    },
    [onChange],
  );

  const onKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (!open || results.length === 0) {
      if (event.key === 'ArrowDown' && results.length > 0) setOpen(true);
      return;
    }
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      setActive((index) => (index + 1) % results.length);
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      setActive((index) => (index <= 0 ? results.length - 1 : index - 1));
    } else if (event.key === 'Enter' && active >= 0) {
      event.preventDefault();
      choose(results[active]);
    } else if (event.key === 'Escape') {
      setOpen(false);
    }
  };

  const showList = open && results.length > 0;

  return (
    <div className={styles.picker}>
      <label className={styles.pickerLabel} htmlFor={`${listId}-input`}>
        {label}
      </label>
      <input
        id={`${listId}-input`}
        className={styles.pickerInput}
        type="text"
        role="combobox"
        aria-expanded={showList}
        aria-controls={listId}
        aria-autocomplete="list"
        aria-activedescendant={
          showList && active >= 0 ? `${listId}-opt-${active}` : undefined
        }
        autoComplete="off"
        value={text}
        placeholder="Search a player…"
        onChange={(event) => {
          setText(event.target.value);
          onChange(event.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => {
          // Delay so a click on an option lands before the list unmounts.
          blurTimer.current = setTimeout(() => setOpen(false), 120);
        }}
        onKeyDown={onKeyDown}
      />
      {showList && (
        <ul className={styles.pickerList} id={listId} role="listbox" aria-label={label}>
          {results.map((player, index) => (
            <li
              key={player.player_id ?? player.name}
              id={`${listId}-opt-${index}`}
              role="option"
              aria-selected={index === active}
              className={styles.pickerOption}
              data-active={index === active}
              // onMouseDown, not onClick: it fires before the input's blur, so
              // the selection is not cancelled by the list unmounting first.
              onMouseDown={(event) => {
                event.preventDefault();
                if (blurTimer.current) clearTimeout(blurTimer.current);
                choose(player);
              }}
              onMouseEnter={() => setActive(index)}
            >
              {player.name}
            </li>
          ))}
        </ul>
      )}
      {/* Always rendered so the picker's height never changes when the status
          appears or clears, an empty line reserves the same space, keeping the
          two side-by-side pickers aligned. `aria-hidden` while empty so screen
          readers announce only the real "Searching…" status. */}
      <p
        className={styles.pickerHint}
        role="status"
        aria-hidden={!(loading && text.trim().length >= 2)}
      >
        {loading && text.trim().length >= 2 ? 'Searching…' : ' '}
      </p>
    </div>
  );
}

// --------------------------------------------------------------------------
// Results.
// --------------------------------------------------------------------------

function pct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function WinProbability({ result }: { result: MatchSimulateResponse }) {
  const aWins = result.win_prob_a >= result.win_prob_b;
  return (
    <section className={styles.winProb} aria-label="Win probability">
      <div className={styles.winRow} data-favourite={aWins}>
        <span className={styles.winName}>{result.player_a}</span>
        <span className={styles.winValue} data-player="a">
          {pct(result.win_prob_a)}
        </span>
      </div>
      <div
        className={styles.winBar}
        role="img"
        aria-label={`${result.player_a} ${pct(result.win_prob_a)}, ${result.player_b} ${pct(
          result.win_prob_b,
        )}`}
      >
        <span
          className={styles.winFill}
          style={{ width: `${result.win_prob_a * 100}%` }}
          data-player="a"
        />
      </div>
      <div className={styles.winRow} data-favourite={!aWins}>
        <span className={styles.winName}>{result.player_b}</span>
        <span className={styles.winValue} data-player="b">
          {pct(result.win_prob_b)}
        </span>
      </div>
    </section>
  );
}

function SetScores({ result }: { result: MatchSimulateResponse }) {
  const max = Math.max(...result.set_scores.map((o) => o.prob), 0.0001);
  return (
    <section className={styles.breakdown} aria-label="Set score distribution">
      <h3 className={styles.breakdownHead}>Set scores</h3>
      <table className={styles.setTable}>
        <thead>
          <tr>
            <th scope="col">Score</th>
            <th scope="col">Winner</th>
            <th scope="col" className={styles.numCol}>
              Chance
            </th>
          </tr>
        </thead>
        <tbody>
          {result.set_scores.map((o) => {
            const winner = o.sets_a > o.sets_b ? result.player_a : result.player_b;
            return (
              <tr key={`${o.sets_a}-${o.sets_b}`} data-score={`${o.sets_a}-${o.sets_b}`}>
                <th scope="row" className={styles.scoreCell}>
                  {o.sets_a}–{o.sets_b}
                </th>
                <td className={styles.winnerCell}>{winner}</td>
                <td className={styles.numCol}>
                  <span className={styles.miniBarWrap} aria-hidden="true">
                    <span
                      className={styles.miniBar}
                      style={{ width: `${(o.prob / max) * 100}%` }}
                    />
                  </span>
                  {pct(o.prob)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}

function TotalGames({ result }: { result: MatchSimulateResponse }) {
  const tg = result.total_games;
  return (
    <section className={styles.breakdown} aria-label="Total games">
      <h3 className={styles.breakdownHead}>Total games</h3>
      <p className={styles.totalLine}>
        <span className={styles.totalMean} data-meta="games_mean">
          {tg.mean.toFixed(1)}
        </span>
        <span className={styles.totalUnit}>games on average</span>
      </p>
      <p className={styles.totalSpread}>
        Typically <strong>{Math.round(tg.mean - tg.std)}</strong> to{' '}
        <strong>{Math.round(tg.mean + tg.std)}</strong> (one standard deviation),
        ranging from {tg.minimum} to {tg.maximum} across the simulations.
      </p>
    </section>
  );
}

// --------------------------------------------------------------------------
// The screen.
// --------------------------------------------------------------------------

type LoadState =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'ready'; result: MatchSimulateResponse }
  | { status: 'error'; error: unknown };

type CopyState = 'idle' | 'copied' | 'failed';

function errorMessage(error: unknown): string {
  if (isApiError(error)) {
    const detail = error.detail;
    if (detail && typeof detail === 'object' && 'message' in detail) {
      return String((detail as { message: unknown }).message);
    }
    return error.message;
  }
  return error instanceof Error ? error.message : String(error);
}

export default function MatchSim() {
  const [form, setForm] = useState<MatchForm>(() => readMatchForm(window.location.search));
  const [query, setQuery] = useState<MatchQuery | null>(() =>
    readMatchQuery(window.location.search),
  );
  const [state, setState] = useState<LoadState>({ status: 'idle' });
  const [copied, setCopied] = useState<CopyState>('idle');

  useEffect(() => {
    if (query === null) {
      setState({ status: 'idle' });
      return;
    }
    const controller = new AbortController();
    setState({ status: 'loading' });
    simulateMatch(
      {
        player_a: query.a,
        player_b: query.b,
        surface: query.surface,
        best_of: query.bestOf,
        seed: query.seed,
      },
      { signal: controller.signal },
    ).then(
      (result) => {
        if (!controller.signal.aborted) {
          setState({ status: 'ready', result });
          // Log the completed result to the browser-local history the dashboard
          // reads back. Fires once per fetch, so a re-render cannot re-log it.
          recordMatch(result);
        }
      },
      (error: unknown) => {
        if (!controller.signal.aborted) setState({ status: 'error', error });
      },
    );
    return () => {
      controller.abort();
    };
  }, [query]);

  const canRun = form.a.trim() !== '' && form.b.trim() !== '';

  const run = useCallback(
    (seed: number) => {
      if (form.a.trim() === '' || form.b.trim() === '') return;
      const next: MatchQuery = { ...form, a: form.a.trim(), b: form.b.trim(), seed };
      window.history.replaceState({}, '', buildMatchUrl(next));
      setCopied('idle');
      setQuery(next);
    },
    [form],
  );

  const share = useCallback(async () => {
    if (query === null) return;
    const url = buildMatchUrl(query);
    try {
      await navigator.clipboard.writeText(url);
      setCopied('copied');
    } catch {
      setCopied('failed');
    }
  }, [query]);

  const result = state.status === 'ready' ? state.result : null;

  return (
    <section className={styles.matchSim} aria-label="Single match simulation">
      <header className={styles.intro}>
        <h1 className={styles.title}>Simulate a single match</h1>
        <p className={styles.lede}>
          Pick two players, a surface and a format. The model plays the match out
          thousands of times and reports how often each player wins, the likely
          set scores, and how many games to expect.
        </p>
      </header>

      <form
        className={styles.controls}
        onSubmit={(event) => {
          event.preventDefault();
          run(nextMatchSeed(query?.seed ?? null));
        }}
      >
        <div className={styles.players}>
          <PlayerPicker
            label="Player A"
            value={form.a}
            onChange={(a) => setForm((f) => ({ ...f, a }))}
          />
          <span className={styles.versus} aria-hidden="true">
            <span className={`${styles.pickerLabel} ${styles.versusSpacer}`}>
              &nbsp;
            </span>
            <span className={styles.versusText}>vs</span>
            <span className={`${styles.pickerHint} ${styles.versusSpacer}`}>
              &nbsp;
            </span>
          </span>
          <PlayerPicker
            label="Player B"
            value={form.b}
            onChange={(b) => setForm((f) => ({ ...f, b }))}
          />
        </div>

        <div className={styles.options}>
          <label className={styles.optionField}>
            <span className={styles.optionLabel}>Surface</span>
            <select
              className={styles.select}
              value={form.surface}
              onChange={(event) =>
                setForm((f) => ({ ...f, surface: event.target.value as Surface }))
              }
            >
              {SURFACES.map((surface) => (
                <option key={surface} value={surface}>
                  {surface}
                </option>
              ))}
            </select>
          </label>
          <label className={styles.optionField}>
            <span className={styles.optionLabel}>Format</span>
            <select
              className={styles.select}
              value={form.bestOf}
              onChange={(event) =>
                setForm((f) => ({ ...f, bestOf: Number(event.target.value) as BestOf }))
              }
            >
              {BEST_OF_OPTIONS.map((bo) => (
                <option key={bo} value={bo}>
                  Best of {bo}
                </option>
              ))}
            </select>
          </label>
          <button
            type="submit"
            className={styles.primary}
            data-action={query === null ? 'simulate' : 'rerun'}
            disabled={!canRun || state.status === 'loading'}
          >
            {query === null ? 'Simulate match' : 'Simulate again'}
          </button>
        </div>
      </form>

      {result !== null && (
        <div className={styles.resultActions}>
          <button
            type="button"
            className={styles.secondary}
            data-action="share"
            onClick={() => void share()}
          >
            Share this result
          </button>
          {query !== null && (
            <span className={styles.seedTag} data-meta="url_seed">
              seed {query.seed}
            </span>
          )}
        </div>
      )}

      {copied !== 'idle' && query !== null && (
        <p className={styles.copy} data-copy={copied}>
          {copied === 'copied' ? (
            <>Link copied.</>
          ) : (
            <>
              Copy this link to replay this exact result:{' '}
              <code className={styles.copyUrl}>{buildMatchUrl(query)}</code>
            </>
          )}
        </p>
      )}

      {state.status === 'loading' && (
        <div className={panelStyles.panel} role="status">
          <p className={styles.loadingLine}>
            Simulating {query?.a} vs {query?.b}…
          </p>
        </div>
      )}

      {state.status === 'error' && (
        <p className={panelStyles.panel} data-state="error" role="alert">
          {errorMessage(state.error)}
        </p>
      )}

      {result !== null && (
        <div className={styles.results}>
          <p className={styles.matchup} data-meta="matchup">
            {result.player_a} vs {result.player_b} · {result.surface} · best of{' '}
            {result.best_of}
          </p>
          <WinProbability result={result} />
          <div className={styles.breakdowns}>
            <SetScores result={result} />
            <TotalGames result={result} />
          </div>
          <Disclosure metadata={result.metadata} variant="compact" text={SINGLE_MATCH_DISCLAIMER} />
        </div>
      )}
    </section>
  );
}
