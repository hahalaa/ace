// @vitest-environment jsdom
// The recent-simulations store: add, cap/eviction, de-dup, clear, and graceful storage failure.

import { afterEach, describe, expect, it, vi } from 'vitest';

import type { MatchSimulateResponse, StorybookResponse } from '../api/types';
import {
  HISTORY_CAP,
  clearHistory,
  logSimulation,
  readHistory,
  recordMatch,
  recordStorybook,
  type NewEntry,
} from './store';

afterEach(() => {
  window.localStorage.clear();
  vi.restoreAllMocks();
});

const sample = (n: number): NewEntry => ({
  kind: 'match',
  url: `?view=match&seed=${n}`,
  summary: `run ${n}`,
  ephemeral: false,
});

describe('logSimulation / readHistory', () => {
  it('records a completed result at the front, newest first', () => {
    logSimulation(sample(1));
    logSimulation(sample(2));

    const history = readHistory();
    expect(history.map((e) => e.summary)).toEqual(['run 2', 'run 1']);
    expect(history[0].timestamp).toBeTypeOf('number');
    expect(history[0].id).toBeTruthy();
  });

  it('caps the history and evicts the oldest first', () => {
    for (let n = 0; n < HISTORY_CAP + 5; n += 1) logSimulation(sample(n));

    const history = readHistory();
    expect(history).toHaveLength(HISTORY_CAP);
    // Newest is the last logged; the five oldest fell off the end.
    expect(history[0].summary).toBe(`run ${HISTORY_CAP + 4}`);
    expect(history.some((e) => e.summary === 'run 0')).toBe(false);
  });

  it('moves a re-logged identical result to the front instead of duplicating it', () => {
    logSimulation(sample(1));
    logSimulation(sample(2));
    logSimulation(sample(1)); // same url as the first

    const history = readHistory();
    expect(history).toHaveLength(2);
    expect(history.map((e) => e.summary)).toEqual(['run 1', 'run 2']);
  });

  it('ignores corrupt or foreign stored data', () => {
    window.localStorage.setItem('ace.recent-simulations.v1', '{not json');
    expect(readHistory()).toEqual([]);

    window.localStorage.setItem(
      'ace.recent-simulations.v1',
      JSON.stringify([{ nonsense: true }, sampleEntryLike()]),
    );
    // The one well-formed entry survives; the junk object is filtered out.
    expect(readHistory()).toHaveLength(1);
  });
});

/** A fully-formed stored entry, for the corrupt-data filter test. */
function sampleEntryLike() {
  return {
    id: 'x',
    kind: 'storybook',
    url: '?view=storybook&seed=3',
    summary: 'ok',
    timestamp: 1,
    ephemeral: false,
  };
}

describe('clearHistory', () => {
  it('forgets every logged result', () => {
    logSimulation(sample(1));
    logSimulation(sample(2));
    clearHistory();
    expect(readHistory()).toEqual([]);
  });
});

describe('graceful storage failure', () => {
  it('does not throw when a write fails (quota / private mode)', () => {
    vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new DOMException('QuotaExceededError');
    });

    // The whole point: logging must not surface an error to the caller.
    expect(() => logSimulation(sample(1))).not.toThrow();
  });

  it('reads an empty history when the store throws on access', () => {
    vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
      throw new DOMException('SecurityError');
    });
    expect(readHistory()).toEqual([]);
  });

  it('does not throw when clearing a store that throws on removal', () => {
    vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(() => {
      throw new DOMException('SecurityError');
    });
    expect(() => clearHistory()).not.toThrow();
  });
});

describe('recordMatch / recordStorybook', () => {
  it('records a single-match result with a reproducing url and readable summary', () => {
    const result = {
      player_a: 'Jannik Sinner',
      player_b: 'Tommy Paul',
      surface: 'Hard',
      best_of: 5,
      metadata: { seed: 7 },
    } as MatchSimulateResponse;

    recordMatch(result);
    const [entry] = readHistory();
    expect(entry.kind).toBe('match');
    expect(entry.summary).toBe('Jannik Sinner vs Tommy Paul, Hard, Best of 5');
    expect(entry.ephemeral).toBe(false);
    const params = new URLSearchParams(entry.url.replace(/^\?/, ''));
    expect(params.get('view')).toBe('match');
    expect(params.get('a')).toBe('Jannik Sinner');
    expect(params.get('b')).toBe('Tommy Paul');
    expect(params.get('surface')).toBe('Hard');
    expect(params.get('bo')).toBe('5');
    expect(params.get('seed')).toBe('7');
  });

  it('flags an uploaded-draw storybook as ephemeral, a curated one as not', () => {
    const curated = {
      tournament_id: 'ausopen_2026_atp_full',
      name: 'Australian Open 2026',
      champion: { player: 'Carlos Alcaraz' },
      metadata: { seed: 42, content_source: 'curated' },
    } as StorybookResponse;
    const uploaded = {
      tournament_id: 'upload-abc123',
      name: 'My Draw',
      champion: { player: 'Someone Else' },
      metadata: { seed: 9, content_source: 'user_upload' },
    } as StorybookResponse;

    recordStorybook(uploaded);
    recordStorybook(curated);
    const history = readHistory();

    const curatedEntry = history.find((e) => e.summary.startsWith('Australian Open 2026'));
    const uploadedEntry = history.find((e) => e.summary.startsWith('My Draw'));
    expect(curatedEntry?.ephemeral).toBe(false);
    expect(curatedEntry?.summary).toBe('Australian Open 2026, champion: Carlos Alcaraz');
    expect(uploadedEntry?.ephemeral).toBe(true);

    const params = new URLSearchParams((curatedEntry?.url ?? '').replace(/^\?/, ''));
    expect(params.get('view')).toBe('storybook');
    expect(params.get('tournament')).toBe('ausopen_2026_atp_full');
    expect(params.get('seed')).toBe('42');
  });
});
