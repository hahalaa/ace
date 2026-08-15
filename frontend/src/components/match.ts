/**
 * The single-match view's URL state — no React, on the `seed.ts`/`rounds.ts`
 * precedent that component files export only components.
 *
 * A single-match result is deep-linkable and reproducible the same way a
 * storybook run is: `/match/simulate` returns a byte-identical body for the same
 * players, surface, format and seed, so everything that decides *what runs* lives
 * in the URL (`?view=match&a=…&b=…&surface=…&bo=…&seed=…`) and is read here, not
 * in a `useState` initialiser that could roll a fresh number on a render.
 */

import type { BestOf, Surface } from '../api/types';

/** `config.API_MATCH_SEED_MAX` — the server rejects anything above it. */
export const MATCH_SEED_MAX = 2 ** 32 - 1;

export const SURFACES: Surface[] = ['Hard', 'Clay', 'Grass'];
export const BEST_OF_OPTIONS: BestOf[] = [3, 5];

/** Sensible defaults for a fresh form (no URL params yet). */
export const DEFAULT_SURFACE: Surface = 'Hard';
export const DEFAULT_BEST_OF: BestOf = 5;

/** The user-editable half of the state: who plays, where, over how many sets. */
export interface MatchForm {
  a: string;
  b: string;
  surface: Surface;
  bestOf: BestOf;
}

/** A form plus the seed that makes one run reproducible. */
export interface MatchQuery extends MatchForm {
  seed: number;
}

function isSurface(value: string): value is Surface {
  return (SURFACES as string[]).includes(value);
}

/**
 * The form a URL pre-fills, with defaults for anything absent or invalid.
 *
 * Always returns a usable form (never null): a bare `?view=match` opens the
 * picker with Hard / best-of-5 selected and both names blank, rather than an
 * empty screen.
 */
export function readMatchForm(search: string): MatchForm {
  const params = new URLSearchParams(search);
  const surfaceRaw = params.get('surface') ?? '';
  const boRaw = Number(params.get('bo'));
  return {
    a: (params.get('a') ?? '').trim(),
    b: (params.get('b') ?? '').trim(),
    surface: isSurface(surfaceRaw) ? surfaceRaw : DEFAULT_SURFACE,
    bestOf: boRaw === 3 ? 3 : DEFAULT_BEST_OF,
  };
}

/**
 * A complete, runnable query the URL carries, or `null` for "not enough to run".
 *
 * A shared result replays only when every part is present, the seed included —
 * both names, a valid surface and format, and a whole seed in range. A URL
 * missing any of them pre-fills the form (via {@link readMatchForm}) and waits
 * for the user to run it, rather than firing a request that would 422.
 */
export function readMatchQuery(search: string): MatchQuery | null {
  const params = new URLSearchParams(search);
  const form = readMatchForm(search);
  if (form.a === '' || form.b === '') return null;
  if (!params.has('surface') || !isSurface(params.get('surface') ?? '')) return null;
  if (Number(params.get('bo')) !== form.bestOf) return null;

  const seedRaw = params.get('seed');
  if (seedRaw === null || seedRaw.trim() === '') return null;
  const seed = Number(seedRaw);
  if (!Number.isInteger(seed) || seed < 0 || seed > MATCH_SEED_MAX) return null;

  return { ...form, seed };
}

/**
 * A fresh seed in the server's range, never equal to `previous`.
 *
 * "Simulate again" should give a different sample, so re-drawing the current
 * seed (which would return the identical aggregate) is excluded.
 */
export function nextMatchSeed(previous: number | null): number {
  for (;;) {
    const seed = Math.floor(Math.random() * (MATCH_SEED_MAX + 1));
    if (seed !== previous) return seed;
  }
}

/** The full shareable address for one run. Absolute, so it can be pasted anywhere. */
export function buildMatchUrl(query: MatchQuery): string {
  const params = new URLSearchParams();
  params.set('view', 'match');
  params.set('a', query.a);
  params.set('b', query.b);
  params.set('surface', query.surface);
  params.set('bo', String(query.bestOf));
  params.set('seed', String(query.seed));
  return `${window.location.origin}${window.location.pathname}?${params.toString()}`;
}
