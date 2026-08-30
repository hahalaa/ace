// The single-match view's URL state, pure, no React. /match/simulate returns a byte-identical body for the same players/surface/format/seed, so the full state lives in the URL.

import type { BestOf, Surface } from '../api/types';

/** `config.API_MATCH_SEED_MAX`, the server rejects anything above it. */
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

/** The form a URL pre-fills, with defaults for anything absent or invalid; always returns a usable form. */
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

/** A complete runnable query the URL carries, or `null`; missing any part pre-fills the form instead of firing a 422. */
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

/** A fresh seed in the server's range, never equal to `previous` (so "simulate again" gives a different sample). */
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
