// The storybook seed and the shareable URL that carries it, pure, no React. /storybook returns a byte-identical body for the same id and seed, which is the basis of a shareable link.

/** `config.API_STORYBOOK_SEED_MAX`, the server rejects anything above it. */
export const SEED_MAX = 2 ** 32 - 1;

/** The seed a URL asks for, or `null`; `0` is legal, and an out-of-range value is treated as absent. */
export function readSeed(search: string): number | null {
  const raw = new URLSearchParams(search).get('seed');
  if (raw === null || raw.trim() === '') return null;
  const seed = Number(raw);
  if (!Number.isInteger(seed) || seed < 0 || seed > SEED_MAX) return null;
  return seed;
}

/** A fresh seed in the server's range, never equal to `previous` (so "re-run" produces a different story). */
export function nextSeed(previous: number | null): number {
  for (;;) {
    const seed = Math.floor(Math.random() * (SEED_MAX + 1));
    if (seed !== previous) return seed;
  }
}

/** The full shareable address for one run, built from the current query string with `view=storybook` set explicitly. */
export function buildShareUrl(tournamentId: string, seed: number): string {
  const params = new URLSearchParams(window.location.search);
  params.set('tournament', tournamentId);
  params.set('view', 'storybook');
  params.set('seed', String(seed));
  return `${window.location.origin}${window.location.pathname}?${params.toString()}`;
}
