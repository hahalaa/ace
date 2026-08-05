/**
 * The storybook seed and the URL that carries it — no React, on the
 * `rounds.ts`/`errors.ts`/`overlay.ts` precedent that component files export
 * only components.
 *
 * `/storybook` returns a byte-identical body for the same id and seed (T3.4),
 * which is the entire basis of a shareable link. Everything that decides *which*
 * seed runs therefore lives here, in four lines that can be read at once and
 * tested without mounting anything — rather than scattered through an effect
 * where a stray `Math.random()` on a render path would be invisible.
 */

/** `config.API_STORYBOOK_SEED_MAX` — the server rejects anything above it. */
export const SEED_MAX = 2 ** 32 - 1;

/**
 * The seed a URL asks for, or `null` for "none asked".
 *
 * `0` is a legal seed, so this tests for absence rather than falsiness. A value
 * that is not a whole number in `[0, SEED_MAX]` is treated as absent: the run
 * then has to be started deliberately, instead of sending the server a seed it
 * will reject and rendering the 422 as if it were the draw's fault.
 */
export function readSeed(search: string): number | null {
  const raw = new URLSearchParams(search).get('seed');
  if (raw === null || raw.trim() === '') return null;
  const seed = Number(raw);
  if (!Number.isInteger(seed) || seed < 0 || seed > SEED_MAX) return null;
  return seed;
}

/**
 * A fresh seed in the server's range, never equal to `previous`.
 *
 * "Re-run" must produce a *different* story, so re-drawing the current seed —
 * which would re-fetch the identical bracket — is excluded rather than left to
 * chance.
 */
export function nextSeed(previous: number | null): number {
  for (;;) {
    const seed = Math.floor(Math.random() * (SEED_MAX + 1));
    if (seed !== previous) return seed;
  }
}

/**
 * The full shareable address for one run: tournament, view and seed.
 *
 * Built from the *current* query string so nothing else in the URL is dropped,
 * and absolute so it can be pasted anywhere. `view=storybook` is set explicitly:
 * a link is only reproducible if it lands on the screen that reads the seed.
 */
export function buildShareUrl(tournamentId: string, seed: number): string {
  const params = new URLSearchParams(window.location.search);
  params.set('tournament', tournamentId);
  params.set('view', 'storybook');
  params.set('seed', String(seed));
  return `${window.location.origin}${window.location.pathname}?${params.toString()}`;
}
