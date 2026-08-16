/**
 * Theme preference: the one piece of app state that is deliberately NOT in the
 * URL.
 *
 * `?view=/?tournament=/?seed=` (App.tsx) exist so a *simulation* is shareable —
 * a link must reproduce the same draw and the same seed for whoever opens it,
 * whatever display they prefer. Theme is a property of the viewer, not of the
 * thing being shared: baking it into the link would mean a colleague opening
 * your bracket inherits your light/dark choice along with your seed, which is
 * not what the link is for. So it lives in localStorage instead, and the URL
 * stays purely about the tournament.
 *
 * First visit with nothing stored falls back to the OS `prefers-color-scheme`.
 * The dark theme is the default identity, so anything other than an explicit OS
 * "light" resolves to dark (a `no-preference` OS included).
 *
 * These functions are intentionally free of React: the same resolution runs in
 * the inline pre-paint script in index.html (which cannot import a module), so
 * the logic is kept small enough to mirror there by hand, and unit-testable on
 * its own. The React seam is `useTheme` in components/ThemeToggle.tsx.
 */

export type Theme = 'light' | 'dark';

/** localStorage key. Namespaced so it cannot collide with anything else. */
export const THEME_STORAGE_KEY = 'ace-theme';

/** The theme with no attribute set — the shipped dark identity. */
export const DEFAULT_THEME: Theme = 'dark';

function isTheme(value: unknown): value is Theme {
  return value === 'light' || value === 'dark';
}

/**
 * The stored preference, or `null` if none is stored (or storage is
 * unavailable, e.g. private mode with cookies blocked — never throw over it).
 */
export function storedTheme(): Theme | null {
  try {
    const value = window.localStorage.getItem(THEME_STORAGE_KEY);
    return isTheme(value) ? value : null;
  } catch {
    return null;
  }
}

/** The OS preference. Only an explicit "light" is light; everything else dark. */
export function systemTheme(): Theme {
  const prefersLight = window.matchMedia?.('(prefers-color-scheme: light)').matches ?? false;
  return prefersLight ? 'light' : DEFAULT_THEME;
}

/** Stored preference wins; otherwise fall back to the OS. */
export function resolveTheme(): Theme {
  return storedTheme() ?? systemTheme();
}

/** Paint the theme by stamping <html data-theme>. The CSS keys off this. */
export function applyTheme(theme: Theme): void {
  document.documentElement.setAttribute('data-theme', theme);
}

/** Persist the explicit choice. Swallows storage failures like storedTheme. */
export function persistTheme(theme: Theme): void {
  try {
    window.localStorage.setItem(THEME_STORAGE_KEY, theme);
  } catch {
    /* storage unavailable — the in-memory choice still applies for this visit */
  }
}

/** Apply and persist in one step — what the toggle calls on every flip. */
export function setTheme(theme: Theme): void {
  applyTheme(theme);
  persistTheme(theme);
}

/**
 * The theme currently PAINTED on the document, read back from the attribute the
 * pre-paint script set. This is the single source of truth for the toggle's
 * displayed state, and it is deliberately the DOM attribute — not a fresh
 * localStorage read.
 *
 * The distinction is what fixes the inverted-label bug: localStorage feeds the
 * pre-paint script, which decides the attribute; the attribute drives the CSS
 * paint; the toggle reads the attribute. One direction, one source per stage.
 * If the toggle instead re-read localStorage (as `resolveTheme()` does), it
 * could claim "light" while the page painted dark — precisely what happened when
 * the pre-paint script was CSP-blocked and never stamped the attribute.
 *
 * So when the attribute is somehow absent, we return `DEFAULT_THEME` — the theme
 * the CSS actually paints for a bare `:root` (see index.css) — not a resolved
 * preference. The label then still matches the pixels, and one click applies the
 * real choice correctly.
 */
export function currentTheme(): Theme {
  const attr = document.documentElement.getAttribute('data-theme');
  return isTheme(attr) ? attr : DEFAULT_THEME;
}

/** The other theme — the one a toggle flips to. */
export function otherTheme(theme: Theme): Theme {
  return theme === 'dark' ? 'light' : 'dark';
}
