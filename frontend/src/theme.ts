// Theme preference: stored in localStorage, deliberately not in the URL (it is a viewer property). React-free so public/theme-init.js can mirror resolveTheme() by hand. The OS prefers-color-scheme is not consulted.

export type Theme = 'light' | 'dark';

/** localStorage key. Namespaced so it cannot collide with anything else. */
export const THEME_STORAGE_KEY = 'ace-theme';

/** The theme with no attribute set, the shipped dark identity. */
export const DEFAULT_THEME: Theme = 'dark';

function isTheme(value: unknown): value is Theme {
  return value === 'light' || value === 'dark';
}

/** The stored preference, or `null` if none is stored or storage is unavailable (never throws). */
export function storedTheme(): Theme | null {
  try {
    const value = window.localStorage.getItem(THEME_STORAGE_KEY);
    return isTheme(value) ? value : null;
  } catch {
    return null;
  }
}

/** Stored preference wins; otherwise the shipped `DEFAULT_THEME` (the OS preference is not consulted). */
export function resolveTheme(): Theme {
  return storedTheme() ?? DEFAULT_THEME;
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
    /* storage unavailable, the in-memory choice still applies for this visit */
  }
}

/** Apply and persist in one step, what the toggle calls on every flip. */
export function setTheme(theme: Theme): void {
  applyTheme(theme);
  persistTheme(theme);
}

/** The theme currently painted on the document, read from `data-theme` (not localStorage) so the toggle label follows the pixels. */
export function currentTheme(): Theme {
  const attr = document.documentElement.getAttribute('data-theme');
  return isTheme(attr) ? attr : DEFAULT_THEME;
}

/** The other theme, the one a toggle flips to. */
export function otherTheme(theme: Theme): Theme {
  return theme === 'dark' ? 'light' : 'dark';
}
