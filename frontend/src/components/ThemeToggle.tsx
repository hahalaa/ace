/**
 * The theme control in the masthead.
 *
 * A single native `<button>`, not the ball mark, not the wordmark, so it is
 * keyboard-operable and in the tab order for free, and reads unambiguously as a
 * control rather than an easter egg on the brand. Its accessible name states the
 * action and changes with the theme ("Switch to light theme" / "…dark theme"),
 * which is what a screen reader announces; the sun/moon glyph is `aria-hidden`
 * decoration on top of that name. `aria-pressed` reports whether light is the
 * active (pressed) state, so the control also announces as a toggle.
 *
 * State lives here, seeded from whatever the pre-paint script already stamped on
 * <html> (see theme.ts). Flipping it writes through to the DOM attribute and
 * localStorage in one step, the URL is deliberately never touched (theme.ts
 * explains why theme is not shareable link state).
 */

import { useCallback, useState } from 'react';

import { currentTheme, otherTheme, setTheme, type Theme } from '../theme';
import styles from './themeToggle.module.css';

/** Reads the painted theme, returns it and a flip callback that persists. */
function useTheme(): [Theme, () => void] {
  const [theme, setThemeState] = useState<Theme>(currentTheme);

  const toggle = useCallback(() => {
    setThemeState((prev) => {
      const next = otherTheme(prev);
      setTheme(next);
      return next;
    });
  }, []);

  return [theme, toggle];
}

/** The sun (shown in dark mode, i.e. tap for light) or moon (shown in light). */
function ThemeGlyph({ theme }: { theme: Theme }) {
  // Drawn with currentColor so it inherits the button's themed ink. Decorative:
  // the button's aria-label carries the meaning.
  if (theme === 'dark') {
    // A sun, tapping moves toward the light theme.
    return (
      <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" focusable="false">
        <circle cx="12" cy="12" r="4.2" fill="currentColor" />
        <g stroke="currentColor" strokeWidth="1.7" strokeLinecap="round">
          <line x1="12" y1="2.5" x2="12" y2="5.4" />
          <line x1="12" y1="18.6" x2="12" y2="21.5" />
          <line x1="2.5" y1="12" x2="5.4" y2="12" />
          <line x1="18.6" y1="12" x2="21.5" y2="12" />
          <line x1="5.3" y1="5.3" x2="7.4" y2="7.4" />
          <line x1="16.6" y1="16.6" x2="18.7" y2="18.7" />
          <line x1="5.3" y1="18.7" x2="7.4" y2="16.6" />
          <line x1="16.6" y1="7.4" x2="18.7" y2="5.3" />
        </g>
      </svg>
    );
  }
  // A crescent moon, tapping moves toward the dark theme.
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" focusable="false">
      <path
        d="M20 14.2A8 8 0 1 1 9.8 4 6.4 6.4 0 0 0 20 14.2Z"
        fill="currentColor"
      />
    </svg>
  );
}

export default function ThemeToggle() {
  const [theme, toggle] = useTheme();
  const target = otherTheme(theme);
  const label = `Switch to ${target} theme`;

  return (
    <button
      type="button"
      className={styles.toggle}
      onClick={toggle}
      aria-label={label}
      aria-pressed={theme === 'light'}
      title={label}
    >
      <ThemeGlyph theme={theme} />
    </button>
  );
}
