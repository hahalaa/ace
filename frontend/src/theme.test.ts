// @vitest-environment jsdom
// Theme resolution rules, which public/theme-init.js mirrors by hand.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  applyTheme,
  currentTheme,
  DEFAULT_THEME,
  otherTheme,
  persistTheme,
  resolveTheme,
  setTheme,
  storedTheme,
  THEME_STORAGE_KEY,
} from './theme';

/** Point matchMedia at a fixed OS preference for a test. */
function stubSystem(prefersLight: boolean) {
  vi.stubGlobal(
    'matchMedia',
    vi.fn((query: string) => ({
      matches: query.includes('light') ? prefersLight : false,
      media: query,
    })),
  );
}

beforeEach(() => {
  window.localStorage.clear();
  document.documentElement.removeAttribute('data-theme');
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('storedTheme', () => {
  it('returns null when nothing is stored', () => {
    expect(storedTheme()).toBeNull();
  });

  it('returns a valid stored value', () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, 'light');
    expect(storedTheme()).toBe('light');
  });

  it('ignores a junk stored value rather than trusting it', () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, 'purple');
    expect(storedTheme()).toBeNull();
  });
});

describe('resolveTheme', () => {
  it('the shipped default is dark', () => {
    expect(DEFAULT_THEME).toBe('dark');
  });

  it('a stored choice is returned, and the OS never overrides it', () => {
    stubSystem(true); // OS wants light...
    window.localStorage.setItem(THEME_STORAGE_KEY, 'light'); // stored light -> light
    expect(resolveTheme()).toBe('light');
    window.localStorage.setItem(THEME_STORAGE_KEY, 'dark'); // stored dark -> dark
    expect(resolveTheme()).toBe('dark');
  });

  it('a genuine first visit falls back to DEFAULT dark, ignoring an OS light preference', () => {
    // prefers-color-scheme must not pull a first-time visitor to light.
    stubSystem(true);
    expect(storedTheme()).toBeNull();
    expect(resolveTheme()).toBe(DEFAULT_THEME);
    expect(resolveTheme()).toBe('dark');
  });
});

describe('applyTheme / persistTheme / setTheme', () => {
  it('applyTheme stamps the attribute but does not persist', () => {
    applyTheme('light');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBeNull();
  });

  it('persistTheme writes storage but does not paint', () => {
    persistTheme('light');
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('light');
    expect(document.documentElement.getAttribute('data-theme')).toBeNull();
  });

  it('setTheme does both', () => {
    setTheme('light');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('light');
  });
});

describe('currentTheme', () => {
  it('reads the painted attribute when present', () => {
    document.documentElement.setAttribute('data-theme', 'light');
    expect(currentTheme()).toBe('light');
  });

  it('falls back to the DEFAULT (what the CSS paints), not a preference read', () => {
    // Regression guard for the inverted-label bug: with no painted attribute the page shows dark.
    stubSystem(true);
    window.localStorage.setItem(THEME_STORAGE_KEY, 'light');
    expect(currentTheme()).toBe('dark');
  });
});

describe('otherTheme', () => {
  it('flips both ways', () => {
    expect(otherTheme('dark')).toBe('light');
    expect(otherTheme('light')).toBe('dark');
  });
});
