// @vitest-environment jsdom
/**
 * Theme resolution: the rules that decide which register a visitor lands in,
 * and that a flip persists. These are the pure functions the pre-paint inline
 * script mirrors by hand, so pinning them here also pins the contract that
 * script has to keep.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import {
  applyTheme,
  currentTheme,
  otherTheme,
  persistTheme,
  resolveTheme,
  setTheme,
  storedTheme,
  systemTheme,
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

describe('systemTheme', () => {
  it('is light only when the OS explicitly prefers light', () => {
    stubSystem(true);
    expect(systemTheme()).toBe('light');
  });

  it('defaults to dark for a no-preference / dark OS', () => {
    stubSystem(false);
    expect(systemTheme()).toBe('dark');
  });
});

describe('resolveTheme', () => {
  it('prefers a stored choice over the OS', () => {
    stubSystem(true); // OS wants light...
    window.localStorage.setItem(THEME_STORAGE_KEY, 'dark'); // ...but the user chose dark
    expect(resolveTheme()).toBe('dark');
  });

  it('falls back to the OS when nothing is stored', () => {
    stubSystem(true);
    expect(resolveTheme()).toBe('light');
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

  it('resolves from scratch when no attribute is set', () => {
    stubSystem(false);
    expect(currentTheme()).toBe('dark');
  });
});

describe('otherTheme', () => {
  it('flips both ways', () => {
    expect(otherTheme('dark')).toBe('light');
    expect(otherTheme('light')).toBe('dark');
  });
});
