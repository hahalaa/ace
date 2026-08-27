// @vitest-environment jsdom
/**
 * The toggle control itself: that it is a real, labelled, keyboard-operable
 * button, that clicking it flips the painted theme and persists the choice, and
 * that its accessible name and pressed-state track the current theme.
 *
 * Keyboard operability is asserted at the semantic level, the control is a
 * native <button>, which the platform makes focusable and Enter/Space-operable
 * for free. (The live keyboard walk-through is done in the browser; jsdom does
 * not turn key events into clicks.)
 */

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import ThemeToggle from './ThemeToggle';
import { THEME_STORAGE_KEY } from '../theme';

beforeEach(() => {
  window.localStorage.clear();
  document.documentElement.removeAttribute('data-theme');
  // Default OS to dark so an un-stamped mount resolves to dark.
  vi.stubGlobal(
    'matchMedia',
    vi.fn((query: string) => ({ matches: false, media: query })),
  );
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const themeAttr = () => document.documentElement.getAttribute('data-theme');

describe('ThemeToggle', () => {
  it('renders a native button in the tab order (keyboard-operable)', () => {
    render(<ThemeToggle />);
    const button = screen.getByRole('button');
    // A real <button>, not a div with a click handler, is focusable and
    // Enter/Space-operable by the platform, with no tabindex hack.
    expect(button.tagName).toBe('BUTTON');
    expect(button.getAttribute('type')).toBe('button');
    expect(button.getAttribute('tabindex')).toBeNull();
  });

  it('starts in dark and offers to switch to light', () => {
    document.documentElement.setAttribute('data-theme', 'dark');
    render(<ThemeToggle />);
    const button = screen.getByRole('button');
    expect(button.getAttribute('aria-label')).toBe('Switch to light theme');
    expect(button.getAttribute('aria-pressed')).toBe('false');
  });

  it('reflects a light start', () => {
    document.documentElement.setAttribute('data-theme', 'light');
    render(<ThemeToggle />);
    const button = screen.getByRole('button');
    expect(button.getAttribute('aria-label')).toBe('Switch to dark theme');
    expect(button.getAttribute('aria-pressed')).toBe('true');
  });

  // --- Fresh-mount / second-page-load correctness ---------------------------
  //
  // The gap that let the persistence bug ship twice: the prior tests only ever
  // clicked an already-mounted component (click-RESPONSE correctness). None
  // simulated a fresh load, a component mounting against pre-existing DOM/
  // storage state, which is what every real navigation is (each screen is a
  // full document load). These tests cover initial-STATE correctness on mount.

  it('a genuine first visit (empty storage, no attribute) mounts on dark', () => {
    // Nothing stored and no attribute painted: the toggle derives its state from
    // the painted attribute, falling back to the dark DEFAULT (the toggle never
    // consults the OS). A first-time visitor lands on dark, offered switch-to-light.
    render(<ThemeToggle />);
    const button = screen.getByRole('button');
    expect(button.getAttribute('aria-label')).toBe('Switch to light theme');
    expect(button.getAttribute('aria-pressed')).toBe('false');
  });

  it('on a fresh mount, the label matches the PAINTED theme, ignoring a disagreeing store', () => {
    // The exact deployed-bug state: the pre-paint script was CSP-blocked, so no
    // attribute is painted (page shows the dark CSS default), yet 'light' is in
    // storage from an earlier choice. A correct toggle reflects the PIXELS
    // (dark), not the stored preference. The old currentTheme() fell back to the
    // stored value and mounted claiming light, an inverted label over a dark
    // page. This test fails on that old code and passes on the fix.
    window.localStorage.setItem(THEME_STORAGE_KEY, 'light');
    // deliberately NO data-theme attribute, the pre-paint script did not run
    render(<ThemeToggle />);
    const button = screen.getByRole('button');
    expect(button.getAttribute('aria-label')).toBe('Switch to light theme');
    expect(button.getAttribute('aria-pressed')).toBe('false');
  });

  it('on a correct second load (pre-paint stamped light), mounts showing light', () => {
    // What a real navigation looks like once the pre-paint script runs under CSP:
    // it reads storage and stamps the attribute before React mounts. The toggle
    // then reflects light immediately on that fresh mount, no click needed.
    window.localStorage.setItem(THEME_STORAGE_KEY, 'light');
    document.documentElement.setAttribute('data-theme', 'light');
    render(<ThemeToggle />);
    const button = screen.getByRole('button');
    expect(button.getAttribute('aria-label')).toBe('Switch to dark theme');
    expect(button.getAttribute('aria-pressed')).toBe('true');
  });

  it('flips the painted theme, persists it, and relabels on click', () => {
    document.documentElement.setAttribute('data-theme', 'dark');
    render(<ThemeToggle />);
    const button = screen.getByRole('button');

    fireEvent.click(button);

    expect(themeAttr()).toBe('light');
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('light');
    expect(button.getAttribute('aria-label')).toBe('Switch to dark theme');
    expect(button.getAttribute('aria-pressed')).toBe('true');

    fireEvent.click(button);

    expect(themeAttr()).toBe('dark');
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark');
    expect(button.getAttribute('aria-label')).toBe('Switch to light theme');
  });
});
