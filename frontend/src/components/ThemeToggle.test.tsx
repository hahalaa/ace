// @vitest-environment jsdom
// The toggle control: a real labelled keyboard-operable <button>; clicking flips the painted theme and persists it; its accessible name and pressed-state track the current theme.

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
    // A real <button>, not a div with a click handler, is focusable and Enter/Space-operable by the platform, with no tabindex hack.
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

  // Fresh-mount / second-page-load correctness: these cover initial-STATE on mount, not click-response.

  it('a genuine first visit (empty storage, no attribute) mounts on dark', () => {
    // Nothing stored, no attribute painted: the toggle falls back to the dark DEFAULT.
    render(<ThemeToggle />);
    const button = screen.getByRole('button');
    expect(button.getAttribute('aria-label')).toBe('Switch to light theme');
    expect(button.getAttribute('aria-pressed')).toBe('false');
  });

  it('on a fresh mount, the label matches the PAINTED theme, ignoring a disagreeing store', () => {
    // The deployed-bug state: no attribute painted but 'light' stored. A correct toggle reflects the pixels (dark), not the store.
    window.localStorage.setItem(THEME_STORAGE_KEY, 'light');
    // deliberately no data-theme attribute
    render(<ThemeToggle />);
    const button = screen.getByRole('button');
    expect(button.getAttribute('aria-label')).toBe('Switch to light theme');
    expect(button.getAttribute('aria-pressed')).toBe('false');
  });

  it('on a correct second load (pre-paint stamped light), mounts showing light', () => {
    // Once the pre-paint script runs it stamps the attribute before React mounts, so the toggle reflects light immediately.
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
