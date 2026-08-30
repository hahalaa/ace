// @vitest-environment jsdom
// The dashboard landing: it fetches nothing, so these are pure render assertions that the two branches are links carrying URL state the shell can read.

import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import Dashboard from './Dashboard';

afterEach(cleanup);

const matchCard = () => document.querySelector('[data-card="match"]') as HTMLElement;
const tournamentCard = () => document.querySelector('[data-card="tournament"]') as HTMLElement;

describe('Dashboard', () => {
  it('offers a single-match path and a tournament path', () => {
    render(<Dashboard tournamentId="ausopen_2026_atp_full" />);

    const matchCard = document.querySelector('[data-card="match"]');
    expect(matchCard).not.toBeNull();
    expect(matchCard?.getAttribute('href')).toBe('?view=match');

    expect(document.querySelector('[data-card="tournament"]')).not.toBeNull();
  });

  it('points the tournament link at the given draw', () => {
    render(<Dashboard tournamentId="ausopen_2026_atp_full" />);

    const example = document.querySelector('[data-link="example"]');
    expect(example?.getAttribute('href')).toBe(
      '?tournament=ausopen_2026_atp_full&view=bracket',
    );
    expect(document.querySelector('[data-link="upload"]')?.getAttribute('href')).toBe(
      '?view=upload',
    );
  });

  it('offers an example matchup deep-link into the single-match view', () => {
    render(<Dashboard tournamentId="ausopen_2026_atp_full" />);

    const href = document.querySelector('[data-link="example-match"]')?.getAttribute('href') ?? '';
    const params = new URLSearchParams(href.replace(/^\?/, ''));
    expect(params.get('view')).toBe('match');
    expect(params.get('a')).toBeTruthy();
    expect(params.get('b')).toBeTruthy();
    expect(params.get('surface')).toBeTruthy();
    expect(params.get('bo')).toBeTruthy();
  });

  it('marks the single-match card as featured', () => {
    render(<Dashboard tournamentId="ausopen_2026_atp_full" />);
    // The featured card carries the "Featured" kicker.
    expect(screen.getByText('Featured')).toBeTruthy();
  });

  it('groups the two cards as a toolbar, each an accessibly-named control', () => {
    render(<Dashboard tournamentId="ausopen_2026_atp_full" />);

    const toolbar = screen.getByRole('toolbar', { name: 'Ways to start' });
    expect(toolbar).not.toBeNull();
    // Both cards are activatable controls with an accessible name.
    expect(matchCard().getAttribute('aria-label')).toBe('Simulate a single match');
    expect(tournamentCard().getAttribute('role')).toBe('link');
    expect(tournamentCard().getAttribute('aria-label')).toContain('Run a full draw');
  });

  it('starts with only the featured card in the tab order (roving tabindex)', () => {
    render(<Dashboard tournamentId="ausopen_2026_atp_full" />);

    expect(matchCard().tabIndex).toBe(0);
    expect(tournamentCard().tabIndex).toBe(-1);
  });

  it('moves focus and the tab stop between cards with the arrow keys', () => {
    render(<Dashboard tournamentId="ausopen_2026_atp_full" />);

    matchCard().focus();
    fireEvent.keyDown(matchCard(), { key: 'ArrowRight' });
    expect(document.activeElement).toBe(tournamentCard());
    expect(tournamentCard().tabIndex).toBe(0);
    expect(matchCard().tabIndex).toBe(-1);

    fireEvent.keyDown(tournamentCard(), { key: 'ArrowLeft' });
    expect(document.activeElement).toBe(matchCard());
    expect(matchCard().tabIndex).toBe(0);
    expect(tournamentCard().tabIndex).toBe(-1);
  });

  // Regression guard for "arrows do nothing": drive the toolbar with a genuine bubbling KeyboardEvent at document.activeElement and assert focus actually moves.
  it('moves real focus on a dispatched (not hand-called) arrow keydown', () => {
    render(<Dashboard tournamentId="ausopen_2026_atp_full" />);

    matchCard().focus();
    expect(document.activeElement).toBe(matchCard());

    // Each dispatch is wrapped in act() so React flushes the roving-index state between keypresses.
    act(() => {
      document.activeElement!.dispatchEvent(
        new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true, cancelable: true }),
      );
    });
    expect(document.activeElement).toBe(tournamentCard());

    act(() => {
      document.activeElement!.dispatchEvent(
        new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true, cancelable: true }),
      );
    });
    expect(document.activeElement).toBe(matchCard());
  });

  it('pins the accent to no card at rest, both share identical styling', () => {
    // Regression guard: neither card may carry a distinguishing accent class; the accent is a CSS :hover/:focus-visible state.
    render(<Dashboard tournamentId="ausopen_2026_atp_full" />);
    expect(matchCard().className).toBe(tournamentCard().className);
  });

  it('keeps the tournament card its own inner links, independently addressable', () => {
    render(<Dashboard tournamentId="ausopen_2026_atp_full" />);

    // The card's primary action is the example bracket; its two inner links stay reachable and carry their own destinations.
    expect(document.querySelector('[data-link="example"]')?.getAttribute('href')).toBe(
      '?tournament=ausopen_2026_atp_full&view=bracket',
    );
    expect(document.querySelector('[data-link="upload"]')?.getAttribute('href')).toBe(
      '?view=upload',
    );
  });
});
