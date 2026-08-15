// @vitest-environment jsdom
/**
 * The dashboard landing.
 *
 * It fetches nothing (that is the point of the restructure), so these are pure
 * render assertions: the two branches are present as links, the single-match
 * card is the featured one, and every link carries URL state the shell can read.
 */

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import Dashboard from './Dashboard';

afterEach(cleanup);

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
});
