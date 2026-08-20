// @vitest-environment jsdom
/**
 * The "Recent simulations" dashboard panel: empty renders nothing, a populated
 * history renders scannable links back to the reproduced runs, the ephemeral
 * upload label shows for that case, and clearing empties the panel.
 */

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { logSimulation } from '../history/store';
import RecentSimulations from './RecentSimulations';

afterEach(() => {
  cleanup();
  window.localStorage.clear();
});

describe('RecentSimulations', () => {
  it('renders nothing when there is no history', () => {
    const { container } = render(<RecentSimulations />);
    expect(container.firstChild).toBeNull();
  });

  it('renders each logged result as a link that reproduces it', () => {
    logSimulation({
      kind: 'match',
      url: '?view=match&a=A&b=B&surface=Hard&bo=5&seed=7',
      summary: 'A vs B, Hard, Best of 5',
      ephemeral: false,
    });
    render(<RecentSimulations />);

    const link = screen.getByRole('link', { name: /A vs B, Hard, Best of 5/ });
    expect(link.getAttribute('href')).toBe('?view=match&a=A&b=B&surface=Hard&bo=5&seed=7');
    // Proper list semantics.
    expect(screen.getByRole('list')).not.toBeNull();
    expect(screen.getAllByRole('listitem')).toHaveLength(1);
  });

  it('labels an ephemeral uploaded-draw entry as possibly no longer playable', () => {
    logSimulation({
      kind: 'storybook',
      url: '?tournament=upload-x&view=storybook&seed=3',
      summary: 'My Draw, champion: Someone',
      ephemeral: true,
    });
    render(<RecentSimulations />);

    expect(document.querySelector('[data-ephemeral="true"]')?.textContent).toBe(
      'may no longer be playable',
    );
    // The warning is part of the link's accessible name, not just a visual tag.
    const link = screen.getByRole('link', { name: /may no longer be playable/ });
    expect(link).not.toBeNull();
  });

  it('does not add the ephemeral label to a curated entry', () => {
    logSimulation({
      kind: 'storybook',
      url: '?tournament=ausopen_2026_atp_full&view=storybook&seed=42',
      summary: 'Australian Open 2026, champion: Alcaraz',
      ephemeral: false,
    });
    render(<RecentSimulations />);
    expect(document.querySelector('[data-ephemeral="true"]')).toBeNull();
  });

  it('clears the history and disappears', () => {
    logSimulation({
      kind: 'match',
      url: '?view=match&seed=1',
      summary: 'A vs B, Hard, Best of 5',
      ephemeral: false,
    });
    const { container } = render(<RecentSimulations />);
    expect(screen.getByRole('list')).not.toBeNull();

    fireEvent.click(screen.getByRole('button', { name: 'Clear history' }));

    // Panel is gone and nothing is left in storage.
    expect(container.firstChild).toBeNull();
    expect(window.localStorage.getItem('ace.recent-simulations.v1')).toBeNull();
  });
});
