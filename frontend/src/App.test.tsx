// @vitest-environment jsdom
/**
 * The shell's dispatch, both axes read from the query string.
 *
 * **These are deep-link tests, not navigation tests.** `App` reads
 * `window.location.search` during render and holds no state, so the thing worth
 * pinning is that a *fresh load* of a URL lands on the right screen — which is
 * what a shared link does. Each case rewrites the URL with `history.replaceState`
 * and mounts the component from scratch, exactly as a page load would.
 *
 * The restructure this suite guards: the app now lands on a **dashboard**, not a
 * bracket, and the nav is two tiers — a primary branch row (home / single match /
 * tournament) plus a secondary view row that appears only inside the tournament
 * branch. Which screen mounted is asserted from the screen's own markup, not only
 * from the nav, so a mutation that styled the nav right while rendering the wrong
 * screen still fails.
 *
 * The client is fully stubbed with promises that never settle: the mounted
 * screen is identifiable without any of them resolving.
 */

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import App from './App';
import {
  getBracket,
  getSimulate,
  getStorybook,
  searchPlayers,
  simulateMatch,
} from './api/client';

vi.mock('./api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./api/client')>();
  return {
    ...actual,
    getBracket: vi.fn(),
    getSimulate: vi.fn(),
    getStorybook: vi.fn(),
    searchPlayers: vi.fn(),
    simulateMatch: vi.fn(),
  };
});

const getBracketMock = vi.mocked(getBracket);
const getSimulateMock = vi.mocked(getSimulate);
const getStorybookMock = vi.mocked(getStorybook);
const searchPlayersMock = vi.mocked(searchPlayers);
const simulateMatchMock = vi.mocked(simulateMatch);

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  window.history.replaceState({}, '', '/');
});

const DEFAULT_ID = 'ausopen_2026_atp_full';

/** Load `url` from scratch, as a browser would, and mount the shell. */
function loadUrl(url: string) {
  getBracketMock.mockReturnValue(new Promise(() => {}));
  getSimulateMock.mockReturnValue(new Promise(() => {}));
  getStorybookMock.mockReturnValue(new Promise(() => {}));
  searchPlayersMock.mockReturnValue(new Promise(() => {}));
  simulateMatchMock.mockReturnValue(new Promise(() => {}));
  window.history.replaceState({}, '', url);
  return render(<App />);
}

type Screen = 'dashboard' | 'match' | 'bracket' | 'odds' | 'storybook' | 'upload';

/** Which screen actually mounted, read off the screen's own markup. */
function mountedScreen(): Screen {
  if (document.querySelector('[data-card="match"]') !== null) return 'dashboard';
  if (document.querySelector('section[aria-label="Single match simulation"]') !== null) {
    return 'match';
  }
  if (document.querySelector('section[aria-label="Upload a draw"]') !== null) return 'upload';
  // Storybook renders a run control above the bracket it draws over.
  if (document.querySelector('[data-action="simulate"], [data-action="rerun"]') !== null) {
    return 'storybook';
  }
  const status = screen.getByRole('status').textContent ?? '';
  if (status.includes('bracket')) return 'bracket';
  if (status.includes('simulation')) return 'odds';
  throw new Error(`No screen identifiable from status: ${status}`);
}

/** The primary nav link marked as the current page. */
function currentPrimary(): string {
  return (
    document
      .querySelector('nav[aria-label="Sections"] a[aria-current="page"]')
      ?.textContent?.trim() ?? ''
  );
}

const tournamentNav = () => document.querySelector('nav[aria-label="Tournament view"]');

// --------------------------------------------------------------------------
// The landing is the dashboard, not a bracket.
// --------------------------------------------------------------------------
describe('landing', () => {
  it('opens on the dashboard when the URL names no view, loading no draw', () => {
    loadUrl('/');

    expect(mountedScreen()).toBe('dashboard');
    expect(currentPrimary()).toBe('Home');
    // The whole point of the restructure: nothing is fetched on landing.
    expect(getBracketMock).not.toHaveBeenCalled();
    expect(getSimulateMock).not.toHaveBeenCalled();
    expect(getStorybookMock).not.toHaveBeenCalled();
  });

  it('falls back to the dashboard for a view value it does not know', () => {
    loadUrl('/?view=nonsense');

    expect(mountedScreen()).toBe('dashboard');
    expect(currentPrimary()).toBe('Home');
  });

  it('shows no tournament sub-nav outside the tournament branch', () => {
    loadUrl('/');
    expect(tournamentNav()).toBeNull();
  });
});

// --------------------------------------------------------------------------
// The single-match view — the featured branch.
// --------------------------------------------------------------------------
describe('single match', () => {
  it('loads the single-match view from ?view=match without fetching', () => {
    loadUrl('/?view=match');

    expect(mountedScreen()).toBe('match');
    expect(currentPrimary()).toBe('Single match');
    // A bare match URL only pre-fills the form; it does not simulate.
    expect(simulateMatchMock).not.toHaveBeenCalled();
    expect(tournamentNav()).toBeNull();
  });

  it('replays a complete match URL by firing one simulation', () => {
    loadUrl('/?view=match&a=Carlos%20Alcaraz&b=Jannik%20Sinner&surface=Clay&bo=5&seed=7');

    expect(mountedScreen()).toBe('match');
    expect(simulateMatchMock).toHaveBeenCalledTimes(1);
    expect(simulateMatchMock.mock.calls[0][0]).toMatchObject({
      player_a: 'Carlos Alcaraz',
      player_b: 'Jannik Sinner',
      surface: 'Clay',
      best_of: 5,
      seed: 7,
    });
  });
});

// --------------------------------------------------------------------------
// The tournament branch — every existing view stays reachable.
// --------------------------------------------------------------------------
describe('tournament branch', () => {
  it('loads the bracket from ?view=bracket and shows the sub-nav', () => {
    loadUrl('/?view=bracket');

    expect(mountedScreen()).toBe('bracket');
    expect(currentPrimary()).toBe('Tournament');
    expect(tournamentNav()).not.toBeNull();
    expect(getBracketMock).toHaveBeenCalledTimes(1);
    expect(getBracketMock.mock.calls[0][0]).toBe(DEFAULT_ID);
  });

  it('loads the odds board from ?view=odds', () => {
    loadUrl('/?view=odds');

    expect(mountedScreen()).toBe('odds');
    expect(getSimulateMock).toHaveBeenCalledTimes(1);
    expect(getSimulateMock.mock.calls[0][0]).toBe(DEFAULT_ID);
  });

  it('loads the storybook screen from ?view=storybook', () => {
    loadUrl('/?view=storybook');

    expect(mountedScreen()).toBe('storybook');
    expect(getSimulateMock).not.toHaveBeenCalled();
  });

  it('loads the upload screen from ?view=upload', () => {
    loadUrl('/?view=upload');

    expect(mountedScreen()).toBe('upload');
    expect(currentPrimary()).toBe('Tournament');
  });

  it('marks the active sub-view, and only it, in the sub-nav', () => {
    loadUrl('/?view=odds');

    const current = tournamentNav()?.querySelectorAll('a[aria-current="page"]');
    expect(current).toHaveLength(1);
    expect(current?.[0].textContent?.trim()).toBe('Title odds');
  });

  it('carries a non-default id into the odds board and the sub-nav', () => {
    loadUrl('/?tournament=example_usopen_2026_atp&view=odds');

    expect(getSimulateMock.mock.calls[0][0]).toBe('example_usopen_2026_atp');
    const hrefs = Array.from(tournamentNav()?.querySelectorAll('a') ?? []).map((a) =>
      a.getAttribute('href'),
    );
    expect(hrefs).toEqual([
      '?tournament=example_usopen_2026_atp&view=bracket',
      '?tournament=example_usopen_2026_atp&view=odds',
      '?tournament=example_usopen_2026_atp&view=storybook',
      '?tournament=example_usopen_2026_atp&view=upload',
    ]);
  });

  it('treats a whitespace-only id as absent rather than requesting it', () => {
    loadUrl('/?tournament=%20%20&view=odds');
    expect(getSimulateMock.mock.calls[0][0]).toBe(DEFAULT_ID);
  });
});

// --------------------------------------------------------------------------
// The primary nav.
// --------------------------------------------------------------------------
describe('primary nav', () => {
  it('offers exactly the three branches, and the wordmark home link', () => {
    loadUrl('/');

    const primary = Array.from(
      document.querySelectorAll('nav[aria-label="Sections"] a'),
    ).map((a) => a.textContent?.trim());
    expect(primary).toEqual(['Home', 'Single match', 'Tournament']);
    // The wordmark also returns home.
    expect(document.querySelector('a[aria-label="Ace home"]')).not.toBeNull();
  });

  it('marks exactly one primary branch as current', () => {
    loadUrl('/?view=storybook');

    expect(
      document.querySelectorAll('nav[aria-label="Sections"] a[aria-current="page"]'),
    ).toHaveLength(1);
    expect(currentPrimary()).toBe('Tournament');
  });

  it("points the Tournament branch at the current draw's bracket", () => {
    loadUrl('/?tournament=example_usopen_2026_atp&view=upload');

    const tournamentLink = Array.from(
      document.querySelectorAll('nav[aria-label="Sections"] a'),
    ).find((a) => a.textContent?.trim() === 'Tournament');
    expect(tournamentLink?.getAttribute('href')).toBe(
      '?tournament=example_usopen_2026_atp&view=bracket',
    );
  });
});
