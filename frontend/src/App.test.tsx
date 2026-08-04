// @vitest-environment jsdom
/**
 * The shell's two axes, both read from the query string.
 *
 * **These are deep-link tests, not navigation tests.** `App` reads
 * `window.location.search` during render and holds no state, so the thing worth
 * pinning is that a *fresh load* of a URL lands on the right screen — which is
 * what a shared link does, and what a `?view=`-ignoring regression would break
 * invisibly. Each case rewrites the URL with `history.replaceState` and mounts
 * the component from scratch, exactly as a page load would.
 *
 * Which screen mounted is asserted from the **screen itself** (its loading
 * status names either "bracket" or "simulation"), not only from the nav's
 * `aria-current` — a mutation that hardcoded the view while still styling the
 * nav correctly has to fail on something the user actually sees.
 *
 * `getBracket`/`getSimulate` are stubbed with promises that never settle: the
 * loading state is enough to identify the mounted screen, and neither
 * component's data path is under test here.
 */

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import App from './App';
import { getBracket, getSimulate } from './api/client';

vi.mock('./api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('./api/client')>();
  return { ...actual, getBracket: vi.fn(), getSimulate: vi.fn(), searchPlayers: vi.fn() };
});

const getBracketMock = vi.mocked(getBracket);
const getSimulateMock = vi.mocked(getSimulate);

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  window.history.replaceState({}, '', '/');
});

const DEFAULT_ID = 'usopen_2024_atp_full';

/** Load `url` from scratch, as a browser would, and mount the shell. */
function loadUrl(url: string) {
  getBracketMock.mockReturnValue(new Promise(() => {}));
  getSimulateMock.mockReturnValue(new Promise(() => {}));
  window.history.replaceState({}, '', url);
  return render(<App />);
}

/** Which screen actually mounted, read off its own loading status. */
function mountedScreen(): 'bracket' | 'odds' {
  const status = screen.getByRole('status').textContent ?? '';
  if (status.includes('bracket')) return 'bracket';
  if (status.includes('simulation')) return 'odds';
  throw new Error(`No screen identifiable from status: ${status}`);
}

/** The nav link marked as the current page. */
function currentNavLink(): string {
  return (
    document.querySelector('nav[aria-label="View"] a[aria-current="page"]')?.textContent?.trim() ??
    ''
  );
}

describe('?view= deep-linking', () => {
  it('loads the odds board directly from ?view=odds', () => {
    loadUrl('/?view=odds');

    expect(mountedScreen()).toBe('odds');
    expect(currentNavLink()).toBe('Title odds');
    expect(getSimulateMock).toHaveBeenCalledTimes(1);
    expect(getBracketMock).not.toHaveBeenCalled();
  });

  it('loads the bracket directly from ?view=bracket', () => {
    loadUrl('/?view=bracket');

    expect(mountedScreen()).toBe('bracket');
    expect(currentNavLink()).toBe('Bracket');
    expect(getBracketMock).toHaveBeenCalledTimes(1);
    expect(getSimulateMock).not.toHaveBeenCalled();
  });

  it('defaults to the bracket when the URL names no view', () => {
    loadUrl('/');

    expect(mountedScreen()).toBe('bracket');
    expect(currentNavLink()).toBe('Bracket');
  });

  it('falls back to the bracket for a view value it does not know', () => {
    // `storybook` is T4.4's, and is exactly the value a half-landed feature
    // would leave in a shared link. An unknown view must not render blank.
    loadUrl('/?view=storybook');

    expect(mountedScreen()).toBe('bracket');
    expect(currentNavLink()).toBe('Bracket');
    expect(screen.getAllByRole('link')).toHaveLength(2);
  });

  it('marks exactly one nav link as the current page', () => {
    loadUrl('/?view=odds');

    expect(document.querySelectorAll('nav[aria-label="View"] a[aria-current="page"]')).toHaveLength(
      1,
    );
  });
});

describe('?tournament= and ?view= are independent axes', () => {
  it('carries a non-default id into the odds board', () => {
    loadUrl('/?tournament=example_usopen_2026_atp&view=odds');

    expect(mountedScreen()).toBe('odds');
    expect(getSimulateMock.mock.calls[0][0]).toBe('example_usopen_2026_atp');
  });

  it('falls back to the default id when only a view is named', () => {
    loadUrl('/?view=odds');

    expect(getSimulateMock.mock.calls[0][0]).toBe(DEFAULT_ID);
  });

  it('keeps the current tournament in both nav hrefs, so switching view keeps the draw', () => {
    loadUrl('/?tournament=example_usopen_2026_atp&view=bracket');

    const hrefs = screen.getAllByRole('link').map((a) => a.getAttribute('href'));
    expect(hrefs).toEqual([
      '?tournament=example_usopen_2026_atp&view=bracket',
      '?tournament=example_usopen_2026_atp&view=odds',
    ]);
  });

  it('treats a whitespace-only id as absent rather than requesting it', () => {
    loadUrl('/?tournament=%20%20&view=odds');

    expect(getSimulateMock.mock.calls[0][0]).toBe(DEFAULT_ID);
  });
});
