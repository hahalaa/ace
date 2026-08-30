// @vitest-environment jsdom
// Rankings tests: the note it must show, the default active-only filter and the inactive toggle, track tabs, and the failures that must not read as "no data".

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ApiError, getRankings } from '../api/client';
import type { RankingsResponse } from '../api/types';
import Rankings from './Rankings';

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return { ...actual, getRankings: vi.fn() };
});

const getRankingsMock = vi.mocked(getRankings);

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

function player(overrides: Partial<RankingsResponse['tracks'][number]['players'][number]>) {
  return {
    rank: 1,
    player_id: 'X',
    player_name: 'Someone',
    rating: 1800,
    matches: 100,
    last_played: '2026-06-01',
    is_active: true,
    ...overrides,
  };
}

const RANKINGS: RankingsResponse = {
  tracks: [
    {
      track: 'overall',
      players: [
        player({ rank: 1, player_id: 'A', player_name: 'Ada Ace', rating: 2100, is_active: true }),
        player({
          rank: 2,
          player_id: 'L',
          player_name: 'Old Legend',
          rating: 2050,
          last_played: '2020-06-01',
          is_active: false,
        }),
        player({ rank: 3, player_id: 'B', player_name: 'Ben Base', rating: 1800, is_active: true }),
      ],
    },
    {
      track: 'Clay',
      players: [
        player({ rank: 1, player_id: 'C', player_name: 'Clay King', rating: 1950, is_active: true }),
      ],
    },
  ],
  metadata: {
    generated_at: '2026-06-02T00:00:00Z',
    as_of: '2026-06-01',
    active_cutoff: '2025-06-01',
    active_window_days: 365,
    data_through_year: 2026,
    n_matches: 1234,
    note: "Ace's own rating, not the official ATP ranking.",
  },
};

function renderReady() {
  getRankingsMock.mockResolvedValue(RANKINGS);
  return render(<Rankings />);
}

describe('Rankings', () => {
  it('shows the "not the ATP ranking" note', async () => {
    renderReady();
    expect(await screen.findByText(/not the official ATP ranking/i)).toBeTruthy();
  });

  it('hides inactive players by default, so a retired peak does not top the board', async () => {
    renderReady();
    await screen.findByText('Ada Ace');
    // The active players are shown...
    expect(screen.queryByText('Ada Ace')).not.toBeNull();
    expect(screen.queryByText('Ben Base')).not.toBeNull();
    // ...the long-inactive legend is not, by default.
    expect(screen.queryByText('Old Legend')).toBeNull();
  });

  it('reveals inactive players (dimmed and tagged) when the toggle is on', async () => {
    renderReady();
    await screen.findByText('Ada Ace');
    fireEvent.click(screen.getByLabelText(/include inactive/i));

    const legendRow = screen.getByText('Old Legend').closest('tr');
    expect(legendRow?.getAttribute('data-active')).toBe('false');
    expect(within(legendRow as HTMLElement).getByText(/inactive/i)).toBeTruthy();
  });

  it('switches leaderboards when a surface tab is picked', async () => {
    renderReady();
    await screen.findByText('Ada Ace');
    fireEvent.click(screen.getByRole('tab', { name: 'Clay' }));

    expect(screen.queryByText('Clay King')).not.toBeNull();
    expect(screen.queryByText('Ada Ace')).toBeNull();
  });

  it('renders a helpful panel when the rankings have not been generated', async () => {
    getRankingsMock.mockRejectedValue(
      new ApiError({
        status: 425,
        statusText: 'Too Early',
        url: 'http://api.test/rankings',
        body: {
          detail: {
            reason: 'rankings_missing',
            message: 'No precomputed Elo rankings.',
            command: 'python scripts/precompute_elo.py',
          },
        },
        message: 'rankings missing',
      }),
    );
    render(<Rankings />);

    const panel = await screen.findByRole('alert');
    expect(panel.getAttribute('data-error-kind')).toBe('rankings-missing');
    expect(panel.textContent).toContain('python scripts/precompute_elo.py');
  });
});
