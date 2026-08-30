// @vitest-environment jsdom
// The single-match view, client stubbed: it reads a complete matchup off the URL and simulates it, renders the server's distributions without re-deriving them, and writes a reproducible seed into the address bar.

import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { searchPlayers, simulateMatch } from '../api/client';
import type { MatchSimulateResponse } from '../api/types';
import MatchSim from './MatchSim';

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return {
    ...actual,
    searchPlayers: vi.fn(),
    simulateMatch: vi.fn(),
  };
});

const searchPlayersMock = vi.mocked(searchPlayers);
const simulateMatchMock = vi.mocked(simulateMatch);

const RESULT: MatchSimulateResponse = {
  player_a: 'Carlos Alcaraz',
  player_b: 'Jannik Sinner',
  surface: 'Clay',
  best_of: 5,
  final_set_rule: '10pt_at_6_6',
  win_prob_a: 0.58,
  win_prob_b: 0.42,
  p_clf: 0.55,
  analytic_win_prob_a: 0.57,
  set_scores: [
    { sets_a: 3, sets_b: 0, count: 900, prob: 0.3 },
    { sets_a: 3, sets_b: 1, count: 600, prob: 0.2 },
    { sets_a: 3, sets_b: 2, count: 240, prob: 0.08 },
    { sets_a: 2, sets_b: 3, count: 360, prob: 0.12 },
    { sets_a: 1, sets_b: 3, count: 600, prob: 0.2 },
    { sets_a: 0, sets_b: 3, count: 300, prob: 0.1 },
  ],
  total_games: {
    mean: 38.4,
    std: 9.1,
    minimum: 21,
    maximum: 62,
    distribution: [
      [21, 10],
      [38, 200],
      [62, 5],
    ],
  },
  metadata: {
    mode: 'blend',
    w: 0.5,
    data_through_year: 2026,
    estimator_class: 'RandomForestClassifier',
    adapter: 'common.classifier_adapter.make_classifier_prob',
    is_forecast: false,
    classifier_limitation: 'A model estimate for a hypothetical matchup, not a betting tip.',
    classifier_limitation_detail: 'The numbers come from simulating this matchup…',
    source: 'single-match simulation',
    runs: 3000,
    seed: 7,
  },
};

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  window.history.replaceState({}, '', '/');
});

const COMPLETE_URL = '/?view=match&a=Carlos%20Alcaraz&b=Jannik%20Sinner&surface=Clay&bo=5&seed=7';

function mountAt(url: string) {
  searchPlayersMock.mockReturnValue(new Promise(() => {}));
  window.history.replaceState({}, '', url);
  return render(<MatchSim />);
}

// Deep-link replay
describe('deep-link replay', () => {
  it('simulates the matchup a complete URL carries, once', async () => {
    simulateMatchMock.mockResolvedValue(RESULT);
    mountAt(COMPLETE_URL);

    await waitFor(() => expect(simulateMatchMock).toHaveBeenCalledTimes(1));
    expect(simulateMatchMock.mock.calls[0][0]).toMatchObject({
      player_a: 'Carlos Alcaraz',
      player_b: 'Jannik Sinner',
      surface: 'Clay',
      best_of: 5,
      seed: 7,
    });
  });

  it('does not simulate a URL missing the seed (waits for the user)', () => {
    simulateMatchMock.mockReturnValue(new Promise(() => {}));
    mountAt('/?view=match&a=Carlos%20Alcaraz&b=Jannik%20Sinner&surface=Clay&bo=5');
    expect(simulateMatchMock).not.toHaveBeenCalled();
  });

  it('does not simulate a bare match URL', () => {
    simulateMatchMock.mockReturnValue(new Promise(() => {}));
    mountAt('/?view=match');
    expect(simulateMatchMock).not.toHaveBeenCalled();
  });
});

// Rendering the server's distributions
describe('results', () => {
  it('renders win probability, set scores and total games from the response', async () => {
    simulateMatchMock.mockResolvedValue(RESULT);
    mountAt(COMPLETE_URL);

    await screen.findByLabelText('Win probability');
    expect(screen.getByText('58.0%')).toBeTruthy();
    expect(screen.getByText('42.0%')).toBeTruthy();

    // Every set-score row the server sent is present.
    const setSection = screen.getByLabelText('Set score distribution');
    expect(setSection.querySelectorAll('tbody tr')).toHaveLength(6);
    expect(setSection.querySelector('[data-score="3-0"]')).not.toBeNull();

    // Total games mean is shown, not re-derived.
    const games = screen.getByLabelText('Total games');
    expect(games.querySelector('[data-meta="games_mean"]')?.textContent).toBe('38.4');
  });

  it('renders the single-match-specific disclaimer wording, not the server’s raw text', async () => {
    simulateMatchMock.mockResolvedValue(RESULT);
    mountAt(COMPLETE_URL);

    const caveat = await screen.findByText(
      'Disclaimer: this is a model estimate for a hypothetical matchup, not a betting tip.',
    );
    expect(caveat).toBeTruthy();
    // The override changes only what the browser renders, the field this page reads off `result.metadata` is untouched.
    expect(RESULT.metadata.classifier_limitation).toBe(
      'A model estimate for a hypothetical matchup, not a betting tip.',
    );
  });

  it('renders a structured error without crashing', async () => {
    simulateMatchMock.mockRejectedValue(new Error('boom'));
    mountAt(COMPLETE_URL);

    const alert = await screen.findByRole('alert');
    expect(alert.textContent).toContain('boom');
  });
});

// Loading state
describe('loading state', () => {
  it('shows a plain "Simulating…" status with no infrastructure explanation, however long the request takes', async () => {
    vi.useFakeTimers();
    try {
      // A request that never resolves, so the component stays in `loading`.
      simulateMatchMock.mockReturnValue(new Promise(() => {}));
      searchPlayersMock.mockReturnValue(new Promise(() => {}));
      window.history.replaceState({}, '', COMPLETE_URL);
      act(() => {
        render(<MatchSim />);
      });

      expect(screen.getByText(/Simulating/)).toBeTruthy();

      // No wording about servers, sleeping, or waking up, at any point.
      act(() => {
        vi.advanceTimersByTime(10000);
      });
      expect(screen.queryByText(/server/i)).toBeNull();
      expect(screen.queryByText(/sleep/i)).toBeNull();
      expect(screen.queryByText(/waking/i)).toBeNull();
      expect(screen.getByText(/Simulating/)).toBeTruthy();
    } finally {
      vi.useRealTimers();
    }
  });
});

// Player picker keeps a stable height (no layout shift on "Searching…")
describe('player picker layout', () => {
  it('always renders a status slot under each input so its height never jumps', () => {
    // A bare match URL: no simulation runs and no search is in flight, so the pickers sit in their resting state (no "Searching…", no loading panel).
    searchPlayersMock.mockReturnValue(new Promise(() => {}));
    simulateMatchMock.mockReturnValue(new Promise(() => {}));
    window.history.replaceState({}, '', '/?view=match');
    render(<MatchSim />);

    // Each picker keeps a reserved status line at rest (aria-hidden while empty), so the two inputs stay the same height.
    const reserved = screen.getAllByRole('status', { hidden: true });
    expect(reserved).toHaveLength(2);
    for (const slot of reserved) {
      // The slot holds non-breaking whitespace at rest, never collapsing to zero height.
      expect(slot.textContent?.trim()).toBe('');
    }

    // And both player inputs are present and label-associated, so the two reserved slots belong to the two side-by-side pickers.
    expect(screen.getAllByLabelText(/Player [AB]/, { selector: 'input' })).toHaveLength(2);
  });
});

// Reproducible-by-seed
describe('reproducibility', () => {
  it('writes players, surface, format and a seed into the address bar on run', async () => {
    simulateMatchMock.mockResolvedValue(RESULT);
    // Start from a complete URL so the form is pre-filled, then re-run.
    mountAt(COMPLETE_URL);
    await waitFor(() => expect(simulateMatchMock).toHaveBeenCalledTimes(1));

    const button = screen.getByRole('button', { name: /simulate again/i });
    await act(async () => {
      fireEvent.click(button);
    });

    const params = new URLSearchParams(window.location.search);
    expect(params.get('view')).toBe('match');
    expect(params.get('a')).toBe('Carlos Alcaraz');
    expect(params.get('b')).toBe('Jannik Sinner');
    expect(params.get('surface')).toBe('Clay');
    expect(params.get('bo')).toBe('5');
    // A fresh, valid seed is written (and it differs from the previous one).
    const seed = Number(params.get('seed'));
    expect(Number.isInteger(seed)).toBe(true);
    expect(seed).not.toBe(7);
  });
});
