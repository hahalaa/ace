// @vitest-environment jsdom
/**
 * Storybook tests: the run drawn onto the bracket, the champion, and the URL
 * contract that makes a link replay the same tournament.
 *
 * **The seed tests are the load-bearing ones.** T2.4/T3.4 already guarantee that
 * one id plus one seed is one deterministic bracket; this ticket's only way to
 * break that guarantee is to lose the seed — by rolling a fresh one on a render,
 * by not writing it to the URL, or by not reading it back on load. Each of those
 * has a test, and they assert on `window.location.search` and on the argument
 * `getStorybook` actually received, not on anything the component says about
 * itself.
 *
 * `getBracket`/`getStorybook` are stubbed but the rest of `../api/client` is the
 * real module, so `ApiError` here is the class the panels really branch on.
 */

import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ApiError, getBracket, getStorybook } from '../api/client';
import type {
  BracketResponse,
  BracketSlot,
  StorybookMatch,
  StorybookResponse,
} from '../api/types';
import Storybook from './Storybook';
import { buildRounds } from './rounds';
import { buildStorybookOverlay, describesBracket, overlayKey } from './overlay';

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return { ...actual, getBracket: vi.fn(), getStorybook: vi.fn() };
});

const getBracketMock = vi.mocked(getBracket);
const getStorybookMock = vi.mocked(getStorybook);

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
  vi.restoreAllMocks();
  window.history.replaceState({}, '', '/');
});

// --------------------------------------------------------------------------
// Fixtures — an 8 draw whose winners are deliberately not all on one side.
// --------------------------------------------------------------------------

const PLAYERS = [
  'Jannik Sinner',
  'Alex de Minaur',
  'Taylor Fritz',
  'Casper Ruud',
  'Daniil Medvedev',
  'Grigor Dimitrov',
  'Andrey Rublev',
  'Carlos Alcaraz',
];

const SEEDS: Record<string, number | null> = {
  'Jannik Sinner': 1,
  'Carlos Alcaraz': 2,
  'Daniil Medvedev': 3,
  'Casper Ruud': 4,
};

function slot(position: number): BracketSlot {
  const player = PLAYERS[position - 1];
  return {
    position,
    player,
    is_placeholder: false,
    player_id: `p${position}`,
    seed: SEEDS[player] ?? null,
  };
}

function toyEightBracket(): BracketResponse {
  return {
    tournament_id: 'toy_8',
    name: 'Toy Open',
    surface: 'Hard',
    best_of: 5,
    final_set_tiebreak: '10pt_at_6_6',
    draw_size: 8,
    slots: Array.from({ length: 8 }, (_, i) => slot(i + 1)),
  };
}

function match(
  roundIndex: number,
  roundLabel: string,
  matchIndex: number,
  winner: string,
  loser: string,
  scoreline: string,
): StorybookMatch {
  return {
    round_index: roundIndex,
    round_label: roundLabel,
    match_index: matchIndex,
    winner,
    loser,
    winner_seed: SEEDS[winner] ?? null,
    loser_seed: SEEDS[loser] ?? null,
    scoreline,
    line: `${winner} def. ${loser} ${scoreline}`,
  };
}

/**
 * A full 8-draw run: 7 matches over 3 rounds.
 *
 * Round one is won by the **bottom** side in three of four matches, so a
 * component that always credited the top slot — or that keyed results by array
 * order — cannot pass. The winners then meet in the order the simulator
 * preserves: SF match 0 is the winners of QF 0 and QF 1.
 */
function toyEightStory(overrides: Partial<StorybookResponse> = {}): StorybookResponse {
  return {
    tournament_id: 'toy_8',
    name: 'Toy Open',
    surface: 'Hard',
    best_of: 5,
    final_set_tiebreak: '10pt_at_6_6',
    draw_size: 8,
    match_count: 7,
    rounds: [
      {
        index: 0,
        label: 'QF',
        matches: [
          match(0, 'QF', 0, 'Alex de Minaur', 'Jannik Sinner', '7-6(5) 3-6 6-4 6-4'),
          match(0, 'QF', 1, 'Taylor Fritz', 'Casper Ruud', '6-3 6-4 6-4'),
          match(0, 'QF', 2, 'Grigor Dimitrov', 'Daniil Medvedev', '4-6 6-3 6-2 7-5'),
          match(0, 'QF', 3, 'Carlos Alcaraz', 'Andrey Rublev', '6-2 6-2 6-1'),
        ],
      },
      {
        index: 1,
        label: 'SF',
        matches: [
          match(1, 'SF', 0, 'Taylor Fritz', 'Alex de Minaur', '6-4 7-6(3) 6-4'),
          match(1, 'SF', 1, 'Carlos Alcaraz', 'Grigor Dimitrov', '6-1 6-4 6-4'),
        ],
      },
      {
        index: 2,
        label: 'F',
        matches: [match(2, 'F', 0, 'Carlos Alcaraz', 'Taylor Fritz', '6-3 3-6 7-6(4) 6-4')],
      },
    ],
    champion: { position: 8, player: 'Carlos Alcaraz', player_id: 'p8', seed: 2 },
    runs: [
      {
        position: 8,
        player: 'Carlos Alcaraz',
        player_id: 'p8',
        seed: 2,
        is_champion: true,
        furthest_round: 'F',
        matches_won: 3,
        beat: ['Andrey Rublev', 'Grigor Dimitrov', 'Taylor Fritz'],
        eliminated_by: null,
      },
      {
        position: 3,
        player: 'Taylor Fritz',
        player_id: 'p3',
        seed: null,
        is_champion: false,
        furthest_round: 'F',
        matches_won: 2,
        beat: ['Casper Ruud', 'Alex de Minaur'],
        eliminated_by: 'Carlos Alcaraz',
      },
    ],
    metadata: {
      mode: 'blend',
      w: 0.5,
      data_through_year: 2026,
      estimator_class: 'RandomForestClassifier',
      adapter: 'common.classifier_adapter:make_classifier_prob',
      is_forecast: false,
      classifier_limitation:
        'Every input is an as-of-now snapshot, so simulating an already-played draw is retrospective.',
      source: 'toy_8.json',
      seed: 42,
    },
    ...overrides,
  };
}

/** Mount the screen at `url`, with both endpoints stubbed. */
function mountAt(url: string, story: StorybookResponse | null = toyEightStory()) {
  getBracketMock.mockResolvedValue(toyEightBracket());
  if (story !== null) getStorybookMock.mockResolvedValue(story);
  window.history.replaceState({}, '', url);
  return render(<Storybook tournamentId="toy_8" />);
}

/** The seed currently in the address bar, as the server would receive it. */
function seedInUrl(): string | null {
  return new URLSearchParams(window.location.search).get('seed');
}

/** One round's cards, in draw order. */
function cardsIn(label: string): HTMLElement[] {
  return within(screen.getByRole('list', { name: `${label} matches` })).getAllByRole('listitem');
}

/** `[name, result]` for both sides of a card — `result` is `null` when unclaimed. */
function sides(card: HTMLElement): [string, string | null][] {
  return [...card.querySelectorAll('[data-state]')].map((row) => [
    row.querySelector('[data-cell="name"]')?.textContent ?? '',
    (row as HTMLElement).dataset.result ?? null,
  ]);
}

// --------------------------------------------------------------------------
// The headline: results on the bracket.
// --------------------------------------------------------------------------

describe('a run populates the bracket', () => {
  it('marks the winner of every first-round match, on whichever side won', async () => {
    mountAt('/?seed=7');
    await screen.findByRole('list', { name: 'QF matches' });
    await waitFor(() => expect(screen.getAllByText(/def\./).length).toBeGreaterThan(0));

    expect(cardsIn('QF').map(sides)).toEqual([
      [
        ['Jannik Sinner', 'lost'],
        ['Alex de Minaur', 'won'],
      ],
      [
        ['Taylor Fritz', 'won'],
        ['Casper Ruud', 'lost'],
      ],
      [
        ['Daniil Medvedev', 'lost'],
        ['Grigor Dimitrov', 'won'],
      ],
      [
        ['Andrey Rublev', 'lost'],
        ['Carlos Alcaraz', 'won'],
      ],
    ]);
  });

  it('shows each match’s scoreline, copied from the response', async () => {
    mountAt('/?seed=7');
    await waitFor(() => expect(cardsIn('F')[0].querySelector('[data-cell="scoreline"]')).not.toBeNull());

    const scorelines = cardsIn('QF').map(
      (card) => card.querySelector('[data-cell="scoreline"]')?.textContent,
    );
    expect(scorelines).toEqual([
      '7-6(5) 3-6 6-4 6-4',
      '6-3 6-4 6-4',
      '4-6 6-3 6-2 7-5',
      '6-2 6-2 6-1',
    ]);
    expect(cardsIn('F')[0].querySelector('[data-cell="scoreline"]')?.textContent).toBe(
      '6-3 3-6 7-6(4) 6-4',
    );
  });

  it('fills the later rounds /bracket leaves as TBD with whoever won their way there', async () => {
    mountAt('/?seed=7');
    await waitFor(() => expect(screen.queryAllByText('TBD')).toHaveLength(0));

    // SF 0 is the winners of QF 0 and QF 1, in that order — the ordering
    // `sim.tournament.simulate_bracket` preserves round to round.
    expect(cardsIn('SF').map(sides)).toEqual([
      [
        ['Alex de Minaur', 'lost'],
        ['Taylor Fritz', 'won'],
      ],
      [
        ['Grigor Dimitrov', 'lost'],
        ['Carlos Alcaraz', 'won'],
      ],
    ]);
    expect(cardsIn('F').map(sides)).toEqual([
      [
        ['Taylor Fritz', 'lost'],
        ['Carlos Alcaraz', 'won'],
      ],
    ]);
  });

  it('shows the draw, unplayed, before a run is started', async () => {
    mountAt('/');
    await screen.findByRole('list', { name: 'QF matches' });

    expect(getStorybookMock).not.toHaveBeenCalled();
    expect(document.querySelectorAll('[data-result]')).toHaveLength(0);
    expect(document.querySelectorAll('[data-cell="scoreline"]')).toHaveLength(0);
    expect(screen.getAllByText('TBD')).toHaveLength(6);
  });
});

// --------------------------------------------------------------------------
// Champion.
// --------------------------------------------------------------------------

describe('champion', () => {
  it('names the response’s own champion, prominently and at the end', async () => {
    mountAt('/?seed=7');
    const panel = await screen.findByRole('region', { name: 'Champion' });

    expect(panel.dataset.champion).toBe('Carlos Alcaraz');
    expect(panel.textContent).toContain('Carlos Alcaraz');
    expect(panel.textContent).toContain('[2]');
    expect(panel.textContent).toContain('3 matches won');
    // The final's line, as sent — not reassembled from set scores.
    expect(panel.textContent).toContain('Carlos Alcaraz def. Taylor Fritz 6-3 3-6 7-6(4) 6-4');
  });

  it('highlights the champion in the final, not merely in the panel', async () => {
    mountAt('/?seed=7');
    await waitFor(() => expect(cardsIn('F')[0].querySelectorAll('[data-result]')).toHaveLength(2));

    const winner = cardsIn('F')[0].querySelector('[data-result="won"]');
    expect(winner?.querySelector('[data-cell="name"]')?.textContent).toBe('Carlos Alcaraz');
  });

  it('trusts `champion` rather than re-deriving it from the last match', async () => {
    // A body whose `champion` disagrees with the final is a server bug — but it
    // pins *which* field this component reads.
    mountAt(
      '/?seed=7',
      toyEightStory({
        champion: { position: 3, player: 'Taylor Fritz', player_id: 'p3', seed: null },
      }),
    );
    const panel = await screen.findByRole('region', { name: 'Champion' });

    expect(panel.dataset.champion).toBe('Taylor Fritz');
  });
});

// --------------------------------------------------------------------------
// The seed: the whole point of the screen.
// --------------------------------------------------------------------------

describe('the seed round-trips through the URL', () => {
  it('runs the seed the URL asked for, not one of its own', async () => {
    mountAt('/?seed=1234');
    await screen.findByRole('region', { name: 'Champion' });

    expect(getStorybookMock).toHaveBeenCalledTimes(1);
    expect(getStorybookMock.mock.calls[0][0]).toBe('toy_8');
    expect(getStorybookMock.mock.calls[0][1]).toBe(1234);
    expect(seedInUrl()).toBe('1234');
  });

  it('runs seed 0 — a legal seed, not an absent one', async () => {
    mountAt('/?seed=0');
    await screen.findByRole('region', { name: 'Champion' });

    expect(getStorybookMock.mock.calls[0][1]).toBe(0);
  });

  it('reloading the same URL asks for the same seed again', async () => {
    const first = mountAt('/?seed=1234');
    await screen.findByRole('region', { name: 'Champion' });
    first.unmount();

    // A fresh mount of the same URL is what a browser reload does.
    render(<Storybook tournamentId="toy_8" />);
    await screen.findByRole('region', { name: 'Champion' });

    expect(getStorybookMock.mock.calls.map((call) => call[1])).toEqual([1234, 1234]);
  });

  it('does not roll a new seed when React re-renders', async () => {
    const { rerender } = mountAt('/?seed=1234');
    await screen.findByRole('region', { name: 'Champion' });

    rerender(<Storybook tournamentId="toy_8" />);
    rerender(<Storybook tournamentId="toy_8" />);

    expect(getStorybookMock).toHaveBeenCalledTimes(1);
    expect(seedInUrl()).toBe('1234');
  });

  it('ignores a seed the server would reject, and starts nothing', async () => {
    mountAt('/?seed=not-a-number');
    await screen.findByRole('list', { name: 'QF matches' });

    expect(getStorybookMock).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: /Simulate this tournament/ })).toBeDefined();
  });

  it('writes the seed it chose into the URL when a run is started', async () => {
    vi.spyOn(Math, 'random').mockReturnValue(0.5);
    mountAt('/');
    await screen.findByRole('list', { name: 'QF matches' });

    fireEvent.click(screen.getByRole('button', { name: /Simulate this tournament/ }));
    await screen.findByRole('region', { name: 'Champion' });

    const chosen = 2 ** 31; // floor(0.5 * 2**32)
    expect(seedInUrl()).toBe(String(chosen));
    expect(getStorybookMock.mock.calls[0][1]).toBe(chosen);
    expect(new URLSearchParams(window.location.search).get('view')).toBe('storybook');
  });
});

describe('re-run', () => {
  it('picks a different seed and puts it in the URL', async () => {
    mountAt('/?seed=1234');
    await screen.findByRole('region', { name: 'Champion' });

    fireEvent.click(screen.getByRole('button', { name: /Re-run/ }));
    await waitFor(() => expect(getStorybookMock).toHaveBeenCalledTimes(2));

    const after = seedInUrl();
    expect(after).not.toBe('1234');
    expect(after).not.toBeNull();
    // The URL and the request agree — a link shared now replays what is shown.
    expect(String(getStorybookMock.mock.calls[1][1])).toBe(after);
  });

  it('keeps producing a new seed run after run', async () => {
    mountAt('/?seed=1234');
    await screen.findByRole('region', { name: 'Champion' });

    const seen = new Set<string | null>(['1234']);
    for (let i = 0; i < 5; i += 1) {
      fireEvent.click(screen.getByRole('button', { name: /Re-run/ }));
      await waitFor(() => expect(getStorybookMock).toHaveBeenCalledTimes(i + 2));
      expect(seen.has(seedInUrl())).toBe(false);
      seen.add(seedInUrl());
    }
  });
});

describe('share', () => {
  it('copies the tournament, view and seed as one absolute link', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', { value: { writeText }, configurable: true });

    mountAt('/?seed=1234');
    await screen.findByRole('region', { name: 'Champion' });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Share/ }));
    });

    expect(writeText).toHaveBeenCalledTimes(1);
    const url = new URL(writeText.mock.calls[0][0] as string);
    expect(url.origin).toBe(window.location.origin);
    expect(url.searchParams.get('tournament')).toBe('toy_8');
    expect(url.searchParams.get('view')).toBe('storybook');
    expect(url.searchParams.get('seed')).toBe('1234');
    expect(screen.getByText(/Link copied/)).toBeDefined();
  });

  it('shows the link to copy by hand when the clipboard refuses', async () => {
    Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: vi.fn().mockRejectedValue(new Error('denied')) },
      configurable: true,
    });

    mountAt('/?seed=1234');
    await screen.findByRole('region', { name: 'Champion' });
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Share/ }));
    });

    const fallback = await screen.findByText(/Copy this link/);
    expect(fallback.textContent).toContain('seed=1234');
  });

  it('offers nothing to share before a run exists', async () => {
    mountAt('/');
    await screen.findByRole('list', { name: 'QF matches' });

    expect(screen.getByRole('button', { name: /Share/ }).hasAttribute('disabled')).toBe(true);
  });
});

// --------------------------------------------------------------------------
// Failures — the same panels the rest of the app renders.
// --------------------------------------------------------------------------

describe('loading and error states', () => {
  it('says it is simulating while the request is in flight', async () => {
    getBracketMock.mockResolvedValue(toyEightBracket());
    getStorybookMock.mockReturnValue(new Promise(() => {}));
    window.history.replaceState({}, '', '/?seed=7');
    render(<Storybook tournamentId="toy_8" />);

    expect(screen.getAllByRole('status').some((el) => /Simulating/.test(el.textContent ?? ''))).toBe(
      true,
    );
    expect(screen.queryByRole('alert')).toBeNull();
  });

  it('renders a placeholder draw’s 409 through ErrorPanel, naming the empty slots', async () => {
    getBracketMock.mockResolvedValue(toyEightBracket());
    getStorybookMock.mockRejectedValue(
      new ApiError({
        status: 409,
        statusText: 'Conflict',
        url: 'http://api.test/tournaments/example_usopen_2026_atp/storybook?seed=7',
        body: {
          detail: {
            reason: 'draw_not_simulatable',
            message:
              "Draw 'example_usopen_2026_atp' still holds placeholder entrants and cannot be simulated.",
            tournament_id: 'example_usopen_2026_atp',
            source: 'example_usopen_2026.json',
            placeholder_slots: [
              { position: 2, player: 'Qualifier' },
              { position: 6, player: 'Qualifier/Lucky Loser' },
            ],
          },
        },
        message: 'ace API request failed (409 Conflict)',
      }),
    );
    window.history.replaceState({}, '', '/?seed=7');
    render(<Storybook tournamentId="example_usopen_2026_atp" />);

    const panel = await screen.findByRole('alert');
    expect(panel.dataset.errorKind).toBe('not-simulatable');
    expect(panel.textContent).toContain('cannot be simulated');
    expect(within(panel).getByText('position 2: Qualifier')).toBeDefined();
    expect(within(panel).getByText('position 6: Qualifier/Lucky Loser')).toBeDefined();
    // Not the generic panel, and not a blank screen: the bracket is still there.
    expect(panel.textContent).not.toContain('The API returned');
  });

  it('renders a 404 as the shared not-found panel', async () => {
    getBracketMock.mockResolvedValue(toyEightBracket());
    getStorybookMock.mockRejectedValue(
      new ApiError({
        status: 404,
        statusText: 'Not Found',
        url: 'http://api.test/tournaments/nope/storybook',
        body: { detail: "Unknown tournament 'nope'." },
        message: 'ace API request failed (404 Not Found)',
      }),
    );
    window.history.replaceState({}, '', '/?seed=7');
    render(<Storybook tournamentId="nope" />);

    const panel = await screen.findByRole('alert');
    expect(panel.dataset.errorKind).toBe('not-found');
  });
});

// --------------------------------------------------------------------------
// The disclosure — inherited from T4.3's requirement via StorybookMetadata.
// --------------------------------------------------------------------------

describe('disclosure', () => {
  it('renders is_forecast: false and the limitation prose, above the bracket', async () => {
    mountAt('/?seed=7');
    await screen.findByRole('region', { name: 'Champion' });

    const disclosure = document.querySelector('[data-forecast]') as HTMLElement;
    expect(disclosure.dataset.forecast).toBe('false');
    expect(screen.getByText(/Not a forecast/)).toBeDefined();
    expect(
      screen.getByText(/Every input is an as-of-now snapshot/),
    ).toBeDefined();

    const board = screen.getByRole('list', { name: 'QF matches' });
    expect(disclosure.compareDocumentPosition(board) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('reports the seed and model that produced this run', async () => {
    mountAt('/?seed=7');
    await screen.findByRole('region', { name: 'Champion' });

    const footnote = document.querySelector('[data-meta="estimator"]')?.parentElement;
    expect(footnote?.textContent).toContain('RandomForestClassifier');
    expect(footnote?.textContent).toContain('blend (w=0.5)');
    expect(footnote?.textContent).toContain('data through 2026');
    expect(footnote?.textContent).toContain('toy_8.json');
    // Two sources, separately labelled: what the URL asks for, and what the
    // server says it ran.
    expect(document.querySelector('[data-meta="seed"]')?.textContent).toContain('42');
    expect(document.querySelector('[data-meta="url_seed"]')?.textContent).toContain('seed 7');
  });
});

// --------------------------------------------------------------------------
// The join itself, without React.
// --------------------------------------------------------------------------

describe('overlay keying', () => {
  const rounds = () => buildRounds(8, toyEightBracket().slots);

  it('keys every match of the run onto the bracket’s own coordinates', () => {
    const overlay = buildStorybookOverlay(rounds(), toyEightStory());

    expect([...overlay.keys()].sort()).toEqual(
      ['0:0', '0:1', '0:2', '0:3', '1:0', '1:1', '2:0'].sort(),
    );
    expect(overlay.get(overlayKey(2, 0))?.top?.player).toBe('Taylor Fritz');
    expect(overlay.get(overlayKey(2, 0))?.bottomResult).toBe('won');
  });

  it('refuses a run that is not about this draw', () => {
    const story = toyEightStory({ tournament_id: 'other_8' });
    expect(describesBracket(story, 'toy_8', 8)).toBe(false);
    expect(describesBracket(toyEightStory({ draw_size: 16 }), 'toy_8', 8)).toBe(false);
    expect(describesBracket(toyEightStory(), 'toy_8', 8)).toBe(true);
  });

  it('claims no winner — and stops propagating — when a name matches neither side', () => {
    // A body disagreeing with its own draw. The scoreline still shows; the side
    // does not, and the round above stays unknown rather than being guessed.
    const story = toyEightStory();
    const broken: StorybookResponse = {
      ...story,
      rounds: [
        {
          ...story.rounds[0],
          matches: [
            match(0, 'QF', 0, 'Nobody At All', 'Jannik Sinner', '6-0 6-0 6-0'),
            ...story.rounds[0].matches.slice(1),
          ],
        },
        ...story.rounds.slice(1),
      ],
    };

    const overlay = buildStorybookOverlay(rounds(), broken);
    const first = overlay.get(overlayKey(0, 0));
    expect(first?.scoreline).toBe('6-0 6-0 6-0');
    expect(first?.topResult).toBeUndefined();
    expect(first?.bottomResult).toBeUndefined();
    expect(overlay.get(overlayKey(1, 0))?.top).toBeNull();
    // The other half of the draw is unaffected.
    expect(overlay.get(overlayKey(1, 1))?.bottom?.player).toBe('Carlos Alcaraz');
  });

  it('leaves a match the response omits exactly as the bracket had it', () => {
    const story = toyEightStory();
    const partial: StorybookResponse = {
      ...story,
      rounds: [story.rounds[0], { ...story.rounds[1], matches: [] }, story.rounds[2]],
    };

    const overlay = buildStorybookOverlay(rounds(), partial);
    expect(overlay.has(overlayKey(1, 0))).toBe(false);
    expect(overlay.get(overlayKey(0, 0))?.topResult).toBe('lost');
  });
});
