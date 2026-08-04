// @vitest-environment jsdom
/**
 * Bracket tests: the shape it derives, the states it renders, and the two cases
 * that would ship broken if only the happy path were covered.
 *
 * `getBracket` is stubbed but the rest of `../api/client` is the real module,
 * so `ApiError`/`ApiNetworkError` here are the classes the component actually
 * branches on — a hand-rolled lookalike would pass every `instanceof` check
 * that matters by accident, or fail one for the wrong reason.
 *
 * jsdom is requested per file rather than globally: `vite.config.ts` sets
 * `environment: 'node'` for the client tests, which touch no DOM.
 */

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ApiConfigError, ApiError, ApiNetworkError, apiBaseUrl, getBracket } from '../api/client';
import type { BracketResponse, BracketSlot } from '../api/types';
import Bracket from './Bracket';
import { buildRounds, roundLabels } from './rounds';

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return { ...actual, getBracket: vi.fn() };
});

const getBracketMock = vi.mocked(getBracket);

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

// --------------------------------------------------------------------------
// Fixtures.
// --------------------------------------------------------------------------

function slot(position: number, player: string, seed: number | null = null): BracketSlot {
  return { position, player, is_placeholder: false, player_id: `p${position}`, seed };
}

/** A `Qualifier`-style slot, exactly as `/bracket` serves one. */
function placeholderSlot(position: number, player: string): BracketSlot {
  return { position, player, is_placeholder: true, player_id: null, seed: null };
}

/** A toy 8 draw of real, resolved players. */
function toyEightDraw(overrides: Partial<BracketResponse> = {}): BracketResponse {
  return {
    tournament_id: 'toy_8',
    name: 'Toy Open',
    surface: 'Hard',
    best_of: 5,
    final_set_tiebreak: '10pt_at_6_6',
    draw_size: 8,
    slots: [
      slot(1, 'Jannik Sinner', 1),
      slot(2, 'Alex de Minaur'),
      slot(3, 'Taylor Fritz'),
      slot(4, 'Casper Ruud', 4),
      slot(5, 'Daniil Medvedev', 3),
      slot(6, 'Grigor Dimitrov'),
      slot(7, 'Andrey Rublev'),
      slot(8, 'Carlos Alcaraz', 2),
    ],
    ...overrides,
  };
}

/** A synthetic 128 draw — the layout's other extreme. */
function bigDraw(): BracketResponse {
  return toyEightDraw({
    tournament_id: 'toy_128',
    draw_size: 128,
    slots: Array.from({ length: 128 }, (_, i) => slot(i + 1, `Player ${i + 1}`)),
  });
}

async function renderBracket(response: BracketResponse) {
  getBracketMock.mockResolvedValue(response);
  const result = render(<Bracket tournamentId={response.tournament_id} />);
  await screen.findByRole('list', { name: `${roundLabels(response.draw_size)[0]} matches` });
  return result;
}

// --------------------------------------------------------------------------
// Structure, derived from the response rather than assumed.
// --------------------------------------------------------------------------

describe('round derivation', () => {
  it('labels rounds by field size, so an 8 draw starts at the QF', () => {
    expect(roundLabels(8)).toEqual(['QF', 'SF', 'F']);
    expect(roundLabels(16)).toEqual(['R16', 'QF', 'SF', 'F']);
    expect(roundLabels(128)).toEqual(['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F']);
  });

  it('refuses a draw size that cannot halve cleanly instead of looping', () => {
    expect(roundLabels(0)).toEqual([]);
    expect(roundLabels(12)).toEqual([]);
    expect(buildRounds(12, [])).toEqual([]);
  });

  it('pairs the first round by position, not by array order', () => {
    const shuffled = [slot(2, 'Second'), slot(1, 'First'), slot(4, 'Fourth'), slot(3, 'Third')];
    const [firstRound] = buildRounds(4, shuffled);
    expect(firstRound.matches[0].top?.player).toBe('First');
    expect(firstRound.matches[0].bottom?.player).toBe('Second');
    expect(firstRound.matches[1].top?.player).toBe('Third');
  });

  it('leaves later rounds unpopulated — /bracket knows entrants, not results', () => {
    const rounds = buildRounds(8, toyEightDraw().slots);
    expect(rounds.map((r) => r.matches.length)).toEqual([4, 2, 1]);
    expect(rounds[1].matches.every((m) => m.top === null && m.bottom === null)).toBe(true);
  });
});

// --------------------------------------------------------------------------
// The ticket's headline case.
// --------------------------------------------------------------------------

describe('rendering a toy 8 draw', () => {
  it('shows 3 rounds and 4 first-round matches', async () => {
    await renderBracket(toyEightDraw());

    const headings = screen.getAllByRole('heading', { level: 3 }).map((h) => h.textContent);
    expect(headings).toHaveLength(3);
    expect(headings[0]).toContain('QF');
    expect(headings[1]).toContain('SF');
    expect(headings[2]).toContain('F');

    const firstRound = screen.getByRole('list', { name: 'QF matches' });
    expect(within(firstRound).getAllByRole('listitem')).toHaveLength(4);
    expect(within(screen.getByRole('list', { name: 'SF matches' })).getAllByRole('listitem'))
      .toHaveLength(2);
    expect(within(screen.getByRole('list', { name: 'F matches' })).getAllByRole('listitem'))
      .toHaveLength(1);
  });

  it('fetches through T4.1’s client, once, for the requested id', async () => {
    await renderBracket(toyEightDraw());
    expect(getBracketMock).toHaveBeenCalledTimes(1);
    expect(getBracketMock.mock.calls[0][0]).toBe('toy_8');
  });

  it('shows both players and their seeds in a first-round card', async () => {
    const { container } = await renderBracket(toyEightDraw());
    const card = within(screen.getByRole('list', { name: 'QF matches' })).getAllByRole(
      'listitem',
    )[0];

    expect(within(card).getByText('Jannik Sinner')).toBeDefined();
    expect(within(card).getByText('Alex de Minaur')).toBeDefined();

    const seeds = [...card.querySelectorAll('[data-cell="seed"]')].map((el) => el.textContent);
    expect(seeds).toEqual(['1', '']); // seeded top, unseeded bottom — not "null"
    expect(container.textContent).not.toMatch(/undefined|null|NaN/);
  });

  it('reserves the probability cell without putting a number in it', async () => {
    const { container } = await renderBracket(toyEightDraw());
    const cells = [...container.querySelectorAll('[data-cell="prob"]')];
    // Two per card, across all 7 matches of an 8 draw.
    expect(cells).toHaveLength(14);
    expect(cells.every((el) => el.textContent === '')).toBe(true);
  });

  it('renders later rounds as TBD rather than inventing entrants', async () => {
    await renderBracket(toyEightDraw());
    const final = within(screen.getByRole('list', { name: 'F matches' })).getAllByRole(
      'listitem',
    )[0];
    expect(within(final).getAllByText('TBD')).toHaveLength(2);
    expect(final.querySelectorAll('[data-state="unknown"]')).toHaveLength(2);
  });
});

// --------------------------------------------------------------------------
// The other extreme, and the navigation that makes it usable.
// --------------------------------------------------------------------------

describe('rendering a 128 draw', () => {
  it('derives 7 rounds and 64 first-round matches from the same code path', async () => {
    await renderBracket(bigDraw());

    const headings = screen.getAllByRole('heading', { level: 3 }).map((h) => h.textContent);
    expect(headings).toHaveLength(7);
    expect(headings[0]).toContain('R128');
    expect(headings[6]).toContain('F');

    expect(
      within(screen.getByRole('list', { name: 'R128 matches' })).getAllByRole('listitem'),
    ).toHaveLength(64);
    expect(
      within(screen.getByRole('list', { name: 'R64 matches' })).getAllByRole('listitem'),
    ).toHaveLength(32);
  });

  it('offers a round selector that narrows the board to one round', async () => {
    await renderBracket(bigDraw());

    const rail = screen.getByRole('navigation', { name: 'Round selector' });
    const buttons = within(rail)
      .getAllByRole('button')
      .map((b) => b.textContent);
    expect(buttons).toEqual(['All rounds', 'R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F']);

    fireEvent.click(within(rail).getByRole('button', { name: 'QF' }));

    expect(screen.getAllByRole('heading', { level: 3 })).toHaveLength(1);
    expect(screen.queryByRole('list', { name: 'R128 matches' })).toBeNull();
    expect(
      within(screen.getByRole('list', { name: 'QF matches' })).getAllByRole('listitem'),
    ).toHaveLength(4);
    expect(within(rail).getByRole('button', { name: 'QF' }).getAttribute('aria-pressed')).toBe(
      'true',
    );

    fireEvent.click(within(rail).getByRole('button', { name: 'All rounds' }));
    expect(screen.getAllByRole('heading', { level: 3 })).toHaveLength(7);
  });
});

// --------------------------------------------------------------------------
// Placeholder entrants — real, documented API behaviour (example_usopen_2026).
// --------------------------------------------------------------------------

describe('placeholder slots', () => {
  const withPlaceholders = () =>
    toyEightDraw({
      tournament_id: 'placeholder_8',
      slots: [
        slot(1, 'Jannik Sinner', 1),
        placeholderSlot(2, 'Qualifier'),
        slot(3, 'Taylor Fritz'),
        slot(4, 'Casper Ruud', 4),
        slot(5, 'Daniil Medvedev', 3),
        placeholderSlot(6, 'Qualifier/Lucky Loser'),
        slot(7, 'Andrey Rublev'),
        slot(8, 'Carlos Alcaraz', 2),
      ],
    });

  it('renders the draw file’s own wording, never undefined, for a null player_id', async () => {
    const { container } = await renderBracket(withPlaceholders());

    expect(screen.getByText('Qualifier')).toBeDefined();
    expect(screen.getByText('Qualifier/Lucky Loser')).toBeDefined();
    expect(container.textContent).not.toMatch(/undefined|null|NaN/);
  });

  it('marks an unfilled slot distinctly from a resolved player and from TBD', async () => {
    const { container } = await renderBracket(withPlaceholders());
    const firstRound = screen.getByRole('list', { name: 'QF matches' });

    const placeholders = [...firstRound.querySelectorAll('[data-state="placeholder"]')];
    expect(placeholders.map((el) => el.textContent)).toEqual([
      'Qualifier',
      'Qualifier/Lucky Loser',
    ]);
    expect(firstRound.querySelectorAll('[data-state="player"]')).toHaveLength(6);
    // "unfilled" and "not decided yet" are different facts and stay separate.
    expect(firstRound.querySelectorAll('[data-state="unknown"]')).toHaveLength(0);
    expect(container.querySelectorAll('[data-state="unknown"]')).toHaveLength(6);
  });
});

// --------------------------------------------------------------------------
// Loading and the typed failures.
// --------------------------------------------------------------------------

describe('loading and error states', () => {
  it('shows a status message while the request is in flight', async () => {
    let resolve: (value: BracketResponse) => void = () => {};
    getBracketMock.mockReturnValue(
      new Promise<BracketResponse>((r) => {
        resolve = r;
      }),
    );

    render(<Bracket tournamentId="toy_8" />);
    expect(screen.getByRole('status').textContent).toContain('Loading');
    expect(screen.queryByRole('alert')).toBeNull();
    expect(screen.queryByRole('list')).toBeNull();

    resolve(toyEightDraw());
    await screen.findByRole('list', { name: 'QF matches' });
    expect(screen.queryByRole('status')).toBeNull();
  });

  it('renders a 404 as its own panel — not the loading state, not a bracket', async () => {
    getBracketMock.mockRejectedValue(
      new ApiError({
        status: 404,
        statusText: 'Not Found',
        url: 'http://api.test/tournaments/nope/bracket',
        body: { detail: "Unknown tournament 'nope'. Registered ids: toy_8, toy_128." },
        message: 'ace API request failed (404 Not Found)',
      }),
    );

    render(<Bracket tournamentId="nope" />);
    const panel = await screen.findByRole('alert');

    expect(panel.dataset.errorKind).toBe('not-found');
    expect(panel.textContent).toContain('No tournament called');
    expect(panel.textContent).toContain('nope');
    expect(panel.textContent).toContain('Registered ids');
    expect(screen.queryByRole('status')).toBeNull();
    expect(screen.queryByRole('list')).toBeNull();
  });

  it('renders a 422 invalid draw file with the validator’s problem list', async () => {
    getBracketMock.mockRejectedValue(
      new ApiError({
        status: 422,
        statusText: 'Unprocessable Entity',
        url: 'http://api.test/tournaments/broken/bracket',
        body: {
          detail: {
            message: 'Draw file failed validation.',
            tournament_id: 'broken',
            source: 'broken.json',
            problems: ['bracket: 7 slots, expected 8', "unresolved entrant: 'Nobody At All'"],
          },
        },
        message: 'ace API request failed (422 Unprocessable Entity)',
      }),
    );

    render(<Bracket tournamentId="broken" />);
    const panel = await screen.findByRole('alert');

    expect(panel.dataset.errorKind).toBe('invalid-draw');
    expect(panel.textContent).toContain('failed validation');
    expect(within(panel).getByText("unresolved entrant: 'Nobody At All'")).toBeDefined();
    expect(within(panel).getByText('bracket: 7 slots, expected 8')).toBeDefined();
  });

  it('renders an unreachable API differently from either HTTP failure', async () => {
    getBracketMock.mockRejectedValue(
      new ApiNetworkError('http://api.test/tournaments/toy_8/bracket', new TypeError('failed')),
    );

    render(<Bracket tournamentId="toy_8" />);
    const panel = await screen.findByRole('alert');

    expect(panel.dataset.errorKind).toBe('network');
    expect(panel.textContent).toContain('Could not reach the ace API');
    expect(panel.textContent).not.toContain('No tournament called');
  });

  it('renders a missing base URL as its own panel, before any request is made', async () => {
    // Not a hand-written message: this is the error `apiBaseUrl()` actually
    // throws when the env var is unset, so the panel stays pinned to the prose
    // an operator really sees rather than to a copy of it that can drift.
    vi.stubEnv('VITE_API_BASE_URL', '');
    let configError: unknown;
    try {
      apiBaseUrl();
    } catch (error) {
      configError = error;
    } finally {
      vi.unstubAllEnvs();
    }
    expect(configError).toBeInstanceOf(ApiConfigError);
    getBracketMock.mockRejectedValue(configError);

    render(<Bracket tournamentId="toy_8" />);
    const panel = await screen.findByRole('alert');

    expect(panel.dataset.errorKind).toBe('config');
    expect(panel.textContent).toContain('The API base URL is not configured');
    // The actionable half: which file to fix. A misconfigured client and an
    // unreachable server are the two failures easiest to confuse, so this
    // panel must not read like the network one.
    expect(panel.textContent).toContain('.env');
    expect(panel.textContent).not.toContain('Could not reach the ace API');
    expect(screen.queryByRole('status')).toBeNull();
    expect(screen.queryByRole('list')).toBeNull();
  });

  it('refuses to lay out a draw size that is not a power of two', async () => {
    getBracketMock.mockResolvedValue(toyEightDraw({ draw_size: 12 }));

    render(<Bracket tournamentId="toy_8" />);
    const panel = await screen.findByRole('alert');

    expect(panel.dataset.errorKind).toBe('draw-size');
    expect(panel.textContent).toContain('12');
  });
});
