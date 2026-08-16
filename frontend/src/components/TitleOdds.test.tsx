// @vitest-environment jsdom
/**
 * TitleOdds tests: the columns it derives, the order it ranks in, the caveat it
 * is required to show, and the failures that must not read as "no data".
 *
 * **The 8-draw case is necessarily mocked, and that is not a shortcut.** The
 * shipped draws are all 128-slot, and a draw holding a `Qualifier` is refused by
 * T3.3 (409 `draw_not_simulatable`) with no cache, so the only cached simulation
 * is the 128 draw. A mocked payload is the only way to exercise the small-draw
 * column derivation at all, which is exactly the regression a hardcoded
 * three-column table would survive.
 *
 * `getSimulate`/`searchPlayers` are stubbed but the rest of `../api/client` is
 * real, so the error classes here are the ones the component branches on.
 */

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ApiError, ApiNetworkError, getSimulate, searchPlayers } from '../api/client';
import type {
  PlayerSearchResponse,
  SimulationPlayer,
  SimulationResponse,
  Surface,
} from '../api/types';
import TitleOdds from './TitleOdds';
import { rankPlayers, survivalColumns } from './odds';

vi.mock('../api/client', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../api/client')>();
  return { ...actual, getSimulate: vi.fn(), searchPlayers: vi.fn() };
});

const getSimulateMock = vi.mocked(getSimulate);
const searchPlayersMock = vi.mocked(searchPlayers);

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

// --------------------------------------------------------------------------
// Fixtures — shaped exactly like `data/cache/usopen_2024_atp_full.json`.
// --------------------------------------------------------------------------

// The one-line summary the reader always sees, and the full account behind the
// collapsed toggle — the two-field split `api.schemas.ModelDisclosure` ships.
const SUMMARY =
  'Not a forecast. This draw already happened, so the model is scoring a result it could have seen.';
const LIMITATION =
  'The classifier is fed a fully populated feature row, all 27 features. The remaining ' +
  'caveat is timing: rankings and recent form are an as-of-now snapshot of the loaded data.';

function metadata(overrides: Partial<SimulationResponse['metadata']> = {}) {
  return {
    mode: 'blend' as const,
    w: 0.5,
    data_through_year: 2026,
    estimator_class: 'RandomForestClassifier',
    adapter: 'common.classifier_adapter.make_classifier_prob',
    is_forecast: false,
    classifier_limitation: SUMMARY,
    classifier_limitation_detail: LIMITATION,
    source: 'example_usopen_2024_full.json',
    runs: 5000,
    seed: 0,
    workers: 1,
    generated_at: '2026-08-03T11:11:38.637248Z',
    ...overrides,
  };
}

function player(
  position: number,
  name: string,
  pTitle: number,
  reach: Record<string, number>,
  seed: number | null = null,
): SimulationPlayer {
  return {
    position,
    player: name,
    player_id: `p${position}`,
    seed,
    titles: Math.round(pTitle * 5000),
    matches_won: 1000,
    p_title: pTitle,
    expected_rounds_won: 1.25,
    reached: Object.fromEntries(Object.entries(reach).map(([k, v]) => [k, v * 5000])),
    p_reach: reach,
  };
}

/** An 8 draw: `round_labels` is `["QF","SF","F"]`, so the table shows SF and F. */
function eightDraw(overrides: Partial<SimulationResponse> = {}): SimulationResponse {
  const players = [
    player(1, 'Jannik Sinner', 0.41, { QF: 1, SF: 0.75, F: 0.6 }, 1),
    player(8, 'Carlos Alcaraz', 0.3, { QF: 1, SF: 0.7, F: 0.5 }, 2),
    player(5, 'Daniil Medvedev', 0.16, { QF: 1, SF: 0.3, F: 0.25 }, 3),
    player(4, 'Casper Ruud', 0.13, { QF: 1, SF: 0.25, F: 0.2 }, 4),
  ];
  return {
    tournament_id: 'toy_8',
    name: 'Toy Open',
    surface: 'Hard' as Surface,
    best_of: 5,
    final_set_tiebreak: '10pt_at_6_6',
    draw_size: 8,
    round_labels: ['QF', 'SF', 'F'],
    count: players.length,
    players,
    metadata: metadata(),
    ...overrides,
  };
}

/** A 128 draw: seven labels, so six survival columns after the first is dropped. */
function bigDraw(): SimulationResponse {
  const labels = ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F'];
  const reach = Object.fromEntries(labels.map((l, i) => [l, 1 - i * 0.1]));
  return eightDraw({
    tournament_id: 'toy_128',
    draw_size: 128,
    round_labels: labels,
    count: 3,
    players: [
      player(1, 'Jannik Sinner', 0.41, reach, 1),
      player(65, 'Carlos Alcaraz', 0.3, reach, 2),
      player(33, 'Novak Djokovic', 0.2, reach, 3),
    ],
  });
}

function playersResponse(name: string): PlayerSearchResponse {
  return {
    query: name,
    count: 1,
    strategy: 'exact',
    players: [
      {
        player_id: 'p1',
        name,
        skills: {
          Hard: { spw: 0.682, rpw: 0.417, n_serve_pts: 12000, n_return_pts: 11800 },
          Clay: { spw: 0.64, rpw: 0.43, n_serve_pts: 4000, n_return_pts: 3900 },
          Grass: { spw: 0.66, rpw: 0.39, n_serve_pts: 0, n_return_pts: 0 },
        },
      },
    ],
  };
}

async function renderOdds(response: SimulationResponse) {
  getSimulateMock.mockResolvedValue(response);
  const result = render(<TitleOdds tournamentId={response.tournament_id} />);
  await screen.findByRole('table');
  return result;
}

/** Column headings, which are what "data-driven" actually means here. */
function headers(): string[] {
  return within(screen.getByRole('table'))
    .getAllByRole('columnheader')
    .map((th) => th.textContent?.trim() ?? '');
}

// --------------------------------------------------------------------------
// Pure derivation.
// --------------------------------------------------------------------------

describe('column derivation', () => {
  it('drops the first round, which is 1.0 for every entrant by construction', () => {
    expect(survivalColumns(['QF', 'SF', 'F'])).toEqual(['SF', 'F']);
    expect(survivalColumns(['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F'])).toEqual([
      'R64',
      'R32',
      'R16',
      'QF',
      'SF',
      'F',
    ]);
  });

  it('yields no columns for a draw with nothing between round one and the title', () => {
    // No such draw ships today — the smallest is 8 — but `slice(1)` is the kind
    // of expression that turns into an off-by-one at the boundary, and a table
    // that renders a stray column or throws on a 2-slot payload is worse than
    // one that renders none.
    expect(survivalColumns(['F'])).toEqual([]);
    expect(survivalColumns([])).toEqual([]);
  });

  it('ranks by title probability, breaking ties by bracket position', () => {
    const tied = [
      player(9, 'Later position', 0.25, {}),
      player(2, 'Earlier position', 0.25, {}),
      player(5, 'Clear leader', 0.5, {}),
    ];
    expect(rankPlayers(tied).map((p) => p.player)).toEqual([
      'Clear leader',
      'Earlier position',
      'Later position',
    ]);
  });

  it('does not mutate the response array while sorting', () => {
    const players = eightDraw().players;
    const before = players.map((p) => p.player);
    rankPlayers(players);
    expect(players.map((p) => p.player)).toEqual(before);
  });
});

// --------------------------------------------------------------------------
// The two payload sizes — the ticket's headline regression.
// --------------------------------------------------------------------------

describe('rendering a 128-draw payload', () => {
  it('renders six survival columns, derived from round_labels', async () => {
    await renderOdds(bigDraw());
    expect(headers()).toEqual(['#', 'Seed', 'Player', 'R64', 'R32', 'R16', 'QF', 'SF', 'F', 'Title', 'E[W]']);
    expect(screen.queryByRole('columnheader', { name: 'R128' })).toBeNull();
  });

  it('renders rows in descending title-probability order', async () => {
    await renderOdds(bigDraw());
    const values = [...document.querySelectorAll('[data-cell="p_title"]')].map(
      (el) => Number.parseFloat(el.textContent ?? ''),
    );
    expect(values).toEqual([41.0, 30.0, 20.0]);
    expect(values).toEqual([...values].sort((a, b) => b - a));
  });
});

describe('rendering an 8-draw payload', () => {
  // The regression a hardcoded QF/SF/F table would pass and a 128 one would not.
  it('renders only SF and F — not six empty columns', async () => {
    await renderOdds(eightDraw());
    expect(headers()).toEqual(['#', 'Seed', 'Player', 'SF', 'F', 'Title', 'E[W]']);
    for (const dropped of ['R128', 'R64', 'R32', 'R16']) {
      expect(screen.queryByRole('columnheader', { name: dropped })).toBeNull();
    }
    // QF is this draw's *first* round, so it is skipped like R128 would be.
    expect(screen.queryByRole('columnheader', { name: 'QF' })).toBeNull();
  });

  it('reads each cell from p_reach by label, never by column position', async () => {
    await renderOdds(eightDraw());
    const top = document.querySelector('tr[data-position="1"]')!;
    expect(top.querySelector('[data-cell="reach_SF"]')?.textContent).toBe('75.0%');
    expect(top.querySelector('[data-cell="reach_F"]')?.textContent).toBe('60.0%');
    expect(top.querySelector('[data-cell="p_title"]')?.textContent).toBe('41.0%');
    expect(top.querySelector('[data-cell="seed"]')?.textContent).toBe('1');
    expect(document.body.textContent).not.toMatch(/undefined|NaN/);
  });
});

describe('rendering a degenerate payload', () => {
  // Hypothetical — the smallest draw shipped is 8 — but the header row, the
  // body cells and the expanded card's `colSpan` are all derived from the same
  // column list, so a zero-length one has to be a rendering case, not a crash.
  it('renders a single-round draw with no survival columns at all', async () => {
    await renderOdds(
      eightDraw({ tournament_id: 'toy_2', draw_size: 2, round_labels: ['F'], count: 2 }),
    );

    expect(headers()).toEqual(['#', 'Seed', 'Player', 'Title', 'E[W]']);
    expect(document.querySelectorAll('tbody tr[data-position]')).toHaveLength(4);
    expect(document.querySelectorAll('[data-cell^="reach_"]')).toHaveLength(0);
    expect(document.body.textContent).not.toMatch(/undefined|NaN/);
  });

  it('keeps the expanded card spanning the whole row when there are no columns', async () => {
    searchPlayersMock.mockResolvedValue(playersResponse('Jannik Sinner'));
    await renderOdds(
      eightDraw({ tournament_id: 'toy_2', draw_size: 2, round_labels: ['F'], count: 2 }),
    );

    fireEvent.click(screen.getByRole('button', { name: 'Jannik Sinner' }));
    const card = await screen.findByTestId('player-card');

    // 3 fixed columns + 0 survival + Title + E[W]. The card's own skill table
    // is a second `table` by now, so scope the count to the odds table's head.
    // `:scope >` matters: the card's table is nested inside this one's tbody,
    // so a plain descendant selector would count its headers too.
    expect(card.closest('td')?.getAttribute('colspan')).toBe('5');
    const oddsHead = document.querySelector('table')!.querySelectorAll(':scope > thead > tr > th');
    expect([...oddsHead].map((th) => th.textContent)).toEqual([
      '#',
      'Seed',
      'Player',
      'Title',
      'E[W]',
    ]);
  });
});

// --------------------------------------------------------------------------
// The disclosure — a required field is not the same as a rendered one.
// --------------------------------------------------------------------------

describe('disclosure', () => {
  it('renders the summary and the detail prose, and says outright this is not a forecast', async () => {
    await renderOdds(eightDraw());

    // The one-line summary is the always-visible lead.
    expect(screen.getByText(SUMMARY)).toBeDefined();
    expect(screen.getByText(/Not a forecast/)).toBeDefined();
    // The full account is present but tucked behind a collapsed <details>.
    expect(screen.getByText(LIMITATION)).toBeDefined();
    const details = document.querySelector('details') as HTMLDetailsElement;
    expect(details.open).toBe(false);
    const panel = document.querySelector('[data-forecast]') as HTMLElement;
    expect(panel.dataset.forecast).toBe('false');
  });

  it('places the caveat above the numbers, where they are read', async () => {
    await renderOdds(eightDraw());
    const disclosure = document.querySelector('[data-forecast]')!;
    const table = screen.getByRole('table');
    // DOCUMENT_POSITION_FOLLOWING: the table comes after the disclosure.
    expect(disclosure.compareDocumentPosition(table) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('changes its wording — not just a flag — when a draw really is a forecast', async () => {
    await renderOdds(eightDraw({ metadata: metadata({ is_forecast: true }) }));
    expect(screen.queryByText(/Not a forecast/)).toBeNull();
    expect(screen.getByText(/has not yet been played/)).toBeDefined();
    expect(screen.getByText(LIMITATION)).toBeDefined();
  });

  it('shows runs, seed and data year in the footnote', async () => {
    await renderOdds(eightDraw());
    expect(document.querySelector('[data-meta="runs"]')?.textContent).toContain('5,000');
    expect(document.querySelector('[data-meta="seed"]')?.textContent).toBe('seed 0');
    expect(document.querySelector('[data-meta="data_through_year"]')?.textContent).toBe(
      'data through 2026',
    );
  });
});

// --------------------------------------------------------------------------
// PlayerCard, expanded from a row.
// --------------------------------------------------------------------------

describe('player card', () => {
  it('expands a row on demand and shows skill plus this run’s probabilities', async () => {
    searchPlayersMock.mockResolvedValue(playersResponse('Jannik Sinner'));
    await renderOdds(eightDraw());

    expect(screen.queryByTestId('player-card')).toBeNull();
    fireEvent.click(screen.getByRole('button', { name: 'Jannik Sinner' }));
    const card = await screen.findByTestId('player-card');

    expect(within(card).getByText('68.2%')).toBeDefined(); // Hard SPW
    expect(within(card).getByText('41.7%')).toBeDefined(); // Hard RPW
    expect(card.querySelector('[data-stat="p_title"]')?.textContent).toBe('41.0%');
    expect(card.querySelector('[data-stat="p_reach_SF"]')?.textContent).toBe('75.0%');
    expect(searchPlayersMock.mock.calls[0][0]).toBe('Jannik Sinner');
  });

  it('marks an unmeasured surface as a baseline rather than a record', async () => {
    searchPlayersMock.mockResolvedValue(playersResponse('Jannik Sinner'));
    await renderOdds(eightDraw());
    fireEvent.click(screen.getByRole('button', { name: 'Jannik Sinner' }));
    const card = await screen.findByTestId('player-card');

    expect(card.querySelector('[data-surface="Hard"]')?.getAttribute('data-measured')).toBe('true');
    // n_serve_pts === 0 → the surface baseline, not a measurement of this player.
    expect(card.querySelector('[data-surface="Grass"]')?.getAttribute('data-measured')).toBe(
      'false',
    );
  });

  it('collapses again, and only fetches for the row asked about', async () => {
    searchPlayersMock.mockResolvedValue(playersResponse('Jannik Sinner'));
    await renderOdds(eightDraw());

    const button = screen.getByRole('button', { name: 'Jannik Sinner' });
    fireEvent.click(button);
    await screen.findByTestId('player-card');
    expect(button.getAttribute('aria-expanded')).toBe('true');

    fireEvent.click(button);
    expect(screen.queryByTestId('player-card')).toBeNull();
    expect(searchPlayersMock).toHaveBeenCalledTimes(1);
  });
});

// --------------------------------------------------------------------------
// Failures — "not simulated yet" must not read like "no data".
// --------------------------------------------------------------------------

describe('loading and error states', () => {
  it('shows a status message while the request is in flight', async () => {
    let resolve: (value: SimulationResponse) => void = () => {};
    getSimulateMock.mockReturnValue(
      new Promise<SimulationResponse>((r) => {
        resolve = r;
      }),
    );

    render(<TitleOdds tournamentId="toy_8" />);
    expect(screen.getByRole('status').textContent).toContain('Loading');
    expect(screen.queryByRole('alert')).toBeNull();

    resolve(eightDraw());
    await screen.findByRole('table');
    expect(screen.queryByRole('status')).toBeNull();
  });

  it('renders a 425 cache_missing with the exact precompute command to run', async () => {
    const command = 'python scripts/precompute_sim.py --draw toy_8 --runs 5000 --seed 0';
    getSimulateMock.mockRejectedValue(
      new ApiError({
        status: 425,
        statusText: 'Too Early',
        url: 'http://api.test/tournaments/toy_8/simulate',
        body: {
          detail: {
            reason: 'cache_missing',
            message: 'No precomputed simulation for toy_8.',
            tournament_id: 'toy_8',
            command,
          },
        },
        message: 'ace API request failed (425 Too Early)',
      }),
    );

    render(<TitleOdds tournamentId="toy_8" />);
    const panel = await screen.findByRole('alert');

    expect(panel.dataset.errorKind).toBe('cache-missing');
    expect(panel.textContent).toContain('No simulation has been run');
    // Verbatim and copyable — an operator is meant to paste this.
    expect(within(panel).getByText(command)).toBeDefined();
    expect(screen.queryByRole('table')).toBeNull();
  });

  it('renders a 409 not-simulatable listing the placeholder slots', async () => {
    getSimulateMock.mockRejectedValue(
      new ApiError({
        status: 409,
        statusText: 'Conflict',
        url: 'http://api.test/tournaments/placeholder_open_atp/simulate',
        body: {
          detail: {
            reason: 'draw_not_simulatable',
            message: 'Draw holds placeholder entrants.',
            tournament_id: 'placeholder_open_atp',
            source: 'placeholder_open.json',
            placeholder_slots: [
              { position: 2, player: 'Qualifier' },
              { position: 6, player: 'Qualifier/Lucky Loser' },
            ],
          },
        },
        message: 'ace API request failed (409 Conflict)',
      }),
    );

    render(<TitleOdds tournamentId="placeholder_open_atp" />);
    const panel = await screen.findByRole('alert');

    expect(panel.dataset.errorKind).toBe('not-simulatable');
    expect(within(panel).getByText('position 2: Qualifier')).toBeDefined();
    expect(within(panel).getByText('position 6: Qualifier/Lucky Loser')).toBeDefined();
  });

  it('distinguishes an unreachable API from a missing cache', async () => {
    getSimulateMock.mockRejectedValue(
      new ApiNetworkError('http://api.test/tournaments/toy_8/simulate', new TypeError('failed')),
    );

    render(<TitleOdds tournamentId="toy_8" />);
    const panel = await screen.findByRole('alert');

    expect(panel.dataset.errorKind).toBe('network');
    expect(panel.textContent).not.toContain('No simulation has been run');
  });
});
