/**
 * Client tests: the request that goes out, and the value that comes back.
 *
 * Both halves matter and only one of them is a type-check. `tsc` proves the
 * declared return type is consistent; only a test proves the client asked for
 * `?query=` rather than `?q=`, dropped an omitted parameter instead of sending
 * `top=undefined`, and handed back the body unmangled.
 */

import { afterEach, describe, expect, it, vi } from 'vitest';

import {
  ApiConfigError,
  ApiError,
  ApiNetworkError,
  getBracket,
  getHealth,
  getSimulate,
  getStorybook,
  getTournaments,
  hasReason,
  searchPlayers,
} from './client';
import type {
  CacheMissingDetail,
  HealthResponse,
  PlayerSearchResponse,
  SimulationResponse,
  StorybookResponse,
} from './types';

const BASE = 'http://api.test';

/** A `fetch` stub typed by its call signature, so the URL argument is a string. */
type FetchStub = (url: string, init?: RequestInit) => Promise<Response>;

/** Install a `fetch` returning one JSON response, and hand back the spy. */
function mockJson(body: unknown, init: { status?: number } = {}) {
  const status = init.status ?? 200;
  const fetchMock = vi.fn<FetchStub>(
    async () =>
      new Response(JSON.stringify(body), {
        status,
        statusText: status === 200 ? 'OK' : 'Error',
        headers: { 'Content-Type': 'application/json' },
      }),
  );
  vi.stubEnv('VITE_API_BASE_URL', BASE);
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

/** The URL the single call under test requested. */
function requestedUrl(fetchMock: ReturnType<typeof mockJson>): URL {
  expect(fetchMock).toHaveBeenCalledTimes(1);
  return new URL(fetchMock.mock.calls[0][0]);
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// --------------------------------------------------------------------------
// getHealth — the simplest round trip, and the one a UI polls to decide
// whether the API is up at all.
// --------------------------------------------------------------------------

describe('getHealth', () => {
  it('GETs /health with no query string and parses the provenance fields', async () => {
    const health: HealthResponse = {
      status: 'ok',
      data_through_year: 2026,
      n_players: 1408,
    };
    const fetchMock = mockJson(health);

    const result = await getHealth();

    const url = requestedUrl(fetchMock);
    expect(url.pathname).toBe('/health');
    expect(url.search).toBe('');
    expect(result).toEqual(health);
    // data_through_year is the observed max season, not config.END_YEAR — a
    // client deciding whether a probability is current reads this one.
    expect(result.data_through_year).toBe(2026);
  });
});

// --------------------------------------------------------------------------
// searchPlayers — request shape and parsed response
// --------------------------------------------------------------------------

const ALCARAZ_SEARCH: PlayerSearchResponse = {
  query: 'alcaraz',
  count: 1,
  strategy: 'substring',
  players: [
    {
      player_id: '207989',
      name: 'Carlos Alcaraz',
      skills: {
        Hard: { spw: 0.6612, rpw: 0.4189, n_serve_pts: 12043.5, n_return_pts: 11890.2 },
        Clay: { spw: 0.6489, rpw: 0.4551, n_serve_pts: 6120.8, n_return_pts: 6033.1 },
        Grass: { spw: 0.6801, rpw: 0.3972, n_serve_pts: 2410.3, n_return_pts: 2388.7 },
      },
    },
  ],
};

describe('searchPlayers', () => {
  it('sends the query as ?query= (not ?q=) and parses the response', async () => {
    const fetchMock = mockJson(ALCARAZ_SEARCH);

    const result = await searchPlayers('alcaraz');

    const url = requestedUrl(fetchMock);
    expect(url.origin).toBe(BASE);
    expect(url.pathname).toBe('/players');
    expect(url.searchParams.get('query')).toBe('alcaraz');
    expect(url.searchParams.get('q')).toBeNull();
    expect([...url.searchParams.keys()]).toEqual(['query']);
    expect(fetchMock.mock.calls[0][1]).toMatchObject({ method: 'GET' });

    // Parsed into the declared type, not merely "some object".
    expect(result).toEqual(ALCARAZ_SEARCH);
    expect(result.strategy).toBe('substring');
    const player = result.players[0];
    expect(player.player_id).toBe('207989');
    expect(player.skills.Clay.spw).toBeCloseTo(0.6489);
    // n_serve_pts === 0 is the documented "this is the surface baseline" signal.
    expect(player.skills.Grass.n_serve_pts).toBeGreaterThan(0);
  });

  it('returns the empty and ambiguous outcomes as ordinary 200 bodies', async () => {
    mockJson({ query: 'zzz', count: 0, strategy: null, players: [] });
    const empty = await searchPlayers('zzz');
    expect(empty.count).toBe(0);
    expect(empty.strategy).toBeNull();
    expect(empty.players).toEqual([]);
  });

  it('escapes the query rather than building the URL by concatenation', async () => {
    const fetchMock = mockJson({ query: 'a b&c', count: 0, strategy: null, players: [] });
    await searchPlayers('a b&c');
    expect(requestedUrl(fetchMock).searchParams.get('query')).toBe('a b&c');
  });
});

// --------------------------------------------------------------------------
// The other endpoints' request shapes
// --------------------------------------------------------------------------

describe('request shapes', () => {
  it('omits ?top= entirely when no limit is given', async () => {
    const fetchMock = mockJson({});
    await getSimulate('usopen_2024_atp_full');
    const url = requestedUrl(fetchMock);
    expect(url.pathname).toBe('/tournaments/usopen_2024_atp_full/simulate');
    expect(url.search).toBe('');
  });

  it('sends ?top= when a limit is given', async () => {
    const fetchMock = mockJson({});
    await getSimulate('usopen_2024_atp_full', 10);
    expect(requestedUrl(fetchMock).searchParams.get('top')).toBe('10');
  });

  it('omits ?seed= so the server default applies, and sends seed 0 when asked', async () => {
    const noSeed = mockJson({});
    await getStorybook('usopen_2024_atp_full');
    expect(requestedUrl(noSeed).search).toBe('');

    vi.unstubAllGlobals();
    const zeroSeed = mockJson({});
    await getStorybook('usopen_2024_atp_full', 0);
    expect(requestedUrl(zeroSeed).searchParams.get('seed')).toBe('0');
  });

  it('percent-encodes the tournament id in the path', async () => {
    const fetchMock = mockJson({});
    await getBracket('a/b');
    expect(requestedUrl(fetchMock).pathname).toBe('/tournaments/a%2Fb/bracket');
  });

  it('strips a trailing slash from the configured base URL', async () => {
    const fetchMock = mockJson({ count: 0, tournaments: [], invalid: [] });
    vi.stubEnv('VITE_API_BASE_URL', `${BASE}/`);
    await getTournaments();
    expect(fetchMock.mock.calls[0][0]).toBe(`${BASE}/tournaments`);
  });

  it('throws ApiConfigError, without fetching, when the base URL is unset', async () => {
    const fetchMock = mockJson({});
    vi.stubEnv('VITE_API_BASE_URL', '');
    await expect(getTournaments()).rejects.toBeInstanceOf(ApiConfigError);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

// --------------------------------------------------------------------------
// Typed structure of the simulate/storybook payloads
// --------------------------------------------------------------------------

const DISCLOSURE = {
  mode: 'blend' as const,
  w: 0.5,
  data_through_year: 2026,
  estimator_class: 'RandomForestClassifier',
  adapter: 'common.classifier_adapter.make_classifier_prob',
  is_forecast: false,
  classifier_limitation: 'Not a forecast. This draw already happened.',
  classifier_limitation_detail: 'The classifier is fed a fully populated feature row …',
  source: 'example_usopen_2024_full.json',
};

const EIGHT_DRAW_SIM: SimulationResponse = {
  tournament_id: 'toy_8',
  name: 'Toy 8-draw',
  surface: 'Hard',
  best_of: 3,
  final_set_tiebreak: '7pt_at_6_6',
  draw_size: 8,
  round_labels: ['QF', 'SF', 'F'],
  count: 2,
  players: [
    {
      position: 1,
      player: 'Player One',
      player_id: '100',
      seed: 1,
      titles: 1800,
      matches_won: 4200,
      p_title: 0.36,
      expected_rounds_won: 0.84,
      reached: { QF: 5000, SF: 3100, F: 2400 },
      p_reach: { QF: 1.0, SF: 0.62, F: 0.48 },
    },
    {
      position: 3,
      player: 'Player Two',
      player_id: '101',
      seed: null,
      titles: 900,
      matches_won: 2600,
      p_title: 0.18,
      expected_rounds_won: 0.52,
      reached: { QF: 5000, SF: 2000, F: 1300 },
      p_reach: { QF: 1.0, SF: 0.4, F: 0.26 },
    },
  ],
  metadata: { ...DISCLOSURE, runs: 5000, seed: 0, workers: 1, generated_at: '2026-08-03T18:22:41Z' },
};

describe('getSimulate', () => {
  it('parses a payload whose p_reach is keyed by the draw’s own round labels', async () => {
    mockJson(EIGHT_DRAW_SIM);

    const result = await getSimulate('toy_8');

    expect(result).toEqual(EIGHT_DRAW_SIM);
    // The keys are data, not a fixed set: an 8 draw has no R128 column.
    expect(result.round_labels).toEqual(['QF', 'SF', 'F']);
    expect(Object.keys(result.players[0].p_reach)).toEqual(result.round_labels);
    expect(result.players[0].p_reach['R128']).toBeUndefined();
    // Round-survival columns follow the CLI precedent: every label after the first.
    expect(result.round_labels.slice(1)).toEqual(['SF', 'F']);
    // The disclosure survives the round trip — T4.3 has to render all of these.
    expect(result.metadata.is_forecast).toBe(false);
    expect(result.metadata.classifier_limitation).toContain('Not a forecast');
    expect(result.metadata.classifier_limitation_detail).toContain('feature row');
    expect(result.metadata.runs).toBe(5000);
    expect(result.metadata.generated_at).toBe('2026-08-03T18:22:41Z');
  });
});

const STORYBOOK: StorybookResponse = {
  tournament_id: 'toy_4',
  name: 'Toy 4-draw',
  surface: 'Clay',
  best_of: 5,
  final_set_tiebreak: '10pt_at_6_6',
  draw_size: 4,
  match_count: 3,
  rounds: [
    {
      index: 0,
      label: 'SF',
      matches: [
        {
          round_index: 0,
          round_label: 'SF',
          match_index: 0,
          winner: 'Player One',
          loser: 'Player Four',
          winner_seed: 1,
          loser_seed: null,
          scoreline: '6-4 3-6 7-6(5) 6-2',
          line: 'Player One def. Player Four 6-4 3-6 7-6(5) 6-2',
        },
      ],
    },
  ],
  champion: { position: 1, player: 'Player One', player_id: '100', seed: 1 },
  runs: [
    {
      position: 1,
      player: 'Player One',
      player_id: '100',
      seed: 1,
      is_champion: true,
      furthest_round: 'F',
      matches_won: 2,
      beat: ['Player Four', 'Player Two'],
      eliminated_by: null,
    },
  ],
  metadata: { ...DISCLOSURE, seed: 42 },
};

describe('getStorybook', () => {
  it('parses rounds, champion and the echoed seed', async () => {
    mockJson(STORYBOOK);

    const result = await getStorybook('toy_4', 42);

    expect(result).toEqual(STORYBOOK);
    expect(result.rounds[0].matches[0].scoreline).toBe('6-4 3-6 7-6(5) 6-2');
    expect(result.champion.player).toBe('Player One');
    expect(result.runs[0].eliminated_by).toBeNull();
    // metadata.seed is what turns a response into a shareable URL.
    expect(result.metadata.seed).toBe(42);
    expect(result.metadata.is_forecast).toBe(false);
  });
});

// --------------------------------------------------------------------------
// Errors carry structure, not prose
// --------------------------------------------------------------------------

describe('error handling', () => {
  it('throws an ApiError carrying the 425 cache_missing reason and command', async () => {
    const detail: CacheMissingDetail = {
      reason: 'cache_missing',
      message: "No precomputed simulation for 'toy_8'.",
      tournament_id: 'toy_8',
      command: 'python scripts/precompute_sim.py --draw toy_8 --runs 5000 --seed 0',
    };
    mockJson({ detail }, { status: 425 });

    const error = await getSimulate('toy_8').catch((thrown: unknown) => thrown);

    expect(error).toBeInstanceOf(ApiError);
    const apiError = error as ApiError;
    expect(apiError.status).toBe(425);
    expect(apiError.reason).toBe('cache_missing');
    expect(apiError.url).toBe(`${BASE}/tournaments/toy_8/simulate`);
    expect(apiError.message).toContain('No precomputed simulation');
    // Narrowed, so a later UI reads detail.command without re-parsing anything.
    if (!hasReason(error, 'cache_missing')) throw new Error('expected cache_missing');
    expect(error.detail.command).toContain('precompute_sim.py');
  });

  it('exposes the draw validator problem list from a 422', async () => {
    mockJson(
      {
        detail: {
          message: "Draw file 'broken.json' failed validation with 2 problem(s).",
          tournament_id: 'broken',
          source: 'broken.json',
          problems: ['draw_size 7 is not a power of two', "unresolved entrant 'Nobody'"],
        },
      },
      { status: 422 },
    );

    const error = (await getBracket('broken').catch((thrown: unknown) => thrown)) as ApiError;

    expect(error.status).toBe(422);
    // No reason code on this body — mirrored faithfully rather than invented.
    expect(error.reason).toBeNull();
    expect(error.problems).toEqual([
      'draw_size 7 is not a power of two',
      "unresolved entrant 'Nobody'",
    ]);
  });

  it("keeps a 404's bare-string detail", async () => {
    mockJson({ detail: "No tournament with id 'nope'. Known ids: 'toy_8'." }, { status: 404 });

    const error = (await getBracket('nope').catch((thrown: unknown) => thrown)) as ApiError;

    expect(error.status).toBe(404);
    expect(error.detail).toBe("No tournament with id 'nope'. Known ids: 'toy_8'.");
    expect(error.problems).toBeNull();
  });

  it('throws ApiNetworkError — not ApiError — when the request never lands', async () => {
    // A rejected fetch is what offline / DNS failure / a refused connection /
    // a blocked CORS preflight all look like: there is no response at all, so
    // there is no status to branch on. A UI has to tell that apart from a 4xx.
    const cause = new TypeError('fetch failed');
    vi.stubEnv('VITE_API_BASE_URL', BASE);
    vi.stubGlobal(
      'fetch',
      vi.fn<FetchStub>(async () => {
        throw cause;
      }),
    );

    const error = await getTournaments().catch((thrown: unknown) => thrown);

    expect(error).toBeInstanceOf(ApiNetworkError);
    expect(error).not.toBeInstanceOf(ApiError);
    const networkError = error as ApiNetworkError;
    expect(networkError.url).toBe(`${BASE}/tournaments`);
    expect(networkError.message).toContain(BASE);
    // The original failure is retained rather than swallowed by our message.
    expect(networkError.cause).toBe(cause);
  });

  it('keeps a non-JSON error body as text instead of losing it', async () => {
    vi.stubEnv('VITE_API_BASE_URL', BASE);
    vi.stubGlobal(
      'fetch',
      vi.fn<FetchStub>(async () => new Response('<html>502 Bad Gateway</html>', { status: 502 })),
    );

    const error = (await getTournaments().catch((thrown: unknown) => thrown)) as ApiError;

    expect(error.status).toBe(502);
    expect(error.body).toBe('<html>502 Bad Gateway</html>');
  });
});
