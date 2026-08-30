// Typed fetch wrapper over the ace API: builds a URL, parses JSON, throws a structured error on non-2xx. Transport only: no retry, cache or debounce, and server semantics are passed through, not re-modelled.

import type {
  ApiErrorReason,
  BracketResponse,
  CacheMissingDetail,
  CacheProblemDetail,
  HealthResponse,
  MatchPlayerProblemDetail,
  MatchSimulateRequest,
  MatchSimulateResponse,
  NotSimulatableDetail,
  PlayerSearchResponse,
  RankingsMissingDetail,
  RankingsProblemDetail,
  RankingsResponse,
  SimulationResponse,
  StorybookResponse,
  TournamentListResponse,
  UploadDrawInvalidDetail,
  UploadNotFoundDetail,
  UploadProblemDetail,
  UploadResponse,
} from './types';

/** The API base URL is missing or blank, see `.env.example`. */
export class ApiConfigError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ApiConfigError';
  }
}

/** The request never produced a response: offline, DNS, CORS preflight, abort. */
export class ApiNetworkError extends Error {
  readonly url: string;

  constructor(url: string, cause: unknown) {
    super(`Could not reach the ace API at ${url}.`, { cause });
    this.name = 'ApiNetworkError';
    this.url = url;
  }
}

/** A non-2xx response: `message` is for developers, a UI branches on `status`/`reason` and renders from `detail`. */
export class ApiError extends Error {
  /** HTTP status. 404 unknown id · 409 not simulatable · 422 invalid draw/request · 425 no cache. */
  readonly status: number;
  readonly statusText: string;
  /** The URL that failed, query string included. */
  readonly url: string;
  /** The parsed response body, or the raw text when it was not JSON. */
  readonly body: unknown;
  /** `body.detail` when present, else the whole body. Shape varies by endpoint (see `ApiErrorDetail`); narrow before use. */
  readonly detail: unknown;
  /** `detail.reason` when present, the machine-readable discriminator. */
  readonly reason: ApiErrorReason | null;
  /** The draw validator's problem list, when this failure carried one. */
  readonly problems: string[] | null;

  constructor(init: {
    status: number;
    statusText: string;
    url: string;
    body: unknown;
    message: string;
  }) {
    super(init.message);
    this.name = 'ApiError';
    this.status = init.status;
    this.statusText = init.statusText;
    this.url = init.url;
    this.body = init.body;
    this.detail = extractDetail(init.body);
    this.reason = extractReason(this.detail);
    this.problems = extractProblems(this.detail);
  }
}

/** Detail type each `reason` code comes with, see the error bodies in `./types`. */
interface DetailForReason {
  draw_not_simulatable: NotSimulatableDetail;
  cache_missing: CacheMissingDetail;
  cache_unreadable: CacheProblemDetail;
  cache_stale: CacheProblemDetail;
  rankings_missing: RankingsMissingDetail;
  rankings_unreadable: RankingsProblemDetail;
  upload_not_found: UploadNotFoundDetail;
  draw_invalid: UploadDrawInvalidDetail;
  unsupported_media_type: UploadProblemDetail;
  payload_too_large: UploadProblemDetail;
  invalid_json: UploadProblemDetail;
  invalid_event_date: UploadProblemDetail;
  monte_carlo_unavailable_for_upload: UploadProblemDetail;
  rate_limited: UploadProblemDetail;
  player_not_found: MatchPlayerProblemDetail;
  player_ambiguous: MatchPlayerProblemDetail;
  player_not_simulatable: MatchPlayerProblemDetail;
  seed_out_of_range: { reason: 'seed_out_of_range'; message: string };
}

export function isApiError(error: unknown): error is ApiError {
  return error instanceof ApiError;
}

/** Narrow an unknown error to one specific API failure and its detail shape. */
export function hasReason<R extends ApiErrorReason>(
  error: unknown,
  reason: R,
): error is ApiError & { detail: DetailForReason[R] } {
  return isApiError(error) && error.reason === reason;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function extractDetail(body: unknown): unknown {
  return isRecord(body) && 'detail' in body ? body.detail : body;
}

function extractReason(detail: unknown): ApiErrorReason | null {
  if (!isRecord(detail)) return null;
  const reason = detail.reason;
  return typeof reason === 'string' ? (reason as ApiErrorReason) : null;
}

function extractProblems(detail: unknown): string[] | null {
  if (!isRecord(detail)) return null;
  const problems = detail.problems;
  if (!Array.isArray(problems)) return null;
  return problems.map(String);
}

/** A one-line summary for logs and test failures, never for the UI. */
function errorMessage(status: number, statusText: string, detail: unknown): string {
  const head = `ace API request failed (${status} ${statusText})`;
  if (typeof detail === 'string' && detail) return `${head}: ${detail}`;
  if (isRecord(detail) && typeof detail.message === 'string') {
    return `${head}: ${detail.message}`;
  }
  return head;
}

/** Per-call options. `signal` is how a type-ahead cancels a superseded search. */
export interface RequestOptions {
  signal?: AbortSignal;
}

/** Query parameters; `undefined` values are omitted from the URL entirely. */
type QueryParams = Record<string, string | number | undefined>;

/** The configured API base URL without a trailing slash; read per call so a stub works. Throws ApiConfigError if unset. */
export function apiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL;
  if (typeof configured !== 'string' || configured.trim() === '') {
    throw new ApiConfigError(
      'VITE_API_BASE_URL is not set. Copy frontend/.env.example to ' +
        'frontend/.env and point it at a running ace API.',
    );
  }
  return configured.trim().replace(/\/+$/, '');
}

/** Join the base URL, a path and its query parameters into a request URL. */
export function buildUrl(path: string, params: QueryParams = {}): string {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) query.set(key, String(value));
  }
  const search = query.toString();
  return `${apiBaseUrl()}${path}${search ? `?${search}` : ''}`;
}

/** Send one request, parse JSON, throw ApiNetworkError on a network failure and ApiError on non-2xx. The return type is asserted, not validated. */
async function request<T>(url: string, init: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (cause) {
    throw new ApiNetworkError(url, cause);
  }

  // Read the body once then parse: a proxy or crash page can return HTML, not JSON.
  const text = await response.text();
  let body: unknown = text;
  try {
    body = text === '' ? null : JSON.parse(text);
  } catch {
    body = text;
  }

  if (!response.ok) {
    throw new ApiError({
      status: response.status,
      statusText: response.statusText,
      url,
      body,
      message: errorMessage(response.status, response.statusText, extractDetail(body)),
    });
  }

  return body as T;
}

/** GET `path`, parse JSON, throw on non-2xx. `async` so a sync throw from `buildUrl` surfaces as a rejected promise. */
async function get<T>(
  path: string,
  params: QueryParams = {},
  options: RequestOptions = {},
): Promise<T> {
  return request<T>(buildUrl(path, params), {
    method: 'GET',
    headers: { Accept: 'application/json' },
    signal: options.signal,
  });
}

/** `GET /health`, liveness plus the provenance of the loaded dataset. */
export function getHealth(options?: RequestOptions): Promise<HealthResponse> {
  return get<HealthResponse>('/health', {}, options);
}

/** `GET /players?query=` (param is `query`, not `q`): one unpaginated 200 body covers unique/ambiguous/no-match; a blank query is a 422. */
export function searchPlayers(
  query: string,
  options?: RequestOptions,
): Promise<PlayerSearchResponse> {
  return get<PlayerSearchResponse>('/players', { query }, options);
}

/** `GET /rankings`, precomputed display-only Elo leaderboards. Throws 425 rankings_missing / 422 rankings_unreadable. */
export function getRankings(options?: RequestOptions): Promise<RankingsResponse> {
  return get<RankingsResponse>('/rankings', {}, options);
}

/** `GET /tournaments`, every draw file, split into `tournaments` and `invalid`. */
export function getTournaments(options?: RequestOptions): Promise<TournamentListResponse> {
  return get<TournamentListResponse>('/tournaments', {}, options);
}

/** `GET /tournaments/{id}/bracket`, every slot in order (placeholders served). Throws 404 unknown id / 422 invalid draw (`error.problems`). */
export function getBracket(
  tournamentId: string,
  options?: RequestOptions,
): Promise<BracketResponse> {
  return get<BracketResponse>(
    `/tournaments/${encodeURIComponent(tournamentId)}/bracket`,
    {},
    options,
  );
}

/** `GET /tournaments/{id}/simulate`, precomputed Monte Carlo. `top` truncates to the N likeliest champions (`top: 0` is a 422, not "all"). Throws 404/422/409/425 cache_missing (detail carries the precompute `command`). */
export function getSimulate(
  tournamentId: string,
  top?: number,
  options?: RequestOptions,
): Promise<SimulationResponse> {
  return get<SimulationResponse>(
    `/tournaments/${encodeURIComponent(tournamentId)}/simulate`,
    { top },
    options,
  );
}

/** `GET /tournaments/{id}/storybook`, one bracket simulated live per request; same id + seed gives a byte-identical body. `seed` is 0..2**32-1 (default 42, echoed in `metadata.seed`). Throws 404/422/409. */
export function getStorybook(
  tournamentId: string,
  seed?: number,
  options?: RequestOptions,
): Promise<StorybookResponse> {
  return get<StorybookResponse>(
    `/tournaments/${encodeURIComponent(tournamentId)}/storybook`,
    { seed },
    options,
  );
}

/** `POST /tournaments/upload`, submit a draw JSON verbatim and get an ephemeral `upload-...` id (lost on restart). Throws 415/413/422 draw_invalid (`error.problems`)/429. */
export async function uploadDraw(
  drawJson: string,
  options?: RequestOptions,
): Promise<UploadResponse> {
  return request<UploadResponse>(buildUrl('/tournaments/upload'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: drawJson,
    signal: options?.signal,
  });
}

/** `POST /match/simulate`, one matchup simulated live; same players/surface/format/seed gives a byte-identical body. Pass names the search returned. Throws 422 (unknown/ambiguous player, bad surface/format/seed) / 429. */
export async function simulateMatch(
  body: MatchSimulateRequest,
  options?: RequestOptions,
): Promise<MatchSimulateResponse> {
  return request<MatchSimulateResponse>(buildUrl('/match/simulate'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body),
    signal: options?.signal,
  });
}
