/**
 * Typed `fetch` wrapper over the ace API (Phase 3 endpoints).
 *
 * The client is a **transport, not a policy layer**. It builds a URL, sends the
 * request, parses JSON, and throws a structured error on anything that is not a
 * 2xx. It does not retry, cache, debounce, or reinterpret what the server said.
 * Three consequences worth stating, because later tickets will be tempted to
 * change them:
 *
 *   * **Server semantics are preserved, not re-modelled.** `/players` collapses
 *     unique / ambiguous / no-match into one 200 body on purpose
 *     (`api/main.py`: "ambiguity is a result, not an error"), so
 *     {@link searchPlayers} returns that one body and lets the caller inspect
 *     `count`. Splitting it into a client-side union here would re-introduce a
 *     distinction the API deliberately does not make, and would have to guess
 *     which of `strategy`/`query` each branch is allowed to keep.
 *   * **Errors carry structure, not prose.** Every non-2xx becomes an
 *     {@link ApiError} holding the status, the parsed body, the `detail`, and —
 *     when the API sent one — the machine-readable `detail.reason` and the draw
 *     validator's `problems` list. Rendering a friendly message is a later
 *     ticket's job; it must not require re-fetching or re-parsing prose.
 *   * **No base URL is baked in.** It comes from `VITE_API_BASE_URL`, read at
 *     call time so a test can stub it, and a missing value is a loud
 *     {@link ApiConfigError} rather than a request to a surprise origin.
 */

import type {
  ApiErrorReason,
  BracketResponse,
  CacheMissingDetail,
  CacheProblemDetail,
  HealthResponse,
  NotSimulatableDetail,
  PlayerSearchResponse,
  SimulationResponse,
  StorybookResponse,
  TournamentListResponse,
  UploadDrawInvalidDetail,
  UploadNotFoundDetail,
  UploadProblemDetail,
  UploadResponse,
} from './types';

// --------------------------------------------------------------------------
// Errors
// --------------------------------------------------------------------------

/** The API base URL is missing or blank — see `.env.example`. */
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

/**
 * A non-2xx response, with everything a UI needs to say something specific.
 *
 * `message` is for a developer (console, test failure). A UI should branch on
 * {@link ApiError.status} and {@link ApiError.reason} and render from
 * {@link ApiError.detail} — that is why all of it is retained rather than
 * flattened into a string.
 */
export class ApiError extends Error {
  /** HTTP status. 404 unknown id · 409 not simulatable · 422 invalid draw/request · 425 no cache. */
  readonly status: number;
  readonly statusText: string;
  /** The URL that failed, query string included. */
  readonly url: string;
  /** The parsed response body, or the raw text when it was not JSON. */
  readonly body: unknown;
  /**
   * `body.detail` when the body had one, else the whole body.
   *
   * Shape varies by endpoint and is deliberately not normalised: a 404 detail
   * is a bare string, a request-validation 422 is an array, and the draw/cache
   * failures are objects — `ApiErrorDetail` in `./types` enumerates all six.
   *
   * Typed `unknown` rather than `ApiErrorDetail` because a failure that never
   * reached the app (a proxy's HTML, a crash page) is none of them, and
   * claiming otherwise would let a caller read `.message` off a string. Narrow
   * before use: {@link hasReason} for the `reason`-bearing bodies,
   * {@link ApiError.problems} for the draw validator's list, or a plain
   * `typeof` check for the 404.
   */
  readonly detail: unknown;
  /** `detail.reason` when present — the machine-readable discriminator. */
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

/** Detail type each `reason` code comes with — see the error bodies in `./types`. */
interface DetailForReason {
  draw_not_simulatable: NotSimulatableDetail;
  cache_missing: CacheMissingDetail;
  cache_unreadable: CacheProblemDetail;
  cache_stale: CacheProblemDetail;
  upload_not_found: UploadNotFoundDetail;
  draw_invalid: UploadDrawInvalidDetail;
  unsupported_media_type: UploadProblemDetail;
  payload_too_large: UploadProblemDetail;
  invalid_json: UploadProblemDetail;
  invalid_event_date: UploadProblemDetail;
  monte_carlo_unavailable_for_upload: UploadProblemDetail;
  rate_limited: UploadProblemDetail;
}

export function isApiError(error: unknown): error is ApiError {
  return error instanceof ApiError;
}

/**
 * Narrow an unknown error to one specific API failure and its detail shape.
 *
 * ```ts
 * if (hasReason(error, 'cache_missing')) showCommand(error.detail.command);
 * ```
 */
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

/** A one-line summary for logs and test failures — never for the UI. */
function errorMessage(status: number, statusText: string, detail: unknown): string {
  const head = `ace API request failed (${status} ${statusText})`;
  if (typeof detail === 'string' && detail) return `${head}: ${detail}`;
  if (isRecord(detail) && typeof detail.message === 'string') {
    return `${head}: ${detail.message}`;
  }
  return head;
}

// --------------------------------------------------------------------------
// Transport
// --------------------------------------------------------------------------

/** Per-call options. `signal` is how a type-ahead cancels a superseded search. */
export interface RequestOptions {
  signal?: AbortSignal;
}

/** Query parameters; `undefined` values are omitted from the URL entirely. */
type QueryParams = Record<string, string | number | undefined>;

/**
 * The configured API base URL, without a trailing slash.
 *
 * Read from the environment on **every** call rather than captured at module
 * load, so `vi.stubEnv` works and so a missing value fails at the call that
 * needed it.
 *
 * @throws {ApiConfigError} if `VITE_API_BASE_URL` is unset or blank.
 */
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

/**
 * Send one request to `url`, parse JSON, throw on anything that is not a 2xx.
 *
 * Shared by {@link get} and {@link uploadDraw} so the error handling — network
 * failure → {@link ApiNetworkError}, non-2xx → structured {@link ApiError} — is
 * one implementation, not one per HTTP method.
 *
 * The return type is asserted, not validated: `T` mirrors an `extra="forbid"`
 * Pydantic model, so the server has already guaranteed the shape.
 */
async function request<T>(url: string, init: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (cause) {
    throw new ApiNetworkError(url, cause);
  }

  // Read the body once, then parse: an error body is JSON in every case the API
  // produces, but a proxy or a crash can return HTML, and losing that text
  // would leave the thrown error with nothing to show.
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

/**
 * GET `path`, parse JSON, throw on anything that is not a 2xx.
 *
 * `async`, so a synchronous throw from `buildUrl` (a missing base URL →
 * {@link ApiConfigError}) surfaces as a rejected promise, not a throw at the
 * call site — every caller `await`s and expects to catch, never to try/catch
 * the call itself.
 */
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

// --------------------------------------------------------------------------
// Endpoints
// --------------------------------------------------------------------------

/** `GET /health` — liveness plus the provenance of the loaded dataset. */
export function getHealth(options?: RequestOptions): Promise<HealthResponse> {
  return get<HealthResponse>('/health', {}, options);
}

/**
 * `GET /players?query=` — fuzzy player search.
 *
 * The parameter is **`query`**, not `q`. One 200 body covers all three resolver
 * outcomes; inspect it rather than expecting a thrown error:
 *
 *   * `count === 1` — a unique match.
 *   * `count > 1` — ambiguous; `players` holds every candidate, in resolver order.
 *   * `count === 0` — nothing matched (`strategy` is `null`). Not an error.
 *
 * A blank or whitespace-only query is the one failure: a 422 from validation.
 * This is passed through rather than pre-empted, so the client keeps no second
 * copy of the server's validation rule.
 *
 * The response is **unpaginated** — `?query=a` returns ~1,220 players — so a
 * type-ahead should debounce and pass `options.signal` to drop superseded
 * requests.
 */
export function searchPlayers(
  query: string,
  options?: RequestOptions,
): Promise<PlayerSearchResponse> {
  return get<PlayerSearchResponse>('/players', { query }, options);
}

/** `GET /tournaments` — every draw file, split into `tournaments` and `invalid`. */
export function getTournaments(options?: RequestOptions): Promise<TournamentListResponse> {
  return get<TournamentListResponse>('/tournaments', {}, options);
}

/**
 * `GET /tournaments/{id}/bracket` — the resolved draw, every slot in order.
 *
 * Placeholder slots are served, not refused; only the simulation endpoints
 * reject them.
 *
 * @throws {ApiError} 404 unknown id · 422 the draw file failed validation
 * (`error.problems` holds the validator's list).
 */
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

/**
 * `GET /tournaments/{id}/simulate` — precomputed Monte Carlo probabilities.
 *
 * Nothing is simulated per request; the server reads a cache file written by
 * `scripts/precompute_sim.py`.
 *
 * @param top Return only the `top` most likely champions. **Omit it for the
 *   whole field — `top: 0` is a 422, not "all"** (unlike the CLI's `--top 0`).
 *   `draw_size` still reports the full field, so truncation stays visible.
 * @throws {ApiError} 404 unknown id · 422 invalid draw file or unreadable/stale
 *   cache · 409 `draw_not_simulatable` · 425 `cache_missing`, whose detail
 *   carries the exact precompute `command`.
 */
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

/**
 * `GET /tournaments/{id}/storybook` — one bracket played out live, for one seed.
 *
 * The **only** endpoint that simulates per request (~0.5–1.5 s for a 128 draw).
 * Same id + same seed → a byte-identical body, which is what makes the URL
 * shareable.
 *
 * @param seed RNG seed, `0 … 2**32-1`. Omit for the server default (42); the
 *   seed that ran is echoed in `metadata.seed` either way, so any response can
 *   be turned into an explicit shareable URL.
 * @throws {ApiError} 404 unknown id · 422 invalid draw file · 409
 *   `draw_not_simulatable`. There is no 425 — nothing is cached.
 */
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

/**
 * `POST /tournaments/upload` — submit a draw JSON and get a generated id back.
 *
 * The `drawJson` string is sent verbatim as the request body, so the server
 * runs its own validation on exactly what the user provided (a malformed file
 * comes back as an {@link ApiError} whose `problems` list is the validator's).
 * The returned `tournament_id` is an `upload-…` id usable with
 * {@link getBracket} and {@link getStorybook}.
 *
 * Uploaded draws are **ephemeral**: held in the server's memory only, so a link
 * to one can stop working after a restart. The caller should say so in the UI.
 *
 * @throws {ApiError} 415 non-JSON · 413 too large · 422 invalid JSON, bad
 *   `event_date`, or `draw_invalid` (`error.problems` holds the validator's
 *   list) · 429 rate limited.
 */
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
