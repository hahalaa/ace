/**
 * Narrowing a thrown client error into the panel that should render it — pure,
 * no React, so the branch table is testable without mounting anything.
 *
 * Split out of `ErrorPanel.tsx` on the `rounds.ts` precedent: component files
 * export only components. **The branches are the API's, not invented here.**
 * `ApiConfigError` (misconfigured, no request attempted), `ApiNetworkError`
 * (the request never landed), and `ApiError` split by `status` and by the
 * machine-readable `detail.reason` the server attaches precisely so a client
 * need not parse prose. Anything unrecognised still renders, with its status
 * and message, rather than collapsing to "something went wrong".
 */

import { ApiConfigError, ApiNetworkError, hasReason, isApiError } from '../api/client';

export interface PanelContent {
  kind: string;
  heading: string;
  message: string;
  /** The draw validator's list (422), rendered verbatim. */
  problems?: string[];
  /** A command the server says will fix this (425), shown to be copied. */
  command?: string;
}

/** Narrow `error` to the panel it should render. */
export function describeError(error: unknown, tournamentId: string): PanelContent {
  const fallbackMessage = error instanceof Error ? error.message : String(error);

  if (error instanceof ApiConfigError) {
    return {
      kind: 'config',
      heading: 'The API base URL is not configured',
      message: fallbackMessage,
    };
  }

  if (error instanceof ApiNetworkError) {
    return {
      kind: 'network',
      heading: 'Could not reach the ace API',
      message: fallbackMessage,
    };
  }

  if (!isApiError(error)) {
    return { kind: 'unknown', heading: 'Could not load this tournament', message: fallbackMessage };
  }

  // Nothing has been simulated yet. The server hands back the exact precompute
  // command, so the panel shows that rather than describing it.
  if (hasReason(error, 'cache_missing')) {
    return {
      kind: 'cache-missing',
      heading: 'No simulation has been run for this draw yet',
      message: error.detail.message,
      command: error.detail.command,
    };
  }

  if (hasReason(error, 'cache_stale') || hasReason(error, 'cache_unreadable')) {
    return {
      kind: error.detail.reason.replace('_', '-'),
      heading:
        error.detail.reason === 'cache_stale'
          ? 'The cached simulation is out of date'
          : 'The cached simulation could not be read',
      message: error.detail.message,
    };
  }

  // A draw still holding `Qualifier`/`Bye` slots has no player_id to simulate.
  if (hasReason(error, 'draw_not_simulatable')) {
    return {
      kind: 'not-simulatable',
      heading: 'This draw cannot be simulated yet',
      message: error.detail.message,
      problems: error.detail.placeholder_slots.map(
        (slot) => `position ${slot.position}: ${slot.player}`,
      ),
    };
  }

  if (error.status === 404) {
    return {
      kind: 'not-found',
      heading: `No tournament called "${tournamentId}"`,
      // A 404's detail is a bare string naming the registered ids.
      message: typeof error.detail === 'string' ? error.detail : error.message,
    };
  }

  if (error.status === 422) {
    const problems = error.problems;
    return {
      kind: 'invalid-draw',
      heading: 'This draw file failed validation',
      message:
        problems === null
          ? error.message
          : 'The server could not load the draw. Fix the file and restart the API.',
      ...(problems === null ? {} : { problems }),
    };
  }

  return {
    kind: 'api',
    heading: `The API returned ${error.status} ${error.statusText}`,
    message: fallbackMessage,
  };
}
