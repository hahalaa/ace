/**
 * Narrowing a thrown client error into the panel that should render it, pure,
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

/** Narrow `error` to the panel it should render.
 *
 * `tournamentId` names the thing that 404'd when a tournament view is loading; it
 * is optional so tournament-free views (the rankings board) can reuse the same
 * panel without inventing an id. */
export function describeError(error: unknown, tournamentId?: string): PanelContent {
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
    return {
      kind: 'unknown',
      heading: tournamentId === undefined
        ? 'Could not load this page'
        : 'Could not load this tournament',
      message: fallbackMessage,
    };
  }

  // The rankings cache has not been generated yet. The server hands back the
  // exact precompute command, shown to be copied, the same shape as a missing
  // simulation cache.
  if (hasReason(error, 'rankings_missing')) {
    return {
      kind: 'rankings-missing',
      heading: 'The rankings have not been generated yet',
      message: error.detail.message,
      command: error.detail.command,
    };
  }

  if (hasReason(error, 'rankings_unreadable')) {
    return {
      kind: 'rankings-unreadable',
      heading: 'The rankings could not be read',
      message: error.detail.message,
    };
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

  // An uploaded draw that is no longer held: evicted, or the server restarted.
  // Uploaded draws live in memory only, so a shared link can stop working.
  if (hasReason(error, 'upload_not_found')) {
    return {
      kind: 'upload-not-found',
      heading: 'This uploaded draw is no longer available',
      message: error.detail.message,
    };
  }

  // The aggregate title-odds board is cache-only and uploads have no cache, so
  // an uploaded draw can only be played out as a live storybook.
  if (hasReason(error, 'monte_carlo_unavailable_for_upload')) {
    return {
      kind: 'upload-no-odds',
      heading: 'Title odds are not available for uploaded draws',
      message: error.detail.message,
    };
  }

  // An uploaded draw that failed validation: show the accumulated problem list,
  // same as a curated draw file's 422, so a hand-entered draw is fixable at once.
  if (hasReason(error, 'draw_invalid')) {
    return {
      kind: 'draw-invalid',
      heading: 'This draw could not be uploaded',
      message: error.detail.message,
      problems: error.detail.problems,
    };
  }

  // The other structured upload failures all carry a plain message from the
  // server; show it verbatim rather than paraphrasing.
  if (
    hasReason(error, 'invalid_json') ||
    hasReason(error, 'invalid_event_date') ||
    hasReason(error, 'unsupported_media_type') ||
    hasReason(error, 'payload_too_large') ||
    hasReason(error, 'rate_limited')
  ) {
    return {
      kind: error.detail.reason.replaceAll('_', '-'),
      heading: 'This draw could not be uploaded',
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
