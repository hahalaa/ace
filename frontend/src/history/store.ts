/**
 * The recent-simulations log — a localStorage history of the user's own results,
 * scoped to this one browser. No React, no network, no backend: a result that
 * completes is recorded here the way a browser records a page you visited, and
 * the dashboard reads it back to offer a quick way in again.
 *
 * **What gets logged, and why only two things.** Single-match runs and storybook
 * runs (curated draws and uploads alike), because each is a specific result a
 * user might want to revisit: a match outcome, a tournament champion. Bracket and
 * title-odds screens are browsing views, not a memorable result, so they are not
 * logged — see the RecentSimulations component and the task report.
 *
 * **Reproduction reuses the app's URL-state model exactly.** Each entry stores the
 * relative `?view=…&seed=…` search string the existing views already read, built
 * by the same {@link buildMatchUrl}/{@link buildShareUrl} helpers the share
 * buttons use. There is no parallel scheme: a logged entry is just a deep link.
 *
 * **Honest about ephemeral uploads.** An uploaded-draw storybook links to
 * server-side in-memory state that may be gone after a restart. Such an entry is
 * flagged `ephemeral` so the panel can label it as possibly no longer playable;
 * clicking through degrades via the views' existing not-found handling, no new
 * error UI.
 *
 * **Every storage touch is guarded.** Private-mode browsers, a disabled store, or
 * a full quota all throw on access. Reads fall back to an empty history and writes
 * silently do nothing, so the app never breaks just because history cannot persist.
 */

import type { MatchSimulateResponse, StorybookResponse } from '../api/types';
import { buildMatchUrl } from '../components/match';
import { buildShareUrl } from '../components/seed';

/** The two result kinds we log. Browsing views (bracket, odds) are excluded. */
export type HistoryKind = 'match' | 'storybook';

export interface HistoryEntry {
  /** Stable per entry, for list keys and de-duplication of a re-logged result. */
  id: string;
  kind: HistoryKind;
  /** Relative search string (`?view=…`) that reproduces the result. Origin-free. */
  url: string;
  /** Human-scannable one-liner, e.g. "Sinner vs Paul, Hard, Best of 5". */
  summary: string;
  /** Epoch millis the result completed. */
  timestamp: number;
  /**
   * True only for uploaded-draw storybook runs: the draw lives in the server's
   * memory and may already be gone, so the panel labels the entry accordingly.
   */
  ephemeral: boolean;
}

/** A completed result, before an id and timestamp are stamped on it. */
export interface NewEntry {
  kind: HistoryKind;
  url: string;
  summary: string;
  ephemeral: boolean;
}

const STORAGE_KEY = 'ace.recent-simulations.v1';

/**
 * How many entries we keep, newest first, oldest evicted past the cap.
 *
 * 25 covers a browsing session's worth of results with room to spare while
 * staying short enough to scan at a glance, and each entry is a few hundred bytes
 * against a multi-megabyte store, so the cap is about legibility rather than
 * space. It mirrors the order of magnitude of the backend's own upload store.
 */
export const HISTORY_CAP = 25;

/** True if `value` is a well-formed entry — guards against corrupt or foreign data. */
function isEntry(value: unknown): value is HistoryEntry {
  if (value === null || typeof value !== 'object') return false;
  const e = value as Record<string, unknown>;
  return (
    typeof e.id === 'string' &&
    (e.kind === 'match' || e.kind === 'storybook') &&
    typeof e.url === 'string' &&
    typeof e.summary === 'string' &&
    typeof e.timestamp === 'number' &&
    typeof e.ephemeral === 'boolean'
  );
}

/**
 * The stored history, newest first — or an empty list if nothing is stored, the
 * store is unreadable, or the payload is not what we wrote.
 */
export function readHistory(): HistoryEntry[] {
  let raw: string | null;
  try {
    raw = window.localStorage.getItem(STORAGE_KEY);
  } catch {
    // Storage disabled or blocked (some private-browsing modes throw on access).
    return [];
  }
  if (raw === null) return [];
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(isEntry);
  } catch {
    return [];
  }
}

function writeHistory(entries: HistoryEntry[]): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(entries));
  } catch {
    // Quota exceeded, disabled storage, private mode: persistence is a comfort,
    // not a requirement, so a failed write is swallowed rather than surfaced.
  }
}

/** A best-effort unique id; the value only has to be distinct within this store. */
function makeId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/**
 * Record one completed result at the front of the history.
 *
 * Re-logging the same result (identical `url`) moves the existing entry to the
 * front with a fresh timestamp rather than duplicating it, matching a browser
 * history's "most recent visit wins" behaviour. Past the cap, the oldest entry
 * is dropped. Never throws: a storage failure just means nothing persists.
 */
export function logSimulation(entry: NewEntry): void {
  const existing = readHistory().filter((e) => e.url !== entry.url);
  const next: HistoryEntry[] = [
    { ...entry, id: makeId(), timestamp: Date.now() },
    ...existing,
  ].slice(0, HISTORY_CAP);
  writeHistory(next);
}

/** Forget every logged result. Never throws. */
export function clearHistory(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Nothing to do: if the store cannot be touched, there is nothing persisted.
  }
}

// --------------------------------------------------------------------------
// Recording completed results — the two call sites' one-liners live here so the
// URL and summary shapes stay in one place and are tested without a component.
// --------------------------------------------------------------------------

/** The relative search part of a builder's absolute URL, so entries are origin-free. */
function toSearch(absolute: string): string {
  const at = absolute.indexOf('?');
  return at === -1 ? '' : absolute.slice(at);
}

/** Log a completed single-match run from its response. */
export function recordMatch(result: MatchSimulateResponse): void {
  const url = toSearch(
    buildMatchUrl({
      a: result.player_a,
      b: result.player_b,
      surface: result.surface,
      bestOf: result.best_of,
      seed: result.metadata.seed,
    }),
  );
  logSimulation({
    kind: 'match',
    url,
    summary: `${result.player_a} vs ${result.player_b}, ${result.surface}, Best of ${result.best_of}`,
    ephemeral: false,
  });
}

/** Log a completed storybook run; flag it ephemeral when the draw was an upload. */
export function recordStorybook(story: StorybookResponse): void {
  const url = toSearch(buildShareUrl(story.tournament_id, story.metadata.seed));
  logSimulation({
    kind: 'storybook',
    url,
    summary: `${story.name}, champion: ${story.champion.player}`,
    ephemeral: story.metadata.content_source === 'user_upload',
  });
}
