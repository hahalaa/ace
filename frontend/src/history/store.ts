// Recent-simulations log in localStorage, scoped to this browser. Each entry is a relative deep link; only match and storybook runs are logged. Every storage touch is guarded (read falls back to empty, write silently does nothing).

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
  /** True only for uploaded-draw storybook runs, whose draw may already be gone from server memory. */
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

/** Entries kept, newest first; the cap is about legibility, not space. */
export const HISTORY_CAP = 25;

/** True if `value` is a well-formed entry, guards against corrupt or foreign data. */
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

/** The stored history, newest first, or an empty list if unreadable or malformed. */
export function readHistory(): HistoryEntry[] {
  let raw: string | null;
  try {
    raw = window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return []; // storage disabled or blocked

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
    // Persistence is a comfort, not a requirement: a failed write is swallowed.
  }
}

/** A best-effort unique id; the value only has to be distinct within this store. */
function makeId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

/** Record one result at the front; re-logging an identical `url` moves it to the front rather than duplicating. Never throws. */
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
    // If the store cannot be touched, there is nothing persisted.
  }
}

// The two call sites' recording helpers, kept here so URL/summary shapes stay in one place.

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
