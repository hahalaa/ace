/**
 * The relative-time wording used by the recent-simulations list.
 */

import { describe, expect, it } from 'vitest';

import { relativeTime } from './time';

const NOW = Date.UTC(2026, 7, 20, 12, 0, 0);
const ago = (ms: number) => NOW - ms;

describe('relativeTime', () => {
  it('reads "just now" under a minute', () => {
    expect(relativeTime(ago(30_000), NOW)).toBe('just now');
  });

  it('counts minutes and hours, singular and plural', () => {
    expect(relativeTime(ago(60_000), NOW)).toBe('1 minute ago');
    expect(relativeTime(ago(5 * 60_000), NOW)).toBe('5 minutes ago');
    expect(relativeTime(ago(60 * 60_000), NOW)).toBe('1 hour ago');
    expect(relativeTime(ago(3 * 60 * 60_000), NOW)).toBe('3 hours ago');
  });

  it('counts days up to a week', () => {
    expect(relativeTime(ago(24 * 60 * 60_000), NOW)).toBe('1 day ago');
    expect(relativeTime(ago(3 * 24 * 60 * 60_000), NOW)).toBe('3 days ago');
  });

  it('falls back to a plain date beyond a week', () => {
    const result = relativeTime(ago(30 * 24 * 60 * 60_000), NOW);
    expect(result).not.toMatch(/ago/);
    expect(result).toMatch(/2026/);
  });
});
