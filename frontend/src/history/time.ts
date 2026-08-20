/**
 * Relative timestamps for the recent-simulations list ("2 hours ago"), kept out
 * of the component so the wording is testable without a render. Pure: pass `now`
 * to pin it in a test.
 */

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/**
 * A short, plain relative time such as "just now", "5 minutes ago" or "3 days
 * ago". Anything older than a week reads as a plain date, since "51 days ago" is
 * harder to place than the date itself.
 */
export function relativeTime(then: number, now: number = Date.now()): string {
  const delta = now - then;
  if (delta < MINUTE) return 'just now';
  if (delta < HOUR) {
    const mins = Math.floor(delta / MINUTE);
    return `${mins} ${mins === 1 ? 'minute' : 'minutes'} ago`;
  }
  if (delta < DAY) {
    const hours = Math.floor(delta / HOUR);
    return `${hours} ${hours === 1 ? 'hour' : 'hours'} ago`;
  }
  if (delta < 7 * DAY) {
    const days = Math.floor(delta / DAY);
    return `${days} ${days === 1 ? 'day' : 'days'} ago`;
  }
  return new Date(then).toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}
