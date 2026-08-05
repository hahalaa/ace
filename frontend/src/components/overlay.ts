/**
 * Keying a storybook run onto the bracket's layout — pure, no React, on the
 * `rounds.ts`/`errors.ts` precedent. (Named `overlay.ts` rather than
 * `storybook.ts` because a case-insensitive filesystem cannot hold that
 * alongside `Storybook.tsx`.)
 *
 * **Two endpoints, one structure, and they already agree.** `/bracket` sends
 * entrants (`draw_size` slots in `position` order) and `/storybook` sends
 * results (`round_index`, `match_index`, winner, loser, scoreline). Nothing in
 * either body links a result back to a slot, so the join has to be derived —
 * and it is derivable exactly, because both sides implement the *same* pairing
 * rule:
 *
 *   * `sim.tournament.simulate_bracket` walks `labels = round_labels(draw_size)`
 *     with `enumerate`, and inside each round pairs `survivors[2m]` vs
 *     `survivors[2m+1]`, appending winners in match order. So `round_index` is
 *     an index into the same label list `rounds.ts` builds, and `match_index`
 *     counts matches top-of-draw-first within it.
 *   * Round one's survivors are `draw.bracket`, position-ordered — the very
 *     slots `/bracket` serves — so `buildRounds`'s `ordered[2m]`/`ordered[2m+1]`
 *     and the simulator's `survivors[2m]`/`survivors[2m+1]` are the same two
 *     entrants.
 *   * Because winners are appended in match order, **the ordering is preserved
 *     round to round** (`sim/tournament.py`'s module docstring says so
 *     explicitly). Match `m` of round `r` is therefore contested by the winners
 *     of matches `2m` and `2m+1` of round `r−1`, in that order — which is how
 *     the later rounds `/bracket` leaves as `null` get filled here.
 *
 * **Which side won is read off the names, and a failure to read it is not
 * invented.** `StorybookMatch` carries `winner`/`loser` but no position, so the
 * winner is identified by matching against the two slots the propagation above
 * already knows. If neither matches — a body disagreeing with its own draw, i.e.
 * a server bug — the match keeps its scoreline, claims no winner, and stops
 * propagating rather than guessing a side; the rounds above it then render as
 * the `TBD`s they were before the run. Silence beats a plausible-looking lie
 * about who beat whom.
 *
 * **Nothing here re-derives a fact the response already states.** The champion
 * comes from `StorybookResponse.champion`, not from the last match; round labels
 * come from the draw size, not from `round_label`; scorelines are copied, never
 * assembled from set scores.
 */

import type { BracketSlot, StorybookMatch, StorybookResponse } from '../api/types';
import type { MatchResult } from './MatchCard';
import type { BracketRoundModel } from './rounds';

/** One played match, in the bracket's own coordinates. */
export interface MatchOverlay {
  /** The lower-position side: a round-one slot, or the winner that reached here. */
  top: BracketSlot | null;
  bottom: BracketSlot | null;
  /** Absent when the winner could not be attributed to a side — see the module docstring. */
  topResult?: MatchResult;
  bottomResult?: MatchResult;
  /** Winner-first, copied from the response. */
  scoreline: string;
}

/** `roundIndex:matchIndex` → what happened there. */
export type StorybookOverlay = ReadonlyMap<string, MatchOverlay>;

/** The overlay key for one position in the bracket. */
export function overlayKey(roundIndex: number, matchIndex: number): string {
  return `${roundIndex}:${matchIndex}`;
}

/**
 * Whether a storybook body describes the bracket it is about to be drawn onto.
 *
 * Guards the one way the two endpoints can disagree in practice: a response
 * still in flight for the *previous* tournament, or a draw file edited between
 * the two requests. Mismatched → no overlay at all, rather than results keyed
 * onto the wrong players.
 */
export function describesBracket(
  story: StorybookResponse,
  tournamentId: string,
  drawSize: number,
): boolean {
  return story.tournament_id === tournamentId && story.draw_size === drawSize;
}

/** Index the response's matches by their bracket coordinates. */
function indexMatches(story: StorybookResponse): Map<string, StorybookMatch> {
  const byPosition = new Map<string, StorybookMatch>();
  for (const round of story.rounds) {
    for (const match of round.matches) {
      byPosition.set(overlayKey(match.round_index, match.match_index), match);
    }
  }
  return byPosition;
}

/**
 * Lay a storybook run over the bracket's rounds.
 *
 * @param rounds The layout from {@link buildRounds} — round one populated, the
 *   rest `null`, exactly as `/bracket` knows it.
 * @param story The `/storybook` body for the same draw.
 * @returns Every match the run played, keyed by {@link overlayKey}. A match the
 *   response omits is simply absent, and renders as it did before the run.
 */
export function buildStorybookOverlay(
  rounds: BracketRoundModel[],
  story: StorybookResponse,
): StorybookOverlay {
  const played = indexMatches(story);
  const overlay = new Map<string, MatchOverlay>();
  /** Winners of the round just processed, in match order — the next round's entrants. */
  let winners: (BracketSlot | null)[] = [];

  for (const round of rounds) {
    const next: (BracketSlot | null)[] = [];

    for (const match of round.matches) {
      const first = round.index === 0;
      const top = first ? match.top : (winners[match.matchIndex * 2] ?? null);
      const bottom = first ? match.bottom : (winners[match.matchIndex * 2 + 1] ?? null);

      const result = played.get(overlayKey(round.index, match.matchIndex));
      if (result === undefined) {
        // Not played (a partial response). Leave the card as the bracket had it
        // and let the rounds above it stay unknown.
        next.push(null);
        continue;
      }

      const topWon = top !== null && top.player === result.winner;
      const bottomWon = !topWon && bottom !== null && bottom.player === result.winner;

      overlay.set(overlayKey(round.index, match.matchIndex), {
        top,
        bottom,
        ...(topWon || bottomWon
          ? {
              topResult: topWon ? ('won' as const) : ('lost' as const),
              bottomResult: topWon ? ('lost' as const) : ('won' as const),
            }
          : {}),
        scoreline: result.scoreline,
      });

      next.push(topWon ? top : bottomWon ? bottom : null);
    }

    winners = next;
  }

  return overlay;
}
