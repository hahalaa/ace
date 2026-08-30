// Keys a storybook run onto the bracket layout (pure): the join is derived, not sent (both sides pair survivors[2m] vs survivors[2m+1] and preserve order). An unattributable winner keeps its scoreline and stops propagating.

import type { BracketSlot, StorybookMatch, StorybookResponse } from '../api/types';
import type { MatchResult } from './MatchCard';
import type { BracketRoundModel } from './rounds';

/** One played match, in the bracket's own coordinates. */
export interface MatchOverlay {
  /** The lower-position side: a round-one slot, or the winner that reached here. */
  top: BracketSlot | null;
  bottom: BracketSlot | null;
  /** Absent when the winner could not be attributed to a side. */
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

/** Whether a storybook body describes this bracket; mismatched -> no overlay rather than results keyed onto the wrong players. */
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

/** Lay a storybook run over the bracket's rounds, returning every played match keyed by {@link overlayKey}. */
export function buildStorybookOverlay(
  rounds: BracketRoundModel[],
  story: StorybookResponse,
): StorybookOverlay {
  const played = indexMatches(story);
  const overlay = new Map<string, MatchOverlay>();
  /** Winners of the round just processed, in match order, the next round's entrants. */
  let winners: (BracketSlot | null)[] = [];

  for (const round of rounds) {
    const next: (BracketSlot | null)[] = [];

    for (const match of round.matches) {
      const first = round.index === 0;
      const top = first ? match.top : (winners[match.matchIndex * 2] ?? null);
      const bottom = first ? match.bottom : (winners[match.matchIndex * 2 + 1] ?? null);

      const result = played.get(overlayKey(round.index, match.matchIndex));
      if (result === undefined) {
        // Not played (a partial response): leave the card as the bracket had it.
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
