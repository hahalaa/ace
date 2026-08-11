/**
 * App shell — deliberately thin, and only as much of one as the shipped
 * components need to be reachable.
 *
 * Both axes come from the query string rather than React state, so the shell
 * holds no state at all, every view is deep-linkable, and the pattern that
 * already picks the tournament also picks the screen:
 *
 *   /                                       → the 128-player 2024 US Open bracket
 *   /?tournament=example_usopen_2026_atp    → the 8-slot draw, placeholders and all
 *   /?view=odds                             → that draw's title-probability board
 *   /?view=storybook&seed=42                → one played-out run, replayable by link
 *
 * **The dispatch is a real switch over `View`, not a ternary.** T4.3 shipped two
 * screens and a `?:`; T4.4 makes three, and the switch is what makes a fourth a
 * *compile* error rather than a silent fall-through to the bracket — the
 * function's return type is `ReactElement`, so a `View` with no `case` fails to
 * return. The `VIEWS` tuple stays the single list: it defines the type, guards
 * the URL value, and orders the nav.
 *
 * Still not a tournament browser: a draw picker appears in no ticket's scope
 * yet. Switching views is a link, so it costs a reload — acceptable for a
 * three-screen app, and the alternative (routing) is machinery no ticket has
 * asked for. The nav deliberately carries only `tournament` and `view`: the
 * `seed` axis belongs to the storybook screen, which writes it into the URL
 * itself once a run exists.
 */

import type { ReactElement } from 'react';

import Bracket from './components/Bracket';
import Storybook from './components/Storybook';
import TitleOdds from './components/TitleOdds';

/** Shown when the URL names no tournament. Any id from `/tournaments` works. */
const DEFAULT_TOURNAMENT_ID = 'usopen_2024_atp_full';

const VIEWS = ['bracket', 'odds', 'storybook'] as const;
type View = (typeof VIEWS)[number];

/** Nav wording per view, in `VIEWS` order. */
const VIEW_LABELS: Record<View, string> = {
  bracket: 'Bracket',
  odds: 'Title odds',
  storybook: 'Storybook',
};

function isView(value: string): value is View {
  return (VIEWS as readonly string[]).includes(value);
}

/** The screen a view names. Exhaustive by construction — see the docstring. */
function screenFor(view: View, tournamentId: string): ReactElement {
  switch (view) {
    case 'bracket':
      return <Bracket tournamentId={tournamentId} />;
    case 'odds':
      return <TitleOdds tournamentId={tournamentId} />;
    case 'storybook':
      return <Storybook tournamentId={tournamentId} />;
  }
}

function App() {
  const params = new URLSearchParams(window.location.search);
  const tournamentId = params.get('tournament')?.trim() || DEFAULT_TOURNAMENT_ID;
  const requestedView = params.get('view')?.trim() ?? '';
  const view: View = isView(requestedView) ? requestedView : 'bracket';

  const href = (target: View) =>
    `?tournament=${encodeURIComponent(tournamentId)}&view=${target}`;

  return (
    <main>
      {/* Masthead. The wordmark is deliberately not a link — App.test pins the
          shell at exactly three anchors, the three views — so it is plain text
          with a single accent dot standing in for the ball. */}
      <header className="masthead">
        <div className="brand">
          <p className="wordmark">
            Ace<span className="ballMark" aria-hidden="true" />
          </p>
          <span className="brandTag">Grand Slam simulator</span>
        </div>
        <nav className="viewNav" aria-label="View">
          {VIEWS.map((target) => (
            <a
              key={target}
              href={href(target)}
              aria-current={view === target ? 'page' : undefined}
            >
              {VIEW_LABELS[target]}
            </a>
          ))}
        </nav>
      </header>

      {screenFor(view, tournamentId)}
    </main>
  );
}

export default App;
