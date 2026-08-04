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
 *
 * Still not a tournament browser: T4.3's Files list is `TitleOdds` and
 * `PlayerCard`, and a draw picker appears in no ticket's scope yet. Switching
 * views is a link, so it costs a reload — acceptable for a two-screen app, and
 * the alternative (routing) is machinery no ticket has asked for.
 */

import Bracket from './components/Bracket';
import TitleOdds from './components/TitleOdds';

/** Shown when the URL names no tournament. Any id from `/tournaments` works. */
const DEFAULT_TOURNAMENT_ID = 'usopen_2024_atp_full';

const VIEWS = ['bracket', 'odds'] as const;
type View = (typeof VIEWS)[number];

function isView(value: string): value is View {
  return (VIEWS as readonly string[]).includes(value);
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
      <nav className="viewNav" aria-label="View">
        <a href={href('bracket')} aria-current={view === 'bracket' ? 'page' : undefined}>
          Bracket
        </a>
        <a href={href('odds')} aria-current={view === 'odds' ? 'page' : undefined}>
          Title odds
        </a>
      </nav>

      {view === 'bracket' ? (
        <Bracket tournamentId={tournamentId} />
      ) : (
        <TitleOdds tournamentId={tournamentId} />
      )}
    </main>
  );
}

export default App;
