/**
 * App shell — a stateless dispatcher over the query string.
 *
 * Both axes come from `window.location.search`, so the shell holds no state,
 * every screen is deep-linkable, and a shared link lands where it says. The URL
 * model is unchanged from earlier phases (`?view=`, `?tournament=`, `?seed=`) and
 * extended for the single-match view, which carries its own players/surface/
 * format/seed params (see `./components/match`).
 *
 *   /                                    → the dashboard (no draw is loaded)
 *   /?view=match&a=…&b=…&surface=…&bo=…   → one simulated matchup
 *   /?tournament=…&view=bracket          → a draw's bracket
 *   /?tournament=…&view=odds             → its title-odds board
 *   /?tournament=…&view=storybook&seed=… → one played-out run, replayable by link
 *
 * **The landing is the dashboard, not a bracket.** The app used to open straight
 * onto the US Open draw; it now opens on a choice — single match, or a full
 * tournament. Single match is the featured path.
 *
 * **The nav is two tiers.** The primary row picks the *branch* (home, single
 * match, tournament); a secondary row appears inside the tournament branch to
 * pick the *view* (bracket, title odds, storybook, upload). Both are a `switch`
 * over the `View` union returning `ReactElement`, so adding a view without a
 * case is a compile error rather than a blank screen.
 */

import type { ReactElement } from 'react';

import Bracket from './components/Bracket';
import Dashboard from './components/Dashboard';
import MatchSim from './components/MatchSim';
import Storybook from './components/Storybook';
import ThemeToggle from './components/ThemeToggle';
import TitleOdds from './components/TitleOdds';
import Upload from './components/Upload';

/** The tournament shown when the URL names none — the current showcase draw. */
const DEFAULT_TOURNAMENT_ID = 'ausopen_2026_atp_full';

const VIEWS = ['dashboard', 'match', 'bracket', 'odds', 'storybook', 'upload'] as const;
type View = (typeof VIEWS)[number];

/** The sub-views that live under the tournament branch, in nav order. */
const TOURNAMENT_VIEWS = ['bracket', 'odds', 'storybook', 'upload'] as const;
type TournamentView = (typeof TOURNAMENT_VIEWS)[number];

const TOURNAMENT_LABELS: Record<TournamentView, string> = {
  bracket: 'Bracket',
  odds: 'Title odds',
  storybook: 'Storybook',
  upload: 'Upload',
};

/** Which primary branch a view belongs to. */
type Branch = 'home' | 'match' | 'tournament';

function branchOf(view: View): Branch {
  if (view === 'dashboard') return 'home';
  if (view === 'match') return 'match';
  return 'tournament';
}

function isView(value: string): value is View {
  return (VIEWS as readonly string[]).includes(value);
}

/** The screen a view names. Exhaustive by construction — see the docstring. */
function screenFor(view: View, tournamentId: string): ReactElement {
  switch (view) {
    case 'dashboard':
      return <Dashboard tournamentId={tournamentId} />;
    case 'match':
      // The one screen that reads no tournament axis: it reads its own params.
      return <MatchSim />;
    case 'bracket':
      return <Bracket tournamentId={tournamentId} />;
    case 'odds':
      return <TitleOdds tournamentId={tournamentId} />;
    case 'storybook':
      return <Storybook tournamentId={tournamentId} />;
    case 'upload':
      return <Upload />;
  }
}

function App() {
  const params = new URLSearchParams(window.location.search);
  const tournamentId = params.get('tournament')?.trim() || DEFAULT_TOURNAMENT_ID;
  const requestedView = params.get('view')?.trim() ?? '';
  const view: View = isView(requestedView) ? requestedView : 'dashboard';
  const branch = branchOf(view);

  // Primary branch links. Home and single match carry no tournament; the
  // tournament branch enters at the bracket and keeps the current draw.
  const primary: { key: Branch; label: string; href: string }[] = [
    { key: 'home', label: 'Home', href: '?view=dashboard' },
    { key: 'match', label: 'Single match', href: '?view=match' },
    {
      key: 'tournament',
      label: 'Tournament',
      href: `?tournament=${encodeURIComponent(tournamentId)}&view=bracket`,
    },
  ];

  const tournamentHref = (target: TournamentView) =>
    `?tournament=${encodeURIComponent(tournamentId)}&view=${target}`;

  return (
    <main>
      <header className="masthead">
        <div className="brand">
          {/* The wordmark returns home. A seamed tennis ball stands in for the
             accent — two seams, one opening to each side, bulging toward the
             centre but leaving a clear gap between them, the way a real ball's
             seams read head-on. Endpoints tuck inside the rim so the round caps
             don't poke past the edge. Fill/seam are theme tokens (see .ballMark
             in index.css): --ball for the body, --ball-ink for the seam, the one
             colour guaranteed to contrast on the ball in either theme. */}
          <a className="wordmark" href="?view=dashboard" aria-label="Ace home">
            Ace
            <svg
              className="ballMark"
              viewBox="0 0 24 24"
              aria-hidden="true"
              focusable="false"
            >
              <circle className="ballBody" cx="12" cy="12" r="10.5" />
              <path className="ballSeam" d="M7 3.9 C 11 8, 11 16, 7 20.1" />
              <path className="ballSeam" d="M17 3.9 C 13 8, 13 16, 17 20.1" />
            </svg>
          </a>
        </div>
        <div className="mastheadEnd">
          <nav className="viewNav" aria-label="Sections">
            {primary.map((item) => (
              <a
                key={item.key}
                href={item.href}
                aria-current={branch === item.key ? 'page' : undefined}
              >
                {item.label}
              </a>
            ))}
          </nav>
          <ThemeToggle />
        </div>
      </header>

      {branch === 'tournament' && (
        <nav className="tournamentNav" aria-label="Tournament view">
          {TOURNAMENT_VIEWS.map((target) => (
            <a
              key={target}
              href={tournamentHref(target)}
              aria-current={view === target ? 'page' : undefined}
            >
              {TOURNAMENT_LABELS[target]}
            </a>
          ))}
        </nav>
      )}

      {screenFor(view, tournamentId)}
    </main>
  );
}

export default App;
