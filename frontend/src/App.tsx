// Stateless app shell: both axes come from window.location.search, so every screen is deep-linkable. The view dispatch is a switch over the View union returning ReactElement, so a missing case is a compile error.

import type { ReactElement } from 'react';

import Bracket from './components/Bracket';
import Dashboard from './components/Dashboard';
import MatchSim from './components/MatchSim';
import Rankings from './components/Rankings';
import Storybook from './components/Storybook';
import ThemeToggle from './components/ThemeToggle';
import TitleOdds from './components/TitleOdds';
import Upload from './components/Upload';

/** The tournament shown when the URL names none, the current showcase draw. */
const DEFAULT_TOURNAMENT_ID = 'ausopen_2026_atp_full';

const VIEWS = ['dashboard', 'match', 'rankings', 'bracket', 'odds', 'storybook', 'upload'] as const;
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
type Branch = 'home' | 'match' | 'rankings' | 'tournament';

function branchOf(view: View): Branch {
  if (view === 'dashboard') return 'home';
  if (view === 'match') return 'match';
  if (view === 'rankings') return 'rankings';
  return 'tournament';
}

function isView(value: string): value is View {
  return (VIEWS as readonly string[]).includes(value);
}

/** The screen a view names. Exhaustive by construction. */
function screenFor(view: View, tournamentId: string): ReactElement {
  switch (view) {
    case 'dashboard':
      return <Dashboard tournamentId={tournamentId} />;
    case 'match':
      return <MatchSim />;
    case 'rankings':
      return <Rankings />;
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

  // Primary branch links; the tournament branch enters at the bracket and keeps the current draw.
  const primary: { key: Branch; label: string; href: string }[] = [
    { key: 'home', label: 'Home', href: '?view=dashboard' },
    { key: 'match', label: 'Single match', href: '?view=match' },
    {
      key: 'tournament',
      label: 'Tournament',
      href: `?tournament=${encodeURIComponent(tournamentId)}&view=bracket`,
    },
    { key: 'rankings', label: 'Rankings', href: '?view=rankings' },
  ];

  const tournamentHref = (target: TournamentView) =>
    `?tournament=${encodeURIComponent(tournamentId)}&view=${target}`;

  return (
    <main>
      <header className="masthead">
        <div className="brand">
          {/* Wordmark returns home; the ball-mark fill/seam are theme tokens (--ball / --ball-ink). */}
          <a className="wordmark" href="?view=dashboard" aria-label="Ace home">
            Ace
            <svg
              className="ballMark"
              viewBox="0 0 24 24"
              aria-hidden="true"
              focusable="false"
            >
              <circle className="ballBody" cx="12" cy="12" r="10.5" />
              <path className="ballSeam" d="M7 3.9 C 9.5 8, 9.5 16, 7 20.1" />
              <path className="ballSeam" d="M17 3.9 C 14.5 8, 14.5 16, 17 20.1" />
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
