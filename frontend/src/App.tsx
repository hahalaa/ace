/**
 * App shell — deliberately thin, and only as much of one as T4.2 needs.
 *
 * T4.2's acceptance criteria are about the bracket rendering at both draw-size
 * extremes against the running API, which needs the component mounted
 * somewhere; nothing here is a tournament browser (that belongs with T4.3's
 * screens). The id comes from `?tournament=` so both shipped draws are
 * reachable without a code change:
 *
 *   /                                       → the 128-player 2024 US Open draw
 *   /?tournament=example_usopen_2026_atp    → the 8-slot draw, placeholders and all
 */

import Bracket from './components/Bracket';

/** Shown when the URL names no tournament. Any id from `/tournaments` works. */
const DEFAULT_TOURNAMENT_ID = 'usopen_2024_atp_full';

function App() {
  const requested = new URLSearchParams(window.location.search).get('tournament');
  const tournamentId = requested?.trim() || DEFAULT_TOURNAMENT_ID;

  return (
    <main>
      <Bracket tournamentId={tournamentId} />
    </main>
  );
}

export default App;
