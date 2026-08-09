# Ace 🎾

Ever wondered who's actually going to win Wimbledon before a single ball is served? Ace simulates it, point by point, game by game, set by set, five thousand times over. We turn a 128 player draw into a full title probability table and a storybook run of the tournament complete with real scorelines. A specialised model plays out every match while a classifier trained on years of ATP history keeps the numbers grounded, so it's not just guesswork.

**🔗 Try Ace here → Coming soon**

## How it works

Every player gets a serve and return rating per surface, built from ~35,000 ATP matches (2014–2026) and weighted towards recent form. Those ratings turn into the probability that the server wins any single point — and from that one number the engine plays real tennis: points into games, games into sets, tiebreaks and deciding-set rules included, until someone actually wins the match.

On its own a point model is a bit naive about who's the better player, so a match-winner classifier trained on the same history — rankings, surface records, head-to-heads, recent form, all computed using only matches played *before* the one being predicted — is reconciled against it. The engine nudges the serve probabilities until the simulated win rate matches that blended forecast, so the scorelines you see are consistent with the odds you see. Run one bracket and you get a story; run five thousand and you get the probabilities.

The title-odds board is computed ahead of time rather than on every page load — a 128 draw at 5,000 runs is around 30 seconds of simulation, which belongs in a batch job and not in a web request. The storybook bracket *is* simulated live, on demand, from whatever seed you give it.

See [DEPLOYING.md](DEPLOYING.md) to run it locally, in Docker, or deployed.

---

Match data from [TML-Database](https://github.com/Tennismylife/TML-Database), offered in partnership with [CanalTenis](https://canaltenis.com/), and used here for educational, analytical, and research purposes. TML-Database ships no formal licence file; its own repository notice is the operative term — all use non-commercial unless explicitly permitted, and no redistribution or sale of the raw database without permission from TennisMyLife and/or the ATP. TML-Database was originally inspired by Jeff Sackmann's [ATP Matches Dataset](https://github.com/JeffSackmann/tennis_atp), which is CC BY-NC-SA 4.0 — that licence covers Sackmann's dataset and not the data vendored here. Non-commercial portfolio project.
