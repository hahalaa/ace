# Deploying Ace

Four ways in, in rough order of effort: [run it locally without Docker](#run-it-locally-without-docker) ·
[run it with Docker](#run-it-with-docker) · [deploy the frontend to a static host](#deploy-the-frontend-to-a-static-host) ·
[keep the data fresh automatically](#keeping-the-data-fresh--the-scheduled-refresh-t53).

## Run it locally, without Docker

Python 3.11+ and Node 22+. Everything is offline — the match data is vendored
into `data/raw/` and nothing fetches from the internet at request time.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`python3`, not `python`, for that first line only — macOS ships no bare `python`
and the command fails outright there. Once the venv is activated `python` exists
and every command below uses it.

**Two artefacts are gitignored and are in no clean checkout**, so a fresh clone
has to build them once. This is the same pair the Docker build regenerates:

```bash
python src/predictor.py                                   # trains outputs/tennis_model.pkl (~1 min)
                                                          # ends in a REPL — Ctrl-D to exit
python scripts/precompute_sim.py --draw usopen_2024_atp_full --runs 5000 --seed 0   # ~29 s
```

Without the model the API fails fast at startup (`FileNotFoundError` on
`outputs/tennis_model.pkl`, after the ~2 s data load and before it binds a port).
Without the cache the odds screen answers `425 cache_missing` for every
**simulatable** draw — a draw still holding `Qualifier`/`Bye` slots answers
`409 draw_not_simulatable` instead, because simulability is checked before the
cache is looked at, so that verdict does not change once you precompute.

```bash
PYTHONPATH=src uvicorn api.main:app --reload               # API on :8000
cd frontend && npm install && cp .env.example .env && npm run dev   # web app on :5173
```

`frontend/.env` is gitignored and **required** — `.env.example` is the tracked
copy and already points at `http://localhost:8000`. Tests: `pytest -q` from the
root, and `npx vitest run && npm run typecheck && npm run lint` in `frontend/`.

## Run it with Docker

Everything below is offline: the match data is vendored into the repo and baked
into the image, and no container makes a network call at start or per request.

### Full stack — API + web app

```bash
docker compose up --build          # then open http://localhost:4173
```

The API lands on `http://localhost:8000` (try `curl localhost:8000/health`) and
the web app on `http://localhost:4173`, served by nginx. Port 4173 is not
arbitrary — it is already in `config.API_ALLOWED_ORIGINS`, and CORS here is an
explicit allow-list, never `*`. Serve the frontend on some other port and you
must add that origin to the list and rebuild the API image.

**The frontend's API URL is frozen at image-build time.** Vite inlines
`VITE_API_BASE_URL` into the bundle and the minifier folds the lookup away, so
it is a build `ARG`, never an `environment:` entry — `docker compose up` cannot
repoint a built image at a different API. To point the app somewhere else:

```bash
VITE_API_BASE_URL=https://ace-api.example.com docker compose build frontend
```

### API only

```bash
docker build -t ace-api .
docker run --rm -p 8000:8000 ace-api
curl localhost:8000/health
```

### The first build takes a while, and here is why

Two files the API needs at runtime are gitignored, so they are in no clean
checkout: `outputs/tennis_model.pkl` (without it the API fails fast at startup)
and `data/cache/*.json` (without it `/simulate` answers `425 cache_missing` for
every simulatable draw). **The build regenerates both** — it trains the classifier and runs
the Monte Carlo in a builder stage — rather than copying whatever happens to be
on your machine. `.dockerignore` excludes both paths from the build context, so
every build behaves like a clean checkout.

Trim the Monte Carlo when you just want a running API quickly:

```bash
docker build --build-arg PRECOMPUTE_RUNS=200 -t ace-api .
```

One consequence worth knowing: `predictor.py` persists whichever of four
classifiers wins on the held-out season, so two builds from identical source
can bake different estimator types. The build log prints which one
(`[build] persisted estimator: …`), and every `/simulate` and `/storybook`
response carries it as `metadata.estimator_class`. If you need an exactly
reproducible deployment, build once and deploy the **image digest**.

### CI check (manual — deliberately not wired into T5.2's workflow)

T5.2's workflow checks **source** only (`pytest`, `tsc`, `vite build`) and never
builds an image, so this stays a manual step — build the image, start it, and
prove `/health` answers:

```bash
docker build --build-arg PRECOMPUTE_RUNS=200 -t ace-api:ci .
docker run -d --name ace-api-ci -p 8000:8000 ace-api:ci
curl -fsS --retry 30 --retry-delay 2 --retry-connrefused --retry-all-errors \
     http://localhost:8000/health
docker rm -f ace-api-ci
```

**`--retry-all-errors` is load-bearing, not decoration.** Docker publishes the
port the moment the container starts, but uvicorn only binds after ~10 s of
startup work (loading the CSVs, engineering features, unpickling the model), so
the first requests get *connection accepted, then closed* — curl exit **52,
"Empty reply from server"**. That is not one of the errors plain `--retry`
treats as transient, so without this flag the retry loop never engages and the
job fails against a perfectly good image. Verified both ways: the snippet
without it exits 52 on the first attempt; with it, curl rides out two 52s and
succeeds in ~4 s. (Needs curl ≥ 7.71 — `ubuntu-latest` has it.) The
alternative, if you prefer to lean on the image's own `HEALTHCHECK`:

```bash
until [ "$(docker inspect -f '{{.State.Health.Status}}' ace-api-ci)" = healthy ]; do sleep 2; done
curl -fsS http://localhost:8000/health
```

And to prove the container needs no network at all:

```bash
docker run --rm --network none ace-api:ci python -c "
from fastapi.testclient import TestClient
import api.main as m
with TestClient(m.app) as c:
    print(c.get('/health').json())
    print(c.get('/tournaments/usopen_2024_atp_full/simulate?top=3').status_code)"
```

---

## Deploy the frontend to a static host

The container path above is not the only one. `frontend/dist/` is a folder of
static files with no server side, so any static host serves it — and
**`render.yaml` at the repo root is the committed, checked Blueprint for that**:
`rootDir: frontend`, `npm ci && npm run build`, publish `./dist`, immutable
`/assets/*` with a `no-cache` `index.html`, and deliberately **no SPA rewrite**
(the app does no path-based routing, so no deep link ever requests a path other
than `/` — the query string carries the whole view state).

**Read `render.yaml`'s header comment before deploying.** It is the only place
the deploy contract is written down, and it also carries the portable version
of the three facts any other host needs. Those other hosts are documented
translations, not verified deploys. Two things bite on a real deploy regardless
of host:

1. `VITE_API_BASE_URL` must be set **in the host's build environment**, not as a
   runtime variable — same build-time inlining as the Docker path.
2. The frontend's deployed origin must be appended to
   `config.API_ALLOWED_ORIGINS` and the API redeployed, over HTTPS if the
   frontend is on HTTPS. CORS is an explicit allow-list and never `*`.

---

## Keeping the data fresh — the scheduled refresh (T5.3)

`.github/workflows/refresh-and-simulate.yml` refreshes the vendored match data,
retrains the classifier, regenerates the Monte Carlo cache and **opens a pull
request** with the result. `scripts/update_and_cache.py` is the orchestrator —
all of the sequencing and every abort live there, so the same run is
reproducible from a terminal:

```bash
python scripts/update_and_cache.py                                  # the scheduled run
python scripts/update_and_cache.py --draw usopen_2024_atp_full      # one draw
python scripts/update_and_cache.py --no-refresh --force             # re-simulate only
python scripts/update_and_cache.py --summary-json summary.json
```

`--draw` takes a **`tournament_id`**, which is the field inside the draw JSON and not always the filename stem. The two shipped today are `usopen_2024_atp_full` (in `example_usopen_2024_full.json`) and `example_usopen_2026_atp` (in `example_usopen_2026.json`, currently unsimulatable — it still holds placeholder entrants). Passing an unknown id fails with the registry's own list of what *is* addressable.

### What it does not do

**It never touches `data/draws/`.** Tournament entrants are entered by hand;
there is no scraper in this project and this workflow adds none. It refreshes
historical *match data* and regenerates *simulations*, and the `git add` in the
workflow names `data/raw` and `data/cache` explicitly so nothing else can be
swept into an automated commit. A draw still holding `Qualifier`/`Bye` slots is
skipped with a log line and is picked up automatically by the next run once
someone has filled it in and merged it.

### Triggers

| | |
|---|---|
| `schedule` | `17 23 * * 1` — Mondays, 23:17 UTC |
| `workflow_dispatch` | `tournament_id` (blank = every simulatable draw), `refresh_data` (default true), `force` (default false) |

Weekly is matched to the source, not guessed: TML-Database republishes the
current season's main-tour file (`2026.csv`, the only one this project fetches
for the running year) roughly weekly — on 2026-08-08 the manifest showed it last
written 2026-08-03, against a bulk republish of every historical year on
2026-07-30. The files that do change daily (`*_ongoing_tourneys.csv`) are ones
this project never downloads. Tour events finish on Sundays, so a Monday run
captures a completed week in one pass.

**Late Monday, not Monday morning**, and the reason matters if you ever move it:
that observed 2026-08-03 timestamp is **20:43 UTC** — Monday *evening*. A morning
slot would have run ~14 h before the week's republish and so fetched the previous
Monday's file every time, a week behind by construction and invisible in the logs
(the run just looks like a no-op). 23:17 stays on the same day, so nothing waits
an extra week the way a Tuesday slot would. **This rests on one observed publish
timestamp** — enough to rule out the morning, not enough to pin the vendor's
publish window. If real runs start reporting no changed years, the publish time
has drifted and this needs revisiting.

The two dispatch inputs are separate on purpose: "pull new match data" and
"re-simulate draw X" are different jobs, and pairing them would force a
13-season re-download just to re-run one Monte Carlo for a draw whose entrants
were entered an hour ago. `refresh_data: false` + a `tournament_id` is the
before-a-Slam path.

### It opens a PR; it does not push to `main`

This is the only workflow in the repo that can write to the default branch, and
it runs unattended. Its output changes what the public demo shows — a refresh
moves every published title probability — so it lands as a reviewable diff on
the long-lived branch `automation/data-refresh` (force-pushed, so there is at
most one open data-refresh PR at a time, always carrying the latest data). The
trade is real and deliberate: "auto-updates before each Slam" means *prepared,
awaiting merge*.

**Two repository settings are required**, both under Settings → Actions →
General:

1. Workflow permissions — the job declares `contents: write` and
   `pull-requests: write` itself, which is enough if the repo default is
   read-only.
2. ✅ **"Allow GitHub Actions to create and approve pull requests"** — without
   it `gh pr create` fails with a 403 *even though the token has
   `pull-requests: write`*.

Two more GitHub behaviours worth knowing. Scheduled workflows are **disabled
after 60 days** of repository inactivity. And **`ci.yml` will not have run by
itself on a PR this workflow opens.** The precise mechanism, because the
approximate version misleads: events triggered by `GITHUB_TOKEN` create no
workflow runs, *except* `pull_request` events with the `opened`, `synchronize` or
`reopened` activity types — those do queue a run, but GitHub holds it in an
**approval-required** state until someone with write access clicks "Approve
workflows to run" on the PR. (The force-push to the automation branch is the
plain case: its `push` event creates no run at all.) So a reviewer looking at the
PR sees no green checks unless they first approve them — which is why this job
runs `pytest -q` itself before opening the PR. Approving the queued `ci.yml` run
is still worth doing; it just cannot be relied on to happen.

**Observed, not just documented (2026-08-08).** The first real run opened PR #1
from `automation/data-refresh`. Its `CI` run was created two seconds after the
PR appeared and then sat there: `run_started_at` is roughly **eight hours
later**, on `run_attempt: 2`, with `actor: github-actions[bot]` but
`triggering_actor` a human. Queued instantly, started only on approval —
exactly as above. That same run is also the proof that both repository settings
are on: `gh pr create` would have 403'd otherwise.

### How a bad refresh is stopped

`scripts/refresh_data.py` reports a failed year by *omission*: it logs it, skips
it, and still exits 0. An unattended wrapper that trusted that exit code would
precompute against half-updated data and stamp the cache with today's date — a
false claim of freshness that nothing downstream could detect. So the
orchestrator gates every phase and each gate fails closed. There are **six**:

0. **Coverage** — `--end` (default: the calendar year) must not exceed
   `config.END_YEAR`, because the pipeline loads data only through `END_YEAR`.
   Refreshing past it would regenerate, date and commit a cache that ignores the
   new season entirely — the same falsely-fresh failure by a slower route.
   **This is the one gate guaranteed to fire on its own, and the one most likely
   to be the reason a run is red when nothing else is wrong:** on the first
   scheduled run after New Year, `--end` becomes the new year and every run stops
   here until someone bumps `config.END_YEAR` (and checks `TEST_YEAR`, which is
   deliberately decoupled — `2026` is a partial season). The abort message names
   that fix. It happens before the network, so nothing is downloaded.
1. **Refresh** — every requested year must be in the returned `{year: path}`
   map, or the run aborts before anything else happens.
2. **Verify** — each file is re-read and checked for the columns the
   preprocessor indexes by name, for `tourney_date` values that parse as
   `YYYYMMDD` (the loader coerces failures to `NaT`, which would silently
   destroy the date ordering every leakage-safe feature depends on), and for a
   row count that has not *shrunk* against the file it replaced. Pass
   `--allow-shrink` if the vendor genuinely removed rows.
3. **Change detection** — sha256 per file, before and after. Unchanged bytes
   ⇒ the run stops successfully, having retrained and regenerated nothing.
   This is the "no empty commits" gate, and it has to live here: every cache
   file carries a `generated_at` that moves on every run, so `git diff
   data/cache/` can never answer the question.
4. **Retrain** — the persisted model is **deleted** and rebuilt. The skill table
   is rebuilt implicitly (nothing persists it), but `outputs/tennis_model.pkl`
   is loaded by `predictor.py` whenever it exists, with no staleness check at
   all — nothing compares the pickle against the data or config it was trained
   under. New data plus an old pickle is a silent mismatch. **Running this by hand is destructive:** the delete happens before
   training and is not rolled back, so a training failure leaves your machine
   with *no* model rather than a stale one (regenerate with
   `python src/predictor.py`, ~1 min; or pass `--retrain never` to leave the
   pickle alone). That is the intended trade — a missing model fails loudly
   everywhere, a stale one does not.
5. **Precompute** — only now, and via `scripts/precompute_sim.py`, so the
   placeholder refusal, the error ladder and the disclosure metadata are the
   same ones the API already relies on.

A failure at any gate exits non-zero, names what failed, and **commits
nothing** — including the draws that succeeded, so the repository never holds a
mix of fresh and stale caches. (Within the precompute phase every target is
still attempted, so one run reports the whole failure list rather than one draw
per run.)

### One known gap: the cache write is not atomic

`scripts/precompute_sim.py`'s `write_cache` is a bare `path.write_text(...)`
straight over the destination — there is no temp-file-plus-`os.replace`. A
crash, a `kill`, or a full disk *during* that write leaves a **truncated JSON
file** in `data/cache/`.

It is left as-is on purpose, and the reasoning is worth knowing before anyone
"fixes" it in a hurry or, worse, relies on it not being there. The failure is
**loud and self-healing**: the API parses every cache file through Pydantic, so
a truncated one is rejected with a `422`, never served as partial numbers, and
the next run overwrites it. The window is also small — the write is the last
act of a phase that spent ~30 s computing. But it *is* a real window, and the
calculus changes the moment the cache becomes authoritative rather than
regenerable: serving it off a shared volume, or publishing it somewhere nothing
re-runs the Monte Carlo. Do that and close this first — write to
`path.with_suffix(".json.tmp")` and `os.replace` it into place, which is atomic
within a filesystem.

The scheduled workflow is not exposed to it in practice: the orchestrator exits
non-zero on a failed precompute, and the commit/PR steps are gated on that, so
a torn file never reaches a pull request.

### Why the cache is `git add -f`'d

`.gitignore:15` (`data/cache/*.json`) stays exactly as it is: it stops a
developer committing a locally-generated cache by accident, which is still worth
having. This workflow is the one place where writing that file is the intended
act, so it is the one place that overrides the rule. Without the `-f` the add is
a **silent no-op** and the freshly computed cache never leaves the runner.

Two consequences of that cache being in git, neither of them a problem:
`outputs/tennis_model.pkl` is *not* committed (it stays gitignored), so the
numbers in the PR were produced by a model that only ever existed on the
runner — `metadata.estimator_class` in each cache file records which of the
best-of-four it was (§4). And the container build ignores the committed cache
entirely: `.dockerignore` excludes `data/cache/` so every image regenerates it
from source, as described above.
