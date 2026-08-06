# Ace — frontend

React 18 + Vite + TypeScript. A **static** single-page app: three screens
(bracket, title odds, storybook run) served as plain files, talking to the ace
API over HTTP. No SSR, no server-side anything — `npm run build` emits a folder
of files and any static host can serve it.

Everything below assumes you are in `frontend/`.

---

## Running locally (dev)

The frontend never bundles its own data — every screen is a call to the API, so
you need **both** processes running.

**1. Start the API** (from the repo root):

```bash
PYTHONPATH=src uvicorn api.main:app --reload
```

It loads the data, skill table and classifier once at startup (~2 s) and then
serves on `http://localhost:8000`.

**2. Point the frontend at it.** There is no default baked into the source —
`apiBaseUrl()` throws a loud `ApiConfigError` rather than guessing an origin —
so this step is not optional:

```bash
cp .env.example .env      # already contains VITE_API_BASE_URL=http://localhost:8000
```

`.env` is gitignored; `.env.example` is the only place a value is written down.

**3. Start the dev server:**

```bash
npm install
npm run dev               # http://localhost:5173
```

`http://localhost:5173` and `http://127.0.0.1:5173` are both already in the
API's CORS allow-list (`config.API_ALLOWED_ORIGINS`), which is an explicit list
and never `*` — so a browser calling from any *other* origin is refused.

**Deep links.** The app holds no router: `App.tsx` reads the query string, so
every screen is shareable as-is.

```
http://localhost:5173/?tournament=usopen_2024_atp_full
http://localhost:5173/?tournament=usopen_2024_atp_full&view=odds
http://localhost:5173/?tournament=usopen_2024_atp_full&view=storybook&seed=42
```

The odds screen needs a precomputed Monte Carlo cache, or it renders a 425 panel
telling you the command to run:

```bash
python scripts/precompute_sim.py --draw usopen_2024_atp_full --runs 5000 --seed 0
```

### Other scripts

```bash
npm run typecheck    # tsc -b --noEmit --force
npm run test         # vitest run
npm run lint         # oxlint
npm run preview      # serve an already-built dist/ on :4173 (see below)
```

---

## Building for production

```bash
VITE_API_BASE_URL=https://your-ace-api.example.com npm run build
```

Output lands in `frontend/dist/` — `index.html`, hashed `assets/*.js` and
`assets/*.css`, and `favicon.svg`. Upload that folder anywhere.

### ⚠️ The API URL is baked in at build time, not read at runtime

This is the one thing to understand before deploying, and it surprises people.

`VITE_API_BASE_URL` is **not** configuration the running site reads. It is
substituted into the source by Vite during `npm run build` and then constant-
folded by the minifier. `apiBaseUrl()` in `src/api/client.ts` — which looks like
a runtime lookup with a validation guard — compiles down to this:

```js
function b(){return`https://your-ace-api.example.com`.replace(/\/+$/,``)}
```

The guard is gone, `import.meta.env` is gone, and the string `VITE_API_BASE_URL`
does not appear anywhere in `dist/`. What follows from that:

- **One built bundle talks to exactly one API.** There is no env file, no
  `window.__CONFIG__`, and no `/config.json` to edit after the fact.
- **Staging and production need two separate builds.** You cannot promote the
  staging bundle to production by copying files or flipping a host's env var —
  changing the API URL means **rebuild and redeploy**, in that order. Redeploy
  alone is a no-op.
- **Rolling back to a previous build rolls back its API URL too**, since the URL
  is part of the artefact rather than part of the environment.
- **If the variable is unset at build time the build still succeeds** (exit 0).
  You get a bundle whose every screen renders the `ApiConfigError` panel at
  runtime. There is no build-time failure to catch this — check the deploy.

If you ever need one artefact to serve several environments, that is a real
change of design (fetch the base URL from a served `config.json` at startup),
not a build flag.

### ⚠️ A local `.env` is read by `npm run build` too

Vite loads `.env` files for `build`, not just `dev`. So running `npm run build`
on a machine that has a dev `.env` sitting in `frontend/` bakes
`http://localhost:8000` into what you were about to ship.

Shell environment variables take precedence over `.env`, so the explicit form
above is safe on a dev machine, and CI and hosted builds never see a `.env` at
all (it is gitignored). Just don't run a bare `npm run build` and upload the
result.

### Checking a production build locally

```bash
VITE_API_BASE_URL=http://localhost:8000 npm run build
npm run preview                                        # :4173
```

`preview` serves `dist/` as a static host would. It runs on **4173**, not 5173 —
a different origin, so it needs its own CORS entry. `http://localhost:4173` is
already in `config.API_ALLOWED_ORIGINS`, so this works out of the box.

The one edge: only the `localhost` spelling is listed for 4173 (5173 has both),
so browsing the preview server at `http://127.0.0.1:4173` is refused. Use
`localhost`, or add the second spelling to the allow-list and restart the API.

---

## Deploying

`render.yaml` in the repo root is a ready static-site Blueprint: connect the
repo on Render, set `VITE_API_BASE_URL` when prompted, done. It pins
`rootDir: frontend`, `npm ci && npm run build`, `./dist`, and cache headers
(immutable for hashed `/assets/*`, `no-cache` for `index.html`).

**Render is the only host with a committed, checked config.** Everything below
is a translation of the same three facts, written from each host's documented
settings — **none of it has been deployed or verified**, so treat it as a
starting point rather than a tested recipe. Any other static host needs:

| | |
|---|---|
| Build command | `npm ci && npm run build` |
| Publish directory | `frontend/dist` |
| Build environment | `VITE_API_BASE_URL` — required, no default |

- **Netlify** — base directory `frontend`, publish `dist`, and set the variable
  under Site settings → Environment variables (it must be present at *build*
  time, not just runtime).
- **Vercel** — root directory `frontend`, framework preset Vite; same variable.
- **Anything else** — `npm run build`, then serve `dist/` with any file server
  (`npx serve dist`, nginx, S3 + CloudFront, `python -m http.server` from inside
  `dist`). No Node runtime is needed to serve it.

**No SPA rewrite rule is required.** The app does no path-based routing — every
screen is `/` plus a query string — so no deep link ever requests a path other
than `/`. A catch-all `/* → /index.html` rewrite is harmless but only converts
genuine 404s into blank app shells.

### After deploying: two things on the API side

1. **Add the frontend's origin to `config.API_ALLOWED_ORIGINS`** and restart the
   API. The allow-list ships with the local dev origins only; a deployed
   frontend is refused until it is added explicitly.
2. **Serve the API over HTTPS** if the frontend is on HTTPS. A browser blocks a
   page on `https://` from calling `http://`, and the failure surfaces as the
   generic "could not reach the ace API" panel.

### Containers are ticket T5.1's, not this file's

A frontend image and a `docker-compose.yml` running API + frontend together are
**not** covered here and were deliberately left unbuilt so the two tickets do
not produce conflicting configs. Three things whoever writes them should take
from this page rather than rediscover:

- **`VITE_API_BASE_URL` must be a build `ARG`, not a container `ENV`.** It is
  inlined at build time (see above), so setting it in `docker-compose.yml`'s
  `environment:` does nothing — the value was already baked when the image was
  built. Wiring the API URL "between the services" means passing a build arg at
  image build time, and `docker-compose up` cannot repoint an existing image at
  a different API.
- **`.dockerignore` must exclude `.env`.** A `COPY frontend/ .` that picks up a
  developer's gitignored `.env` bakes `http://localhost:8000` into the image —
  the same trap as above, with a longer feedback loop.
- **The artefact is static files.** `dist/` needs no Node runtime to serve, so
  the image is a builder stage plus nginx (or any file server), not a `node`
  image running `vite preview`.

---

## CI

`npm run build` is CI-ready as-is: no interactive prompts, no network beyond
`npm ci`, byte-identical output across repeated builds of the same commit, exit
`0` on success and non-zero on failure (`tsc -b` failures exit `2` before Vite
runs). Wiring it into GitHub Actions alongside `npm ci`, `tsc -b --noEmit` and
`vitest run` is T5.2's job.
