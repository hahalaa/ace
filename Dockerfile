# syntax=docker/dockerfile:1
#
# ace, container build.
#
# ONE Dockerfile, six stages, two shippable images. The default target is the
# API (it is the last stage), so `docker build -t ace-api .` does the expected
# thing; the frontend is `--build-arg VITE_API_BASE_URL=… --target frontend`.
# BuildKit only builds the stages its target depends on, so an API build never
# touches Node and a frontend build never trains a model.
#
#   deps              python:3.11-slim + /opt/venv from requirements.txt
#   runtime-deps      ⬑ the same venv minus build-only packages (1.2 GB → 693 MB)
#   artefacts         ⬑ trains the model and precomputes the sim cache
#   frontend-builder  node + `npm ci && npm run build`
#   frontend          nginx serving dist/            (--target frontend)
#   api               uvicorn + src/ + data/ + the artefacts   (default target)
#
# --------------------------------------------------------------------------
# The artefact problem, and which of the three resolutions this file takes
# --------------------------------------------------------------------------
# Two files the API needs at runtime are gitignored and are therefore in NO
# clean checkout (.gitignore:10 `outputs/`, .gitignore:15 `data/cache/*.json`):
#
#   outputs/tennis_model.pkl   api/deps.py loads it at startup and fails fast
#                              without it, no model, no API at all.
#   data/cache/<id>.json       what GET /tournaments/{id}/simulate serves;
#                              absent, every simulatable draw answers 425
#                              cache_missing (a draw with placeholder entrants
#                              answers 409 first, cache or no cache).
#
# Only data/raw/ (the vendored per-year CSVs) and data/draws/ are in git. So
# "COPY data/ and hope" ships a broken image. The ticket allows three fixes,
# (a) regenerate in a builder stage, (b) mount volumes, (c) fetch a release
# artefact. **This file takes (a).** Reasons, in order:
#
#   * The image stays self-contained and offline. (b) makes `docker run` alone
#     insufficient, the acceptance criterion is that it serves /health, and
#     (c) would put a network fetch in the build and a release-hosting story in
#     a ticket that has neither.
#   * Everything (a) needs is already in the repo: the raw CSVs are vendored,
#     and both generators (`src/predictor.py`, `scripts/precompute_sim.py`)
#     already run fully offline.
#   * `.dockerignore` excludes `outputs/` and `data/cache/` from the build
#     context, so a developer machine that happens to have both cannot leak
#     them in. Every build regenerates them from source, on every machine.
#
# THE TRADEOFF, STATED (ace-04-current-state.md §4): `predictor.py` persists
# whichever of Logistic Regression / Decision Tree / Random Forest / XGBoost
# wins on TEST_YEAR accuracy, so this stage decides which model the image
# ships. Measured, so the risk is stated at its true size rather than guessed:
#
#   * For a FIXED environment it is deterministic. Three independent builds,
#     two of them --no-cache, from two different contexts, produced a
#     byte-identical tennis_model.pkl (sha eca0f9a3…). A rebuild today does not
#     roll dice.
#   * It is the ENVIRONMENT that flips it. The same source trained inside this
#     image selects LogisticRegression where the host venv had persisted a
#     RandomForestClassifier, and the board moves with it (Sinner 41.3% →
#     31.2%). requirements.txt pins lower bounds only (`pandas>=2.0`,
#     `scikit-learn>=1.3`, no lockfile) and the base image moves, so a rebuild
#     months from now resolves different wheels and may bake a different model.
#
# The image is therefore reproducible in behaviour-from-source, reproducible
# bit-for-bit *while the environment holds*, and **not reproducible across a
# dependency or base-image drift**. Accepted for this ticket, not papered over,
# because:
#
#   * Which estimator was baked is observable, not hidden: the artefacts stage
#     prints it during the build, and every /simulate and /storybook response
#     carries `metadata.estimator_class`.
#   * Pinning is a modelling decision (seam 3 / §4), not a packaging one, it
#     means changing what `train_and_evaluate` selects, which is outside a
#     Docker ticket's scope and would change the CLIs and the API too.
#   * The operational fix needs no code: build ONCE and deploy the resulting
#     **image digest**. A digest is exactly reproducible; a rebuild is not.
#
# Build cost of (a): the model train dominates. `--build-arg PRECOMPUTE_RUNS=`
# trims the Monte Carlo for CI.


# --------------------------------------------------------------------------- #
# Stage 1, deps: one venv, built once, shared by the builder and the runtime.
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS deps

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# libgomp1: xgboost's manylinux wheel links against the system OpenMP runtime
# and fails to import without it. Needed wherever the venv is *used*, so the
# runtime stage installs it too.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt ./
RUN pip install -r requirements.txt


# --------------------------------------------------------------------------- #
# Stage 2, runtime-deps: the same venv, minus what only the *builder* needs.
#
# `COPY --from=` copies a stage's final filesystem, not its layers, so pruning
# here genuinely shrinks the shipped image (an uninstall in the api stage would
# not, deleted files still sit in the layer below). Measured: 1.2 GB → 693 MB (of which the matplotlib-orphan prune below is the last ~34 MB).
#
#   nvidia-nccl-cu12  401 MB of CUDA collective-communication libraries, pulled
#                     in as an xgboost dependency and used only for distributed
#                     *GPU training*. This image does CPU inference.
#   matplotlib,       plotting, used by model/train.py + model/viz.py during the
#   seaborn           artefacts stage. `import api.main` loads neither (checked).
#   pytest            test-only; tests/ is not even in the image.
#   fonttools,        matplotlib's own dependencies, left orphaned once it goes
#   kiwisolver,       (pip uninstall does not cascade to a package's deps). None
#   cycler,           is required by any surviving package, verified: their
#   contourpy         reverse-dependency set is empty, and fonttools alone is
#                     ~28 MB. Pruned the same pip-list|grep|xargs way as nvidia
#                     so an absent one (a future matplotlib with different deps)
#                     is a no-op, not a build failure. pillow/narwhals/pygments
#                     are deliberately NOT here: they are still declared deps of
#                     scikit-learn / httpx, so the guard universe keeps them.
#
# The guard below is the safety net: any of those turning out to be load-bearing
# fails the build here rather than at container start. It must keep exercising
# xgboost specifically, since the persisted model may be an XGBClassifier
# (best-of-four, §4) and xgboost is what the nvidia prune touches.
# --------------------------------------------------------------------------- #
FROM deps AS runtime-deps

RUN pip list --format=freeze | grep -i '^nvidia' | cut -d= -f1 | xargs -r pip uninstall -y \
 && pip uninstall -y matplotlib seaborn pytest \
 && pip list --format=freeze | grep -iE '^(fonttools|kiwisolver|cycler|contourpy)=' | cut -d= -f1 | xargs -r pip uninstall -y \
 && python -c "import xgboost, sklearn, pandas, numpy, joblib, fastapi, uvicorn, pydantic; print('[build] runtime import guard OK')"


# --------------------------------------------------------------------------- #
# Stage 3, artefacts: regenerate the two gitignored runtime files. OFFLINE.
# --------------------------------------------------------------------------- #
FROM deps AS artefacts

WORKDIR /build

# Which draw to precompute, and how hard. The shipped 128-slot 2026 Australian
# Open is the default simulatable draw in data/draws/. A draw holding placeholder
# slots is refused by monte_carlo by design, so precomputing one is not a
# thing that can succeed.
ARG PRECOMPUTE_DRAW=ausopen_2026_atp_full
ARG PRECOMPUTE_RUNS=5000
ARG PRECOMPUTE_SEED=0

# Agg: headless. matplotlib would otherwise probe for a GUI backend, and
# model/train.py + model/viz.py write PNGs during the training run.
ENV MPLBACKEND=Agg

COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/

# outputs/ and data/cache/ are excluded from the build context on purpose (see
# the header), so create them, train.py's savefig runs before predictor.py's
# own makedirs and would fail on a missing directory.
RUN mkdir -p outputs data/cache

# `< /dev/null` closes stdin: predictor.py ends in the interactive REPL, which
# takes an EOF as "input closed" and exits 0 cleanly (cli/interactive.py:86).
# Training is the expensive half of this build. The second command prints which
# of the best-of-four was persisted, the §4 reproducibility risk made visible
# in the build log, and the same value the API reports at runtime as
# metadata.estimator_class.
RUN python src/predictor.py < /dev/null \
 && test -s outputs/tennis_model.pkl \
 && python -c "import joblib; print('[build] persisted estimator:', type(joblib.load('outputs/tennis_model.pkl')).__name__)"

RUN python scripts/precompute_sim.py \
        --draw "${PRECOMPUTE_DRAW}" \
        --runs "${PRECOMPUTE_RUNS}" \
        --seed "${PRECOMPUTE_SEED}" \
 && test -s "data/cache/${PRECOMPUTE_DRAW}.json"

# The Elo leaderboards GET /rankings serves. A display feature walled off from
# the model (src/features/elo.py), computed offline on the same cadence as the
# sim cache, never per request. Without this the endpoint answers 425
# rankings_missing for every visitor.
RUN python scripts/precompute_elo.py \
 && test -s data/cache/elo_ratings.json


# --------------------------------------------------------------------------- #
# Stage 4, frontend-builder: `npm run build` and nothing else ships from here.
# --------------------------------------------------------------------------- #
FROM node:22-alpine AS frontend-builder

WORKDIR /app

# VITE_API_BASE_URL IS A BUILD ARG, AND CAN ONLY EVER BE ONE.
# Vite inlines it into the bundle at build time and the minifier constant-folds
# the lookup away entirely, `import.meta.env` and the string
# "VITE_API_BASE_URL" appear NOWHERE in dist/ (ace-04-current-state.md §7 seam 9). So a compose `environment:`
# entry, a container ENV, or a runtime `-e` flag does nothing at all: the value
# was decided when the image was built. Repointing the app at a different API
# is a REBUILD. Unset is not fatal, the build warns and ships a bundle whose
# every screen renders the ApiConfigError panel (vite.config.ts explains why it
# warns rather than throws).
# Declared as an ARG and deliberately NOT re-exported with `ENV`: BuildKit
# already puts a declared ARG into the environment of this stage's RUN steps,
# so `npm run build` sees it, and leaving it out of the image metadata keeps the
# shipped image honest, there is no runtime knob here to mistake it for. It
# also preserves vite.config.ts's unset-value warning, which an
# `ENV …=${…}` line suppresses by turning "unset" into "set to empty".
ARG VITE_API_BASE_URL

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# .dockerignore drops frontend/.env (gitignored, but present on every developer
# machine and read by `vite build`, not just `vite dev`). Without that
# exclusion this COPY would bake a developer's http://localhost:8000 into the
# image and silently beat the ARG above, a shell/ARG value wins over .env, but
# only when it is set, so an unset ARG would inherit the stray file.
COPY frontend/ ./
RUN npm run build


# --------------------------------------------------------------------------- #
# Stage 5, frontend: static files, served by nginx. `--target frontend`.
# --------------------------------------------------------------------------- #
FROM nginx:1.27-alpine AS frontend

# The artefact is a folder of static files; it needs no Node runtime, so this
# is NOT a node image running `vite preview` (a dev-facing command; seam 9). nginx's stock config serves /usr/share/nginx/html
# on :80, which is all this app needs: no SPA rewrite (every screen is `/` plus
# a query string, so no deep link ever requests another path, see render.yaml)
# and no proxying (the browser calls the API directly, cross-origin, which is
# what config.API_ALLOWED_ORIGINS exists for).
COPY --from=frontend-builder /app/dist/ /usr/share/nginx/html/

EXPOSE 80


# --------------------------------------------------------------------------- #
# Stage 6, api: the default target. `docker build -t ace-api .`
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS api

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Same base image as `deps`, so the venv's interpreter symlink still resolves.
# From `runtime-deps`, not `deps`, see stage 2 for what that prunes and why.
COPY --from=runtime-deps /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Everything config.py resolves is relative to the working directory:
# RAW_DATA_DIR=data/raw, DRAWS_DIR=data/draws, CACHE_DIR=data/cache,
# MODEL_PATH=outputs/tennis_model.pkl. Hence WORKDIR /app and this layout.
COPY src/ src/
COPY data/raw/ data/raw/
COPY data/draws/ data/draws/
COPY --from=artefacts /build/outputs/tennis_model.pkl outputs/tennis_model.pkl
COPY --from=artefacts /build/data/cache/ data/cache/

# No scripts/, tests/, docs/ or frontend/, the server imports none of them.

RUN useradd --create-home --uid 10001 ace && chown -R ace:ace /app
USER ace

EXPOSE 8000

# Startup loads the vendored CSVs, engineers features and pins the classifier
# (~2 s measured), so the start period is generous relative to that. urllib
# rather than curl: the slim image has no curl and adding one to run a
# healthcheck is not worth the layer.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# NO NETWORK, EVER, not at start, not per request. The data is vendored into
# the image (data/loader.py raises rather than downloading) and both
# generated artefacts were baked in above. Verified with `--network none`.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
