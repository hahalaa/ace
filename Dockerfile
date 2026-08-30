# syntax=docker/dockerfile:1
#
# ace container build: one Dockerfile, six stages, two shippable images
# (`api`, the default target, and `--target frontend`).
#
#   docker build -t ace-api .
#   docker build --build-arg VITE_API_BASE_URL=... --target frontend -t ace-web .
#   docker build --build-arg PRECOMPUTE_RUNS=200 -t ace-api:ci .   # fast CI image


# --------------------------------------------------------------------------- #
# Stage 1, deps: one venv, shared by the builder and runtime stages.
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS deps

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# libgomp1: xgboost's wheel fails to import without the system OpenMP runtime.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt ./
RUN pip install -r requirements.txt


# --------------------------------------------------------------------------- #
# Stage 2, runtime-deps: the venv minus packages only the builder needs.
# `COPY --from=` copies a stage's filesystem, so pruning here shrinks the image.
# The import guard fails the build (not a container start) if a prune stops being safe.
# --------------------------------------------------------------------------- #
FROM deps AS runtime-deps

RUN pip list --format=freeze | grep -i '^nvidia' | cut -d= -f1 | xargs -r pip uninstall -y \
 && pip uninstall -y matplotlib seaborn pytest \
 && pip list --format=freeze | grep -iE '^(fonttools|kiwisolver|cycler|contourpy)=' | cut -d= -f1 | xargs -r pip uninstall -y \
 && python -c "import xgboost, sklearn, pandas, numpy, joblib, fastapi, uvicorn, pydantic; print('[build] runtime import guard OK')"


# --------------------------------------------------------------------------- #
# Stage 3, artefacts: regenerate the two gitignored runtime files, offline.
# --------------------------------------------------------------------------- #
FROM deps AS artefacts

WORKDIR /build

# A draw holding placeholder slots is refused by monte_carlo, so it cannot be precomputed.
ARG PRECOMPUTE_DRAW=ausopen_2026_atp_full
ARG PRECOMPUTE_RUNS=5000
ARG PRECOMPUTE_SEED=0

# Agg: headless matplotlib for the PNGs train.py/viz.py write.
ENV MPLBACKEND=Agg

COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/

# outputs/ and data/cache/ are excluded from the build context, so create them first.
RUN mkdir -p outputs data/cache

# `< /dev/null` closes stdin so predictor.py's REPL takes EOF and exits 0.
# The second command prints which best-of-four estimator was persisted (type can vary by build environment).
RUN python src/predictor.py < /dev/null \
 && test -s outputs/tennis_model.pkl \
 && python -c "import joblib; print('[build] persisted estimator:', type(joblib.load('outputs/tennis_model.pkl')).__name__)"

RUN python scripts/precompute_sim.py \
        --draw "${PRECOMPUTE_DRAW}" \
        --runs "${PRECOMPUTE_RUNS}" \
        --seed "${PRECOMPUTE_SEED}" \
 && test -s "data/cache/${PRECOMPUTE_DRAW}.json"

# The Elo cache GET /rankings serves; without it the endpoint answers 425 for every visitor.
RUN python scripts/precompute_elo.py \
 && test -s data/cache/elo_ratings.json


# --------------------------------------------------------------------------- #
# Stage 4, frontend-builder: `npm run build`.
# --------------------------------------------------------------------------- #
FROM node:22-alpine AS frontend-builder

WORKDIR /app

# Build arg only; a container ENV or runtime -e does nothing here.
# Not re-exported with ENV: that would turn "unset" into "set to empty" and suppress vite.config.ts's warning.
ARG VITE_API_BASE_URL

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

# .dockerignore drops frontend/.env, which `vite build` would otherwise read and bake in.
COPY frontend/ ./
RUN npm run build


# --------------------------------------------------------------------------- #
# Stage 5, frontend: static files served by nginx. `--target frontend`.
# --------------------------------------------------------------------------- #
FROM nginx:1.27-alpine AS frontend

# Static files need no Node runtime. nginx's stock config serves /usr/share/nginx/html on :80,
# with no SPA rewrite and no proxying.
COPY --from=frontend-builder /app/dist/ /usr/share/nginx/html/

EXPOSE 80


# --------------------------------------------------------------------------- #
# Stage 6, api: the default target. `docker build -t ace-api .`
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS api

RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# From runtime-deps, not deps: see stage 2 for what it prunes.
COPY --from=runtime-deps /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# config.py resolves every path relative to the working directory, hence WORKDIR /app and this layout.
COPY src/ src/
COPY data/raw/ data/raw/
COPY data/draws/ data/draws/
COPY --from=artefacts /build/outputs/tennis_model.pkl outputs/tennis_model.pkl
COPY --from=artefacts /build/data/cache/ data/cache/

# No scripts/, tests/, docs/ or frontend/: the server imports none of them.

RUN useradd --create-home --uid 10001 ace && chown -R ace:ace /app
USER ace

EXPOSE 8000

# urllib, not curl: the slim image has no curl. Start period covers the ~2s data load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

# No network, ever: data is vendored and both artefacts were baked in above.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
