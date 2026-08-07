# Ace 🎾

Ever wondered who's actually going to win Wimbledon before a single ball is served? Ace simulates it, point by point, game by game, set by set, thousands of times over. We turn a 128 player draw into a live title probability table and a full storybook run of the tournament complete with real scorelines. A specialised model plays out every match while a classifier trained on years of ATP history keeps the numbers grounded, so it's not just guesswork.

**🔗 Try Ace here → Coming soon**

---

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
every draw). **The build regenerates both** — it trains the classifier and runs
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

### CI check (the step T5.2 should wire in)

Until the GitHub Actions workflow lands, this is the manual equivalent — build
the image, start it, and prove `/health` answers:

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

Match data from [TML-Database](https://github.com/Tennismylife/TML-Database), used for educational, analytical, and research purposes. Originally inspired by Jeff Sackmann's [ATP Matches Dataset](https://github.com/JeffSackmann/tennis_atp) (CC BY-NC-SA 4.0). Non-commercial portfolio project.
