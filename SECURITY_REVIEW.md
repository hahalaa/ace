# Security review — decision record

Durable record of the security-hardening pass (2026-08-12) and, crucially, the
reasoning behind every **accepted** (rather than fixed) finding. It exists
because that reasoning previously lived only in code comments and a review
conversation; this project records its architectural seams and decisions in
`docs/ace-04-current-state.md`, and the security decisions get the same
treatment. If you are re-reviewing before a deploy, start here.

The service being reviewed: a **read-only, unauthenticated JSON API** (FastAPI /
Starlette / uvicorn) plus a **static** React SPA. No database, no auth, no user
accounts, no write endpoints, no file uploads, no user-supplied file paths
except the tournament id, which is validated against a fixed registry and a
path-traversal guard (`api.registry.cache_path_for`, `tests/test_api_security.py`).
Keep that shape in mind: it is why several classes of finding are not applicable.

> **Update (2026-08-16) — the app has grown since the shape above was written.**
> It is still unauthenticated and read-only in the "no accounts, no database"
> sense, but it now has **one write-shaped endpoint** (`POST /tournaments/upload`,
> which stores nothing on disk — an in-memory, capped, ephemeral LRU) and **three
> endpoints that run real per-request compute**: `GET /storybook` (curated draws
> *and* uploaded draws) and `POST /match/simulate`. The traversal surface is
> unchanged (still only the tournament id, still guarded by `cache_path_for`).
> See §6 for the second-review findings.

---

## 1. What this pass changed (fixed)

- **CORS/CSP `connect-src` pinned** — `render.yaml` served the SPA with
  `connect-src 'self' https://*.onrender.com`. `onrender.com` is shared public
  hosting, so the wildcard would let any future XSS exfiltrate to any sub-app on
  the platform. Pinned to the exact deployed API origin
  `https://ace-e21v.onrender.com` (the only origin the bundle actually calls,
  confirmed by runtime network capture). Regression test pins the exact origin
  and forbids a wildcard.
- **Security headers on every response** — `X-Content-Type-Options: nosniff`,
  `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, and a strict
  `Content-Security-Policy: default-src 'none'; frame-ancestors 'none';
  base-uri 'none'` on all JSON, via `api.main.add_security_headers`. The SPA gets
  the equivalent set from `render.yaml`.
- **Docs-CSP selector reads the routed path, not `request.url`** — the middleware
  chooses the relaxed docs CSP off `request.scope["path"]` (the ASGI routed
  path), not `request.url.path` (reconstructed from the spoofable `Host` header).
  This closes the one reachable sink of PYSEC-2026-161/248 (see §3).
- **`/storybook` rate limit** — a small in-memory per-IP sliding window
  (`SlidingWindowRateLimiter`) on the one compute-heavy endpoint. Scope is
  deliberately narrow; see §2.
- **Non-spoofable client key on-platform** — `_client_key` prefers Cloudflare's
  `CF-Connecting-IP` (which Cloudflare overwrites, so a client cannot forge it)
  over the leftmost `X-Forwarded-For`, falling back to XFF then the socket peer
  off-platform. Verified live: rotating a spoofed XFF no longer buys fresh
  buckets when `CF-Connecting-IP` is present.
- **Catch-all 500 handler** — `unexpected_error_handler` re-attaches the security
  headers to unhandled 500s (which otherwise bypass the header middleware, since
  Starlette's `ServerErrorMiddleware` sits outside it) and returns a fixed
  `{"detail": "Internal Server Error"}` body that never echoes the exception.

---

## 2. Accepted tradeoffs (deliberately not "fixed")

### 2a. Rate limiter is hygiene, not abuse prevention
`config.API_STORYBOOK_RATE_LIMIT` (`"12/minute"`) is enforced by an **in-memory,
per-process** sliding window. Its counters live only in one process's memory, so:

- Render's free tier cold-starts and restarts routinely, and every restart
  resets the counters to zero;
- a horizontally-scaled deployment gives each replica its own independent store.

So this is **resource-exhaustion hygiene against a single careless or looping
client** (a buggy frontend retry loop, a naïve script), **not** robust abuse
prevention. It gates **no authorization** — there is none. A determined attacker
defeats it by waiting out a restart or spreading across replicas.

The client key prefers `CF-Connecting-IP` (non-spoofable on Render, see §1), so
header-rotation only bypasses it off-platform where `X-Forwarded-For` is the
fallback — and even there the limiter's own scope note means a bypass costs the
attacker nothing but their own evasion. Real volumetric/distributed defense is
Render's infrastructure layer, by design. Accepted: adding a shared store (Redis
etc.) would be real infrastructure for a demo that does not need it.

### 2b. `/docs` and `/redoc` get a relaxed CSP
Swagger UI / ReDoc HTML pulls its bundle from `cdn.jsdelivr.net` and runs an
inline bootstrap, so `DOCS_CSP` allows exactly that (`script-src`/`style-src`
`'self' https://cdn.jsdelivr.net 'unsafe-inline'`, `img-src` `data:` +
`fastapi.tiangolo.com`, `font-src cdn.jsdelivr.net`). Every **other** route —
every JSON body, including `/openapi.json` — gets the strict `default-src 'none'`
policy. The selector keys on the routed path (`request.scope["path"]`), so the
relaxed policy cannot leak to a data route via a spoofed `Host` header
(`tests/test_api_security.py::test_relaxed_csp_is_scoped_to_docs_only`). Accepted:
the docs UI is genuinely useful and the relaxed policy is scoped to it alone.

### 2c. HSTS is not set by the app
`Strict-Transport-Security` is deliberately omitted from both the API and the
SPA. Render terminates TLS at its edge and owns the certificate and the
HTTP→HTTPS redirect; a `max-age` pinned from the app could outlive the
deployment. HSTS is the platform's to set (its dashboard offers it). Accepted:
setting it from behind the proxy is the wrong layer.

### 2d. Supply-chain pinning is by tag/range, not immutable digest ("SHA-pinning")
- **GitHub Actions** are pinned to major-version tags
  (`actions/checkout@v4`, `actions/setup-python@v5`, `actions/setup-node@v4`),
  not full commit SHAs. Tags are mutable: a compromised upstream could re-point
  one. SHA-pinning would be immutable but forfeits automatic patch pickup and
  readability.
- **Docker base images** are pinned by tag (`python:3.11-slim`,
  `node:22-alpine`, `nginx:1.27-alpine`), not by digest.
- **Python deps** (`requirements.txt`) use compatible version ranges, not
  `--hash` pins.

Accepted for a demo project relying on first-party `actions/*` and official base
images: the marginal supply-chain risk does not justify the maintenance cost of
digest-pinning here. Revisit if this ever handles untrusted data or secrets.

---

## 3. CVE acceptance reasoning (pip-audit)

`pip-audit -r requirements.txt` reports **8 findings, all in `starlette` 0.48.0**.
FastAPI 0.116.2 pins `starlette<0.49`, so none can be patched out without a
FastAPI major-version bump — a larger change than this hardening pass. Each was
assessed by tracing the actual code path, not by trusting a grep. None of the
symbols the reachable-only CVEs require exist anywhere in `src/`.

| Advisory | Summary | Reachable here? |
|---|---|---|
| **PYSEC-2026-249** | `request.form()` ignores `max_fields`/`max_part_size` for `x-www-form-urlencoded` | **No.** The API has no form endpoints — all routes are GET/JSON. `request.form`, `Form`, `File` appear nowhere in `src/`. |
| **PYSEC-2026-1942** | `FileResponse`/`StaticFiles` `Range` header → O(n²) CPU DoS | **No.** No `StaticFiles` mount and no `FileResponse` anywhere in `src/`; all responses are `JSONResponse`/`HTMLResponse` (the docs UI). |
| **PYSEC-2026-2281** | `StaticFiles` on **Windows** → UNC path SSRF leaking service creds | **No.** No `StaticFiles`; deploy target is Linux. Doubly not applicable. |
| **PYSEC-2026-2280** | `HTTPEndpoint` dispatches non-standard verbs via `getattr` | **No.** All routes are function-based `@app.get(...)`; no `HTTPEndpoint` subclass registered. |
| **PYSEC-2026-161** | `Host` header not validated before rebuilding `request.url`; `request.url.path` can differ from the routed path → path-based checks bypassable | **Was reachable, now mitigated.** The security middleware chose the docs-vs-strict CSP off `request.url.path`. Changed to `request.scope["path"]` (the routed path, immovable by a header). Impact was already low (JSON bodies are inert, `nosniff` is set, Cloudflare/Render normalize malformed `Host`), but this removes the sink entirely. |
| **PYSEC-2026-248** | Path not starting with `/` poisons `request.url.hostname`/`netloc` | **No.** No code reads `request.url.hostname`/`netloc`; the poisoned path routes to 404. The CSP selector (the only `request.url` reader) now uses `scope["path"]` regardless. |

**Net:** four are structurally unreachable given this app's shape, and the two
`request.url` advisories are neutralized by keying the one security-relevant path
decision on the raw routed path. Upgrading Starlette out of these is tracked as a
future FastAPI-version bump, not a blocker for this deploy.

---

## 4. Not applicable (premise re-confirmed 2026-08-12)

- **No auth system** — no OAuth/JWT/session/cookie/password handling anywhere in
  `src/` (grep confirms; the only matches are comments stating there is none).
  So "authentication bypass" style findings have nothing to bypass.
- **No database** — no SQLAlchemy/psycopg/sqlite/ORM; nothing is queried. No
  SQL-injection surface.
- **No user-controlled file paths** except the tournament id, which is validated
  against the registry and guarded against traversal by `cache_path_for`
  (`tests/test_api_security.py::test_cache_path_for_rejects_traversal`).

---

## 5. How to re-verify

```bash
pytest -q tests/test_api_security.py          # headers, rate limit, CSP, 500, traversal
pip-audit -r requirements.txt                 # expect the 8 starlette findings above
```

For a live check against a deployment, confirm the security headers are present
on a normal response and on a forced 4xx, that `/docs` gets the relaxed CSP while
`/openapi.json` gets the strict one, and that the SPA's only cross-origin XHR
target is the pinned API origin.

---

## 6. Review 2 — full OWASP Top 10 (2021) re-verification (2026-08-16)

A second full pass, specifically re-verifying everything in §§1–5 against the
**current** code (not assuming the first pass still holds) and covering what
shipped since: the draw-upload endpoint, single-match live simulation, and the
design-elevate CSS/token changes. Everything below was checked against a **live
running API**, not only by reading source.

### 6a. Prior findings — re-verified, all still hold

| §1/§2 item | Status 2026-08-16 | Evidence |
|---|---|---|
| Security headers (`nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, strict `default-src 'none'` CSP) on **every** response | **Still holds — now confirmed on the new routes too.** The `add_security_headers` middleware wraps the whole app, so it covers `/tournaments/upload`, `/match/simulate` and `/storybook` with no per-route work. Verified live on 200s **and** on 415/422/429/503 error bodies. | curl `-D-` on `/health`, `/players`, a 422 from `/match/simulate`, a 415 from `/tournaments/upload` — all carry the four headers + strict CSP. |
| Docs CSP scoped to `/docs`/`/redoc` off `request.scope["path"]` (not spoofable `request.url`) | **Still holds.** `/docs` gets `DOCS_CSP`; `/openapi.json` and every data route get the strict policy. | live curl; `test_api_security.py::test_docs_get_a_relaxed_csp_but_json_does_not`. |
| Catch-all 500 handler re-attaches headers, never echoes the exception | **Still holds.** | `test_error_responses_still_carry_headers`, `test_500_body_does_not_leak`. |
| CORS is an explicit allow-list, never `*`; `render.yaml` `connect-src` pinned to the exact API origin | **Still holds and positionally intact.** `config.API_ALLOWED_ORIGINS` is the 4-entry list (localhost:5173 ×2, localhost:4173, the Render frontend origin); `allow_credentials` is **not** set, so the `*` methods/headers are safe. Live: an allowed `Origin` is echoed, `https://evil.example.com` gets **no** `Access-Control-Allow-Origin` and its preflight is a 400. `render.yaml` still pins `connect-src 'self' https://ace-e21v.onrender.com;` with no wildcard. | live curl; `test_frontend_csp_connect_src_is_pinned_not_wildcard`. |
| Rate limiter prefers non-spoofable `CF-Connecting-IP`, used **consistently** across all rate-limited endpoints | **Still holds — and now three endpoints share the identical `_client_key`.** `/storybook`, `/tournaments/upload` and `/match/simulate` each call `_client_key(request)` through the same `SlidingWindowRateLimiter`. Live: with a **constant** `CF-Connecting-IP` and a **rotating** `X-Forwarded-For`, `/match/simulate` still 429s after its 20/min — the rotation buys nothing. Thresholds confirmed live: storybook 429 after 12, upload after 6, match after 20. | live curl loops; `test_client_key_prefers_cf_connecting_ip_over_spoofable_xff`. |
| Path-traversal guard `cache_path_for` | **Still holds.** Rejects empty/`.`/`..`, any `os.sep`/`altsep`/`/`/`\`/NUL, and asserts the assembled path's parent resolves to `cache_dir`. The id also only reaches it after a registry lookup, so a traversal id is doubly unreachable. | `test_cache_path_for_rejects_traversal` (incl. `../../etc/passwd`, UNC, absolute). |
| HSTS deliberately not app-set; supply-chain pinned by tag not digest | **Unchanged, still accepted** (§2c, §2d). |

### 6b. NEW finding **R2-1 (Medium→Low): aggregate live-compute was unbounded in concurrency** — FIXED

The genuinely new question: three families of live per-request compute now share
one 512 MB / 0.1-CPU instance, each individually rate-limited and individually
verified affordable **in isolation**, but never bounded **in aggregate**.

**Measured cost (dev machine, warm):** `/storybook` (128-draw) ≈ 0.6 s CPU and
one transient `BracketResult`; `/match/simulate` (3,000 runs) ≈ 0.27 s CPU, no
per-request allocation. Process RSS is **bounded and stable at ~246 MB** under
repeated and concurrent load (baseline 198 MB) — memory is *not* the concern; the
upload LRU (cap 32) and the shared-across-requests adapter memo are both bounded.

**The gap.** Both simulation handlers are **sync `def`**, so FastAPI dispatches
them into the anyio threadpool (default **40** workers). The per-IP rate limits
bound each client's *rate* but nothing bounded how many ran *at once*: a single
client can fire many concurrent requests while under its own limit, and three
different visitors (curated storybook, uploaded storybook, single-match — a
realistic split) each stay under theirs. Summed at their per-IP limits the three
endpoints demand **far more CPU-seconds/minute than a 0.1-CPU instance supplies**
(0.1 × 60 = 6 CPU-s/min) — even one storybook client at 12/min oversubscribes it.
The failure mode is **degradation, not a crash**: CPU-bound handlers queue behind
the throttle, latencies balloon, and requests time out at the edge. Self-healing
on restart, no data at risk — hence Medium severity as a raw resource issue,
Low in context (unauthenticated read-only demo; per the standing decision,
volumetric/distributed defense is the platform's job).

**Fix applied.** A shared, in-memory, per-process concurrency cap across **both**
live-compute routes — `api.main.LiveComputeGate`
(`config.API_LIVE_COMPUTE_MAX_CONCURRENCY = 4`), attached to `/storybook` and
`/match/simulate` as the `live_compute_slot` dependency. It bounds simultaneous
live simulations to a small constant **regardless of source IP**, and **sheds**
the excess immediately with `503 + Retry-After` rather than queuing it (queuing
behind a throttled CPU is precisely what produces the timeout pile-up). Unlike the
per-IP windows it also blunts the distributed case (many IPs, one request each),
which a per-IP window structurally cannot. Same honest scope as the rate limiters:
per-process, resets on restart, independent per replica — peak-load hygiene, not a
substitute for infra-layer defense. Chosen over "tighter per-IP limits" (which do
not bound concurrency and hurt legitimate use) and over "document as sufficient"
(the numbers show the aggregate was genuinely unbounded).

**Verified live:** 10 concurrent `/storybook` requests from 10 distinct
`CF-Connecting-IP`s → exactly **4× 200, 6× 503**; the 503 carries the security
headers, `Retry-After: 2`, and a generic `{"reason":"server_busy"}` body; after
load clears, requests return 200 (slots release, no leak). Tests:
`test_api_security.py::{test_live_compute_gate_admits_up_to_cap_then_sheds,
test_live_compute_gate_rejects_bad_capacity,
test_concurrent_live_sims_over_the_cap_get_503,
test_storybook_and_match_routes_declare_the_concurrency_dependency}`.

### 6c. Upload + single-match adversarial re-check — all pass

- **Upload** enforces, in order: `application/json` only (**415**); a
  `config.API_UPLOAD_MAX_BYTES` (1 MiB) cap **streamed** so an oversized body is
  never fully buffered (**413**); unparseable JSON **and** deeply-nested JSON —
  `RecursionError` is caught alongside the decode errors so a nest-bomb is a clean
  **422**, not an uncaught 500 with a logged stack (**422** `invalid_json`); a bad
  `event_date` (**422**); and `sim.draw.parse_draw` validation, returning the full
  accumulated `problems` list (**422** `draw_invalid`). Injection-style player
  names are inert — names are data looked up in the skill table, there is no SQL,
  no shell, no file path built from them. Stored in a **capped, in-memory** LRU
  that writes nothing to disk. Tests: `test_api_upload.py` (`…non_json…`,
  `…oversized_body…`, `…deeply_nested_json_is_rejected_without_a_500`,
  `…malformed_event_date…`, `…malformed_draw…`, `…writes_nothing_to_disk`,
  `…do_not_survive_a_restart`).
- **Single-match** resolves each player **server-side** through the shared name
  index and refuses an unknown name (**422** `player_not_found`) or an ambiguous
  one (**422** `player_ambiguous`, candidates listed) — the frontend type-ahead is
  a convenience, not the guard. Confirmed live (`zzzznotaplayer` → 422). The
  solve-once-per-aggregate performance guard is present and passing
  (`test_single_match.py::test_solves_reconciliation_once_per_aggregate_not_per_run`
  asserts `solve_reconciled_serve_probs` runs exactly once per request, not once
  per run — so a single request cannot be turned into thousands of δ-bisections).

### 6d. Design-elevate CSS — no security-relevant change

The design pass touched only CSS modules, `index.css`, one SVG path in `App.tsx`
(a favicon seam curve), and `favicon.svg`. **No new external requests**: zero
`@import`, `@font-face`, or `url(https://…)` anywhere in `frontend/`; fonts are
system stacks (`-apple-system`, `ui-monospace, …`). **No inline styles/scripts**
that would require loosening a CSP. The built bundle references only
`http://localhost` (the locally-baked `VITE_API_BASE_URL`) and a `reactjs.org`
error-decoder **string constant** — no CDN, no remote asset load. The
self-contained-bundle guarantee is intact.

### 6e. Dependency, secrets, and CI supply-chain scans

- **`pip-audit -r requirements.txt`**: the **same 8 findings, all in
  `starlette 0.48.0`** as the first pass (PYSEC-2026-161/248/249/1942/2280/2281) —
  no change, no new package affected. FastAPI 0.116.2 pins `starlette<0.49`, so
  they cannot be patched without a FastAPI major bump; the §3 reachability triage
  (four structurally unreachable, two `request.url` sinks neutralised by keying
  the CSP decision on `scope["path"]`) **still holds verbatim** against the current
  code. Not a deploy blocker; tracked as a future FastAPI bump.
- **`npm audit`**: one **high**, `nanoid < 3.3.18` (loops on `size 0`). It is a
  **build-time-only transitive** dep (`vite → postcss → nanoid`), **not present in
  the built `dist/` bundle**, and never called at runtime by this app. Real
  severity here is negligible; a one-line `npm audit fix` (or a `nanoid` bump)
  clears the scanner and is worth doing on the next frontend touch, but it is not a
  runtime exposure.
- **Secrets scan** over all tracked files (excluding `node_modules`/binaries):
  **none.** Every hit for `token`/`secret`/`password`/`key` is a design-token
  comment, a `GITHUB_TOKEN` env reference, `secrets.token_hex` (the upload id
  generator — good, not a committed value), or a `../../etc/passwd` traversal-test
  string. `frontend/.env` remains gitignored; `.env.example` is the tracked copy.
- **CI supply-chain**: both workflows (`ci.yml`, `refresh-and-simulate.yml`) pin
  **only first-party `actions/*` to major-version tags** (`checkout@v4`,
  `setup-python@v5`, `setup-node@v4`) — no `@main`, no unpinned, no third-party
  action (the PR is opened with the runner's built-in `gh`, not a marketplace
  action). Dispatch inputs reach the shell through `env:`, never interpolated into
  a `run:` line — no script-injection sink. Consistent with §2d; unchanged.

### 6f. Follow-up fixes (2026-08-16, second pass on the same branch)

Three items requested after the initial Review-2 pass, all applied and verified.

**1. Pre-existing test failure fixed — schema example re-aligned to the canonical
doc.** `test_upload_ui_schema_example_matches_doc_worked_example` was red on the
base commit (nothing this review introduced): `Upload.tsx`'s `SCHEMA_EXAMPLE`
constant had drifted from `DRAW_SCHEMA.md`'s "Minimal worked example". The two
differed in `note`/`tournament_id`/`name`, the seed block and every bracket
entrant, and — substantively — the frontend copy **omitted the `event_date`
field** and carried a `"Qualifier/Lucky Loser"` compound placeholder.
`DRAW_SCHEMA.md` is the canonical schema reference (built in the AO-2026 addendum,
guarded by `test_draw_schema_doc.py` against drifting from `config`), so it is the
source of truth; the frontend's copy was updated to match it byte-for-byte. This
was also the *more correct* direction independently: `SCHEMA_EXAMPLE` is
**display-only** (rendered in a `<code>` block, never loaded or submitted), and the
doc's version demonstrates the upload-specific `event_date` field the frontend copy
was missing. The stale "copied from `example_usopen_2026.json`" comment was
corrected to point at `DRAW_SCHEMA.md` and to name the guard test.

**2. Full trivy scan of the built Docker API image (base-OS layer).** Image built
fresh (`docker build --build-arg PRECOMPUTE_RUNS=200 -t ace-api:secreview2 .`),
smoke-tested (`/health` → `200 ok`), and scanned with
`trivy image --severity HIGH,CRITICAL`. **Result: 23 OS-package + 5 distinct
Python-package HIGH/CRITICAL — none reachable in this app's actual runtime.**
Triaged to the same standard as §3, not counted raw:

  * **OS layer (Debian 13 base, all `python:3.11-slim`'s):** every finding has
    **no fixed version available from Debian yet** (`FixedVersion: -`,
    `status: affected`/`fix_deferred`), so a rebuild today cannot patch them. They
    sit in base-image packages a headless JSON API never exercises on untrusted
    input: `util-linux`/`libblkid` (`CVE-2026-53615`, DOS-partition-table integer
    overflow — the app parses no disk images), `gzip` (`CVE-2026-41992`),
    `libacl1` (`CVE-2026-54369`, local symlink traversal), `ncurses`
    (`CVE-2025-69720`, terminal-input overflow — no TTY in the container), and
    **`perl-base` (7 CVEs incl. 4 CRITICAL: regex/Archive-Tar/Storable/IO-Compress)**.
    `perl` is present at `/usr/bin/perl` as base-image cruft but the runtime process
    is uvicorn/Python — it never spawns perl, extracts a tar, or compiles a perl
    regex, so the perl CVEs are structurally unreachable. This is **continuity with
    T5.1's accepted posture**, not a new decision: base images pinned by tag, the
    weekly rebuild picks up Debian fixes as they ship (none available for these
    yet), and the packages are inert to the request path.
  * **Python layer:** the 3 `starlette 0.48.0` findings (`CVE-2025-62727` Range-DoS,
    `CVE-2026-48818` StaticFiles UNC SSRF, `CVE-2026-54283` `form()` limits) are the
    **same advisories §3 already accepted** under their CVE aliases (PYSEC-2026-1942 /
    -2281 / -249) — still unreachable (no `StaticFiles`/`FileResponse`, no form
    endpoints, Linux target) and still pinned behind `fastapi<0.49`. **Two are new
    to this scan but inert:** `wheel 0.45.1` (`CVE-2026-24049`, malicious-wheel code
    exec) and `jaraco.context 5.3.0` (`CVE-2026-23949`, tar path-traversal) are
    **build/packaging tools left in the venv**, never invoked at request time (the
    server installs no wheels and extracts no archives while serving). Candidates to
    prune from the runtime image in a future pass — the same treatment the Dockerfile
    already gives `nvidia-nccl` — but not a runtime exposure. **Nothing new is
    reachable; nothing changes the deploy decision.**

**3. `nanoid` bumped past the advisory.** `npm audit fix` moved the transitive
`vite → postcss → nanoid` from `3.3.17` to **`3.3.18`** (resolves GHSA-2v37-7h3g-55p8).
Re-verified: `npm audit` → **0 vulnerabilities**, and after a fresh
`npm run build` the string `nanoid` appears **nowhere in `dist/`** — it remains a
build-time-only dependency, absent from the shipped bundle.

### 6g. Final verification — everything green

- `pytest -q` → **791 passed, 0 failed** (the previously-failing schema-doc test
  now passes; the four `LiveComputeGate` tests and every other test pass).
- `npx vitest run` → **154 passed**; `tsc -b` (typecheck) → **clean**;
  `oxlint` → **clean**; `npm run build` → **succeeds**; `npm audit` → **0**.
- Live re-verification (headers, CORS, all three rate limits, the new concurrency
  gate) from the initial pass still holds; Docker image builds and serves `/health`.

**Files changed across all of Review 2:** `src/config.py`
(`API_LIVE_COMPUTE_MAX_CONCURRENCY`), `src/api/main.py` (`LiveComputeGate` +
`live_compute_slot` on `/storybook` and `/match/simulate`),
`tests/test_api_security.py` (four new concurrency tests),
`frontend/src/components/Upload.tsx` (`SCHEMA_EXAMPLE` re-aligned to
`DRAW_SCHEMA.md`), `frontend/package-lock.json` (`nanoid` 3.3.17 → 3.3.18), and
this document. No change to any data, model, or trained artefact.
