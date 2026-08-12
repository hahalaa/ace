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
