"""Security hardening: response headers, /docs CSP exception, the /storybook
rate limit, and a path-traversal regression guard for ``cache_path_for``.

These cover the behaviour added by the security review (2026-08-12). The app is
built through ``create_app``'s injection seams (fixture draws + a stub
classifier) so nothing here touches the vendored dataset or the pinned model —
the same pattern ``tests/test_api_storybook.py`` uses.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import config
from api.deps import ApiContext
import api.main as api_main
from api.main import (
    API_CSP,
    DOCS_CSP,
    LiveComputeGate,
    SlidingWindowRateLimiter,
    _client_key,
    _parse_rate,
    create_app,
)
from api.registry import cache_path_for
from features.serve import PlayerSkill, SkillTable

FIXTURE_DRAWS = Path(__file__).parent / "fixtures" / "draws"
FIXTURE_CACHE = Path(__file__).parent / "fixtures" / "cache"

TOY_PLAYERS = {
    "Alice Ace": "A1", "Bob Baseline": "B2", "Cara Chip": "C3", "Dan Drop": "D4",
    "Eve Emphatic": "E5", "Frank Fault": "F6", "Gina Grip": "G7", "Hank Half": "H8",
}


@pytest.fixture
def skill_table() -> SkillTable:
    skills = {}
    for i, pid in enumerate(TOY_PLAYERS.values()):
        for surface in config.VALID_SURFACES:
            mu = config.SURFACE_MU[surface]
            skills[(pid, surface)] = PlayerSkill(
                spw=mu + 0.01 * i, rpw=1.0 - mu + 0.01 * i,
                n_serve_pts=1000.0, n_return_pts=1000.0,
            )
    return SkillTable(skills, dict(TOY_PLAYERS), dict(config.SURFACE_MU),
                      cutoff=pd.Timestamp("2026-01-01"))


@pytest.fixture
def context(skill_table: SkillTable) -> ApiContext:
    return ApiContext(
        data=pd.DataFrame({"tourney_date": pd.to_datetime(["2026-01-01"])}),
        surface_history={}, h2h_history={}, skill_table=skill_table,
        estimator=object(), estimator_class="StubClassifier", data_through_year=2026,
    )


class _StubClassifier:
    def __call__(self, a: str, b: str, surface: str) -> float:
        return 0.55 if a < b else 0.45


def _stub_factory(estimator, data, surface_history, h2h_history):
    return _StubClassifier()


def _make_app(context: ApiContext):
    return create_app(
        context_factory=lambda: context,
        draws_dir=FIXTURE_DRAWS,
        cache_dir=FIXTURE_CACHE,
        classifier_factory=_stub_factory,
    )


@pytest.fixture
def client(context: ApiContext):
    with TestClient(_make_app(context)) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# Security headers
# --------------------------------------------------------------------------- #
STANDARD_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


@pytest.mark.parametrize("path", ["/health", "/players?query=a", "/tournaments"])
def test_json_endpoints_carry_standard_headers_and_strict_csp(client, path):
    response = client.get(path)
    assert response.status_code == 200
    for header, value in STANDARD_HEADERS.items():
        assert response.headers[header] == value
    assert response.headers["Content-Security-Policy"] == API_CSP
    assert "default-src 'none'" in response.headers["Content-Security-Policy"]


def test_error_responses_still_carry_headers(client):
    # A 404 is produced by an HTTPException, handled inside the middleware stack.
    response = client.get("/tournaments/does_not_exist/bracket")
    assert response.status_code == 404
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Content-Security-Policy"] == API_CSP


def test_docs_get_a_relaxed_csp_but_json_does_not(client):
    docs = client.get("/docs")
    assert docs.status_code == 200
    csp = docs.headers["Content-Security-Policy"]
    assert csp == DOCS_CSP
    assert "cdn.jsdelivr.net" in csp  # Swagger UI's bundle must be allowed
    assert docs.headers["X-Content-Type-Options"] == "nosniff"

    # The schema itself is public and strict-CSP'd like any other JSON.
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert schema.headers["Content-Security-Policy"] == API_CSP


# --------------------------------------------------------------------------- #
# Rate limit primitives
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "spec, expected",
    [("12/minute", (12, 60)), ("5/second", (5, 1)), ("100/hour", (100, 3600)),
     ("3/minutes", (3, 60)), ("2/day", (2, 86400))],
)
def test_parse_rate(spec, expected):
    assert _parse_rate(spec) == expected


@pytest.mark.parametrize("bad", ["12", "12/", "/minute", "12/fortnight", "x/minute"])
def test_parse_rate_rejects_garbage(bad):
    with pytest.raises(ValueError):
        _parse_rate(bad)


def test_sliding_window_allows_then_blocks():
    limiter = SlidingWindowRateLimiter("3/minute")
    assert limiter.check("ip-a") is None
    assert limiter.check("ip-a") is None
    assert limiter.check("ip-a") is None
    retry = limiter.check("ip-a")
    assert retry is not None and retry >= 1
    # A different key has its own independent window.
    assert limiter.check("ip-b") is None


def test_client_key_prefers_forwarded_for():
    class _Req:
        def __init__(self, headers, host):
            self.headers = headers
            self.client = type("C", (), {"host": host})()

    assert _client_key(_Req({"x-forwarded-for": "1.2.3.4, 10.0.0.1"}, "10.0.0.1")) == "1.2.3.4"
    assert _client_key(_Req({}, "9.9.9.9")) == "9.9.9.9"


def test_client_key_prefers_cf_connecting_ip_over_spoofable_xff():
    class _Req:
        def __init__(self, headers, host="10.0.0.1"):
            self.headers = headers
            self.client = type("C", (), {"host": host})()

    # Cloudflare overwrites CF-Connecting-IP with the real peer, so a client that
    # rotates X-Forwarded-For to dodge the limit is still keyed on the CF value.
    real = "203.0.113.7"
    assert _client_key(_Req({"cf-connecting-ip": real, "x-forwarded-for": "1.2.3.4"})) == real
    # No spoof: still the CF value.
    assert _client_key(_Req({"cf-connecting-ip": real})) == real
    # Off-platform (no Cloudflare): falls back to XFF, then socket peer.
    assert _client_key(_Req({"x-forwarded-for": "5.6.7.8, 10.0.0.1"})) == "5.6.7.8"
    assert _client_key(_Req({}, host="9.9.9.9")) == "9.9.9.9"


# --------------------------------------------------------------------------- #
# Rate limit, end to end
# --------------------------------------------------------------------------- #
def test_storybook_is_rate_limited_but_other_routes_are_not(context, monkeypatch):
    # A tiny limit so the test is fast and independent of the shipped value.
    monkeypatch.setattr(config, "API_STORYBOOK_RATE_LIMIT", "3/minute")
    with TestClient(_make_app(context)) as client:
        codes = [
            client.get("/tournaments/toy_open_2026/storybook?seed=%d" % i).status_code
            for i in range(5)
        ]
        assert codes == [200, 200, 200, 429, 429]

        blocked = client.get("/tournaments/toy_open_2026/storybook?seed=99")
        assert blocked.status_code == 429
        assert blocked.json()["detail"]["reason"] == "rate_limited"
        assert int(blocked.headers["Retry-After"]) >= 1

        # The cheap, non-simulating endpoints are never throttled.
        for _ in range(5):
            assert client.get("/health").status_code == 200
            assert client.get("/tournaments/toy_open_2026/bracket").status_code == 200


# --------------------------------------------------------------------------- #
# Path-traversal regression guard (re-verifies T3.3's cache_path_for)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "malicious",
    [
        "../../etc/passwd", "..", ".", "../secrets", "foo/bar", "foo\\bar",
        "\\\\server\\share", "/etc/passwd", "outputs/tennis_model",
        "cache\x00.json", "a/../../b",
    ],
)
def test_cache_path_for_rejects_traversal(malicious, tmp_path):
    assert cache_path_for(malicious, tmp_path) is None


def test_cache_path_for_contains_literal_names_inside_cache_dir(tmp_path):
    # Non-separator oddities are allowed but MUST resolve inside the cache dir.
    for name in ["usopen_2024_atp_full", "..%2F..%2Fetc", "con"]:
        path = cache_path_for(name, tmp_path)
        assert path is not None
        assert path.parent.resolve() == tmp_path.resolve()


# --------------------------------------------------------------------------- #
# Deployed rate-limit value (the end-to-end test above overrides it, so nothing
# otherwise pins what actually ships — a silent bump to a huge value would pass
# every other test).
# --------------------------------------------------------------------------- #
def test_shipped_storybook_rate_limit_is_sane():
    count, window = _parse_rate(config.API_STORYBOOK_RATE_LIMIT)
    # A real, small per-IP cap on the one compute-heavy endpoint — not disabled
    # by being set absurdly high, and a well-formed spec the limiter can parse.
    assert 1 <= count <= 60
    assert window in (1, 60, 3600, 86400)


# --------------------------------------------------------------------------- #
# The docs-vs-strict CSP decision must key on the ROUTED path, not on
# request.url (which is rebuilt from the spoofable Host header —
# PYSEC-2026-161/248). We can at least pin that a normal non-docs route gets the
# strict policy and only the real /docs family gets the relaxed one.
# --------------------------------------------------------------------------- #
def test_relaxed_csp_is_scoped_to_docs_only(client):
    for path in ["/health", "/openapi.json", "/tournaments"]:
        assert client.get(path).headers["Content-Security-Policy"] == API_CSP
    for path in ["/docs", "/redoc"]:
        assert client.get(path).headers["Content-Security-Policy"] == DOCS_CSP


# --------------------------------------------------------------------------- #
# Unhandled 500s bypass the header middleware (ServerErrorMiddleware sits outside
# it), so a catch-all handler must re-attach the headers AND keep the body
# generic — no internal detail may leak.
# --------------------------------------------------------------------------- #
def test_unhandled_500_carries_headers_and_a_non_leaking_body(context, monkeypatch):
    secret = "SECRET-INTERNAL-abc123"

    def boom(*args, **kwargs):
        raise RuntimeError(secret)

    # storybook_run is the symbol the /storybook handler calls; make it blow up.
    monkeypatch.setattr(api_main, "storybook_run", boom)
    with TestClient(_make_app(context), raise_server_exceptions=False) as c:
        r = c.get("/tournaments/toy_open_2026/storybook?seed=1")
        assert r.status_code == 500
        # Headers present despite bypassing the middleware.
        for header, value in STANDARD_HEADERS.items():
            assert r.headers[header] == value
        assert r.headers["Content-Security-Policy"] == API_CSP
        # Body is the fixed generic message; the exception text does not leak.
        assert r.json() == {"detail": "Internal Server Error"}
        assert secret not in r.text


# --------------------------------------------------------------------------- #
# render.yaml CSP: connect-src must be pinned to the EXACT API origin, never a
# wildcard. onrender.com is shared public hosting, so https://*.onrender.com
# would let a future XSS exfiltrate to any sub-app on the platform.
# --------------------------------------------------------------------------- #
def test_frontend_csp_connect_src_is_pinned_not_wildcard():
    render_yaml = Path(__file__).resolve().parents[1] / "render.yaml"
    text = render_yaml.read_text(encoding="utf-8")
    assert "connect-src 'self' https://ace-e21v.onrender.com;" in text
    # No wildcard host as a CSP source (the comment may still name it as the
    # anti-pattern being avoided; what must never reappear is a wildcard origin).
    assert "https://*.onrender.com" not in text


# --------------------------------------------------------------------------- #
# Shared live-compute concurrency cap (security-review-2)
#
# The per-IP rate limits bound each client's request RATE but not how many live
# simulations run AT ONCE. LiveComputeGate caps concurrent /storybook +
# /match/simulate executions across every IP and sheds the excess with a 503,
# so a burst of concurrent requests (one client or many) cannot pile a
# threadpool's worth of 128-match brackets onto a 0.1-CPU instance.
# --------------------------------------------------------------------------- #
def test_live_compute_gate_admits_up_to_cap_then_sheds():
    gate = LiveComputeGate(2)
    assert gate.acquire() is True
    assert gate.acquire() is True
    # Cap reached: further acquires are refused without blocking.
    assert gate.acquire() is False
    # Releasing frees exactly one slot.
    gate.release()
    assert gate.acquire() is True
    assert gate.acquire() is False


def test_live_compute_gate_rejects_bad_capacity():
    with pytest.raises(ValueError):
        LiveComputeGate(0)


def test_concurrent_live_sims_over_the_cap_get_503(context, monkeypatch):
    """A burst of concurrent /storybook calls beyond the cap is shed with 503.

    Uses a classifier stub that blocks until the test releases it, so the
    admitted requests genuinely hold their slots while the rest arrive — which is
    what forces the gate to shed rather than serialise. Distinct client keys per
    request rule the per-IP rate limiter out as the cause.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    cap = 2
    monkeypatch.setattr(config, "API_LIVE_COMPUTE_MAX_CONCURRENCY", cap)

    release = threading.Event()

    class _BlockingClassifier:
        def __call__(self, a: str, b: str, surface: str) -> float:
            # Hold the slot until the test lets go (bounded so a bug can't hang
            # the suite).
            release.wait(timeout=10)
            return 0.55 if a < b else 0.45

    def _blocking_factory(estimator, data, surface_history, h2h_history):
        return _BlockingClassifier()

    app = create_app(
        context_factory=lambda: context,
        draws_dir=FIXTURE_DRAWS,
        cache_dir=FIXTURE_CACHE,
        classifier_factory=_blocking_factory,
    )

    n = 5  # cap admitted, the rest shed
    with TestClient(app) as client:
        def fire(i: int):
            return client.get(
                f"/tournaments/toy_open_2026/storybook?seed={i}",
                headers={"CF-Connecting-IP": f"client-{i}"},
            ).status_code

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = [pool.submit(fire, i) for i in range(n)]
            # Give the admitted requests time to occupy their slots and the
            # excess time to be shed, then release everyone.
            time.sleep(0.5)
            release.set()
            codes = [f.result(timeout=15) for f in futures]

    assert sorted(codes) == [200] * cap + [503] * (n - cap), codes


def test_storybook_and_match_routes_declare_the_concurrency_dependency(context):
    """Both live-compute routes carry the gate dependency; read-only ones do not."""
    app = _make_app(context)
    guarded = {"/tournaments/{tournament_id}/storybook", "/match/simulate"}
    seen = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        if path not in guarded:
            continue
        dep_names = {
            getattr(d.call, "__name__", "")
            for d in route.dependant.dependencies
        }
        assert "live_compute_slot" in dep_names, (path, dep_names)
        seen.add(path)
    assert seen == guarded
