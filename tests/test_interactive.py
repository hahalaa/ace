"""Regression tests for the CLI REPL loop in ``cli.interactive``.

Covers the pre-existing EOFError infinite-loop defect found during the Phase 0
cross-ticket audit (2026-07-19): on closed/empty stdin, ``input()`` raised
``EOFError`` every iteration, the broad ``except Exception`` swallowed it, and
the ``while True`` loop spun forever (~120k iterations/sec, unbounded output).
The fix is an explicit ``except EOFError: ... break`` ahead of the broad handler.

The test mirrors the audit's reproduction — feed the REPL closed/empty stdin —
but runs it in a daemon thread with a hard join timeout so that if the bug ever
regresses the suite fails fast instead of hanging (a Python thread can't be
force-killed, hence the daemon + is_alive check rather than a bare call).
"""
import io
import threading

import pandas as pd

import cli.interactive as interactive


def _minimal_frame() -> pd.DataFrame:
    """Smallest frame the loop needs: it reads p1_name/p2_name before any input."""
    return pd.DataFrame({"p1_name": ["Player A"], "p2_name": ["Player B"]})


def test_repl_exits_promptly_on_closed_stdin(monkeypatch):
    """Closed/empty stdin must make the REPL exit, not spin on EOFError.

    Reproduces the audit finding: with stdin at EOF, ``input()`` raises
    ``EOFError`` on the very first prompt. The loop must leave via the dedicated
    handler rather than re-looping. We never reach ``model.predict_proba`` (we
    break at the first prompt), so a ``None`` model is fine.
    """
    # Empty stream -> input() raises EOFError naturally, exactly like `echo -n "" |`.
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    finished = threading.Event()

    def run() -> None:
        try:
            interactive.interactive_prediction_loop(
                model=None, data=_minimal_frame(), surf_hist={}, h2h_hist={}
            )
        finally:
            finished.set()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=5.0)

    assert finished.is_set(), (
        "REPL did not exit within 5s on closed stdin — EOFError infinite loop "
        "has regressed (missing `except EOFError` before the broad handler)."
    )


def test_repl_breaks_on_first_eof_without_relooping(monkeypatch):
    """The loop must call input() exactly once before leaving on EOF.

    A bounded, deterministic complement to the timeout test: a fake ``input``
    counts calls and, if the loop ever re-enters after the first EOFError,
    raises a ``BaseException`` subclass that the broad ``except Exception``
    cannot swallow — so a regression surfaces as a hard error, not a hang.
    """

    class _LoopGuard(BaseException):
        """Not an Exception subclass, so the REPL's broad handler won't catch it."""

    calls = {"n": 0}

    def fake_input(prompt: str = "") -> str:
        calls["n"] += 1
        if calls["n"] > 1:
            raise _LoopGuard("REPL re-looped after EOF instead of breaking")
        raise EOFError

    monkeypatch.setattr("builtins.input", fake_input)

    # Must return normally (no _LoopGuard escaping) with the fix in place.
    interactive.interactive_prediction_loop(
        model=None, data=_minimal_frame(), surf_hist={}, h2h_hist={}
    )

    assert calls["n"] == 1, f"expected a single input() call, got {calls['n']}"
