"""Keep ``data/draws/DRAW_SCHEMA.md`` honest against the real validator.

The schema reference documents the allowed value sets for a draw file. Those
sets are owned by ``config`` (and, through it, ``sim/draw.py`` and
``sim/match.py``). This test fails if the prose and ``config`` ever drift, so the
document a person hand-authoring a draw reaches for cannot quietly go stale.

It asserts every allowed value is *named* in the doc — not that the doc lists
nothing extra — so surrounding explanation is free, but a value the validator
accepts and the doc omits (or a value the doc invents) is caught.
"""

from __future__ import annotations

import re
from pathlib import Path

import config

SCHEMA_DOC = config.DRAWS_DIR / "DRAW_SCHEMA.md"

# The uploader UI embeds a copy of the doc's worked example (see the module note
# below). This is the path to that copy; the equality test keeps the two honest.
UPLOAD_COMPONENT = Path("frontend/src/components/Upload.tsx")


def _doc_text() -> str:
    return Path(SCHEMA_DOC).read_text(encoding="utf-8")


def test_schema_doc_exists() -> None:
    assert Path(SCHEMA_DOC).is_file(), f"missing {SCHEMA_DOC}"


def test_surfaces_are_documented() -> None:
    text = _doc_text()
    for surface in config.VALID_SURFACES:
        assert f'`"{surface}"`' in text, surface


def test_best_of_values_are_documented() -> None:
    text = _doc_text()
    for value in config.VALID_BEST_OF:
        assert f"`{value}`" in text, value


def test_final_set_tiebreaks_are_documented() -> None:
    text = _doc_text()
    for value in config.VALID_FINAL_SET_TIEBREAKS:
        assert f'`"{value}"`' in text, value


def test_draw_sizes_are_documented() -> None:
    text = _doc_text()
    for value in config.VALID_DRAW_SIZES:
        assert f"`{value}`" in text, value


def test_placeholder_tokens_are_documented() -> None:
    text = _doc_text().lower()
    for token in config.DRAW_PLACEHOLDER_ENTRANTS:
        assert token in text, token


def test_event_date_field_name_is_documented() -> None:
    assert f"`{config.UPLOAD_EVENT_DATE_FIELD}`" in _doc_text()


def test_required_fields_are_documented() -> None:
    from sim.draw import REQUIRED_FIELDS

    text = _doc_text()
    for field in REQUIRED_FIELDS:
        assert f"`{field}`" in text, field


def test_upload_ui_schema_example_matches_doc_worked_example() -> None:
    """The uploader's inline ``SCHEMA_EXAMPLE`` must be byte-identical to the
    doc's "Minimal worked example". The component claims in a comment that it is
    "copied verbatim" and "cannot drift" — this asserts that directly rather than
    trusting the comment, so editing one copy without the other fails the suite.
    """
    doc = _doc_text()
    start = doc.index("Minimal worked example")
    doc_match = re.search(r"```json\n(.*?)\n```", doc[start:], re.S)
    assert doc_match, "no ```json worked example found under 'Minimal worked example'"
    doc_example = doc_match.group(1)

    component = UPLOAD_COMPONENT.read_text(encoding="utf-8")
    ui_match = re.search(r"const SCHEMA_EXAMPLE = `(.*?)`;", component, re.S)
    assert ui_match, f"no SCHEMA_EXAMPLE template literal found in {UPLOAD_COMPONENT}"
    ui_example = ui_match.group(1)

    assert ui_example == doc_example, (
        "Upload.tsx SCHEMA_EXAMPLE has drifted from DRAW_SCHEMA.md's worked "
        "example; re-copy one to match the other."
    )
