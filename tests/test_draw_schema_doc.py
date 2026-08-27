"""Keep ``data/draws/DRAW_SCHEMA.md`` honest against the real validator.

The schema reference documents the allowed value sets for a draw file. Those
sets are owned by ``config`` (and, through it, ``sim/draw.py`` and
``sim/match.py``). This test fails if the prose and ``config`` ever drift, so the
document a person hand-authoring a draw reaches for cannot quietly go stale.

It asserts every allowed value is *named* in the doc, not that the doc lists
nothing extra, so surrounding explanation is free, but a value the validator
accepts and the doc omits (or a value the doc invents) is caught.
"""

from __future__ import annotations

from pathlib import Path

import config

SCHEMA_DOC = config.DRAWS_DIR / "DRAW_SCHEMA.md"


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
