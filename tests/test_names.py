"""Tests for the extracted, UI-free name resolver (T0.6).

Covers the four matching strategies (exact → initials → substring → fuzzy) and
the ambiguous case as *data* (candidates returned, nothing printed), plus the
CLI wrapper regression proving the old ``resolve_player_name`` still resolves to
the same names and prints the same ambiguity messages.
"""
import pytest

from common.names import (
    MatchStrategy,
    NameIndex,
    NameMatch,
    resolve_name,
)

# A stable roster with the collisions each strategy's ambiguity test needs:
#  - two "*  Djokovic" for an ambiguous initials query ("N Djokovic")
#  - "Alex*" pair for an ambiguous substring query ("alex")
NAMES = [
    "Carlos Alcaraz",
    "Jannik Sinner",
    "Novak Djokovic",
    "Nikola Djokovic",
    "Rafael Nadal",
    "Alex de Minaur",
    "Alexander Zverev",
]

# Same roster with ids attached, to prove the resolver is id-agnostic.
NAME_TO_ID = {name: f"id-{i}" for i, name in enumerate(NAMES)}


@pytest.fixture
def index() -> NameIndex:
    return NameIndex.from_names(NAMES)


# --------------------------------------------------------------------------- #
# Unique matches — one per strategy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "query, expected, strategy",
    [
        ("carlos alcaraz", "Carlos Alcaraz", MatchStrategy.EXACT),      # exact, case-insensitive
        ("CARLOS ALCARAZ", "Carlos Alcaraz", MatchStrategy.EXACT),
        ("C Alcaraz", "Carlos Alcaraz", MatchStrategy.INITIALS),        # "F Lastname"
        ("A de minaur", "Alex de Minaur", MatchStrategy.INITIALS),      # initial + multi-word lastname
        ("nadal", "Rafael Nadal", MatchStrategy.SUBSTRING),             # substring
        ("Jannik Sinnner", "Jannik Sinner", MatchStrategy.FUZZY),       # typo -> fuzzy fallback
    ],
)
def test_unique_match_per_strategy(index, query, expected, strategy):
    match = resolve_name(query, index)
    assert isinstance(match, NameMatch)
    assert match.name == expected
    assert match.strategy is strategy
    assert not match.is_ambiguous
    assert match.candidates == []


def test_dotted_initial_falls_through_to_fuzzy_not_initials(index):
    """Preserved quirk: the initials stage guards on ``len(parts[0]) == 1``,
    so ``"C. Alcaraz"`` (dotted -> first token is length 2) does NOT match as
    initials — it falls through to the fuzzy stage. The un-dotted ``"C Alcaraz"``
    is the form that resolves via initials. This is behaviour, not intent; the
    lift must keep it verbatim (see ace-04-current-state.md §3)."""
    dotted = resolve_name("C. Alcaraz", index)
    assert dotted is not None
    assert dotted.name == "Carlos Alcaraz"
    assert dotted.strategy is MatchStrategy.FUZZY      # fell through, not initials
    assert dotted.strategy is not MatchStrategy.INITIALS

    undotted = resolve_name("C Alcaraz", index)
    assert undotted.strategy is MatchStrategy.INITIALS  # the form that IS initials


def test_no_match_returns_none(index):
    assert resolve_name("Roger Federer", index) is None


def test_empty_query_returns_none(index):
    assert resolve_name("", index) is None


# --------------------------------------------------------------------------- #
# Ambiguity is returned as data, never printed
# --------------------------------------------------------------------------- #
def test_ambiguous_initials_returns_candidates(index, capsys):
    match = resolve_name("N Djokovic", index)
    assert match.is_ambiguous
    assert match.strategy is MatchStrategy.INITIALS
    assert set(match.candidates) == {"Novak Djokovic", "Nikola Djokovic"}
    assert match.name == ""
    assert capsys.readouterr().out == ""  # pure: nothing printed


def test_ambiguous_substring_returns_candidates(index, capsys):
    match = resolve_name("alex", index)
    assert match.is_ambiguous
    assert match.strategy is MatchStrategy.SUBSTRING
    assert set(match.candidates) == {"Alex de Minaur", "Alexander Zverev"}
    assert capsys.readouterr().out == ""


def test_ambiguous_fuzzy_returns_candidates(capsys):
    # Two near-identical names, queried with a typo that is a substring of
    # neither, so only the fuzzy stage can match — and it matches both.
    idx = NameIndex.from_names(["Novak Djokovic", "Novak Djokavic"])
    match = resolve_name("Novak Djokevic", idx)
    assert match.is_ambiguous
    assert match.strategy is MatchStrategy.FUZZY
    assert set(match.candidates) == {"Novak Djokovic", "Novak Djokavic"}
    assert capsys.readouterr().out == ""


def test_strategy_order_exact_beats_substring():
    # "Sinner" is both an exact-able and substring-able target; a query equal to
    # a full name must resolve exact, not fall through to substring.
    idx = NameIndex.from_names(["Jannik Sinner", "Jannik Sinner Jr"])
    match = resolve_name("Jannik Sinner", idx)
    assert match.strategy is MatchStrategy.EXACT
    assert match.name == "Jannik Sinner"


# --------------------------------------------------------------------------- #
# Id-agnostic: name-only index -> None; name->id index -> the id
# --------------------------------------------------------------------------- #
def test_name_only_index_yields_null_id(index):
    assert resolve_name("nadal", index).player_id is None


def test_mapping_index_yields_player_id():
    idx = NameIndex.from_mapping(NAME_TO_ID)
    match = resolve_name("nadal", idx)
    assert match.name == "Rafael Nadal"
    assert match.player_id == NAME_TO_ID["Rafael Nadal"]


def test_common_names_has_no_ui_or_layer_imports():
    """The resolver module must not print or import cli/sim/api."""
    import inspect

    import common.names as names

    src = inspect.getsource(names)
    assert "print(" not in src
    for forbidden in ("import cli", "from cli", "import sim", "from sim",
                      "import api", "from api"):
        assert forbidden not in src


# --------------------------------------------------------------------------- #
# CLI wrapper regression — same resolved names AND same printed messages
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "query, expected",
    [
        ("carlos alcaraz", "Carlos Alcaraz"),
        ("C Alcaraz", "Carlos Alcaraz"),
        ("nadal", "Rafael Nadal"),
        ("Jannik Sinnner", "Jannik Sinner"),
    ],
)
def test_cli_wrapper_resolves_same_names(query, expected):
    from cli.interactive import resolve_player_name

    assert resolve_player_name(query, NAMES) == expected


def test_cli_wrapper_no_match_returns_none_silently(capsys):
    from cli.interactive import resolve_player_name

    assert resolve_player_name("Roger Federer", NAMES) is None
    assert capsys.readouterr().out == ""


def test_cli_wrapper_ambiguous_initials_message(capsys):
    from cli.interactive import resolve_player_name

    assert resolve_player_name("N Djokovic", NAMES) is None
    out = capsys.readouterr().out
    assert out.startswith("Ambiguous: Multiple players match 'N Djokovic': ")
    assert "Please be more specific." in out
    assert "Novak Djokovic" in out and "Nikola Djokovic" in out


def test_cli_wrapper_ambiguous_substring_truncates_to_five(capsys):
    from cli.interactive import resolve_player_name

    names = [f"Alex Player{i}" for i in range(7)]  # 7 substring hits on "alex"
    assert resolve_player_name("alex", names) is None
    out = capsys.readouterr().out
    assert out.startswith("Ambiguous: Multiple players match 'alex': ")
    assert "..." in out  # >5 candidates -> truncated with ellipsis
    # Pin the boundary at *exactly* 5: the first five (Player0..Player4) are
    # shown and the sixth (Player5) is not. A weaker "Player5 absent" check
    # would pass for any limit <= 5 and miss a 5->3/5->4 drift; asserting
    # Player4 present + Player5 absent fails on any limit != 5.
    for i in range(5):
        assert f"Player{i}" in out
    assert "Player5" not in out and "Player6" not in out


def test_cli_wrapper_ambiguous_fuzzy_message(capsys):
    from cli.interactive import resolve_player_name

    names = ["Novak Djokovic", "Novak Djokavic"]
    assert resolve_player_name("Novak Djokevic", names) is None
    assert capsys.readouterr().out.startswith("Did you mean: ")
