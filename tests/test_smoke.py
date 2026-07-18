"""Smoke tests for the T0.5 scaffolding.

Assert the src-relative imports resolve via pyproject's ``pythonpath = ["src"]``
(no sys.path manipulation here), and that the sample fixture is discoverable.
"""


def test_src_modules_import():
    import config  # noqa: F401
    import data.loader  # noqa: F401
    import data.preprocess  # noqa: F401


def test_decoupled_win_pct_constants_exist():
    """DEFAULT_WIN_PCT's three overloads are now separate constants (T0.5)."""
    import config

    assert config.DEFAULT_WIN_PCT == 0.5
    assert config.PLAYER_SWAP_THRESHOLD == 0.5
    assert config.ACCURACY_PLOT_YMIN == 0.5
    assert config.PLAYER_SWAP_SEED == 42
    # The dangling legacy blob path is gone.
    assert not hasattr(config, "DATA_PATH")


def test_sample_fixture_exists(sample_matches_csv):
    assert sample_matches_csv.exists()
