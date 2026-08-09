"""Startup-loaded application state for the API (T3.1).

Everything expensive the API needs — the vendored data, the leakage-safe
histories, the T1.1 skill table and the pinned classifier — is loaded **once**,
at app startup, into an immutable :class:`ApiContext` stored on ``app.state``.
No request handler ever loads a file, and (per ``CLAUDE.md``) nothing here
reaches the network. This is the Phase 3 global constraint: *load the skill
table + classifier once at startup, not per request.*

**Why this is not simply ``cli.simulate_match.build_context``.** T1.10 built the
same startup context for the CLIs and T2.5 reused it, so a third loader looks
like duplication. It is not reusable here: ``build_context`` lives in ``cli/``,
so importing it would put ``api/ → cli/`` into the dependency graph, which the
layering rule forbids outright. (T3.1 had a second reason — that ``SimContext``'s
``classifier`` field was then an adapter built from
``cli/interactive.build_feature_row``. That reason is **gone**: since T3.5
``build_context`` calls ``common.classifier_adapter.make_classifier_prob`` like
every other caller. The layering reason stands on its own, so the decision is
unchanged.) So this module reuses every *core* step verbatim
(``data.loader`` → ``data.preprocess`` → ``features.engineering.add_features``,
plus ``sim.reconcile.load_pinned_classifier``) and duplicates only the four-line
orchestration that sequences them.

**What is deliberately *not* built here: the ``ClassifierProb`` adapter.**
:class:`ApiContext` carries the pinned **estimator** and the two name-keyed
histories — the raw materials an adapter needs — but stops there. The adapter is
built by whoever actually needs one: ``create_app``'s lifespan for the live
``/storybook`` endpoint (resolved through ``config.API_CLASSIFIER_ADAPTER``), and
``scripts/precompute_sim.py`` for the offline Monte Carlo. Both construct it from
exactly these materials, and construction is free — the rolling-form table and
every ``predict_proba`` are deferred to first use, then memoised — so nothing is
lost by not building it here, and endpoints that need no ``P_clf`` at all (such
as ``/players``) do not pay for one.

**Historical note, because the old text here said otherwise until 2026-08-09.**
This docstring used to record that the only adapter lived in ``cli/`` and filled
20 of the 27 ``config.MODEL_FEATURES`` with synthetic constants, collapsing
``P_clf`` toward 0.5. **That defect was fixed by T3.5** —
``common/classifier_adapter.py`` is now the single UI-free adapter, it assembles
all 27 features from the pipeline's own leakage-safe state, and its held-out
Brier score equals the training pipeline's own. The prose above was simply
missed when the rest of the codebase was updated.

Nothing in this module calls the classifier: the estimator is loaded because the
Phase 3 constraint says startup is where loading happens, and because failing
fast on a missing ``outputs/tennis_model.pkl`` at startup beats failing on the
first simulate request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd

import config
import data.loader as loader
import data.preprocess as preprocess
import features.engineering as engineering
from features.serve import SkillTable
from sim.reconcile import load_pinned_classifier

logger = logging.getLogger(__name__)

# A factory the app calls exactly once at startup to obtain its context.
# Parameterising it is what lets tests inject a small fixture context instead of
# the full vendored dataset (and count the calls).
ContextFactory = Callable[[], "ApiContext"]


@dataclass(frozen=True)
class ApiContext:
    """Immutable startup state shared by every request handler.

    Attributes:
        data: The engineered match frame (name-keyed; latest rank/age live here).
        surface_history: Name-keyed leakage-safe surface records from
            ``features.engineering.add_features``.
        h2h_history: Name-keyed head-to-head records from the same call.
        skill_table: The T1.1 id-keyed serve/return skill table — the API's
            authoritative player universe and the source of ``/players`` skills.
        estimator: The pinned classifier, an opaque ``predict_proba`` estimator
            (whichever of the best-of-four won on ``TEST_YEAR``; see
            ``ace-04-current-state.md §4``). **Not called by any T3.1 endpoint** —
            see the module docstring.
        estimator_class: Its class name, for provenance.
        data_through_year: The latest season actually present in ``data``.
    """

    data: pd.DataFrame
    surface_history: dict
    h2h_history: dict
    skill_table: SkillTable
    estimator: Any
    estimator_class: str
    data_through_year: int

    @property
    def n_players(self) -> int:
        """Size of the searchable player universe (see ``SkillTable.name_index``)."""
        return len(self.skill_table.name_index.names)


def build_api_context(
    start_year: int = config.START_YEAR,
    end_year: int = config.END_YEAR,
    model_path: Path = config.MODEL_PATH,
) -> ApiContext:
    """Run the offline pipeline once and pin the classifier.

    Reads only vendored files — no network (``CLAUDE.md`` rule 5). Measured at
    **1.8 s** for the full 2014–2026 range (35,319 matches, 1,408 players,
    2026-08-02, dev machine) — cheap enough that startup is unremarkable, and
    still far too expensive to repeat per request.

    Args:
        start_year: First vendored season to load.
        end_year: Last vendored season to load (the *configured* end; the
            resulting :attr:`ApiContext.data_through_year` reports what was
            actually found).
        model_path: Path to the persisted classifier.

    Returns:
        The populated :class:`ApiContext`.

    Raises:
        FileNotFoundError: If a vendored data year is missing (run
            ``scripts/refresh_data.py``) or the model artefact is absent (run
            ``python src/predictor.py``).
    """
    raw = loader.load_atp_data(start_year, end_year)
    processed = preprocess.preprocess_data(raw)
    data, surface_history, h2h_history, skill_table = engineering.add_features(processed)

    pinned = load_pinned_classifier(model_path)

    # END_YEAR is the *configured* data end; this is the season actually present.
    # They differ whenever a year fails to vendor or the last season has not
    # started, so /health reports the observed value rather than the intent.
    data_through_year = int(data["tourney_date"].dt.year.max())

    return ApiContext(
        data=data,
        surface_history=surface_history,
        h2h_history=h2h_history,
        skill_table=skill_table,
        estimator=pinned.estimator,
        estimator_class=pinned.estimator_class,
        data_through_year=data_through_year,
    )


def log_context(context: ApiContext) -> None:
    """Emit the one-line startup record proving the load happened (and once).

    Kept next to the loader so ``main.create_app`` logs the same line whether the
    context came from :func:`build_api_context` or from an injected test factory.
    """
    logger.info(
        "API startup: loaded %d matches through %d, %d players, classifier=%s",
        len(context.data),
        context.data_through_year,
        context.n_players,
        context.estimator_class,
    )
