"""Regression guards for the container build (T5.1).

These do not build an image — that is the documented manual/CI step in
``README.md`` (and T5.2's job). What they pin is the handful of facts that are
*silently* wrong when broken: an image that builds and runs perfectly while
serving a bundle pointed at the wrong API, or a build that quietly inherits a
developer's model pickle instead of producing one. Each assertion below
corresponds to a trap recorded in ``ace-04-current-state.md`` §7 seam 9 or in
the T5.1 notes of ``ace-phase-5-infra.md``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import config

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / ".dockerignore"
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"

# The host port docker-compose publishes the frontend on, and the API base URL
# baked into that image. Both are asserted against config/the compose file
# rather than assumed.
FRONTEND_HOST_PORT = 4173
API_HOST_PORT = 8000


def _read(path: Path) -> str:
    assert path.exists(), f"{path.name} is missing — T5.1 creates it"
    return path.read_text()


def _dockerignore_patterns() -> list[str]:
    return [
        line.strip()
        for line in _read(DOCKERIGNORE).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _compose() -> dict:
    yaml = pytest.importorskip(
        "yaml", reason="PyYAML (a uvicorn[standard] dependency) not installed"
    )
    return yaml.safe_load(_read(COMPOSE_FILE))


def _dockerfile_stages() -> list[str]:
    """Stage names, in file order, from every ``FROM … AS <name>``."""
    return re.findall(r"^FROM\s+\S+\s+AS\s+(\S+)", _read(DOCKERFILE), re.MULTILINE)


def _dockerfile_instructions() -> str:
    """The Dockerfile with comment lines dropped and continuations joined.

    The header comments discuss `vite preview`, `ENV` and the prune at length —
    precisely to explain what the file does *not* do — so every assertion about
    behaviour has to read instructions, never prose.
    """
    text = _read(DOCKERFILE).replace("\\\n", " ")
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def _prune_instruction() -> str | None:
    """The single RUN line that uninstalls packages from the shipped venv."""
    for line in _dockerfile_instructions().splitlines():
        if "pip uninstall" in line:
            return line
    return None


def _environment_keys(service: dict) -> set[str]:
    """Variable names in a compose ``environment:``, in either spelling.

    Compose accepts a mapping (``KEY: value``) *or* a list (``- KEY=value``).
    A membership test against the raw value silently passes on the list form,
    which is the more common idiom — so normalise before asserting.
    """
    env = service.get("environment") or {}
    if isinstance(env, dict):
        return set(env)
    return {str(item).split("=", 1)[0] for item in env}


# --------------------------------------------------------------------------- #
# .dockerignore — leanness, and the two load-bearing exclusions.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "pattern",
    ["frontend/node_modules/", ".git", "venv/", ".venv/", "**/__pycache__"],
)
def test_dockerignore_keeps_the_context_lean(pattern: str) -> None:
    assert pattern in _dockerignore_patterns()


def test_dockerignore_excludes_the_frontend_env_file() -> None:
    """The trap T4.5 verified: ``vite build`` reads ``.env``, not only ``vite dev``.

    ``frontend/.env`` is gitignored but sits on every developer machine holding
    ``http://localhost:8000``. A ``COPY frontend/ .`` that picked it up would
    bake that URL into a shipped image behind the build ARG's back. Excluding
    ``frontend/node_modules`` alone does not cover it.
    """
    patterns = _dockerignore_patterns()
    assert "frontend/.env" in patterns
    assert "frontend/.env.*" in patterns


@pytest.mark.parametrize("pattern", ["outputs/", "data/cache/"])
def test_dockerignore_excludes_the_regenerated_artefacts(pattern: str) -> None:
    """The two gitignored, runtime-required artefacts must not leak in.

    They are produced by the builder stage. Excluding them from the context is
    what makes every build behave like a clean checkout — otherwise a developer
    machine holding both would ship an image whose provenance nobody knows, and
    a genuinely clean CI build would be the first to discover the difference.
    """
    assert pattern in _dockerignore_patterns()


# --------------------------------------------------------------------------- #
# Dockerfile — stage shape and the artefact resolution.
# --------------------------------------------------------------------------- #
def test_api_is_the_default_build_target() -> None:
    """``docker build .`` must produce the API image (the acceptance criterion)."""
    stages = _dockerfile_stages()
    assert stages, "no named stages found in Dockerfile"
    assert stages[-1] == "api"
    assert "frontend" in stages


def test_builder_stage_regenerates_both_gitignored_artefacts() -> None:
    """Resolution (a): regenerate, rather than copy-and-hope.

    ``outputs/tennis_model.pkl`` and ``data/cache/*.json`` are absent from every
    checkout, so the build has to produce them or the image does not work —
    the API fails fast without the model and answers 425 without the cache.
    """
    text = _read(DOCKERFILE)
    assert "python src/predictor.py" in text
    assert "scripts/precompute_sim.py" in text
    # …and they must actually reach the runtime image.
    assert "COPY --from=artefacts /build/outputs/tennis_model.pkl" in text
    assert "COPY --from=artefacts /build/data/cache/" in text


def test_frontend_image_serves_static_files_and_never_runs_vite_preview() -> None:
    """T4.5's explicit finding: ``vite preview`` is a dev command, not a server.

    The built artefact is static files, so the frontend stage is a builder plus
    a file server — not a node image running preview.
    """
    assert re.search(
        r"^FROM\s+nginx:\S+\s+AS\s+frontend\s*$", _read(DOCKERFILE), re.MULTILINE
    )
    instructions = _dockerfile_instructions()
    assert "vite preview" not in instructions
    assert "npm run preview" not in instructions


def test_frontend_api_url_is_a_build_arg_in_the_dockerfile() -> None:
    text = _read(DOCKERFILE)
    assert "ARG VITE_API_BASE_URL" in text


# --------------------------------------------------------------------------- #
# docker-compose.yml — the wiring T4.5 said must not be done with `environment:`.
# --------------------------------------------------------------------------- #
def test_compose_defines_api_and_frontend() -> None:
    services = _compose()["services"]
    assert set(services) == {"api", "frontend"}


def test_compose_passes_the_api_url_as_a_build_arg_not_an_environment_entry() -> None:
    """The whole point of seam 9's first bullet.

    Vite inlines ``VITE_API_BASE_URL`` at build time and the minifier constant-
    folds the runtime lookup away, so an ``environment:`` entry does nothing at
    all — the image would keep whatever URL it was built with, and the mistake
    is invisible until someone loads the page.
    """
    frontend = _compose()["services"]["frontend"]
    args = frontend["build"]["args"]
    assert "VITE_API_BASE_URL" in args
    assert str(API_HOST_PORT) in args["VITE_API_BASE_URL"]
    # Both compose spellings, because the list form is the more common idiom and
    # a plain `"X" not in [...]` membership test sails straight past it.
    assert "VITE_API_BASE_URL" not in _environment_keys(frontend)


def test_dockerfile_does_not_restate_the_build_arg_as_env() -> None:
    """`ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}` looks harmless and is not.

    BuildKit already exposes a declared ARG to the stage's RUN steps, so the
    line adds nothing — but it turns *unset* into *set to the empty string*,
    which silences the unset-value warning `vite.config.ts` exists to emit, and
    it puts a phantom runtime knob into the image metadata for a value that was
    frozen at build time. Written once during T5.1 and removed; pinned here.
    """
    instructions = _dockerfile_instructions()
    assert "ARG VITE_API_BASE_URL" in instructions
    assert not re.search(r"^ENV\s+VITE_API_BASE_URL", instructions, re.MULTILINE)


def test_runtime_prune_is_chained_to_an_import_guard_that_exercises_xgboost() -> None:
    """The prune's safety net, and the one thing that must never be edited away.

    The ``runtime-deps`` stage uninstalls packages from the venv the shipped
    image runs on — including every ``nvidia*`` distribution, matched by a
    *pattern*, so a future dependency could be caught by it. The guard is what
    turns "removed something xgboost needed" into a failed build instead of a
    container that dies on startup. Verified live: pruning ``scipy`` with the
    guard fails the build at this line; pruning it without the guard builds
    clean and the container then exits 3 at startup.

    Chained with ``&&`` on purpose — a separate ``RUN`` would still fail, but a
    guard that can drift into a different layer from the prune it guards is a
    guard waiting to be reordered away.
    """
    prune = _prune_instruction()
    assert prune is not None, "no `pip uninstall` found in the Dockerfile"
    assert "import xgboost" in prune, "the guard must import xgboost specifically"
    assert "&&" in prune, "the guard must be chained to the prune, not a separate RUN"


def test_compose_frontend_origin_is_cors_allowed_by_the_api() -> None:
    """The published port is a CORS decision, not a cosmetic one.

    ``config.API_ALLOWED_ORIGINS`` is an explicit allow-list and never ``*``, so
    a frontend served on a port that is not on it loads fine and then fails
    every fetch — surfacing as the generic "could not reach the ace API" panel,
    because a browser hides the real cause.
    """
    ports = _compose()["services"]["frontend"]["ports"]
    assert any(str(p).startswith(f"{FRONTEND_HOST_PORT}:") for p in ports)
    assert f"http://localhost:{FRONTEND_HOST_PORT}" in config.API_ALLOWED_ORIGINS


def test_compose_api_is_published_where_the_bundle_expects_it() -> None:
    """The baked-in URL names the host, so the mapping has to match it."""
    ports = _compose()["services"]["api"]["ports"]
    assert any(str(p).startswith(f"{API_HOST_PORT}:") for p in ports)
