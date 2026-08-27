"""FastAPI application layer (Phase 3).

``api/`` is a *consumer* of the core: it imports ``config``, ``data/``,
``features/``, ``sim/`` and ``common/``, and, like ``sim/``, it must **never**
import from ``cli/`` (the layering rule). Shared logic the CLI
and the API both need lives in a UI-free module; see ``api/deps.py`` for the one
place that rule visibly costs something.
"""
