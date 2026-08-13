"""In-memory store for user-uploaded tournament draws (draw upload feature).

**The whole design is one platform constraint made explicit.** The deployment
target (Render's free tier) has no persistent disk: a file written to a running
container is gone on the next restart, cold-start wake or redeploy. So an
uploaded draw is held *only* in this process's memory — an LRU keyed by a
generated id — and never touches the filesystem. The honest consequence, which
the API and the web app both disclose rather than paper over, is that an upload
does not survive a restart.

Two boundaries this module keeps:

* **A separate namespace from the curated draws.** Ids are generated as
  ``config.UPLOAD_ID_PREFIX + <random hex>``, never the draw's own
  ``tournament_id``. The git-committed example draws in ``data/draws/`` are
  independently verified against real historical records; a user upload is
  neither, and a generated id in its own namespace makes that distinction
  un-blurrable — an upload can never collide with or shadow a curated id, and a
  glance at an id says which kind it is.
* **A hard cap on how many are held.** :class:`UploadStore` evicts the
  least-recently-added draw once :data:`config.API_UPLOAD_MAX_DRAWS` is reached,
  so the feature cannot be turned into a memory-exhaustion lever on a small
  instance.

This module imports only ``config`` and ``sim.draw`` (for the :class:`~sim.draw.Draw`
it stores) — no FastAPI, no ``cli/`` — so it stays a plain data structure that a
test can exercise without a web app.
"""

from __future__ import annotations

import secrets
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timezone

import config
from sim.draw import Draw


@dataclass(frozen=True)
class UploadedDraw:
    """One draw a visitor submitted, plus the facts its disclosure needs.

    Attributes:
        upload_id: The generated, namespaced id this draw is addressed by
            (``config.UPLOAD_ID_PREFIX`` + random hex). **Not** the draw's own
            ``tournament_id`` — see the module docstring.
        draw: The validated :class:`~sim.draw.Draw`.
        is_forecast: True when :attr:`event_date` names a future event, so the
            simulation is a genuine forecast rather than a retrospective of an
            already-played draw. This is the one place in the project where the
            flag can legitimately be true.
        event_date: The declared event date, or ``None`` if the upload carried
            none. Drives :attr:`is_forecast`.
        created_at: When the draw was accepted (UTC), for LRU ordering and
            diagnostics.
    """

    upload_id: str
    draw: Draw
    is_forecast: bool
    event_date: date | None
    created_at: datetime


class UploadStore:
    """Thread-safe, bounded, in-memory LRU of uploaded draws.

    Sync FastAPI handlers run in a threadpool, so every access is guarded by a
    lock. Memory is bounded by :attr:`max_uploads`: adding past the cap evicts
    the oldest entry (insertion order), which is what keeps a stream of uploads
    from growing the process without limit.

    Nothing here is persisted. A fresh store starts empty, which is exactly the
    ephemeral behaviour the feature promises — a restart (a new process, a new
    store) has forgotten every prior upload.
    """

    def __init__(
        self,
        max_uploads: int = config.API_UPLOAD_MAX_DRAWS,
        id_prefix: str = config.UPLOAD_ID_PREFIX,
    ) -> None:
        if max_uploads < 1:
            raise ValueError(f"max_uploads must be >= 1, got {max_uploads}")
        self.max_uploads = max_uploads
        self.id_prefix = id_prefix
        self._store: "OrderedDict[str, UploadedDraw]" = OrderedDict()
        self._lock = threading.Lock()

    def _new_id(self) -> str:
        """A collision-free, filename-safe id in the upload namespace."""
        while True:
            candidate = f"{self.id_prefix}{secrets.token_hex(8)}"
            if candidate not in self._store:
                return candidate

    def add(
        self, draw: Draw, *, is_forecast: bool, event_date: date | None
    ) -> UploadedDraw:
        """Store ``draw`` under a fresh id, evicting the oldest if at capacity.

        Returns:
            The :class:`UploadedDraw` record, whose ``upload_id`` is how the
            draw is fetched back.
        """
        with self._lock:
            upload_id = self._new_id()
            record = UploadedDraw(
                upload_id=upload_id,
                draw=draw,
                is_forecast=is_forecast,
                event_date=event_date,
                created_at=datetime.now(timezone.utc),
            )
            self._store[upload_id] = record
            # Evict the oldest until we are back within the cap. A single add can
            # only ever push one over, but the loop is robust to a cap lowered at
            # runtime.
            while len(self._store) > self.max_uploads:
                self._store.popitem(last=False)
            return record

    def get(self, upload_id: str) -> UploadedDraw | None:
        """The stored draw for ``upload_id``, or ``None`` if unknown/evicted.

        A miss is the normal shape of the ephemeral promise: an id from before a
        restart, or one evicted by newer uploads, simply is not here.
        """
        with self._lock:
            return self._store.get(upload_id)

    def is_upload_id(self, tournament_id: str) -> bool:
        """True if ``tournament_id`` is in the upload namespace.

        A cheap prefix test the handlers use to decide whether to consult this
        store at all, before a lookup. Being in the namespace does not mean the
        draw is still held (it may have been evicted) — only :meth:`get` answers
        that.
        """
        return tournament_id.startswith(self.id_prefix)

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
