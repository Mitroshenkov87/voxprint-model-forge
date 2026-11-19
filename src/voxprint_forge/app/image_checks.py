"""Queue the Photos image check on a worker host and keep the result cache.

The window does not import the checker. A failed, crashed, or malformed job
sets :attr:`failure` and leaves the cache empty, so it cannot be read as a
photo result.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

from voxprint_common.worker import EventKind, Handlers, WorkerEvent, WorkerHost

MODEL_ID = "image-check"
CHECK_STOPPED = "The image check stopped."
HANDLERS = Handlers(
    run="voxprint_forge.lora.image_job:run",
    load_model="voxprint_forge.lora.image_job:load_model",
    unload_model="voxprint_forge.lora.image_job:unload_model",
    check="voxprint_forge.lora.image_job:check",
)


class ImageCheckError(RuntimeError):
    """The worker did not finish the check. This is not a photo result."""


def image_check_host(state: Path) -> WorkerHost:
    """A host whose child runs the image-check handlers."""
    return WorkerHost(HANDLERS, state_dir=state)


class ImageChecks:
    """One batch of photo checks and the cache of rows the worker returned."""

    def __init__(self, host: WorkerHost) -> None:
        self.host = host
        self.cache: dict[str, tuple[bool, str]] = {}
        self.failure: str | None = None
        self.armed = False
        self._job_id: str | None = None
        self._cancelled = False
        self._done = threading.Event()
        self._lock = threading.Lock()

    def submit(self, items: Sequence[tuple[str, str]]) -> None:
        """Queue one check for the active photos. ``items`` is ``(id, path)``."""
        payload = {"items": [{"id": photo_id, "path": path} for photo_id, path in items]}
        with self._lock:
            self.cache = {}
            self.failure = None
            self._cancelled = False
            self.armed = True
            self._job_id = None
            self._done.clear()
        if not payload["items"]:
            with self._lock:
                self._done.set()
            return
        job_id = self.host.check(payload, model=MODEL_ID, phase="image.check")
        with self._lock:
            self._job_id = job_id

    def cancel(self) -> None:
        """Stop the open check. Rows already in the cache are dropped."""
        with self._lock:
            self._cancelled = True
            self.cache = {}
            job_id = self._job_id
            self._done.set()
        if not job_id:
            return
        try:
            self.host.cancel(job_id)
        except LookupError:
            return

    def wait(self, photo_id: str, token: object) -> tuple[bool, str]:
        """Block until this photo's row is cached. Called off the UI thread."""
        while not self._done.wait(0.05):
            if _token_cancelled(token):
                return True, ""
        with self._lock:
            if _token_cancelled(token) or self._cancelled:
                return True, ""
            if self.failure:
                raise ImageCheckError(self.failure)
            found = self.cache.get(photo_id)
            if found is None:
                raise ImageCheckError(CHECK_STOPPED)
            return found

    def observe(self, event: WorkerEvent) -> None:
        """Apply one host event. Called on the UI thread."""
        with self._lock:
            if self._job_id and event.job_id and event.job_id != self._job_id:
                return
            if event.kind is EventKind.FINISHED:
                if self._cancelled:
                    self._done.set()
                    return
                self._accept(event.value)
                self._done.set()
                return
            if event.kind is EventKind.CANCELLED:
                self._cancelled = True
                self.cache = {}
                self._done.set()
                return
            if event.kind is EventKind.FAILED or event.kind is EventKind.CRASHED:
                if not self._cancelled:
                    self.failure = CHECK_STOPPED
                    self.cache = {}
                self._done.set()

    def wake_for_shutdown(self) -> None:
        """Unblock a thread waiting on the open check. Does not read a result."""
        with self._lock:
            self._cancelled = True
            self._done.set()

    def _accept(self, value: object) -> None:
        """Store a well-formed result list. Anything else is a failed check."""
        if not isinstance(value, dict):
            self._reject()
            return
        rows = value.get("results")
        if not isinstance(rows, list):
            self._reject()
            return
        stored: dict[str, tuple[bool, str]] = {}
        for row in rows:
            parsed = _row(row)
            if parsed is None:
                self._reject()
                return
            photo_id, ok, message = parsed
            stored[photo_id] = (ok, message)
        self.cache = stored
        self.failure = None

    def _reject(self) -> None:
        self.failure = CHECK_STOPPED
        self.cache = {}


def _row(row: object) -> tuple[str, bool, str] | None:
    """One photo row, or None when it is not a finished check result."""
    if not isinstance(row, dict):
        return None
    photo_id = row.get("id")
    ok = row.get("ok")
    message = row.get("message", "")
    if not isinstance(photo_id, str) or not photo_id:
        return None
    if not isinstance(ok, bool) or not isinstance(message, str):
        return None
    return photo_id, ok, message


def _token_cancelled(token: object) -> bool:
    return bool(getattr(token, "is_cancelled", False))
