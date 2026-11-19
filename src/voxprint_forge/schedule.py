"""Overlap CPU work with one GPU holder. --jobs caps how many models run at once."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager


class Gate:
    """One GPU stage at a time. CPU stages (quantize, fetch, upload) can run beside it."""

    def __init__(self, jobs: int) -> None:
        if jobs < 1:
            raise ValueError("jobs must be at least 1")
        self.jobs = jobs
        self._slots = threading.Semaphore(jobs)
        self._gpu = threading.Lock()

    @contextmanager
    def slot(self, *, gpu: bool) -> Iterator[None]:
        """Hold one job slot, and the GPU lock when ``gpu`` is set."""
        self._slots.acquire()
        got_gpu = False
        try:
            if gpu:
                self._gpu.acquire()
                got_gpu = True
            yield
        finally:
            if got_gpu:
                self._gpu.release()
            self._slots.release()
