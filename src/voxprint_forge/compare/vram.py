"""Peak VRAM: ``torch.cuda.max_memory_allocated`` plus a 1 Hz nvidia-smi sample."""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Callable


class SmiSampler:
    """Track the highest nvidia-smi memory reading while a block runs."""

    def __init__(self, reader: Callable[[], int | None] | None = None, *, interval: float = 1.0) -> None:
        self._reader = reader or read_smi_bytes
        self._interval = interval
        self.peak_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> SmiSampler:
        """Start the sampler thread."""
        self._thread = threading.Thread(target=self._loop, name="compare-smi", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        """Stop the sampler and include one last sample."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 1.0)
        self._note(self._reader())

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._note(self._reader())
            if self._stop.wait(self._interval):
                return

    def _note(self, sample: int | None) -> None:
        if sample is None:
            return
        if self.peak_bytes is None or sample > self.peak_bytes:
            self.peak_bytes = sample


def read_smi_bytes() -> int | None:
    """Current first-GPU memory.used in bytes, or ``None`` when nvidia-smi is absent."""
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0 or not done.stdout.strip():
        return None
    token = done.stdout.strip().splitlines()[0].split()[0]
    try:
        mib = float(token)
    except ValueError:
        return None
    return int(mib * 1024 * 1024)


def torch_peak_bytes() -> int | None:
    """``torch.cuda.max_memory_allocated`` for the current device, in bytes."""
    import torch

    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def reset_torch_peak() -> None:
    """Clear the CUDA allocation peak so the next sample is this image only."""
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
