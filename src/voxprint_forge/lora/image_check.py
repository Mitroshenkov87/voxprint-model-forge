"""Dataset quality check for the Photos step.

The decision is :func:`voxprint_forge.lora.photos.filter_reason` (size, collage,
blur, near-duplicate). It runs on the worker that the gate starts, and the
weights — none, for this check — are acquired only on the first ``classify``
of a run. A caption model is not loaded: it does not fit in the 4 GiB budget
on a 16 GiB card that is not fully free.

Device staging, if a card is present, never holds more than
:data:`CHECK_VRAM_BUDGET` bytes of weights plus images.
"""

from __future__ import annotations

import gc
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

try:
    import torch
except ImportError:  # the check still runs on CPU when torch is not installed
    torch = None

from voxprint_forge.lora.photos import filter_reason, open_photo

CHECK_VRAM_BUDGET = 4 * 1024**3


class CancelPoll(Protocol):
    """The bit of a cancel token the checker reads."""

    @property
    def is_cancelled(self) -> bool:
        """True once the gate has asked the run to stop."""


def batch_count(weight_bytes: int, image_bytes: int, budget: int = CHECK_VRAM_BUDGET) -> int:
    """How many images of ``image_bytes`` fit beside ``weight_bytes`` inside ``budget``.

    ``0`` means that image must not be staged on the device. The planned bytes
    (weights plus the returned count times one image) are at most ``budget``.
    """
    if weight_bytes < 0 or image_bytes < 0 or budget < 0:
        raise ValueError("byte counts must be zero or positive")
    if image_bytes == 0 or weight_bytes >= budget:
        return 0
    room = budget - weight_bytes
    if room < image_bytes:
        return 0
    return room // image_bytes


class ImageChecker:
    """Lazy quality classifier. ``loaded`` stays false until :meth:`classify`."""

    def __init__(self) -> None:
        self.loaded = False
        self.peak_bytes = 0
        self.weight_bytes = 0
        self.last_batch = 0
        self._seen: list[int] = []
        self._device_hold: list[object] = []

    def ensure_loaded(self) -> None:
        """Mark the checker resident and clear the duplicate memory of the previous run.

        Called from :meth:`classify` only, so adding or removing a photo does not load it.
        """
        if self.loaded:
            return
        self._seen = []
        self.peak_bytes = 0
        self.weight_bytes = 0
        self._reset_peak()
        self.loaded = True
        self._note_peak()

    def classify(self, path: Path, token: CancelPoll) -> tuple[bool, str]:
        """``(ok, reason key)`` for one photo. A cancelled token returns before any new work."""
        if token.is_cancelled:
            return True, ""
        self.ensure_loaded()
        if token.is_cancelled:
            return True, ""
        image = open_photo(path)
        if token.is_cancelled:
            return True, ""
        staged = 0 if image is None else image.width * image.height * 3 * 4
        self.last_batch = batch_count(self.weight_bytes, staged)
        self._stage_one(staged)
        try:
            reason = filter_reason(image, self._seen)
        finally:
            self._release_staged()
            self._note_peak()
        if token.is_cancelled:
            return True, ""
        if reason:
            return False, reason
        return True, ""

    def unload(self) -> None:
        """Drop staged device memory and the duplicate hashes. Safe to call twice."""
        self._release_staged()
        self._seen = []
        self.loaded = False
        self.weight_bytes = 0
        gc.collect()
        _empty_cache()
        self._note_peak()

    def _stage_one(self, nbytes: int) -> None:
        """Hold one image tensor when it fits in the budget. Otherwise stay on the CPU."""
        self._release_staged()
        allowed = batch_count(self.weight_bytes, nbytes)
        self.last_batch = allowed
        if allowed < 1 or torch is None or not torch.cuda.is_available():
            return
        if self.weight_bytes + nbytes > CHECK_VRAM_BUDGET:
            return
        floats = nbytes // 4
        if floats <= 0:
            return
        self._device_hold.append(torch.empty(floats, dtype=torch.float32, device="cuda"))
        self._note_peak()

    def _release_staged(self) -> None:
        self._device_hold.clear()

    def _reset_peak(self) -> None:
        if torch is None or not torch.cuda.is_available():
            return
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        except RuntimeError:
            return

    def _note_peak(self) -> None:
        if torch is None or not torch.cuda.is_available():
            return
        try:
            self.peak_bytes = max(self.peak_bytes, int(torch.cuda.max_memory_allocated()))
        except RuntimeError:
            return


def measure_check(paths: Sequence[Path]) -> tuple[float, int]:
    """Wall seconds and peak allocated device bytes for one check of ``paths``.

    Peak is ``0`` when CUDA is not available. The checker is loaded on the first
    photo and unloaded before this returns.
    """
    checker = ImageChecker()
    token = _Never()
    started = time.perf_counter()
    try:
        for path in paths:
            checker.classify(path, token)
    finally:
        checker.unload()
    return time.perf_counter() - started, checker.peak_bytes


def allocated_bytes() -> int:
    """Bytes ``torch.cuda`` still reports allocated, or ``0`` without a card."""
    if torch is None or not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.memory_allocated())
    except RuntimeError:
        return 0


class _Never:
    """A token that never cancels, for :func:`measure_check`."""

    @property
    def is_cancelled(self) -> bool:
        return False


def _empty_cache() -> None:
    if torch is None or not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        return
