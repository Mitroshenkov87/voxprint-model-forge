"""Session RAM park for base, caption, and preview weights.

voxprint-common 0.4.1 does not ship ``model_park`` (it is on the unreleased
RAM-tier work, and 0.5.0 is expected to add ``stage``). This is the thin
stand-in: each key is read once per process, kept resident, copied to VRAM
with :func:`stage`, and dropped from VRAM with :func:`drop_vram`. A later
``voxprint_common.model_park.park`` replaces the local cache.

Upstream should ship that module with ``park(key, loader)``, ``stage`` (RAM
to VRAM), and ``drop`` (release the VRAM copy, keep the RAM copy), and log
the same park/unpark timings.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from collections.abc import Callable
from typing import TypeVar

log = logging.getLogger("voxprint.model_park")

GIB = 2**30
T = TypeVar("T")

_cache: dict[str, object] = {}
_events: list[dict[str, object]] = []
_rng: dict[str, tuple[object, list[object] | None]] = {}
_MISSING = object()


def reset() -> None:
    """Drop the session cache and the log. Tests call this between cases."""
    _cache.clear()
    _events.clear()
    _rng.clear()


def events() -> list[dict[str, object]]:
    """Park, hit, stage, and unpark rows recorded in this process."""
    return list(_events)


def park(key: str, loader: Callable[[], T], *, stage_name: str, replay_rng: bool = False) -> T:
    """Return the resident object for ``key``, calling ``loader`` only the first time.

    A hit does not call ``loader``, so a later stage does not read the files again.
    ``replay_rng`` restores the torch RNG to the state the first load left behind.
    Training reseeds and then loads; without this, skipping the loader on a resume
    in the same process would shift every random draw that follows (the cached
    latents included).
    """
    found = _cache.get(key)
    if found is not None:
        if replay_rng:
            _restore_rng(key)
        _record("hit", key, stage_name, 0.0)
        return found  # type: ignore[return-value]
    upstream = _upstream_park(key, loader, stage_name=stage_name)
    if upstream is not _MISSING:
        _cache[key] = upstream
        if replay_rng:
            _remember_rng(key)
        return upstream  # type: ignore[return-value]
    started = time.perf_counter()
    value = loader()
    _cache[key] = value
    if replay_rng:
        _remember_rng(key)
    _record("park", key, stage_name, time.perf_counter() - started)
    return value


def stage(module: object, device: object, *, key: str, stage_name: str) -> object:
    """Copy ``module`` from RAM to ``device`` and log the copy."""
    started = time.perf_counter()
    move = getattr(module, "to", None)
    if callable(move):
        move(device)
    _record("stage", key, stage_name, time.perf_counter() - started)
    return module


def drop_vram(module: object, *, key: str, stage_name: str) -> None:
    """Drop the VRAM copy. The object stays in the session cache."""
    started = time.perf_counter()
    move = getattr(module, "to", None)
    if callable(move):
        try:
            move("cpu")
        except (RuntimeError, TypeError, ValueError):
            pass
    gc.collect()
    _empty_cache()
    _record("unpark", key, stage_name, time.perf_counter() - started)


def _upstream_park(key: str, loader: Callable[[], T], *, stage_name: str) -> object:
    """Use ``voxprint_common.model_park.park`` when the pin provides it.

    Imported here because voxprint-common 0.4.1 does not ship ``model_park``.
    A top-level import would stop the app on the current pin.
    """
    try:
        import voxprint_common.model_park as upstream
    except ImportError:
        return _MISSING
    fn = getattr(upstream, "park", None)
    if not callable(fn):
        return _MISSING
    return fn(key, loader, stage=stage_name)


def _remember_rng(key: str) -> None:
    # Imported here so ``voxprint-forge --version`` does not pull torch via the park.
    import torch

    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    _rng[key] = (torch.random.get_rng_state(), cuda)


def _restore_rng(key: str) -> None:
    import torch

    saved = _rng.get(key)
    if saved is None:
        return
    cpu, cuda = saved
    torch.random.set_rng_state(cpu)
    if cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda)


def _record(action: str, key: str, stage_name: str, seconds: float) -> None:
    ram_gb = round(_rss_bytes() / GIB, 3)
    vram_gb = round(_vram_bytes() / GIB, 3)
    row = {
        "action": action,
        "key": key,
        "stage": stage_name,
        "seconds": round(seconds, 4),
        "ram_gb": ram_gb,
        "vram_gb": vram_gb,
    }
    _events.append(row)
    log.info(
        "%s key=%s stage=%s seconds=%.3f ram_gb=%.3f vram_gb=%.3f",
        action,
        key,
        stage_name,
        seconds,
        ram_gb,
        vram_gb,
    )


def _rss_bytes() -> int:
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            pages = int(handle.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return 0


def _vram_bytes() -> int:
    # Imported here so ``voxprint-forge --version`` does not pull torch via the park.
    import torch

    if not torch.cuda.is_available():
        return 0
    try:
        return int(torch.cuda.memory_allocated())
    except RuntimeError:
        return 0


def _empty_cache() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        return
