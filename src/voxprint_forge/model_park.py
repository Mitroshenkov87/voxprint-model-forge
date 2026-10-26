"""Session residency for base, caption, and preview weights.

voxprint-common 0.5.0 keeps those weights in :class:`voxprint_common.model_park.ModelPark`.
:meth:`ModelPark.get` reads a key from disk once and raises ``entry.refs``.
:class:`~voxprint_common.model_park.Parked` has no destructor, so dropping that
handle does not drop the ref. ``ram32`` evicts only entries with ``refs <= 0``.
``size_bytes`` is the pin budget. These values are live modules with no
``pin_memory``, and ``get`` does not refuse a load that does not fit.

:func:`stage` and :func:`drop_vram` move that live module. :meth:`ModelPark.stage`
copies state-dict tensors and is what :func:`voxprint_forge.mdl_adapter.stage`
calls. A pipeline is not a state dict: LoRA injection mutates the module
``get`` stored, and ``tensor.to`` would return a different object.

``replay_rng`` stays here. Upstream ``get`` does not restore the torch RNG
that a resumed ``train`` reseeded past.
"""

from __future__ import annotations

import gc
import logging
import os
import time
from collections.abc import Callable
from typing import TypeVar

from voxprint_common.model_park import for_process, reset_process_park

log = logging.getLogger("voxprint.model_park")

GIB = 2**30
T = TypeVar("T")

_events: list[dict[str, object]] = []
_rng: dict[str, tuple[object, list[object] | None]] = {}


class _Hold:
    """Box one live object inside a state-dict park.

    ``ModelPark`` copies each stored value with ``to``. A module's ``to``
    moves parameters in place, so this box returns itself and the caller
    moves the module with :func:`stage`.
    """

    def __init__(self, value: object) -> None:
        self.value = value

    def to(self, device: object, non_blocking: bool = False) -> _Hold:
        del device, non_blocking
        return self


def reset() -> None:
    """Drop the process park and the log. Tests call this between cases."""
    _events.clear()
    _rng.clear()
    reset_process_park()


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
    upstream = for_process(app="model-forge")
    before = upstream.stats().disk_reads.get(key, 0)
    started = time.perf_counter()

    def load() -> dict[str, _Hold]:
        return {"value": _Hold(loader())}

    parked = upstream.get(key, load, size_bytes=1, role="main")
    # Do not call release(). The ref lives on the entry, not on this handle.
    held = parked.to("cpu")["value"]
    if not isinstance(held, _Hold):
        raise RuntimeError(f"model park returned an unexpected value for {key}")
    value: T = held.value  # type: ignore[assignment]
    if upstream.stats().disk_reads.get(key, 0) == before:
        if replay_rng:
            _restore_rng(key)
        _record("hit", key, stage_name, 0.0)
    else:
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
    """Drop the VRAM copy. The object stays in the process park."""
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


def _remember_rng(key: str) -> None:
    # Imported here so ``voxprint-forge --version`` does not pull torch via the park.
    import torch

    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    _rng[key] = (torch.random.get_rng_state(), cuda)


def _restore_rng(key: str) -> None:
    # Imported here so ``voxprint-forge --version`` does not pull torch via the park.
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
    # Imported here so ``voxprint-forge --version`` does not pull torch via the park.
    import torch

    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        return
