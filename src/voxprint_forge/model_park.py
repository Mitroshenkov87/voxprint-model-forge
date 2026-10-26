"""Session residency for base, caption, and preview weights.

voxprint-common 0.5.1 keeps those weights in :class:`voxprint_common.model_park.ModelPark`.
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
    """Copy ``module`` from RAM to ``device`` and log the copy.

    ``module.to`` moves registered parameters and skips the int8 ``state.CB``
    buffer. That buffer is moved too, so a copy parked on the CPU can return
    to the card.
    """
    started = time.perf_counter()
    _place(module, device)
    _record("stage", key, stage_name, time.perf_counter() - started)
    return module


def drop_vram(module: object, *, key: str, stage_name: str) -> None:
    """Move ``module`` off the card. The object stays in the process park.

    After the first forward the int8 weights live on ``state.CB``. ``module.to``
    does not visit that tensor, and a bitsandbytes root may refuse ``.to``
    outright. The tensors are moved in place, the duplicate parameter storage
    is dropped, then the collector runs and the caching allocator is emptied.
    The caller measures ``memory_allocated`` after this returns.
    """
    started = time.perf_counter()
    try:
        _place(module, "cpu")
    except (RuntimeError, TypeError):
        _release_tensors(module, "cpu")
        _move_quant_state(module, "cpu")
    gc.collect()
    _empty_cache()
    _record("unpark", key, stage_name, time.perf_counter() - started)


def _place(module: object, device: object) -> None:
    """Move ``module`` to ``device``, including the int8 state ``.to`` leaves behind."""
    target = _device_type(device)
    refused = False
    move = getattr(module, "to", None)
    if callable(move):
        try:
            move(device)
        except ValueError:
            # Some 8-bit and 4-bit roots refuse .to and would leave every weight where it is.
            refused = True
    if target == "cpu" or refused:
        _release_tensors(module, device)
    # A successful .to still skips MatmulLtState. Stage has to bring CB back.
    _move_quant_state(module, device)


def _release_tensors(root: object, device: object) -> None:
    """Move every tensor under ``root`` that is not already on ``device``.

    ``.to`` on a tensor returns a new object. Assigning ``.data`` retargets the
    existing one, so a second reference such as ``state.CB`` drops the card copy.
    """
    target = _device_type(device)
    if target is None:
        return
    seen: set[int] = set()
    modules = getattr(root, "modules", None)
    nodes = list(modules()) if callable(modules) else [root]
    for node in nodes:
        _walk(node, device, target, seen)


# Private names that still hold weights. Everything else private is a hook or a cycle.
_WEIGHT_FIELDS = {"_parameters", "_buffers", "_hf_hook"}


def _walk(obj: object, device: object, target: str, seen: set[int]) -> None:
    if obj is None or id(obj) in seen:
        return
    if _is_tensor(obj):
        seen.add(id(obj))
        _move_tensor(obj, device, target)
        attrs = getattr(obj, "__dict__", None)
        if isinstance(attrs, dict):
            for key, value in list(attrs.items()):
                if isinstance(key, str) and key.startswith("_"):
                    continue
                _walk(value, device, target, seen)
        return
    # nn.Module is callable. Skip bare functions, and keep walking objects that own attributes.
    if callable(obj) and not isinstance(getattr(obj, "__dict__", None), dict):
        return
    seen.add(id(obj))
    if isinstance(obj, dict):
        for value in obj.values():
            _walk(value, device, target, seen)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for value in obj:
            _walk(value, device, target, seen)
        return
    attrs = getattr(obj, "__dict__", None)
    if not isinstance(attrs, dict):
        return
    for key, value in list(attrs.items()):
        if isinstance(key, str) and key.startswith("_") and key not in _WEIGHT_FIELDS:
            continue
        _walk(value, device, target, seen)


def _is_tensor(obj: object) -> bool:
    if callable(getattr(obj, "modules", None)):
        return False
    if not isinstance(getattr(obj, "is_cuda", None), bool):
        return False
    return callable(getattr(obj, "to", None))


def _device_type(device: object) -> str | None:
    if isinstance(device, str):
        head = device.split(":", 1)[0]
        return head or None
    kind = getattr(device, "type", None)
    return kind if isinstance(kind, str) else None


def _tensor_device(tensor: object) -> str | None:
    kind = _device_type(getattr(tensor, "device", None))
    if kind is not None:
        return kind
    is_cuda = getattr(tensor, "is_cuda", None)
    if isinstance(is_cuda, bool):
        return "cuda" if is_cuda else "cpu"
    return None


# Tensors MatmulLtState keeps after Linear8bitLt.forward. module.to never sees them.
_QUANT_FIELDS = ("CB", "SCB", "CxB", "SB", "CBt", "SBt", "CxBt", "subB", "idx")


def _move_quant_state(root: object, device: object) -> None:
    """Move bitsandbytes int8 state onto ``device`` and keep a single weight copy."""
    target = _device_type(device)
    if target is None:
        return
    modules = getattr(root, "modules", None)
    nodes = list(modules()) if callable(modules) else [root]
    for node in nodes:
        state = getattr(node, "state", None)
        if state is not None:
            for name in _QUANT_FIELDS:
                tensor = getattr(state, name, None)
                if _is_tensor(tensor):
                    _move_tensor(tensor, device, target)
        weight = getattr(node, "weight", None)
        if weight is None:
            continue
        for name in ("CB", "SCB"):
            tensor = getattr(weight, name, None)
            if _is_tensor(tensor):
                _move_tensor(tensor, device, target)
        _retie_int8(node)


def _retie_int8(node: object) -> None:
    """Point an int8 weight at ``state.CB`` so the move does not keep two copies."""
    state = getattr(node, "state", None)
    weight = getattr(node, "weight", None)
    cb = getattr(state, "CB", None) if state is not None else None
    if weight is None or not _is_tensor(cb):
        return
    if getattr(state, "has_fp16_weights", False) or getattr(weight, "has_fp16_weights", False):
        return
    data = getattr(weight, "data", None)
    dtype = getattr(data, "dtype", None)
    floating = getattr(dtype, "is_floating_point", None)
    if dtype is None or (callable(floating) and floating()):
        return
    try:
        weight.data = cb
    except (RuntimeError, TypeError, ValueError, AssertionError, AttributeError):
        return
    scb = getattr(state, "SCB", None)
    if scb is not None and hasattr(weight, "SCB"):
        try:
            weight.SCB = scb
        except (RuntimeError, TypeError, ValueError, AssertionError, AttributeError):
            return


def _move_tensor(tensor: object, device: object, target: str) -> None:
    current = _tensor_device(tensor)
    if current is None or current == target:
        return
    move = getattr(tensor, "to", None)
    if not callable(move):
        return
    try:
        moved = move(device)
    except (RuntimeError, TypeError, ValueError, AssertionError, AttributeError):
        # One tensor that cannot move (no CUDA build, a meta parameter) must not
        # skip the rest. The int8 buffer is often a later attribute.
        return
    if moved is None or moved is tensor:
        return
    data = getattr(moved, "data", moved)
    try:
        tensor.data = data
    except (RuntimeError, TypeError, ValueError, AssertionError, AttributeError):
        return


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
