"""Where and how the MOSS-Image text encoder runs.

``te_quant``: ``auto`` (default) picks by memory tier: bf16 on the 24 GB tier, bitsandbytes
LLM.int8 on the 16 GB tier. ``none`` (bf16), ``int8`` and ``nf4`` (bitsandbytes 4-bit) override it.
``te_device``: ``gpu`` (default) or ``cpu``. The encoder runs on the card while the DiT waits in RAM,
and is freed before the DiT moves to the card. On the 24 GB tier, CPU encode is a fallback:
``--te-device cpu``, no CUDA, or ``auto`` with bitsandbytes missing (a warning is logged).
On the 16 GB tier those paths raise instead of moving the encoder to the CPU. The DiT is not
quantized here. bitsandbytes is the optional ``te-quant`` extra.

:func:`compressed_te_device` is the other choice, for a model and text encoder that are already
compressed. It puts the encoder on the GPU when both still leave about 4 GB free, and on the CPU
otherwise. The bf16 prequant reference does not use it.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass
from typing import Any

from voxprint_forge.lora.errors import LoraStageError, LoraUsageError
from voxprint_forge.lora.vram_stage import TIER_GB

MODES = ("auto", "none", "int8", "nf4")
DEVICES = ("gpu", "cpu")
ENV = "VOXPRINT_FORGE_TE_QUANT"
ENV_DEVICE = "VOXPRINT_FORGE_TE_DEVICE"
TIER_DEFAULT = {24: "none", 16: "int8"}
# Free space that must remain when a compressed text encoder shares the card with a
# compressed model. About 3–4 GB; 4 GB is the top of that range.
COMPRESSED_TE_HEADROOM_GB = 4.0
_GIB = 1024**3
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TePlan:
    """The resolved choice: ``quant`` is none/int8/nf4, ``device`` is cuda or cpu."""

    quant: str
    device: str
    note: str = ""


def resolve(value: str | None = None) -> str:
    """The mode from ``value``, else ``VOXPRINT_FORGE_TE_QUANT``, else ``auto``."""
    mode = (value or os.environ.get(ENV) or "auto").strip().lower()
    if mode not in MODES:
        raise LoraUsageError("lora.err.te_quant_mode", mode=mode, modes=", ".join(MODES))
    return mode


def resolve_device(value: str | None = None) -> str:
    """``gpu`` or ``cpu`` from ``value``, else ``VOXPRINT_FORGE_TE_DEVICE``, else ``gpu``."""
    device = (value or os.environ.get(ENV_DEVICE) or "gpu").strip().lower()
    if device not in DEVICES:
        raise LoraUsageError("lora.err.te_quant_mode", mode=device, modes=", ".join(DEVICES))
    return device


def available() -> bool:
    """True when bitsandbytes can be imported."""
    return importlib.util.find_spec("bitsandbytes") is not None


def cuda_available() -> bool:
    """True when torch sees a CUDA card."""
    import torch

    return bool(torch.cuda.is_available())


def check_available(mode: str) -> None:
    """Raise a localized stage error when ``mode`` needs bitsandbytes and it is missing."""
    if mode in ("int8", "nf4") and not available():
        raise LoraStageError("lora.err.te_quant_missing", mode=mode)


def compressed_te_device(
    model_bytes: int,
    te_bytes: int,
    budget_gb: float,
    *,
    headroom_gb: float = COMPRESSED_TE_HEADROOM_GB,
) -> str:
    """``cuda`` when both compressed weights leave ``headroom_gb`` free, else ``cpu``.

    ``model_bytes`` and ``te_bytes`` are compressed sizes, not bf16. A non-positive
    size stays on the CPU. Full bf16 weights do not fit the 13.5 GB prequant cap
    with this headroom, so they resolve to ``cpu`` too. The bf16 reference render
    does not call this; it forces the text encoder onto the CPU itself.
    """
    if int(model_bytes) <= 0 or int(te_bytes) <= 0:
        return "cpu"
    need = int(model_bytes) + int(te_bytes) + int(float(headroom_gb) * _GIB)
    if need <= int(float(budget_gb) * _GIB):
        return "cuda"
    return "cpu"


def plan(quant: str | None, device: str | None, tier: int) -> TePlan:
    """Pick quantization and device for one run on memory tier ``tier`` (16 or 24)."""
    mode = resolve(quant)
    where = resolve_device(device)
    tight = int(tier) == TIER_GB
    if where == "cpu" or not cuda_available():
        reason = "cpu requested" if where == "cpu" else "no CUDA card"
        if tight:
            raise LoraStageError("lora.err.cpu_fallback", tier=TIER_GB, reason=reason)
        return TePlan("none", "cpu", reason)
    if mode == "auto":
        mode = TIER_DEFAULT.get(int(tier), "int8")
        if mode != "none" and not available():
            if tight:
                raise LoraStageError("lora.err.te_quant_missing", mode=mode)
            log.warning("bitsandbytes is missing: the text encoder falls back to bf16 on the CPU (slower).")
            return TePlan("none", "cpu", "bitsandbytes missing")
    check_available(mode)
    return TePlan(mode, "cuda")


def quant_config(mode: str) -> Any | None:
    """A transformers ``BitsAndBytesConfig`` for ``mode``, or ``None`` for ``none``."""
    if mode in ("none", "auto"):
        return None
    check_available(mode)
    import torch
    from transformers import BitsAndBytesConfig

    if mode == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=False,
    )
