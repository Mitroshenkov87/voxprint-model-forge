"""Where and how the MOSS-Image text encoder runs.

``te_quant``: ``auto`` (default) picks by memory tier: bf16 on the 24 GB tier, bitsandbytes
LLM.int8 on the 16 GB tier. ``none`` (bf16), ``int8`` and ``nf4`` (bitsandbytes 4-bit) override it.
``te_device``: ``gpu`` (default) or ``cpu``. The encoder runs on the card while the DiT waits in RAM,
and is freed before the DiT moves to the card. On the 24 GB tier, CPU encode is a fallback:
``--te-device cpu``, no CUDA, or ``auto`` with bitsandbytes missing (a warning is logged).
On the 16 GB tier those paths raise instead of moving the encoder to the CPU. The DiT is not
quantized here. bitsandbytes is the optional ``te-quant`` extra.
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
