"""Weights + KV size against a card budget.

The check is weights + KV, not the whole CUDA context. Headroom (default 3 GB
on a 16 or 24 GB card) is what remains for compute buffers and the desktop.
A quant that fails this on-GPU check can still be kept when an offload split
fits in VRAM and RAM. See ``voxprint_forge.placement``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

GIB = 1024**3

# ggml block sizes: Q8_0 is 34 bytes / 32 values, Q4_0 is 18 / 32.
_BLOCK = {"q8_0": 34, "q4_0": 18}


@dataclass(frozen=True)
class Arch:
    """Layer counts and head sizes used to estimate KV bytes."""
    full_layers: int
    full_kv_heads: int
    full_head_dim: int
    swa_layers: int = 0
    swa_kv_heads: int = 0
    swa_head_dim: int = 0
    swa_window: int = 0

    def kv_bytes(self, ctx: int, cache: str) -> int:
        """KV cache bytes for this context and cache type.

        Full-attention layers use the whole context. Sliding-window layers use ``min(ctx, swa_window)``. ``q8_0`` is 34
        bytes per 32 values and ``q4_0`` is 18. ``f16`` is 2 bytes per value. The fit gate compares weights plus this
        estimate to the card budget minus headroom.
        """
        if cache not in _BLOCK and cache != "f16":
            raise ValueError(f"unsupported KV cache type {cache}")
        full_values = self.full_layers * self.full_kv_heads * self.full_head_dim * 2 * ctx
        swa_tokens = min(ctx, self.swa_window) if self.swa_window else 0
        swa_values = self.swa_layers * self.swa_kv_heads * self.swa_head_dim * 2 * swa_tokens
        return _value_bytes(full_values, cache) + _value_bytes(swa_values, cache)


def _value_bytes(values: int, cache: str) -> int:
    if values <= 0:
        return 0
    if cache == "f16":
        return values * 2
    block = _BLOCK[cache]
    blocks = (values + 31) // 32
    return blocks * block


def gib(num_bytes: int | float) -> float:
    """Convert a byte count to gibibytes."""
    return float(num_bytes) / GIB


def budget_bytes(budget_gb: float, headroom_gb: float) -> int:
    """Maximum weights plus KV on the card.

    Headroom stays free. On a 16 or 24 GB tier the plan uses 3 GB, so weights
    and KV together must fit in 13 or 21 GiB.
    """
    return int((budget_gb - headroom_gb) * GIB)


def fits_budget(
    weights_bytes: int,
    kv_bytes: int,
    *,
    budget_gb: float,
    headroom_gb: float,
) -> bool:
    """True when weights plus KV fit on the card after headroom.

    The budget is ``(budget_gb - headroom_gb)`` GiB. Headroom (3 GB on the 16 and 24 GB tiers) stays free for compute
    buffers and the desktop. This is the on-GPU fit gate. A quant that fails it can still be kept when an offload split
    fits.
    """
    return weights_bytes + kv_bytes <= budget_bytes(budget_gb, headroom_gb)


def min_vram_gb(weights_plus_kv_gib: float, headroom_gb: float) -> int:
    """Smallest suite card (16 or 24) that holds weights plus KV and headroom.

    Anything above 24 GB is rounded up to the next whole gigabyte. The value
    is what the registry records as ``min_vram_gb`` for the tier.
    """
    need = weights_plus_kv_gib + headroom_gb
    if need <= 16:
        return 16
    if need <= 24:
        return 24
    return int(math.ceil(need))


# Higher is better when several quants fit the same tier.
QUANT_RANK = {
    "Q8_0": 80,
    "Q6_K": 70,
    "Q5_K_M": 60,
    "Q5_K": 60,
    "Q5_K_S": 55,
    "Q4_K_M": 50,
    "Q4_K": 50,
    "Q4_K_S": 45,
    "IQ4_XS": 40,
    "Q3_K_L": 35,
    "Q3_K_M": 34,
    "Q3_K": 34,
    "IQ3_M": 30,
    "IQ3_S": 28,
    "IQ3_XXS": 26,
    "IQ2_M": 20,
    "F16": 100,
}


# Bits per weight used by preflight. IQ3_M at 3.716 bpw is 14.4 GB for a 31B model.
QUANT_BPW = {
    "F16": 16.0,
    "BF16": 16.0,
    "Q8_0": 8.5,
    "Q6_K": 6.56,
    "Q5_K_M": 5.69,
    "Q5_K": 5.69,
    "Q5_K_S": 5.5,
    "Q4_K_M": 4.85,
    "Q4_K": 4.85,
    "Q4_K_S": 4.5,
    "IQ4_XS": 4.25,
    "Q3_K_L": 4.0,
    "Q3_K_M": 3.9,
    "Q3_K": 3.9,
    "IQ3_M": 3.716,
    "IQ3_S": 3.5,
    "IQ3_XXS": 3.0,
    "IQ2_M": 2.7,
}


def quant_rank(name: str) -> int:
    """Higher means a better quant when several fit the same tier."""
    return QUANT_RANK.get(name, 0)


def quant_bpw(name: str) -> float:
    """Bits per weight used by the preflight size estimate."""
    return QUANT_BPW.get(name, 16.0)


def estimate_weights_bytes(params_b: float, quant: str) -> int:
    """Parameter count times bits per weight.

    ``params_b`` is billions of parameters. Preflight uses this before a file
    exists. The fit stage later replaces it with the real GGUF size.
    """
    return int(params_b * 1_000_000_000 * quant_bpw(quant) / 8)
