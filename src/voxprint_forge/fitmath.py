"""Weights + KV size against a card budget.

The check is weights + KV, not the whole CUDA context. Headroom (default 3 GB
on a 16 or 24 GB card) is what remains for compute buffers and the desktop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

GIB = 1024**3

# ggml block sizes: Q8_0 is 34 bytes / 32 values, Q4_0 is 18 / 32.
_BLOCK = {"q8_0": 34, "q4_0": 18}


@dataclass(frozen=True)
class Arch:
    full_layers: int
    full_kv_heads: int
    full_head_dim: int
    swa_layers: int = 0
    swa_kv_heads: int = 0
    swa_head_dim: int = 0
    swa_window: int = 0

    def kv_bytes(self, ctx: int, cache: str) -> int:
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
    return float(num_bytes) / GIB


def budget_bytes(budget_gb: float, headroom_gb: float) -> int:
    """Maximum weights + KV. Headroom stays free on the named card."""
    return int((budget_gb - headroom_gb) * GIB)


def fits_budget(
    weights_bytes: int,
    kv_bytes: int,
    *,
    budget_gb: float,
    headroom_gb: float,
) -> bool:
    return weights_bytes + kv_bytes <= budget_bytes(budget_gb, headroom_gb)


def min_vram_gb(weights_plus_kv_gib: float, headroom_gb: float) -> int:
    """Smallest suite card (16 or 24) that holds weights + KV plus headroom."""
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
    "F16": 100,
}


def quant_rank(name: str) -> int:
    return QUANT_RANK.get(name, 0)
