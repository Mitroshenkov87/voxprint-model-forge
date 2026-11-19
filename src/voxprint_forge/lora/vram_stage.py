"""16 GB VRAM rules for one heavy model per stage.

A 16 GB tier loads heavy weights quantized, keeps a single heavy model on the
card, checks that the card is nearly clear between stages, and stops when a
stage's peak allocation passes 14 GB. A missing quantizer is an error. The
24 GB tier is unchanged.
"""

from __future__ import annotations

import logging

from voxprint_forge.lora.errors import LoraStageError

log = logging.getLogger("voxprint.vram")

TIER_GB = 16
PEAK_LIMIT_GB = 14.0
# Leftover memory_allocated above the pre-stage baseline and the caller's cached
# samples. This is not reserved memory. Do not raise it to hide a model that
# is still referenced.
NEAR_ZERO_GB = 0.25

_heavy: str | None = None


def reset() -> None:
    """Forget which heavy model is on the card. Tests call this between cases."""
    global _heavy
    _heavy = None


def heavy() -> str | None:
    """The heavy model currently allowed on the card, if this tier is tracking one."""
    return _heavy


def require_quantized(tier: int, quantized: bool, model: str) -> None:
    """On a 16 GB tier, refuse a heavy model that is still full precision."""
    if tier == TIER_GB and not quantized:
        raise LoraStageError("lora.err.vram_quant", model=model)


def hold_heavy(name: str, *, tier: int) -> None:
    """Mark ``name`` as the one heavy model on a 16 GB card.

    A second name raises ``lora.err.vram_one``. Other tiers do not track this.
    """
    global _heavy
    if tier != TIER_GB:
        return
    if _heavy is not None and _heavy != name:
        raise LoraStageError("lora.err.vram_one", stage=_heavy, model=name)
    _heavy = name


def release_heavy(name: str) -> None:
    """Clear the heavy-model mark when ``name`` is the one that held it."""
    global _heavy
    if _heavy == name:
        _heavy = None


def fail_peak(stage: str, alloc_gb: float, *, tier: int) -> None:
    """Log the stage peak and raise when a 16 GB tier passes 14 GB."""
    log.info("vram stage=%s peak_gb=%.2f limit_gb=%.1f tier=%s", stage, alloc_gb, PEAK_LIMIT_GB, tier)
    if tier == TIER_GB and alloc_gb > PEAK_LIMIT_GB:
        raise LoraStageError("lora.err.vram_peak", stage=stage, gb=round(alloc_gb, 2), limit=int(PEAK_LIMIT_GB))


def assert_stage_clear(stage: str, *, tier: int, samples_gb: float = 0.0, baseline_gb: float = 0.0) -> None:
    """After a heavy model leaves, allocated bytes above the baseline and known caches must be small.

    ``baseline_gb`` is ``memory_allocated`` before the model was placed on the card.
    No CUDA card means there is nothing to check. Other tiers keep the wider residual check.
    The comparison is allocated bytes, never reserved memory.
    """
    if tier != TIER_GB:
        return
    allocated = _allocated_gb()
    if allocated is None:
        return
    leaked = allocated - samples_gb - baseline_gb
    log.info(
        "vram stage=%s clear allocated_gb=%.3f baseline_gb=%.3f samples_gb=%.3f",
        stage,
        allocated,
        baseline_gb,
        samples_gb,
    )
    if leaked > NEAR_ZERO_GB:
        raise LoraStageError("lora.err.vram_held", stage=stage, gb=round(allocated - baseline_gb, 2))


def allocated_gb() -> float:
    """Current ``torch.cuda.memory_allocated`` in GiB, or 0 without a CUDA card."""
    value = _allocated_gb()
    return 0.0 if value is None else value


def _allocated_gb() -> float | None:
    # Imported here so ``voxprint-forge --version`` does not pull torch through te_quant.
    import torch

    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.memory_allocated()) / (2**30)
