"""FP8 publish gates. A failed component stays BF16."""

from __future__ import annotations

from typing import Literal

Decision = Literal["bf16", "fp8"]

LPIPS_MAX = 0.10
CLIP_DROP_MAX = 0.01
LPIPS_EDIT_MAX = 0.12
CLIP_I_MIN = 0.95


def clip_drop(reference: float, candidate: float) -> float:
    """Relative drop `(reference - candidate) / abs(reference)`."""
    denom = abs(reference)
    if denom == 0:
        return 0.0 if candidate == 0 else 1.0
    return (reference - candidate) / denom


def decide_base(lpips_mean: float, clip_drop_value: float) -> Decision:
    """Part A: mean LPIPS <= 0.10 and CLIP score drop <= 1%."""
    if lpips_mean > LPIPS_MAX or clip_drop_value > CLIP_DROP_MAX:
        return "bf16"
    return "fp8"


def decide_control(lpips_mean: float) -> Decision:
    """Part B: mean LPIPS <= 0.10."""
    if lpips_mean > LPIPS_MAX:
        return "bf16"
    return "fp8"


def decide_edit(lpips_mean: float, clip_i: float) -> Decision:
    """Part C: mean LPIPS <= 0.12 and CLIP-I >= 0.95."""
    if lpips_mean > LPIPS_EDIT_MAX or clip_i < CLIP_I_MIN:
        return "bf16"
    return "fp8"
