"""Publish gates for Qwen-Image-2512. The job does not call the INT4 gate while deferred."""

from __future__ import annotations

from typing import Literal

from voxprint_forge.diffusion.policy import Decision, decide_base

INT4_LPIPS_MAX = 0.15
INT4_CLIP_DROP_MAX = 0.02
Int4Decision = Literal["bf16", "int4"]


def decide_qwen_fp8(lpips_mean: float, clip_drop_value: float) -> Decision:
    """Same gate as part A: mean LPIPS <= 0.10 and CLIP score drop <= 1%."""
    return decide_base(lpips_mean, clip_drop_value)


def decide_qwen_int4(lpips_mean: float, clip_drop_value: float) -> Int4Decision:
    """INT4 gate: mean LPIPS <= 0.15 and CLIP score drop <= 2%. Unused while deferred."""
    if lpips_mean > INT4_LPIPS_MAX or clip_drop_value > INT4_CLIP_DROP_MAX:
        return "bf16"
    return "int4"
