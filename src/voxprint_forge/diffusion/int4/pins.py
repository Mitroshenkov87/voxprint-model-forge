"""Pinned tools, models, and time estimates for the diffusion-int4 stage.

Licence checks below were read from the pinned trees on 2026-10-11.
ComfyUI-GGUF ``LICENSE`` and Wan2.2 ``LICENSE.txt`` are Apache-2.0.
llama.cpp tag ``b3962`` is MIT.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

TargetName = Literal["wan-2.2-a14b"]

COMFY_GGUF_REPO = "https://github.com/city96/ComfyUI-GGUF"
COMFY_GGUF_COMMIT = "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
COMFY_GGUF_LICENCE = "Apache-2.0"

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
LLAMA_CPP_TAG = "b3962"
LLAMA_CPP_COMMIT = "c8c07d658a6cefc5a50cfdf6be7d726503612303"
LLAMA_CPP_LICENCE = "MIT"

WAN_CODE_REPO = "https://github.com/Wan-Video/Wan2.2"
WAN_CODE_COMMIT = "1ea34ff48f87168174e12956e200b1d908b1c5ff"
WAN_CODE_LICENCE = "Apache-2.0"

WAN_WEIGHT_REPO = "Wan-AI/Wan2.2-T2V-A14B"
WAN_WEIGHT_REVISION = "c8c270b13ee05bfa474194ac9fb07a5868a97cea"
WAN_LICENCE = "Apache-2.0"
WAN_EXPERTS = ("high_noise", "low_noise")
WAN_QUANTS = ("Q4_K_M", "Q5_K_M")

DRIVE_FOLDER_ID = "13qzgTnnkqvZrf_DK9fmymJ8ICdOuwDJT"
DISK_FLOOR = 0.15
DEFAULT_SVD_RANK = 256
SVD_GROUP = 64
SCHEMA = "voxprint-forge-diffusion-int4/1"

# 15.0 GiB leaves headroom on a 16 GB card. 23.0 GiB leaves about 1 GiB on 24 GB.
FITS_16_MIB = 15.0 * 1024
FITS_24_MIB = 23.0 * 1024
IDENTITY_DELTA = 0.05

HF_USER_DEFAULT = "Mitroshenkov87"
WAN_SMOKE_SIZE = "832*480"
WAN_SMOKE_FRAMES = 5
WAN_SMOKE_STEPS = 4


@dataclass(frozen=True)
class StepEstimate:
    """One resumable step and its planning time on an A100 80GB.

    An H100 80GB uses the same table except the quant steps, which are 1.8x faster.
    """

    step_id: str
    expected_min: int
    note: str


WAN_STEPS: tuple[StepEstimate, ...] = (
    StepEstimate(
        "wan-2.2-gguf",
        100,
        "Q4_K_M and Q5_K_M for the high-noise and low-noise experts.",
    ),
    StepEstimate(
        "wan-2.2-fp8",
        40,
        "Per-tensor FP8 for both experts.",
    ),
    StepEstimate(
        "wan-2.2-smoke",
        20,
        "5 frames at 832*480, 4 steps, with offload.",
    ),
    StepEstimate(
        "wan-2.2-publish",
        25,
        "Card, sha256, public Hugging Face, Drive fallback.",
    ),
    StepEstimate(
        "wan-2.2-reclaim",
        5,
        "Delete Wan sources after a mirror sha256 matches.",
    ),
)

def skip_wan() -> bool:
    """True when ``FORGE_SKIP_WAN=1`` omits the Wan 2.2 steps."""
    return os.environ.get("FORGE_SKIP_WAN", "").strip() == "1"


def planned_steps() -> tuple[StepEstimate, ...]:
    """Steps that run for this invocation, in order."""
    if skip_wan():
        return ()
    return WAN_STEPS


def expected_minutes() -> int:
    """A100 planning total. Wan minutes are the same on an H100."""
    return sum(step.expected_min for step in planned_steps())
