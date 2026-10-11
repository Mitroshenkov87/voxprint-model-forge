"""Pinned tools, models, and time estimates for the diffusion-int4 stage.

Licence checks below were read from the pinned trees on 2026-10-11.
deepcompressor ``LICENSE`` is Apache-2.0 (copyright 2024 Yujun Lin, Muyang Li,
Zhekai Zhang, Haotian Tang, Shang Yang, Song Han). ComfyUI-GGUF ``LICENSE``
and Wan2.2 ``LICENSE.txt`` are Apache-2.0. llama.cpp tag ``b3962`` is MIT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TargetName = Literal["qwen-image-edit-2511", "qwen-image", "wan-2.2-a14b"]

DEEPCOMPRESSOR_REPO = "https://github.com/mit-han-lab/deepcompressor"
DEEPCOMPRESSOR_MIRROR = "https://github.com/nunchux-ai/deepcompressor"
DEEPCOMPRESSOR_COMMIT = "69f3473f5e1c1504bae35cc50c7858ef900a9b17"
DEEPCOMPRESSOR_LICENCE = "Apache-2.0"

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

NUNCHAKU_FORK_REPO = "tonera/vitoom-nunchaku"
NUNCHAKU_FORK_COMMIT = "85bad5510d409570527b018387491fd0c81e4727"
NUNCHAKU_WHEEL = (
    "nunchaku-1.3.0.dev20260622+cu12.8torch2.11-cp311-cp311-linux_x86_64.whl"
)
NUNCHAKU_WHEEL_SHA256 = "bcdfe6fcc36cd37da34c1d0f1e0fa7c9b4201b2a70a89a3274d22420221afd2f"
NUNCHAKU_LICENCE = "Apache-2.0"

QWEN_EDIT_REPO = "Qwen/Qwen-Image-Edit-2511"
QWEN_EDIT_REVISION = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
QWEN_IMAGE_REPO = "Qwen/Qwen-Image"
QWEN_IMAGE_REVISION = "75e0b4be04f60ec59a75f475837eced720f823b6"
QWEN_LICENCE = "Apache-2.0"

QUANTFUNC_REPO = "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511"
QUANTFUNC_REVISION = "4ca543a1e91db4183ee16fd5aa5950e0b02803e5"
QUANTFUNC_LICENCE = "Apache-2.0"
QUANTFUNC_R128 = "nunchaku_qwen_image_edit_2511_balance_int4.safetensors"
QUANTFUNC_R128_SHA256 = "ec36d349309057f245e77255c008fc97330f2bc280dc8d704de715bcbaf6841b"
QUANTFUNC_R32 = "nunchaku_qwen_image_edit_2511_ultimate_speed_int4.safetensors"
QUANTFUNC_R32_SHA256 = "432d5b807f4345de4ec3d0b7ac1a6143930a81a3d35b582b1f603c18d07d56ec"
QUANTFUNC_R256 = "nunchaku_qwen_image_edit_2511_best_quality_int4.safetensors"
QUANTFUNC_R256_SHA256 = "d47054ff853d8d9cd6812386d4611cfe36fa9d1a5447549c8f0aa37c8aa79302"
PHASE0_RESULT = "/workspace/vm-results/model-forge/nunchaku-test-1/result.json"
HARD_CASES = ("c1", "c2", "c3")

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
FP8_EDIT_MIRROR = "Mitroshenkov87/voxprint-mirror-qwen-image-edit-2511"
EDIT_STEPS = 40
EDIT_TRUE_CFG = 4.0
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


QWEN_EDIT_STEPS: tuple[StepEstimate, ...] = (
    StepEstimate(
        "qwen-edit-2511-calib",
        30,
        "Build the deepcompressor prompt file from data/diffusion_calib.",
    ),
    StepEstimate(
        "qwen-edit-2511-quant",
        900,
        "fast_int4, SVD rank 256. 15 h on an A100 80GB (12-18 h). 500 min on an H100 80GB.",
    ),
    StepEstimate(
        "qwen-edit-2511-export",
        20,
        "Nunchaku svdquant safetensors export.",
    ),
    StepEstimate(
        "qwen-edit-2511-eval",
        90,
        "Identity, hands, text, and edit fidelity against QuantFunc r128, r256, and FP8.",
    ),
    StepEstimate(
        "qwen-edit-2511-publish",
        25,
        "Card, sha256, public Hugging Face, Drive fallback.",
    ),
    StepEstimate(
        "qwen-edit-2511-reclaim",
        5,
        "Delete sources after a mirror sha256 matches.",
    ),
)

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

QWEN_IMAGE_STEPS: tuple[StepEstimate, ...] = (
    StepEstimate(
        "qwen-image-quant",
        900,
        "Optional fast_int4, SVD rank 256. 900 min on A100, 500 min on H100.",
    ),
    StepEstimate(
        "qwen-image-export",
        20,
        "Optional Nunchaku export. zero_cond_t stays false.",
    ),
    StepEstimate(
        "qwen-image-eval",
        60,
        "Optional text-to-image smoke against FP8.",
    ),
    StepEstimate(
        "qwen-image-publish",
        25,
        "Optional card, sha256, and upload.",
    ),
    StepEstimate(
        "qwen-image-reclaim",
        5,
        "Optional source delete after the mirror matches.",
    ),
)


MIRROR_STEPS: tuple[StepEstimate, ...] = (
    StepEstimate(
        "qwen-edit-2511-mirror",
        25,
        "Publish QuantFunc r128 and r32. The own 12-18 h quant is skipped.",
    ),
    StepEstimate(
        "qwen-edit-2511-reclaim",
        5,
        "Delete the mirrored QuantFunc files after the sha256 matches.",
    ),
)


def planned_steps(*, qwen_image: bool, mirror: bool = False) -> tuple[StepEstimate, ...]:
    """Steps that run for this invocation, in order."""
    steps = MIRROR_STEPS if mirror else QWEN_EDIT_STEPS
    steps = steps + WAN_STEPS
    if qwen_image:
        return steps + QWEN_IMAGE_STEPS
    return steps


def expected_minutes(*, qwen_image: bool) -> int:
    """A100 planning total. H100 totals are ``gpu.total_minutes``."""
    return sum(step.expected_min for step in planned_steps(qwen_image=qwen_image))
