"""Output file names from the canvas decision, sections 6 and 8."""

from __future__ import annotations

SHARD_BYTES = 5_000_000_000

TURBO_BF16_PREFIX = "zimage-turbo-transformer-bf16"
TURBO_FP8_NAME = "zimage-turbo-transformer-fp8.safetensors"
ZIMAGE_BF16_DIR = "zimage-transformer-bf16"
ZIMAGE_FP8_NAME = "zimage-transformer-fp8.safetensors"
QWEN3_FP8_NAME = "qwen3-4b-text-encoder-fp8.safetensors"
BASE_METRICS = "checks/fp8-vs-bf16.json"
BASE_GRID = "checks/grid.avif"

CONTROL_FULL_UPSTREAM = "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2602-8steps.safetensors"
CONTROL_LITE_UPSTREAM = "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2602-8steps.safetensors"
CONTROL_LITE_FP8 = "zimage-turbo-controlnet-union-2.1-lite-fp8.safetensors"
CONTROL_LITE_BF16 = "zimage-turbo-controlnet-union-2.1-lite-bf16.safetensors"
CONTROL_FULL_BF16 = "zimage-turbo-controlnet-union-2.1-bf16.safetensors"
I2I_METRICS = "checks/i2i-fp8-vs-bf16.json"
I2I_GRID = "checks/grid.avif"
INPAINT_GRID = "checks/grid-inpaint.avif"

NOTICE_TEXT = "Apache-2.0\n\nModified by Voxprint: dtype conversion\n"


def shard_name(prefix: str, index: int, total: int) -> str:
    """1-based shard name with five digits: prefix-00001-of-00003.safetensors."""
    return f"{prefix}-{index:05d}-of-{total:05d}.safetensors"
