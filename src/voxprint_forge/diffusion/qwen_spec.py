"""Part D constants. INT4 stays deferred unless an official build is pinned."""

from __future__ import annotations

QWEN_STEPS = 50
QWEN_TRUE_CFG = 4.0
QWEN_SIZE = 1328
THIS_RENTAL_MIN = 120
REUSE_MIN = 280
RECOMPUTE_MIN = 340
ENCODER_ESTIMATE_MIN = 15
INT4_MODE = "deferred"

QWEN_FP8_PREFIX = "qwen-image-2512-transformer-fp8"
QWEN_BF16_PREFIX = "qwen-image-2512-transformer-bf16"
QWEN_TRANSFORMER_CONFIG = "qwen-image-2512-transformer.config.json"
QWEN_ENCODER_FP8 = "qwen2.5-vl-7b-text-encoder-fp8.safetensors"
QWEN_METRICS = "checks/qwen-vs-bf16.json"
QWEN_GRID = "checks/grid.avif"
ENCODER_REUSE = "encoder-reuse.json"
INT4_MARKER_NAME = "DEFERRED.txt"

JOB_ID = "canvas-qwen-image-2512"
PLAN_NAME = "canvas-qwen-image-2512-v1.toml"
ZIMAGE_PLAN = "canvas-zimage-tiers-v1.toml"
STABLE_DIR = "qwen-image-2512"
