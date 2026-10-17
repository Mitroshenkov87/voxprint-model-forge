"""Check settings from the canvas decision. The plan file is what a run obeys."""

from __future__ import annotations

TURBO_STEPS = 8
TURBO_GUIDANCE = 0.0
CREATIVE_STEPS = 50
CREATIVE_GUIDANCE = 4.0
CONTROL_STEPS = 8
CONTROL_GUIDANCE = 0.0
CONTROL_SCALE = 0.75
IMAGE_SIZE = 1024

PART_MINUTES = {"base": 55, "controlnet": 25}
JOB_ID = "canvas-zimage-tiers-v1"
