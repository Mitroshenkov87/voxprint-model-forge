"""A100 (sm_80) and H100 (sm_90) planning and FP8 baseline mode.

An H100 80GB is planned at 1.8x the A100 quant step. FP8 tensor cores
exist from sm_89 upward. On sm_80 the FP8 baseline is weight-only BF16.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

from voxprint_forge.diffusion.int4.pins import StepEstimate

H100_QUANT_SPEEDUP = 1.8
# Wan GGUF and FP8 are not SVDQuant, so no step uses the H100 quant speedup.
QUANT_STEPS: frozenset[str] = frozenset()
EMULATED_NOTE = (
    "FP8 is emulated and weight-only on this GPU, so speed comparisons vs FP8 are not representative."
)
NATIVE_NOTE = "FP8 baseline uses native FP8 tensor cores (sm_89 or newer)."
UNKNOWN_NOTE = "GPU compute capability was not detected, so speed comparisons vs FP8 are not representative."


@dataclass(frozen=True)
class GpuProfile:
    """One detected device, or the A100 planning default when nothing was probed."""

    gpu: str
    sm: str
    compute_cap: str
    mode: str
    speed_vs_fp8_representative: bool
    note: str

    def as_dict(self) -> dict[str, object]:
        """JSON object stored on the eval comparison."""
        return {
            "gpu": self.gpu,
            "sm": self.sm,
            "compute_cap": self.compute_cap,
            "mode": self.mode,
            "speed_vs_fp8_representative": self.speed_vs_fp8_representative,
            "note": self.note,
        }


def parse_compute_cap(text: str) -> tuple[int, int] | None:
    """Parse ``8.0`` or ``9.0`` from a capability string."""
    token = text.strip().split(",")[-1].strip()
    if not token or "." not in token:
        return None
    major_text, minor_text = token.split(".", 1)
    if not major_text.isdigit() or not minor_text.isdigit():
        return None
    return int(major_text), int(minor_text)


def sm_tag(major: int, minor: int) -> str:
    """``sm_80`` for compute capability 8.0."""
    return f"sm_{major}{minor}"


def fp8_mode(major: int, minor: int) -> str:
    """``native`` from sm_89 up, otherwise ``emulated``."""
    if (major, minor) >= (8, 9):
        return "native"
    return "emulated"


def planning_name(major: int, minor: int) -> str:
    """``a100`` for sm_80, ``h100`` for sm_90, otherwise ``other``."""
    if (major, minor) == (8, 0):
        return "a100"
    if (major, minor) == (9, 0):
        return "h100"
    return "other"


def profile_for_cap(major: int, minor: int) -> GpuProfile:
    """Build the eval record for a known compute capability."""
    mode = fp8_mode(major, minor)
    representative = mode == "native"
    note = NATIVE_NOTE if representative else EMULATED_NOTE
    return GpuProfile(
        gpu=planning_name(major, minor),
        sm=sm_tag(major, minor),
        compute_cap=f"{major}.{minor}",
        mode=mode,
        speed_vs_fp8_representative=representative,
        note=note,
    )


def _probe_smi() -> str:
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    if done.returncode != 0:
        return ""
    lines = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    return lines[0] if lines else ""


def resolve_cap(*, probe: bool) -> tuple[int, int] | None:
    """Capability from ``FORGE_COMPUTE_CAP``, then an optional nvidia-smi probe."""
    forced = os.environ.get("FORGE_COMPUTE_CAP", "").strip()
    parsed = parse_compute_cap(forced) if forced else None
    if parsed is not None:
        return parsed
    if not probe:
        return None
    return parse_compute_cap(_probe_smi())


def device_fp8_report(*, probe: bool) -> dict[str, object]:
    """Eval field describing whether the FP8 baseline is native or emulated."""
    cap = resolve_cap(probe=probe)
    if cap is None:
        return GpuProfile(
            gpu=planning_gpu(),
            sm="",
            compute_cap="",
            mode="unknown",
            speed_vs_fp8_representative=False,
            note=UNKNOWN_NOTE,
        ).as_dict()
    return profile_for_cap(*cap).as_dict()


def planning_gpu() -> str:
    """Which minute table to use. Unset detection stays on the A100 table."""
    forced = os.environ.get("FORGE_GPU", "").strip().lower()
    if forced in {"a100", "h100"}:
        return forced
    cap = resolve_cap(probe=False)
    if cap == (9, 0):
        return "h100"
    return "a100"


def step_minutes(step: StepEstimate, gpu: str) -> int:
    """Minutes for one step. Only quant steps shrink on an H100."""
    if gpu == "h100" and step.step_id in QUANT_STEPS:
        return int(round(step.expected_min / H100_QUANT_SPEEDUP))
    return step.expected_min


def total_minutes(steps: tuple[StepEstimate, ...], gpu: str) -> int:
    """Sum of :func:`step_minutes`."""
    return sum(step_minutes(step, gpu) for step in steps)
