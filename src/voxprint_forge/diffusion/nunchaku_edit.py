"""Nunchaku INT4 matrix for Qwen-Image-Edit-2511 and Z-Image-Turbo.

One uv venv installs the pinned ``tonera/vitoom-nunchaku`` wheel (it implements
``zero_cond_t`` and ships both transformer modules). Upstream ``nunchaku`` is
never installed. The 4090 entry points are ``scripts/nunchaku_edit_test.sh`` and
``scripts/nunchaku_edit_test.ps1``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import transformers
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

from voxprint_forge.diffusion.fp8 import dequant_file, read_format
from voxprint_forge.hashing import sha256_file
from voxprint_forge.scrub import active_secrets, scrub

try:
    import diffusers
except ImportError:
    diffusers = None

try:
    import nunchaku
except ImportError:
    nunchaku = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
except ImportError:
    NunchakuQwenImageTransformer2DModel = None

try:
    from nunchaku.models.transformers.transformer_zimage import NunchakuZImageTransformer2DModel
except ImportError:
    NunchakuZImageTransformer2DModel = None

try:
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline
except ImportError:
    QwenImageEditPlusPipeline = None

try:
    from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
except ImportError:
    ZImagePipeline = None

try:
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
except ImportError:
    QwenImageTransformer2DModel = None

try:
    from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
except ImportError:
    ZImageTransformer2DModel = None

if NunchakuQwenImageTransformer2DModel is None and nunchaku is not None:
    NunchakuQwenImageTransformer2DModel = getattr(nunchaku, "NunchakuQwenImageTransformer2DModel", None)
if NunchakuZImageTransformer2DModel is None and nunchaku is not None:
    NunchakuZImageTransformer2DModel = getattr(nunchaku, "NunchakuZImageTransformer2DModel", None)
if QwenImageEditPlusPipeline is None and diffusers is not None:
    QwenImageEditPlusPipeline = getattr(diffusers, "QwenImageEditPlusPipeline", None)
if ZImagePipeline is None and diffusers is not None:
    ZImagePipeline = getattr(diffusers, "ZImagePipeline", None)
if QwenImageTransformer2DModel is None and diffusers is not None:
    QwenImageTransformer2DModel = getattr(diffusers, "QwenImageTransformer2DModel", None)
if ZImageTransformer2DModel is None and diffusers is not None:
    ZImageTransformer2DModel = getattr(diffusers, "ZImageTransformer2DModel", None)

SCHEMA = "voxprint-forge-nunchaku-edit/1"
TAG = "nunchaku-test-1"
WHEEL_REPO = "tonera/vitoom-nunchaku"
WHEEL_REVISION = "85bad5510d409570527b018387491fd0c81e4727"
QWEN_WEIGHT_REPO = "QuantFunc/Nunchaku-Qwen-Image-EDIT-2511"
QWEN_WEIGHT_REVISION = "4ca543a1e91db4183ee16fd5aa5950e0b02803e5"
ZIMAGE_WEIGHT_REPO = "nunchaku-ai/nunchaku-z-image-turbo"
ZIMAGE_WEIGHT_REVISION = "ca6bac69c3b0b2bdd31ca5196bf87c5f2a9eaedf"
QWEN_BASE_REPO = "Qwen/Qwen-Image-Edit-2511"
QWEN_BASE_REVISION = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
ZIMAGE_BASE_REPO = "Tongyi-MAI/Z-Image-Turbo"
ZIMAGE_BASE_REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
QWEN_FP8_REPO = "Mitroshenkov87/voxprint-mirror-qwen-image-edit-2511"
ZIMAGE_FP8_REPO = "Mitroshenkov87/voxprint-mirror-z-image-turbo"
FITS_16GB_MIB = 15.0 * 1024
MIN_VRAM_BYTES = int(16 * (1024**3) * 0.97)
DISK_MARGIN = 5 * 1024**3
DEFAULT_BUDGET_MIN = 90.0
SEED = 0
QWEN_STEPS = 40
QWEN_SMOKE_STEPS = 20
ZIMAGE_STEPS = 8
TRUE_CFG = 4.0
TURBO_GUIDANCE = 0.0
NEGATIVE_PROMPT = " "
ARM_ORDER = ("qwen-r128", "qwen-r32", "zimage-r128", "zimage-r256")
ESTIMATE_MIN = {
    "qwen-r128": 22.0,
    "qwen-r32": 18.0,
    "zimage-r128": 8.0,
    "zimage-r256": 14.0,
    "qwen-fp8": 16.0,
    "zimage-fp8": 6.0,
}

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_SETUP = 2
EXIT_GPU = 3


class NunchakuError(Exception):
    """Failure with a script exit code."""

    exit_code = EXIT_FAIL


class SetupError(NunchakuError):
    """Wheel, hash, disk, or download failure. Exit 2."""

    exit_code = EXIT_SETUP


class GpuError(NunchakuError):
    """No usable CUDA GPU. Exit 3."""

    exit_code = EXIT_GPU


@dataclass(frozen=True)
class WheelSpec:
    """Pinned vitoom-nunchaku wheel for one platform."""

    filename: str
    sha256: str
    python: str
    index_url: str


@dataclass(frozen=True)
class ArmSpec:
    """One INT4 weight file."""

    arm_id: str
    family: str
    rank: int
    repo: str
    revision: str
    filename: str
    sha256: str
    size_bytes: int
    baseline: str


@dataclass(frozen=True)
class CaseSpec:
    """One timed generation."""

    case_id: str
    prompt: str
    image: str | None
    width: int
    height: int


@dataclass(frozen=True)
class GpuFacts:
    """nvidia-smi snapshot used for the exit-3 gate."""

    name: str
    memory_mib: int
    driver: str
    compute_major: int
    compute_minor: int

    @property
    def compute_capability(self) -> str:
        """``major.minor`` string, for the result file."""
        return f"{self.compute_major}.{self.compute_minor}"


WINDOWS_WHEEL = WheelSpec(
    filename="nunchaku-1.3.0.dev20260629+cu13.0torch2.11-cp312-cp312-win_amd64.whl",
    sha256="ddc015bfa6fc6b84957532823c23418f5b986cb61e3c02c97a646cfe2a1fd955",
    python="3.12",
    index_url="https://download.pytorch.org/whl/cu130",
)
# The pinned revision has no linux x86_64 cp312 wheel. This is the plan's linux fallback.
LINUX_WHEEL = WheelSpec(
    filename="nunchaku-1.3.0.dev20260622+cu12.8torch2.11-cp311-cp311-linux_x86_64.whl",
    sha256="bcdfe6fcc36cd37da34c1d0f1e0fa7c9b4201b2a70a89a3274d22420221afd2f",
    python="3.11",
    index_url="https://download.pytorch.org/whl/cu128",
)

ARMS: dict[str, ArmSpec] = {
    "qwen-r128": ArmSpec(
        "qwen-r128",
        "qwen-edit-2511",
        128,
        QWEN_WEIGHT_REPO,
        QWEN_WEIGHT_REVISION,
        "nunchaku_qwen_image_edit_2511_balance_int4.safetensors",
        "ec36d349309057f245e77255c008fc97330f2bc280dc8d704de715bcbaf6841b",
        12_650_000_000,
        "qwen-fp8",
    ),
    "qwen-r32": ArmSpec(
        "qwen-r32",
        "qwen-edit-2511",
        32,
        QWEN_WEIGHT_REPO,
        QWEN_WEIGHT_REVISION,
        "nunchaku_qwen_image_edit_2511_ultimate_speed_int4.safetensors",
        "432d5b807f4345de4ec3d0b7ac1a6143930a81a3d35b582b1f603c18d07d56ec",
        11_520_000_000,
        "qwen-fp8",
    ),
    "zimage-r128": ArmSpec(
        "zimage-r128",
        "z-image-turbo",
        128,
        ZIMAGE_WEIGHT_REPO,
        ZIMAGE_WEIGHT_REVISION,
        "svdq-int4_r128-z-image-turbo.safetensors",
        "dbf5aba0f7370d1fe88e52823d3d7b9ceb6fa006707f075baf9cfdaf362ea745",
        4_011_374_152,
        "zimage-fp8",
    ),
    "zimage-r256": ArmSpec(
        "zimage-r256",
        "z-image-turbo",
        256,
        ZIMAGE_WEIGHT_REPO,
        ZIMAGE_WEIGHT_REVISION,
        "svdq-int4_r256-z-image-turbo.safetensors",
        "636328b27315b44ab37844feeeda03874050b30ef92fc9f339bb0cafcf5c01e8",
        4_546_148_632,
        "zimage-fp8",
    ),
}

BASES: dict[str, dict[str, object]] = {
    "qwen-edit-2511": {
        "repo": QWEN_BASE_REPO,
        "revision": QWEN_BASE_REVISION,
        "bytes": 18_000_000_000,
    },
    "z-image-turbo": {
        "repo": ZIMAGE_BASE_REPO,
        "revision": ZIMAGE_BASE_REVISION,
        "bytes": 14_000_000_000,
    },
}

QWEN_CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        "c1",
        "Change her black evening dress to a deep red silk evening dress; keep the same woman, "
        "face, hairstyle, pose, lighting, window and background unchanged.",
        "e1.jpg",
        1280,
        720,
    ),
    CaseSpec(
        "c2",
        "Make her raise her right hand toward the camera with an open palm and five clearly "
        "separated fingers; keep the same woman, face, dress and balcony unchanged.",
        "e2.jpg",
        1280,
        720,
    ),
    CaseSpec(
        "c3",
        'Add a neon shop sign on the brick wall above her that reads "VOXPRINT" in bold white '
        "letters; keep the same woman, face, pose and street unchanged.",
        "e3.jpg",
        1024,
        1024,
    ),
)
ZIMAGE_CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        "z1",
        "An adult woman in a tailored grey coat standing on a quiet city corner, fully clothed, "
        "face turned toward the camera, overcast daylight, no text.",
        None,
        1024,
        1024,
    ),
    CaseSpec(
        "z2",
        "Close view of an adult's right hand raised toward the camera, open palm, five clearly "
        "separated fingers, linen cuff, soft window light, no face.",
        None,
        1024,
        1024,
    ),
    CaseSpec(
        "z3",
        'A brick shop wall at dusk with a neon sign that reads "VOXPRINT" in bold white letters '
        'and a second line that reads "печать", empty street, no people.',
        None,
        1024,
        1024,
    ),
)


def repo_root() -> Path:
    """Repository root that contains ``src`` and ``tests``."""
    return Path(__file__).resolve().parents[3]


def asset_dir() -> Path:
    """Committed JPEG inputs for the Qwen edit cases."""
    return repo_root() / "tests" / "assets" / "nunchaku_edit"


def cache_dir() -> Path:
    """Wheel, venv, and weight cache. ``NUNCHAKU_CACHE`` overrides it."""
    override = os.environ.get("NUNCHAKU_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "voxprint-forge" / "nunchaku-edit"


def done_dir() -> Path:
    """Resume markers. ``NUNCHAKU_DONE`` overrides ``~/.done/nunchaku-edit``."""
    override = os.environ.get("NUNCHAKU_DONE")
    if override:
        return Path(override)
    return Path.home() / ".done" / "nunchaku-edit"


def select_wheel(platform: str | None = None) -> WheelSpec:
    """Return the pinned wheel for Windows or the plan's Linux fallback.

    Linux x86_64 has no cp312 wheel at the pinned revision, so Linux uses the
    cp311 cu128 wheel and Python 3.11. Windows Server uses the cp312 cu130 wheel.
    """
    system = sys.platform if platform is None else platform
    if system == "win32":
        return WINDOWS_WHEEL
    if system.startswith("linux"):
        return LINUX_WHEEL
    raise SetupError(f"no pinned vitoom-nunchaku wheel for {system}")


def build_install_commands(python: Path, wheel: Path, spec: WheelSpec) -> list[list[str]]:
    """uv pip commands. The wheel is installed from disk with ``--no-deps``.

    A bare ``nunchaku`` requirement is never passed, so pip cannot select the
    upstream package that lacks ``zero_cond_t``.
    """
    py = str(python)
    return [
        [
            "uv",
            "pip",
            "install",
            "--python",
            py,
            "torch>=2.11,<2.12",
            "torchvision>=0.26,<0.27",
            "--index-url",
            spec.index_url,
        ],
        [
            "uv",
            "pip",
            "install",
            "--python",
            py,
            "diffusers==0.37.1",
            "transformers==4.57.6",
            "accelerate>=1.9",
            "peft>=0.17",
            "pillow",
        ],
        ["uv", "pip", "install", "--python", py, "--no-deps", str(wheel)],
    ]


def reject_upstream_install(commands: list[list[str]]) -> None:
    """Raise when a command would install upstream nunchaku from an index."""
    for command in commands:
        for arg in command:
            name = Path(arg).name
            if arg == "nunchaku" or arg.startswith("nunchaku==") or arg.startswith("nunchaku>="):
                raise SetupError("refusing to install upstream nunchaku")
            if name == "nunchaku" or name.startswith("nunchaku=="):
                raise SetupError("refusing to install upstream nunchaku")


def assert_zero_cond_source(source: str) -> None:
    """Raise when the installed Qwen transformer source lacks ``zero_cond_t``."""
    if "zero_cond_t" not in source:
        raise SetupError("installed nunchaku has no zero_cond_t; refusing to run Qwen-Image-Edit-2511")


def require_hash(path: Path, expected: str) -> str:
    """Return the sha256 of ``path``. Delete the file and exit 2 on mismatch."""
    digest = sha256_file(path)
    if digest != expected:
        path.unlink(missing_ok=True)
        raise SetupError(f"sha256 mismatch for {path.name}: {digest} != {expected}")
    return digest


def fits_16gb(peak_mib: float | None) -> bool:
    """True when the nvidia-smi peak is at most 15.0 GiB."""
    if peak_mib is None:
        return False
    return float(peak_mib) <= FITS_16GB_MIB


def memory_fraction(cap: str) -> float | None:
    """Fraction for ``torch.cuda.set_per_process_memory_fraction``.

    ``cap16`` is 16/24 of the 24 GB card. Unconstrained does not set a fraction.
    """
    if cap == "unconstrained":
        return None
    if cap == "cap16":
        return 16 / 24
    raise SetupError(f"unknown cap {cap}")


def plan_modes(arm: str, cap: str, mode: str) -> tuple[str, ...]:
    """Offload attempts. ``auto`` falls forward only when the attempt does not qualify."""
    if mode != "auto":
        if mode not in {"model-offload", "block-offload"}:
            raise SetupError(f"unknown mode {mode}")
        return (mode,)
    if arm.startswith("zimage") and cap == "unconstrained":
        return ("resident", "model-offload", "block-offload")
    return ("model-offload", "block-offload")


def steps_for(arm: str, smoke: bool) -> int:
    """Sampling steps. Smoke keeps one case and shortens Qwen from 40 to 20."""
    if arm.startswith("qwen"):
        return QWEN_SMOKE_STEPS if smoke else QWEN_STEPS
    return ZIMAGE_STEPS


def cases_for(arm: str, requested: tuple[str, ...] | None, *, smoke: bool) -> tuple[CaseSpec, ...]:
    """Cases for an arm. ``--smoke`` keeps the first case of that family."""
    pool = QWEN_CASES if arm.startswith("qwen") else ZIMAGE_CASES
    if smoke:
        return (pool[0],)
    if not requested:
        return pool
    chosen = tuple(case for case in pool if case.case_id in requested)
    if not chosen:
        joined = ", ".join(requested)
        raise SetupError(f"no cases for {arm} in {joined}")
    return chosen


def baselines_for(arms: tuple[str, ...] | list[str]) -> list[str]:
    """FP8 baseline ids for the families present in ``arms``."""
    names: list[str] = []
    if any(arm.startswith("qwen") for arm in arms):
        names.append("qwen-fp8")
    if any(arm.startswith("zimage") for arm in arms):
        names.append("zimage-fp8")
    return names


def sort_arms(arms: tuple[str, ...] | list[str]) -> list[str]:
    """Stable arm order. Unknown ids are a setup error."""
    unknown = [arm for arm in arms if arm not in ARMS]
    if unknown:
        raise SetupError(f"unknown arm {unknown[0]}")
    return [arm for arm in ARM_ORDER if arm in arms]


def plan_arm_ids(
    arms: list[str], budget_min: float
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Drop ``zimage-r256`` first when the minute estimate exceeds the budget."""
    skipped: list[dict[str, str]] = []
    selected = [arm for arm in arms if arm != "zimage-r256"]
    wanted_r256 = "zimage-r256" in arms
    bases = baselines_for(selected + (["zimage-r256"] if wanted_r256 else []))
    cost = sum(ESTIMATE_MIN[arm] for arm in selected)
    cost += ESTIMATE_MIN["zimage-r256"] if wanted_r256 else 0
    cost += sum(ESTIMATE_MIN[name] for name in bases)
    if wanted_r256 and cost > budget_min:
        skipped.append({"id": "zimage-r256", "reason": "time_budget"})
        bases = baselines_for(selected)
        return selected, bases, skipped
    if wanted_r256:
        selected.append("zimage-r256")
    return selected, bases, skipped


def skip_for_elapsed(arm: str, *, elapsed_min: float, budget_min: float, kind: str) -> str | None:
    """Skip r256 once three quarters of the budget is gone, then FP8 if it will not fit."""
    if arm == "zimage-r256" and elapsed_min > budget_min * 0.75:
        return "time_budget"
    if kind == "fp8" and elapsed_min + ESTIMATE_MIN.get(arm, 0.0) > budget_min:
        return "time_budget"
    return None


def parse_gpu_csv(line: str) -> GpuFacts:
    """Parse one ``nvidia-smi`` CSV line: name, memory.total, driver, compute_cap."""
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 4:
        raise GpuError("nvidia-smi returned an unexpected GPU line")
    major, minor = _parse_cc(parts[3])
    return GpuFacts(parts[0], _mib(parts[1]), parts[2], major, minor)


def gate_gpu(gpu: GpuFacts) -> None:
    """Exit 3 when VRAM is under 16 GB or compute capability is below 8.9."""
    if (gpu.compute_major, gpu.compute_minor) < (8, 9):
        raise GpuError(
            f"compute capability {gpu.compute_capability} is below 8.9 ({gpu.name})"
        )
    if gpu.memory_mib * 1024 * 1024 < MIN_VRAM_BYTES:
        raise GpuError(f"GPU VRAM {gpu.memory_mib} MiB is under 16 GB ({gpu.name})")


def require_free(path: Path, need: int, *, free: int | None = None) -> None:
    """Exit 2 when ``path`` does not have ``need`` bytes free."""
    target = path if path.exists() else path.parent
    target.mkdir(parents=True, exist_ok=True)
    have = shutil.disk_usage(target).free if free is None else free
    if have < need:
        raise SetupError(
            f"need {need / 1024**3:.1f} GiB free, have {have / 1024**3:.1f} GiB at {target}"
        )


def weight_bytes_needed(arms: list[str], root: Path, done: Path) -> int:
    """Bytes still to download for the selected INT4 files and bases, plus margin."""
    need = 0
    families: set[str] = set()
    for arm in arms:
        spec = ARMS[arm]
        path = root / "weights" / spec.filename
        families.add(spec.family)
        if not _marked(done, f"weights-{arm}", spec.sha256) or not path.is_file():
            need += spec.size_bytes
    for family in families:
        base = BASES[family]
        revision = str(base["revision"])
        if not _marked(done, f"base-{family}", revision):
            need += int(base["bytes"])
    return need + DISK_MARGIN


def apply_offload(pipe: object, mode: str) -> str:
    """Move ``pipe`` with the requested offload. Block mode pins one transformer block."""
    if mode == "resident":
        pipe.to("cuda")  # type: ignore[attr-defined]
        return "resident"
    if mode == "model-offload":
        pipe.enable_model_cpu_offload()  # type: ignore[attr-defined]
        return "model-offload"
    if mode == "block-offload":
        transformer = pipe.transformer  # type: ignore[attr-defined]
        set_offload = getattr(transformer, "set_offload", None)
        if set_offload is not None:
            set_offload(True, use_pin_memory=True, num_blocks_on_gpu=1)
        exclude = getattr(pipe, "_exclude_from_cpu_offload", None)
        if isinstance(exclude, list) and "transformer" not in exclude:
            exclude.append("transformer")
        pipe.enable_sequential_cpu_offload()  # type: ignore[attr-defined]
        return "block-offload"
    raise SetupError(f"unknown mode {mode}")


def note_sample(peak: float, value: float | None) -> float:
    """Running maximum of nvidia-smi samples. ``None`` leaves the peak unchanged."""
    if value is None:
        return peak
    return max(peak, float(value))


def identity_and_ssim(
    reference: list[tuple[int, int, int]],
    candidate: list[tuple[int, int, int]],
    side: int,
) -> dict[str, float]:
    """Face-crop cosine and border SSIM on a square RGB grid.

    This is a proxy for YuNet plus a background mask. It does not claim ArcFace.
    """
    face_ref = _luma_where(reference, side, _face)
    face_cand = _luma_where(candidate, side, _face)
    border_ref = _luma_where(reference, side, _border)
    border_cand = _luma_where(candidate, side, _border)
    return {
        "identity_cosine_proxy": _cosine(face_ref, face_cand),
        "background_ssim_proxy": _ssim(border_ref, border_cand),
    }


def frame_cosine(
    reference: list[tuple[int, int, int]], candidate: list[tuple[int, int, int]]
) -> float:
    """Luma cosine of two equal-length RGB grids."""
    return _cosine(_luma(reference), _luma(candidate))


def fp8_strategy(mirror_files: list[Path], base: Path, vram_mib: int) -> str:
    """How to run the FP8 baseline, or why it will not run.

    ``mirror-dequant`` expands voxprint FP8 when twice the file size fits in 70% of
    VRAM. ``diffusers-layerwise`` is the fallback when the mirror is missing and the
    bf16 transformer is already on disk. This run does not download that transformer.
    """
    if mirror_files:
        total = sum(path.stat().st_size for path in mirror_files if path.is_file())
        if total * 2 < vram_mib * 1024 * 1024 * 0.70:
            return "mirror-dequant"
        return "skipped_vram"
    transformer = base / "transformer"
    if transformer.is_dir() and any(transformer.glob("*.safetensors")):
        return "diffusers-layerwise"
    return "not_mirrored"


def export_aliases(out: Path, arm: str, cap: str, case_ids: list[str]) -> None:
    """Copy the Qwen r128 16 GB outputs to ``c1.webp`` and the other contract names."""
    if arm != "qwen-r128" or cap != "cap16":
        return
    for case_id in case_ids:
        source = out / f"{arm}-{cap}-{case_id}.webp"
        if source.is_file():
            shutil.copy2(source, out / f"{case_id}.webp")


def render_message(result: dict[str, object]) -> str:
    """One-screen summary written to ``MESSAGE.md``."""
    gpu = result.get("gpu") if isinstance(result.get("gpu"), dict) else {}
    versions = result.get("versions") if isinstance(result.get("versions"), dict) else {}
    wheel = result.get("wheel") if isinstance(result.get("wheel"), dict) else {}
    lines = [
        f"# {result.get('tag', TAG)}",
        "",
        f"Schema `{result.get('schema', SCHEMA)}`. Exit {result.get('exit_code')}.",
        "",
        (
            f"GPU: {gpu.get('name', '')}, driver {gpu.get('driver', '')}, "
            f"{gpu.get('memory_mib', '')} MiB, sm_{gpu.get('compute_capability', '')}."
        ),
        (
            f"Wheel: {wheel.get('filename', '')} sha256 {wheel.get('sha256', '')}. "
            f"torch {versions.get('torch', '')}, diffusers {versions.get('diffusers', '')}, "
            f"transformers {versions.get('transformers', '')}, nunchaku {versions.get('nunchaku', '')}."
        ),
        "",
        "Warm-up is excluded from s/img. fits_16gb is the nvidia-smi peak at or below 15.0 GiB.",
        "",
        "| Arm | Cap | Offload | s/img | load s | torch MiB | smi MiB | fits_16gb |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rows = list(result.get("arms") or []) + list(result.get("baselines") or [])
    for row in rows:
        if not isinstance(row, dict):
            continue
        caps = row.get("caps") if isinstance(row.get("caps"), dict) else {}
        for cap, payload in caps.items():
            if not isinstance(payload, dict):
                continue
            seconds = payload.get("seconds_per_image")
            mean = seconds.get("mean") if isinstance(seconds, dict) else ""
            lines.append(
                "| {arm} | {cap} | {mode} | {mean} | {load} | {torch_mib} | {smi} | {fits} |".format(
                    arm=row.get("id", ""),
                    cap=cap,
                    mode=payload.get("offload_mode", ""),
                    mean=mean,
                    load=payload.get("load_seconds", ""),
                    torch_mib=payload.get("peak_vram_torch_mib", ""),
                    smi=payload.get("peak_vram_smi_mib", ""),
                    fits=payload.get("fits_16gb", ""),
                )
            )
    skipped = result.get("skipped") if isinstance(result.get("skipped"), list) else []
    if skipped:
        lines.append("")
        names = [f"{item.get('id')} ({item.get('reason')})" for item in skipped if isinstance(item, dict)]
        lines.append("Skipped: " + ", ".join(names))
    lines.append("")
    return "\n".join(lines)


def decide_exit(
    arm_ids: list[str],
    caps: tuple[str, ...],
    cap_results: dict[tuple[str, str], dict[str, object]],
) -> int:
    """0 when every requested cap finished and every cap16 result fits in 15 GiB."""
    for arm in arm_ids:
        for cap in caps:
            payload = cap_results.get((arm, cap))
            if payload is None or payload.get("error"):
                return EXIT_FAIL
            if cap == "cap16" and not payload.get("fits_16gb"):
                return EXIT_FAIL
    return EXIT_OK


def execute(
    options: argparse.Namespace,
    *,
    spawn: Callable[[dict[str, object]], dict[str, object]],
    elapsed_min: Callable[[], float],
    gpu: GpuFacts,
    wheel: WheelSpec,
    versions: dict[str, str] | None = None,
    mirror_lookup: Callable[[str], list[Path]] | None = None,
) -> tuple[int, dict[str, object]]:
    """Run the planned arms and write ``result.json`` plus ``MESSAGE.md``."""
    arms, baselines, skipped = plan_arm_ids(list(options.arm_ids), options.budget_min)
    requested = tuple(options.cases) if options.cases else None
    cap_results: dict[tuple[str, str], dict[str, object]] = {}
    attempts: list[dict[str, object]] = []
    out = Path(options.out)
    out.mkdir(parents=True, exist_ok=True)
    for arm in list(arms):
        reason = skip_for_elapsed(arm, elapsed_min=elapsed_min(), budget_min=options.budget_min, kind="int4")
        if reason is not None:
            arms.remove(arm)
            skipped.append({"id": arm, "reason": reason})
            continue
        for cap in options.caps:
            payload = _run_cap(
                arm=arm,
                cap=cap,
                mode=options.mode,
                smoke=options.smoke,
                requested=requested,
                spawn=spawn,
                out=out,
                attempts=attempts,
            )
            cap_results[(arm, cap)] = payload
    baseline_rows: list[dict[str, object]] = []
    for baseline in baselines:
        reason = skip_for_elapsed(
            baseline, elapsed_min=elapsed_min(), budget_min=options.budget_min, kind="fp8"
        )
        if reason is not None:
            skipped.append({"id": baseline, "reason": reason})
            baseline_rows.append({"id": baseline, "status": reason, "caps": {}})
            continue
        lookup = mirror_lookup if mirror_lookup is not None else _mirror_files
        payload = _run_baseline(baseline, options, spawn, out, gpu, lookup)
        baseline_rows.append(payload)
    code = EXIT_FAIL if not arms else decide_exit(arms, tuple(options.caps), cap_results)
    if "cap16" in options.caps and arms:
        summary_fits: bool | None = all(bool(cap_results[(arm, "cap16")].get("fits_16gb")) for arm in arms)
    else:
        summary_fits = None
    result: dict[str, object] = {
        "schema": SCHEMA,
        "tag": TAG,
        "ok": code == EXIT_OK,
        "exit_code": code,
        "budget_min": options.budget_min,
        "gpu": {
            "name": gpu.name,
            "driver": gpu.driver,
            "memory_mib": gpu.memory_mib,
            "compute_capability": gpu.compute_capability,
        },
        "versions": versions or {},
        "wheel": {
            "repo": WHEEL_REPO,
            "revision": WHEEL_REVISION,
            "filename": wheel.filename,
            "sha256": wheel.sha256,
            "python": wheel.python,
            "index_url": wheel.index_url,
        },
        "fits_16gb": summary_fits,
        "arms": [_arm_row(arm, tuple(options.caps), cap_results) for arm in arms],
        "baselines": baseline_rows,
        "skipped": skipped,
        "attempts": attempts,
    }
    for arm in arms:
        if "cap16" in options.caps:
            export_aliases(out, arm, "cap16", [case.case_id for case in cases_for(arm, requested, smoke=options.smoke)])
    text = render_message(result)
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "MESSAGE.md").write_text(text, encoding="utf-8")
    mirrored = _mirror(out)
    if mirrored is not None:
        result["mirror"] = mirrored
        (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if options.cleanup:
        _cleanup_weights(arms)
    return code, result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the script contract flags. ``--smoke`` is one case per arm."""
    parser = argparse.ArgumentParser(description="Nunchaku INT4 matrix for the RTX 4090 (tag nunchaku-test-1).")
    parser.add_argument("--json", action="store_true", help="print result.json on stdout")
    parser.add_argument(
        "--arms",
        default="qwen-r128,qwen-r32,zimage-r128",
        help="comma list: qwen-r128, qwen-r32, zimage-r128, zimage-r256 (default omits r256)",
    )
    parser.add_argument("--cases", default="", help="comma list of c1,c2,c3 and/or z1,z2,z3")
    parser.add_argument("--out", default="", help="output directory (default results/nunchaku-edit/<tag>)")
    parser.add_argument("--mode", default="auto", choices=["auto", "model-offload", "block-offload"])
    parser.add_argument("--caps", default="unconstrained,cap16", help="unconstrained and/or cap16")
    parser.add_argument("--smoke", action="store_true", help="one case per arm; Qwen uses 20 steps")
    parser.add_argument("--cleanup", action="store_true", help="delete INT4 weight files after the run")
    parser.add_argument("--budget-min", type=float, default=DEFAULT_BUDGET_MIN)
    parser.add_argument("--worker", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    args.arm_ids = sort_arms([part.strip() for part in args.arms.split(",") if part.strip()])
    args.cases = tuple(part.strip() for part in args.cases.split(",") if part.strip())
    caps = tuple(part.strip() for part in args.caps.split(",") if part.strip())
    for cap in caps:
        memory_fraction(cap)
    if not caps:
        raise SetupError("--caps is empty")
    args.caps = caps
    if not args.out:
        args.out = str(Path("results") / "nunchaku-edit" / TAG)
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the matrix. Returns 0, 1, 2, or 3."""
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else EXIT_SETUP
    except NunchakuError as exc:
        _emit_error(None, exc)
        return exc.exit_code
    if args.worker:
        try:
            return worker_main(Path(args.worker))
        except NunchakuError as exc:
            _log(str(exc))
            return exc.exit_code
    json_mode = bool(args.json)
    try:
        gpu = read_gpu()
        gate_gpu(gpu)
        wheel = select_wheel()
        need_zimage = any(arm.startswith("zimage") for arm in args.arm_ids)
        prepare_runtime(sys.argv[1:] if argv is None else list(argv), need_zimage=need_zimage)
        selected, _, _ = plan_arm_ids(list(args.arm_ids), args.budget_min)
        _prepare_weights(selected)
        versions = collect_versions()
        started = _monotonic()
        code, result = execute(
            args,
            spawn=spawn_worker,
            elapsed_min=lambda: (_monotonic() - started) / 60.0,
            gpu=gpu,
            wheel=wheel,
            versions=versions,
        )
    except NunchakuError as exc:
        _emit_error(json_mode, exc)
        return exc.exit_code
    if json_mode:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        message = Path(args.out) / "MESSAGE.md"
        if message.is_file():
            print(message.read_text(encoding="utf-8"))
    return code


def worker_main(path: Path) -> int:
    """Run one arm, cap, and offload mode. The parent process survives an OOM."""
    job = json.loads(path.read_text(encoding="utf-8"))
    result_path = Path(str(job["result_path"]))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    cap = str(job["cap"])
    mode = str(job["mode"])
    try:
        if not torch.cuda.is_available():
            raise GpuError("CUDA is not available in the worker")
        fraction = memory_fraction(cap)
        if fraction is not None:
            torch.cuda.set_per_process_memory_fraction(fraction)
        sampler = _SmiSampler()
        sampler.start()
        torch.cuda.reset_peak_memory_stats()
        started = _monotonic()
        pipe = _load_pipe(job)
        apply_offload(pipe, mode)
        load_seconds = _monotonic() - started
        cases = list(job["cases"])
        warmup_peak = 0.0
        if job.get("warmup") and cases:
            torch.cuda.reset_peak_memory_stats()
            _generate(pipe, str(job["family"]), cases[0], int(job["steps"]))
            warmup_peak = torch.cuda.max_memory_allocated() / (1024**2)
        case_rows: list[dict[str, object]] = []
        case_peaks: list[float] = []
        for case in cases:
            torch.cuda.reset_peak_memory_stats()
            case_started = _monotonic()
            image = _generate(pipe, str(job["family"]), case, int(job["steps"]))
            seconds = _monotonic() - case_started
            peak = torch.cuda.max_memory_allocated() / (1024**2)
            case_peaks.append(peak)
            filename = f"{job['arm']}-{cap}-{case['id']}.webp"
            destination = Path(str(job["out"])) / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            _save_webp(image, destination)
            case_rows.append(
                {
                    "id": case["id"],
                    "seconds": round(seconds, 3),
                    "output": filename,
                    "peak_vram_torch_mib": round(peak, 3),
                    "scores": _score_case(job, case, image),
                }
            )
            _mark(done_dir(), f"case-{job['arm']}-{cap}-{mode}-{case['id']}", f"{job['steps']}:{SEED}")
        smi_peak = sampler.finish()
        torch_peak = max(case_peaks + [warmup_peak], default=0.0)
        seconds_map = {str(row["id"]): row["seconds"] for row in case_rows}
        values = [float(row["seconds"]) for row in case_rows]
        seconds_map["mean"] = round(sum(values) / len(values), 3) if values else 0.0
        payload: dict[str, object] = {
            "ok": True,
            "oom": False,
            "error": None,
            "arm": job["arm"],
            "cap": cap,
            "offload_mode": mode,
            "load_seconds": round(load_seconds, 3),
            "peak_vram_torch_mib": round(torch_peak, 3),
            "peak_vram_smi_mib": round(smi_peak, 3),
            "fits_16gb": fits_16gb(smi_peak),
            "memory_fraction": fraction,
            "seconds_per_image": seconds_map,
            "cases": case_rows,
            "versions": collect_versions(),
            "warmup_excluded": True,
        }
        _write_json(result_path, payload)
        return EXIT_OK
    except Exception as exc:
        message = scrub(str(exc), active_secrets())
        oom = isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())) or "out of memory" in message.lower()
        payload = {
            "ok": False,
            "oom": oom,
            "crashed": False,
            "error": message,
            "arm": job.get("arm"),
            "cap": cap,
            "offload_mode": mode,
            "fits_16gb": False,
            "peak_vram_torch_mib": None,
            "peak_vram_smi_mib": None,
            "seconds_per_image": {},
            "cases": [],
        }
        _write_json(result_path, payload)
        if isinstance(exc, GpuError):
            return EXIT_GPU
        return EXIT_FAIL


def collect_versions() -> dict[str, str]:
    """Package versions from the interpreter that imports them."""
    return {
        "python": sys.version.split()[0],
        "torch": getattr(torch, "__version__", "missing"),
        "diffusers": getattr(diffusers, "__version__", "missing"),
        "transformers": getattr(transformers, "__version__", "missing"),
        "nunchaku": getattr(nunchaku, "__version__", "missing"),
    }


def read_gpu() -> GpuFacts:
    """Query the first GPU. Exit 3 when nvidia-smi is missing or reports nothing."""
    if shutil.which("nvidia-smi") is None:
        raise GpuError("nvidia-smi not found")
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    line = proc.stdout.strip().splitlines()
    if proc.returncode != 0 or not line:
        raise GpuError("no CUDA GPU")
    return parse_gpu_csv(line[0])


def prepare_runtime(argv: list[str], *, need_zimage: bool) -> None:
    """Create the uv venv, check the wheel hash, and re-exec into it."""
    spec = select_wheel()
    root = cache_dir()
    venv = root / "venv"
    python = _venv_python(venv)
    done = done_dir()
    ready = python.is_file() and _marked(done, "wheel", spec.sha256) and _probe(python, need_zimage=need_zimage)
    if not ready:
        if shutil.which("uv") is None:
            raise SetupError("uv is not on PATH")
        venv.parent.mkdir(parents=True, exist_ok=True)
        if not python.is_file():
            _run(["uv", "venv", "--python", spec.python, str(venv)])
            _mark(done, "venv", spec.python)
        wheel_path = _download_wheel(spec)
        require_hash(wheel_path, spec.sha256)
        commands = build_install_commands(python, wheel_path, spec)
        reject_upstream_install(commands)
        for command in commands:
            _run(command)
        _assert_installed_sources(python, need_zimage=need_zimage)
        _mark(done, "wheel", spec.sha256)
    if not _same_file(Path(sys.executable), python):
        _reexec(python, argv)


def spawn_worker(job: dict[str, object]) -> dict[str, object]:
    """Run one worker subprocess and return its JSON payload."""
    result_path = Path(str(job["result_path"]))
    job_path = result_path.with_suffix(".job.json")
    job_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(job_path, job)
    proc = subprocess.run(
        [sys.executable, "-m", "voxprint_forge.diffusion.nunchaku_edit", "--worker", str(job_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        error = scrub((proc.stderr or proc.stdout or "worker produced no result")[-2000:], active_secrets())
        payload = {
            "ok": False,
            "oom": "out of memory" in error.lower(),
            "crashed": True,
            "error": error,
            "offload_mode": job.get("mode"),
            "fits_16gb": False,
            "peak_vram_smi_mib": None,
            "peak_vram_torch_mib": None,
            "load_seconds": None,
            "seconds_per_image": {},
            "cases": [],
        }
    if isinstance(payload, dict):
        payload["returncode"] = proc.returncode
    return payload


def _run_cap(
    *,
    arm: str,
    cap: str,
    mode: str,
    smoke: bool,
    requested: tuple[str, ...] | None,
    spawn: Callable[[dict[str, object]], dict[str, object]],
    out: Path,
    attempts: list[dict[str, object]],
) -> dict[str, object]:
    token = _cap_token(arm, cap, smoke, requested, mode)
    final = out / "runs" / f"{arm}-{cap}.json"
    if _marked(done_dir(), f"cap-{arm}-{cap}", token) and final.is_file():
        cached_cap = json.loads(final.read_text(encoding="utf-8"))
        if isinstance(cached_cap, dict):
            return cached_cap
    modes = plan_modes(arm, cap, mode)
    accepted: dict[str, object] | None = None
    for attempt in modes:
        cached = _cached_attempt(out, arm, cap, attempt, smoke, requested)
        payload = cached if cached is not None else spawn(
            _job(arm, cap, attempt, smoke, requested, out, kind="int4")
        )
        payload = _normalize(payload, attempt)
        code = payload.get("returncode")
        if code == EXIT_GPU:
            raise GpuError(str(payload.get("error") or "no CUDA GPU"))
        if code == EXIT_SETUP:
            raise SetupError(str(payload.get("error") or "setup failed"))
        attempts.append(
            {
                "arm": arm,
                "cap": cap,
                "mode": attempt,
                "fits_16gb": payload.get("fits_16gb"),
                "error": payload.get("error"),
            }
        )
        if payload.get("oom") or payload.get("crashed"):
            continue
        if payload.get("error") and mode != "auto":
            accepted = payload
            break
        if cap == "cap16" and not payload.get("fits_16gb"):
            if accepted is None or _smi(payload) < _smi(accepted):
                accepted = payload
            continue
        if payload.get("error"):
            continue
        accepted = payload
        break
    if accepted is None:
        return {
            "offload_mode": modes[-1],
            "error": "every offload mode failed",
            "fits_16gb": False,
            "peak_vram_torch_mib": None,
            "peak_vram_smi_mib": None,
            "load_seconds": None,
            "seconds_per_image": {},
            "cases": [],
            "memory_fraction": memory_fraction(cap),
        }
    _write_json(final, accepted)
    _mark(done_dir(), f"cap-{arm}-{cap}", token)
    return accepted


def _run_baseline(
    baseline: str,
    options: argparse.Namespace,
    spawn: Callable[[dict[str, object]], dict[str, object]],
    out: Path,
    gpu: GpuFacts,
    mirror_lookup: Callable[[str], list[Path]],
) -> dict[str, object]:
    family = "qwen-edit-2511" if baseline.startswith("qwen") else "z-image-turbo"
    base = cache_dir() / "bases" / family
    mirror = mirror_lookup(baseline)
    strategy = fp8_strategy(mirror, base, gpu.memory_mib)
    if strategy in {"skipped_vram", "not_mirrored"}:
        return {"id": baseline, "status": strategy, "repo": _fp8_repo(baseline), "caps": {}}
    payload = spawn(
        _job(
            baseline,
            "unconstrained",
            "model-offload",
            options.smoke,
            tuple(options.cases) or None,
            out,
            kind="fp8",
            strategy=strategy,
            mirror=mirror,
        )
    )
    payload = _normalize(payload, "model-offload")
    return {
        "id": baseline,
        "status": "ran" if not payload.get("error") else "error",
        "strategy": strategy,
        "repo": _fp8_repo(baseline),
        "caps": {"unconstrained": payload},
    }


def _job(
    arm: str,
    cap: str,
    mode: str,
    smoke: bool,
    requested: tuple[str, ...] | None,
    out: Path,
    *,
    kind: str,
    strategy: str = "",
    mirror: list[Path] | None = None,
) -> dict[str, object]:
    family = ARMS[arm].family if arm in ARMS else ("qwen-edit-2511" if arm.startswith("qwen") else "z-image-turbo")
    cases = cases_for(arm, requested, smoke=smoke)
    case_rows = []
    for case in cases:
        image = None if case.image is None else str(asset_dir() / case.image)
        case_rows.append(
            {
                "id": case.case_id,
                "prompt": case.prompt,
                "image": image,
                "width": case.width,
                "height": case.height,
            }
        )
    weight = ""
    if arm in ARMS:
        weight = str(cache_dir() / "weights" / ARMS[arm].filename)
    return {
        "kind": kind,
        "arm": arm,
        "family": family,
        "cap": cap,
        "mode": mode,
        "steps": steps_for(arm, smoke),
        "seed": SEED,
        "cases": case_rows,
        "weight": weight,
        "weight_sha256": ARMS[arm].sha256 if arm in ARMS else "",
        "base": str(cache_dir() / "bases" / family),
        "out": str(out),
        "warmup": True,
        "result_path": str(out / "runs" / f"{arm}-{cap}-{mode}.json"),
        "strategy": strategy,
        "fp8_files": [str(path) for path in mirror or []],
    }


def _normalize(payload: dict[str, object], mode: str) -> dict[str, object]:
    peak = payload.get("peak_vram_smi_mib")
    peak_value = float(peak) if isinstance(peak, (int, float)) else None
    payload["fits_16gb"] = fits_16gb(peak_value)
    payload.setdefault("offload_mode", mode)
    payload.setdefault("error", None if payload.get("ok") else payload.get("error") or "worker failed")
    return payload


def _arm_row(
    arm: str,
    caps: tuple[str, ...],
    cap_results: dict[tuple[str, str], dict[str, object]],
) -> dict[str, object]:
    spec = ARMS[arm]
    return {
        "id": arm,
        "family": spec.family,
        "rank": spec.rank,
        "repo": spec.repo,
        "revision": spec.revision,
        "weight_file": spec.filename,
        "weight_sha256": spec.sha256,
        "baseline": spec.baseline,
        "caps": {cap: cap_results[(arm, cap)] for cap in caps if (arm, cap) in cap_results},
    }


def _cap_token(arm: str, cap: str, smoke: bool, requested: tuple[str, ...] | None, mode: str) -> str:
    ids = ",".join(case.case_id for case in cases_for(arm, requested, smoke=smoke))
    return f"{steps_for(arm, smoke)}:{SEED}:{mode}:{cap}:{ids}"


def _cached_attempt(
    out: Path,
    arm: str,
    cap: str,
    mode: str,
    smoke: bool,
    requested: tuple[str, ...] | None,
) -> dict[str, object] | None:
    path = out / "runs" / f"{arm}-{cap}-{mode}.json"
    case_id = cases_for(arm, requested, smoke=smoke)[-1].case_id
    marker = done_dir() / f"case-{arm}-{cap}-{mode}-{case_id}"
    token = f"{steps_for(arm, smoke)}:{SEED}"
    if path.is_file() and marker.is_file() and marker.read_text(encoding="utf-8").strip() == token:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    return None


def _prepare_weights(arms: list[str]) -> None:
    root = cache_dir()
    done = done_dir()
    families = {ARMS[arm].family for arm in arms}
    require_free(root, weight_bytes_needed(arms, root, done))
    for family in sorted(families):
        _ensure_base(family)
    for arm in arms:
        _ensure_weight(ARMS[arm])


def _ensure_base(family: str) -> None:
    base = BASES[family]
    revision = str(base["revision"])
    dest = cache_dir() / "bases" / family
    done = done_dir()
    if _marked(done, f"base-{family}", revision) and (dest / "model_index.json").is_file():
        return
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=str(base["repo"]),
        revision=revision,
        ignore_patterns=["transformer/*"],
        local_dir=str(dest),
        token=False,
    )
    _mark(done, f"base-{family}", revision)


def _ensure_weight(spec: ArmSpec) -> None:
    dest = cache_dir() / "weights" / spec.filename
    done = done_dir()
    step = f"weights-{spec.arm_id}"
    if _marked(done, step, spec.sha256) and dest.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest.is_file() and sha256_file(dest) == spec.sha256):
        hf_hub_download(
            repo_id=spec.repo,
            filename=spec.filename,
            revision=spec.revision,
            local_dir=str(dest.parent),
            token=False,
        )
    require_hash(dest, spec.sha256)
    _mark(done, step, spec.sha256)


def _download_wheel(spec: WheelSpec) -> Path:
    dest = cache_dir() / "wheels" / spec.filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        try:
            require_hash(dest, spec.sha256)
            return dest
        except SetupError:
            pass
    url = f"https://huggingface.co/{WHEEL_REPO}/resolve/{WHEEL_REVISION}/{urllib.parse.quote(spec.filename)}"
    _download_url(url, dest)
    return dest


def _download_url(url: str, dest: Path) -> None:
    existing = dest.stat().st_size if dest.is_file() else 0
    headers = {"User-Agent": "voxprint-forge"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            status = getattr(response, "status", 200)
            data = response.read()
    except urllib.error.URLError as exc:
        raise SetupError(f"wheel download failed: {exc}") from exc
    mode = "ab" if status == 206 and existing else "wb"
    with dest.open(mode) as handle:
        handle.write(data)


def _assert_installed_sources(python: Path, *, need_zimage: bool) -> None:
    script = (
        "import nunchaku.models.transformers.transformer_qwenimage as qwen\n"
        "print(qwen.__file__)\n"
    )
    proc = subprocess.run([str(python), "-c", script], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise SetupError(scrub(proc.stderr[-1000:] or "nunchaku import failed", active_secrets()))
    source = Path(proc.stdout.strip()).read_text(encoding="utf-8", errors="replace")
    assert_zero_cond_source(source)
    if need_zimage and not _probe(python, need_zimage=True):
        raise SetupError(
            "pinned vitoom-nunchaku wheel does not export the Z-Image transformer; "
            "upstream nunchaku is not installed"
        )


def _probe(python: Path, *, need_zimage: bool) -> bool:
    script = "import nunchaku, torch, diffusers, transformers\n"
    if need_zimage:
        script += (
            "from nunchaku.models.transformers.transformer_zimage import NunchakuZImageTransformer2DModel\n"
        )
    proc = subprocess.run([str(python), "-c", script], capture_output=True, text=True, check=False)
    return proc.returncode == 0


def _require_cls(cls: object, name: str) -> object:
    if cls is None:
        raise SetupError(f"{name} is not available in this interpreter")
    return cls


def _load_pipe(job: dict[str, object]) -> object:
    if diffusers is None or nunchaku is None:
        raise SetupError("diffusers and the vitoom-nunchaku wheel are required in this interpreter")
    family = str(job["family"])
    base = str(job["base"])
    if str(job.get("kind")) == "fp8":
        return _load_fp8(job)
    weight = str(job["weight"])
    if family == "qwen-edit-2511":
        transformer_cls = _require_cls(NunchakuQwenImageTransformer2DModel, "NunchakuQwenImageTransformer2DModel")
        pipeline_cls = _require_cls(QwenImageEditPlusPipeline, "QwenImageEditPlusPipeline")
    elif family == "z-image-turbo":
        transformer_cls = _require_cls(NunchakuZImageTransformer2DModel, "NunchakuZImageTransformer2DModel")
        pipeline_cls = _require_cls(ZImagePipeline, "ZImagePipeline")
    else:
        raise SetupError(f"unknown family {family}")
    transformer = transformer_cls.from_pretrained(weight, torch_dtype=torch.bfloat16)
    kwargs: dict[str, object] = {"transformer": transformer, "torch_dtype": torch.bfloat16}
    if family == "z-image-turbo":
        kwargs["low_cpu_mem_usage"] = False
    return pipeline_cls.from_pretrained(base, **kwargs)


def _load_fp8(job: dict[str, object]) -> object:
    family = str(job["family"])
    if family == "qwen-edit-2511":
        pipeline_cls = _require_cls(QwenImageEditPlusPipeline, "QwenImageEditPlusPipeline")
        model_cls = _require_cls(QwenImageTransformer2DModel, "QwenImageTransformer2DModel")
    else:
        pipeline_cls = _require_cls(ZImagePipeline, "ZImagePipeline")
        model_cls = _require_cls(ZImageTransformer2DModel, "ZImageTransformer2DModel")
    if str(job.get("strategy")) == "diffusers-layerwise":
        pipe = pipeline_cls.from_pretrained(str(job["base"]), torch_dtype=torch.bfloat16)
        pipe.transformer.enable_layerwise_casting(
            storage_dtype=torch.float8_e4m3fn,
            compute_dtype=torch.bfloat16,
        )
        return pipe
    files = [Path(str(item)) for item in job.get("fp8_files") or []]
    state: dict[str, torch.Tensor] = {}
    for path in files:
        if read_format(path) not in {None, "voxprint-fp8-v1"}:
            continue
        state.update(dequant_file(path))
    config = Path(str(job["base"])) / "transformer"
    transformer = model_cls.from_config(model_cls.load_config(str(config)))
    transformer.load_state_dict(state, strict=False)
    return pipeline_cls.from_pretrained(str(job["base"]), transformer=transformer, torch_dtype=torch.bfloat16)


def _generate(pipe: object, family: str, case: dict[str, object], steps: int) -> object:
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    if family == "qwen-edit-2511":
        if Image is None or not case.get("image"):
            raise SetupError("Pillow and the edit input are required")
        source = Image.open(str(case["image"])).convert("RGB")
        return pipe(  # type: ignore[operator]
            image=source,
            prompt=str(case["prompt"]),
            negative_prompt=NEGATIVE_PROMPT,
            true_cfg_scale=TRUE_CFG,
            num_inference_steps=steps,
            height=int(case["height"]),
            width=int(case["width"]),
            generator=generator,
            output_type="pil",
        ).images[0]
    return pipe(  # type: ignore[operator]
        prompt=str(case["prompt"]),
        height=int(case["height"]),
        width=int(case["width"]),
        num_inference_steps=steps,
        guidance_scale=TURBO_GUIDANCE,
        generator=generator,
    ).images[0]


def _save_webp(image: object, path: Path) -> None:
    image.save(path, format="WEBP", quality=95, method=6)  # type: ignore[attr-defined]


def _score_case(job: dict[str, object], case: dict[str, object], image: object) -> dict[str, object]:
    scores: dict[str, object] = {"ocr_text": _ocr(image), "ocr_match": None}
    text = scores["ocr_text"]
    if isinstance(text, str):
        scores["ocr_match"] = "VOXPRINT" in text.upper()
    if Image is None:
        return scores
    candidate = _grid(image)
    if case.get("image"):
        reference = _grid(Image.open(str(case["image"])).convert("RGB"))
        scores.update(identity_and_ssim(reference, candidate, 64))
    return scores


def _grid(image: object) -> list[tuple[int, int, int]]:
    resized = image.convert("RGB").resize((64, 64))  # type: ignore[attr-defined]
    return [(int(r), int(g), int(b)) for r, g, b in resized.getdata()]


def _ocr(image: object) -> str | None:
    if pytesseract is None:
        return None
    try:
        text = pytesseract.image_to_string(image)
    except Exception:
        return None
    cleaned = " ".join(str(text).split())
    return cleaned or None


def _mirror_files(baseline: str) -> list[Path]:
    repo = _fp8_repo(baseline)
    dest = cache_dir() / "fp8" / baseline
    try:
        names = list_repo_files(repo, token=False)
    except Exception as exc:
        _log(f"fp8 mirror {repo} unavailable: {exc}")
        return []
    globs = (
        ("qwen-image-edit-2511-transformer-fp8",)
        if baseline.startswith("qwen")
        else ("zimage-turbo-transformer-fp8", "zimage-transformer-fp8")
    )
    matched = [
        name
        for name in names
        if name.endswith(".safetensors") and any(name.split("/")[-1].startswith(prefix) for prefix in globs)
    ]
    if not matched:
        return []
    dest.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in matched:
        downloaded = hf_hub_download(repo_id=repo, filename=name, local_dir=str(dest), token=False)
        paths.append(Path(downloaded))
    return paths


def _fp8_repo(baseline: str) -> str:
    if baseline.startswith("qwen"):
        return QWEN_FP8_REPO
    return ZIMAGE_FP8_REPO


def _cleanup_weights(arms: list[str]) -> None:
    for arm in arms:
        path = cache_dir() / "weights" / ARMS[arm].filename
        if path.is_file():
            path.unlink()
        marker = done_dir() / f"weights-{arm}"
        if marker.is_file():
            marker.unlink()


def _mirror(out: Path) -> str | None:
    override = os.environ.get("NUNCHAKU_VM_RESULTS", "/workspace/vm-results/model-forge")
    root = Path(override)
    if not root.is_dir():
        return None
    dest = root / TAG
    dest.mkdir(parents=True, exist_ok=True)
    for path in out.iterdir():
        if path.is_file():
            shutil.copy2(path, dest / path.name)
    runs = out / "runs"
    if runs.is_dir():
        target = dest / "runs"
        target.mkdir(parents=True, exist_ok=True)
        for path in runs.glob("*.json"):
            shutil.copy2(path, target / path.name)
    return str(dest)


def _emit_error(json_mode: bool | None, exc: NunchakuError) -> None:
    message = scrub(str(exc), active_secrets())
    if json_mode:
        print(
            json.dumps(
                {"ok": False, "schema": SCHEMA, "tag": TAG, "exit_code": exc.exit_code, "error": message},
                indent=2,
                sort_keys=True,
            )
        )
        return
    _log(message)


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mark(directory: Path, step: str, token: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / step).write_text(token + "\n", encoding="utf-8")


def _marked(directory: Path, step: str, token: str) -> bool:
    path = directory / step
    return path.is_file() and path.read_text(encoding="utf-8").strip() == token


def _venv_python(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return str(left) == str(right)


def _reexec(python: Path, argv: list[str]) -> None:
    env = os.environ.copy()
    root = str(repo_root() / "src")
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not current else root + os.pathsep + current
    env["PYTHONUNBUFFERED"] = "1"
    os.execve(str(python), [str(python), "-m", "voxprint_forge.diffusion.nunchaku_edit", *argv], env)


def _run(command: list[str]) -> None:
    proc = subprocess.run(command, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = scrub((proc.stderr or proc.stdout)[-1000:], active_secrets())
        raise SetupError(f"command failed ({proc.returncode}): {' '.join(command[:4])}: {detail}")


def _smi(payload: dict[str, object]) -> float:
    value = payload.get("peak_vram_smi_mib")
    if isinstance(value, (int, float)):
        return float(value)
    return 1e18


def _monotonic() -> float:
    return time.monotonic()


def _mib(text: str) -> int:
    digits = "".join(character for character in text if character.isdigit())
    if not digits:
        raise GpuError("nvidia-smi did not report VRAM")
    return int(digits)


def _parse_cc(text: str) -> tuple[int, int]:
    cleaned = text.strip().strip("[]")
    if "." not in cleaned:
        raise GpuError(f"unreadable compute capability {text}")
    major, minor = cleaned.split(".", 1)
    minor_digits = "".join(character for character in minor if character.isdigit())
    return int(major), int(minor_digits or "0")


def _luma(pixels: list[tuple[int, int, int]]) -> list[float]:
    return [0.2126 * red + 0.7152 * green + 0.0722 * blue for red, green, blue in pixels]


def _luma_where(
    pixels: list[tuple[int, int, int]],
    side: int,
    predicate: Callable[[int, int, int], bool],
) -> list[float]:
    values: list[float] = []
    for index, (red, green, blue) in enumerate(pixels):
        x = index % side
        y = index // side
        if predicate(x, y, side):
            values.append(0.2126 * red + 0.7152 * green + 0.0722 * blue)
    return values


def _face(x: int, y: int, side: int) -> bool:
    return 0.30 * side <= x < 0.70 * side and 0.05 * side <= y < 0.45 * side


def _border(x: int, y: int, side: int) -> bool:
    return x < 0.10 * side or x >= 0.90 * side or y < 0.10 * side or y >= 0.90 * side


def _cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _ssim(left: list[float], right: list[float]) -> float:
    count = len(left)
    if count == 0 or count != len(right):
        return 0.0
    mean_left = sum(left) / count
    mean_right = sum(right) / count
    var_left = sum((value - mean_left) ** 2 for value in left) / count
    var_right = sum((value - mean_right) ** 2 for value in right) / count
    cov = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True)) / count
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    return ((2 * mean_left * mean_right + c1) * (2 * cov + c2)) / (
        (mean_left**2 + mean_right**2 + c1) * (var_left + var_right + c2)
    )


class _SmiSampler:
    """Background nvidia-smi sampler at 0.5 s."""

    def __init__(self, interval: float = 0.5) -> None:
        self.interval = interval
        self.peak = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="nunchaku-smi", daemon=True)

    def start(self) -> None:
        """Start sampling."""
        self._thread.start()

    def finish(self) -> float:
        """Stop the thread and return the peak MiB."""
        self._stop.set()
        self._thread.join(timeout=2)
        return self.peak

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = note_sample(self.peak, _query_smi_mib())
            self._stop.wait(self.interval)


def _query_smi_mib() -> float | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return float(_mib(proc.stdout.strip().splitlines()[0]))
    except GpuError:
        return None


if __name__ == "__main__":
    sys.exit(main())
