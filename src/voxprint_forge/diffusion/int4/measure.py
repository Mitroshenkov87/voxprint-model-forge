"""Optional GPU eval. diffusers and nunchaku are not installed on CPU CI."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

from safetensors.torch import load_file

from voxprint_forge.diffusion.fp8 import dequant_state
from voxprint_forge.diffusion.int4.calib import gate_prompts, load_prompts
from voxprint_forge.diffusion.int4.context import StageContext
from voxprint_forge.diffusion.int4.evalrule import Arm
from voxprint_forge.diffusion.int4.pins import (
    EDIT_STEPS,
    EDIT_TRUE_CFG,
    NUNCHAKU_FORK_COMMIT,
    NUNCHAKU_WHEEL,
    QUANTFUNC_R128,
    QUANTFUNC_R256,
    QWEN_EDIT_REPO,
    QWEN_EDIT_REVISION,
)
from voxprint_forge.errors import ToolMissingError

try:
    import torch
except ImportError:
    torch = None

try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError:
    QwenImageEditPlusPipeline = None

try:
    from nunchaku import NunchakuQwenImageTransformer2DModel
except ImportError:
    NunchakuQwenImageTransformer2DModel = None

try:
    from PIL import Image
except ImportError:
    Image = None


def require_edit_runtime() -> None:
    """Raise when the pinned edit stack cannot be imported."""
    missing = [
        name
        for name, value in (
            ("torch", torch),
            ("diffusers.QwenImageEditPlusPipeline", QwenImageEditPlusPipeline),
            ("nunchaku.NunchakuQwenImageTransformer2DModel", NunchakuQwenImageTransformer2DModel),
        )
        if value is None
    ]
    if missing:
        raise ToolMissingError(
            "Qwen-Image-Edit-2511 eval needs "
            + ", ".join(missing)
            + f". Install the vitoom-nunchaku wheel {NUNCHAKU_WHEEL} from commit {NUNCHAKU_FORK_COMMIT}.",
            code="tool_missing",
        )


def aggregate_case(peaks_mib: list[float], seconds: list[float]) -> tuple[float, float]:
    """Max smi peak and the mean seconds per image."""
    if not peaks_mib or not seconds:
        raise ToolMissingError("eval produced no samples", code="eval_empty")
    return max(peaks_mib), sum(seconds) / len(seconds)


def _smi_mib() -> float | None:
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if done.returncode != 0 or not done.stdout.strip():
        return None
    return float(done.stdout.strip().splitlines()[0].split()[0])


def _time_pipe(
    pipe: Callable[..., object],
    cases: list[dict[str, object]],
    image_root: Path,
) -> tuple[float, float]:
    if Image is None or torch is None or not torch.cuda.is_available():
        raise ToolMissingError("edit eval needs Pillow, torch, and a CUDA device", code="tool_missing")
    peaks: list[float] = []
    seconds: list[float] = []
    for case in cases:
        image = Image.open(image_root / str(case["image"])).convert("RGB")
        smi: list[float] = []
        stop = threading.Event()

        def watch() -> None:
            while not stop.wait(0.5):
                sample = _smi_mib()
                if sample is not None:
                    smi.append(sample)

        thread = threading.Thread(target=watch, name="diffusion-int4-smi", daemon=True)
        torch.cuda.reset_peak_memory_stats()
        thread.start()
        start = time.perf_counter()
        pipe(
            image=image,
            prompt=str(case["prompt"]),
            negative_prompt=" ",
            true_cfg_scale=EDIT_TRUE_CFG,
            num_inference_steps=EDIT_STEPS,
            generator=torch.Generator(device="cuda").manual_seed(0),
        )
        seconds.append(time.perf_counter() - start)
        stop.set()
        thread.join(timeout=1)
        torch_mib = float(torch.cuda.max_memory_allocated()) / (1024 * 1024)
        peaks.append(max(smi) if smi else torch_mib)
    return aggregate_case(peaks, seconds)


def _weight(env_name: str, name: str) -> Path:
    root = os.environ.get(env_name, "").strip()
    if not root:
        raise ToolMissingError(f"{env_name} is not set", code="tool_missing")
    path = Path(root) / name if name else Path(root)
    if not path.exists():
        raise ToolMissingError(f"missing {path}", code="tool_missing")
    return path


def measure_live(ctx: StageContext, target: str) -> list[Arm]:
    """Time own INT4, QuantFunc r128/r256, and FP8 on the hard cases.

    Identity, hands, and OCR stay unset, so the winner rule uses fit and
    seconds per image. The comparison file records that rule.
    """
    require_edit_runtime()
    assert QwenImageEditPlusPipeline is not None
    assert torch is not None
    if not torch.cuda.is_available():
        raise ToolMissingError("edit eval needs a CUDA device", code="tool_missing")
    export = sorted((ctx.job(target) / "export").glob("*.safetensors"))
    if not export:
        raise ToolMissingError(f"{target} has no exported safetensors", code="tool_missing")
    specs: list[tuple[str, Path, int | None, str]] = [
        ("own-r256", export[0], ctx.svd_rank, "nunchaku"),
        ("quantfunc-r128", _weight("FORGE_QUANTFUNC", QUANTFUNC_R128), 128, "nunchaku"),
        ("quantfunc-r256", _weight("FORGE_QUANTFUNC", QUANTFUNC_R256), 256, "nunchaku"),
        ("fp8", _weight("FORGE_FP8_EDIT", ""), None, "diffusers"),
    ]
    cases = gate_prompts(load_prompts(ctx.repo))
    image_root = ctx.repo / "data" / "diffusion_calib"
    arms: list[Arm] = []
    for arm_id, path, rank, kind in specs:
        if kind == "nunchaku":
            pipe = load_edit_pipeline(path)
        else:
            pipe = load_fp8_edit(path)
        peak, seconds = _time_pipe(pipe, cases, image_root)
        arms.append(Arm(arm_id, peak, seconds, None, None, None, None, rank))
        del pipe
        torch.cuda.empty_cache()
    return arms


def load_edit_pipeline(weight: Path, *, offload: str = "model"):
    """Load Qwen-Image-Edit-2511 with the fork transformer.

    ``offload`` is ``model`` (``enable_model_cpu_offload``) or ``sequential``.
    """
    require_edit_runtime()
    assert torch is not None
    assert NunchakuQwenImageTransformer2DModel is not None
    assert QwenImageEditPlusPipeline is not None
    transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(str(weight))
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        QWEN_EDIT_REPO,
        revision=QWEN_EDIT_REVISION,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    if offload == "sequential":
        pipe.enable_sequential_cpu_offload()
    else:
        pipe.enable_model_cpu_offload()
    return pipe


def time_cases(
    generate: Callable[[str], tuple[float, float]],
    case_ids: list[str],
) -> tuple[float, float]:
    """Return the max peak MiB and the mean seconds from one callable per case."""
    peaks: list[float] = []
    seconds: list[float] = []
    for case_id in case_ids:
        peak, elapsed = generate(case_id)
        peaks.append(peak)
        seconds.append(elapsed)
    return aggregate_case(peaks, seconds)


def fp8_layout(path: Path) -> str:
    """Return ``diffusers`` or ``shards`` for a FP8 comparison directory."""
    if path.is_dir() and (path / "model_index.json").is_file():
        return "diffusers"
    if path.is_dir() and any(path.glob("*.safetensors")):
        return "shards"
    raise ToolMissingError(f"{path} is not a diffusers folder or an FP8 shard directory", code="tool_missing")


def load_fp8_edit(path: Path):
    """Load the FP8 comparison pipeline.

    A directory with ``model_index.json`` loads as a diffusers folder.
    Canvas ``voxprint-fp8-v1`` transformer shards are dequantized into the
    base Qwen-Image-Edit-2511 pipeline (``FORGE_QWEN_EDIT`` or the Hub repo).
    """
    require_edit_runtime()
    assert torch is not None
    assert QwenImageEditPlusPipeline is not None
    layout = fp8_layout(path)
    if layout == "diffusers":
        pipe = QwenImageEditPlusPipeline.from_pretrained(str(path), torch_dtype=torch.bfloat16)
    elif layout == "shards":
        base = os.environ.get("FORGE_QWEN_EDIT", "").strip() or QWEN_EDIT_REPO
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            base,
            revision=None if os.environ.get("FORGE_QWEN_EDIT", "").strip() else QWEN_EDIT_REVISION,
            torch_dtype=torch.bfloat16,
        )
        tensors: dict[str, torch.Tensor] = {}
        for shard in sorted(path.glob("*.safetensors")):
            tensors.update(load_file(str(shard)))
        pipe.transformer.load_state_dict(dequant_state(tensors), strict=False)
    else:
        raise ToolMissingError(f"unhandled FP8 layout {layout}", code="tool_missing")
    pipe.enable_model_cpu_offload()
    return pipe


def edit_settings() -> dict[str, float]:
    """Steps and true CFG shared with the canvas FP8 edit eval."""
    return {"steps": float(EDIT_STEPS), "true_cfg": EDIT_TRUE_CFG}
