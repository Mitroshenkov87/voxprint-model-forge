"""Optional GPU eval.

diffusers and nunchaku are imported on the first eval call. A CPU dry-run
must not import diffusers, including when the package is installed for the
one-block Qwen proof.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Never

from safetensors.torch import load_file

from voxprint_forge.diffusion.fp8 import FP8_MAX, dequant_state
from voxprint_forge.diffusion.int4.calib import gate_prompts, image_root, load_prompts
from voxprint_forge.diffusion.int4.context import StageContext
from voxprint_forge.diffusion.int4.evalrule import Arm
from voxprint_forge.diffusion.int4.gpu import device_fp8_report
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
from voxprint_forge.diffusion.int4.quantfunc_smooth import install_quantfunc_shim
from voxprint_forge.errors import StageError, ToolMissingError

Fp8ExecMode = Literal["native", "emulated"]
Fp8Layout = Literal["diffusers", "shards"]

try:
    import torch
except ImportError:
    torch = None

QwenImageEditPlusPipeline = None
NunchakuQwenImageTransformer2DModel = None

try:
    from PIL import Image
except ImportError:
    Image = None


def _bind_edit_stack() -> None:
    """Import diffusers and nunchaku once, on the eval path."""
    global QwenImageEditPlusPipeline, NunchakuQwenImageTransformer2DModel
    if QwenImageEditPlusPipeline is None:
        try:
            from diffusers import QwenImageEditPlusPipeline as _pipeline
        except ImportError:
            _pipeline = None
        QwenImageEditPlusPipeline = _pipeline
    if NunchakuQwenImageTransformer2DModel is None:
        try:
            from nunchaku import NunchakuQwenImageTransformer2DModel as _transformer
        except ImportError:
            _transformer = None
        NunchakuQwenImageTransformer2DModel = _transformer


def require_edit_runtime() -> None:
    """Raise when the pinned edit stack cannot be imported."""
    _bind_edit_stack()
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
    save_dir: Path | None = None,
    arm_id: str = "",
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
        result = pipe(
            image=image,
            prompt=str(case["prompt"]),
            negative_prompt=" ",
            true_cfg_scale=EDIT_TRUE_CFG,
            num_inference_steps=EDIT_STEPS,
            generator=torch.Generator(device="cuda").manual_seed(0),
        )
        if save_dir is not None:
            images = getattr(result, "images", None)
            if not images:
                raise StageError(f"{arm_id} produced no eval image", code="eval_empty")
            save_dir.mkdir(parents=True, exist_ok=True)
            images[0].save(save_dir / f"{arm_id}-{case['id']}.png")
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
    images = image_root(ctx.repo)
    save_dir = ctx.job(target) / "eval-images"
    errors_path = ctx.job(target) / "eval-errors.json"
    errors: dict[str, str] = {}
    arms: list[Arm] = []
    for arm_id, path, rank, kind in specs:
        try:
            if kind == "nunchaku":
                pipe = load_edit_pipeline(path)
            else:
                pipe = load_fp8_edit(path, mode=_live_fp8_mode())
            peak, seconds = _time_pipe(pipe, cases, images, save_dir, arm_id)
        except Exception as exc:
            if arm_id.startswith("own-"):
                raise
            errors[arm_id] = f"{type(exc).__name__}: {exc}"[:2000]
            print(f"EVAL-ARM-FAILED {arm_id}: {errors[arm_id][:300]}", flush=True)
            errors_path.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
            if torch is not None:
                torch.cuda.empty_cache()
            continue
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
    install_quantfunc_shim()
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


def fp8_layout(path: Path) -> Fp8Layout:
    """Return ``diffusers`` or ``shards`` for a FP8 comparison directory."""
    if path.is_dir() and (path / "model_index.json").is_file():
        return "diffusers"
    if path.is_dir() and any(path.glob("*.safetensors")):
        return "shards"
    raise ToolMissingError(f"{path} is not a diffusers folder or an FP8 shard directory", code="tool_missing")


def _unexpected_layout(layout: Never) -> Never:
    raise ToolMissingError(f"unhandled FP8 layout {layout}", code="tool_missing")


def _unexpected_fp8_mode(mode: Never) -> Never:
    raise StageError(f"unhandled FP8 baseline mode {mode}", code="fp8_mode")


def _live_fp8_mode() -> Fp8ExecMode:
    """Native on sm_89+. Unknown and emulated capabilities run weight-only BF16."""
    mode = str(device_fp8_report(probe=True)["mode"])
    if mode == "native":
        return "native"
    if mode == "emulated" or mode == "unknown":
        return "emulated"
    raise StageError(f"unhandled FP8 baseline mode {mode}", code="fp8_mode")


def _read_shards(path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(path.glob("*.safetensors")):
        tensors.update(load_file(str(shard)))
    return tensors


def _fp8_pairs(
    tensors: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Split BF16 leftovers from float8 weights and their scales."""
    scales = {
        name[: -len(".scale_weight")]: value for name, value in tensors.items() if name.endswith(".scale_weight")
    }
    rest: dict[str, torch.Tensor] = {}
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, tensor in tensors.items():
        if name.endswith(".scale_weight"):
            continue
        if tensor.dtype == torch.float8_e4m3fn:
            scale = scales.get(name)
            if scale is None:
                raise StageError(f"{name} is fp8 without a scale", code="fp8_scale")
            pairs[name] = (tensor, scale)
            continue
        if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
            rest[name] = tensor.to(dtype=torch.bfloat16)
        else:
            rest[name] = tensor
    return rest, pairs


def _install_scaled_linear(linear: torch.nn.Linear, weight: torch.Tensor, scale: torch.Tensor) -> None:
    """Point one Linear at a float8 weight. The forward uses ``torch._scaled_mm`` on CUDA."""
    linear.weight = torch.nn.Parameter(weight.detach().contiguous(), requires_grad=False)
    linear.register_buffer("_fp8_scale", scale.detach().to(dtype=torch.float32).reshape(()), persistent=False)

    def forward(incoming: torch.Tensor, _linear: torch.nn.Linear = linear) -> torch.Tensor:
        return _fp8_linear_forward(_linear, incoming)

    linear.forward = forward  # type: ignore[method-assign]


def _fp8_linear_forward(linear: torch.nn.Linear, incoming: torch.Tensor) -> torch.Tensor:
    """Tensor-wise FP8 matmul. The second matrix stays column-major."""
    if torch is None:
        raise ToolMissingError("native FP8 needs torch", code="tool_missing")
    leading = incoming.shape[:-1]
    flat = incoming.reshape(-1, incoming.shape[-1]).to(dtype=torch.bfloat16)
    amax = flat.detach().abs().amax()
    if not torch.isfinite(amax):
        amax = torch.zeros((), device=flat.device, dtype=torch.float32)
    scale_a = torch.clamp(amax.float() / FP8_MAX, min=torch.finfo(torch.float32).tiny)
    activation = (flat.float() / scale_a).to(torch.float8_e4m3fn)
    scale_b = getattr(linear, "_fp8_scale").to(device=flat.device, dtype=torch.float32).reshape(1)
    produced = torch._scaled_mm(
        activation,
        linear.weight.t(),
        scale_a=scale_a.reshape(1),
        scale_b=scale_b,
        out_dtype=torch.bfloat16,
    )
    if isinstance(produced, tuple):
        produced = produced[0]
    if linear.bias is not None:
        produced = produced + linear.bias.to(dtype=produced.dtype)
    return produced.reshape(*leading, produced.shape[-1])


def bind_fp8_linears(module: torch.nn.Module, pairs: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> int:
    """Replace matching Linear weights with float8 storage. Returns how many were bound."""
    if torch is None:
        raise ToolMissingError("native FP8 needs torch", code="tool_missing")
    bound = 0
    for name, linear in module.named_modules():
        if not isinstance(linear, torch.nn.Linear):
            continue
        found = pairs.get(f"{name}.weight")
        if found is None:
            continue
        weight, scale = found
        _install_scaled_linear(linear, weight, scale)
        bound += 1
    return bound


def _require_native_bindings(bound: int) -> None:
    if bound == 0:
        raise StageError(
            "native FP8 attached no Linear modules. Refusing to time a BF16 pipeline as a native FP8 baseline.",
            code="fp8_native",
        )


def _base_edit_pipe():
    assert torch is not None
    assert QwenImageEditPlusPipeline is not None
    local = os.environ.get("FORGE_QWEN_EDIT", "").strip()
    base = local or QWEN_EDIT_REPO
    return QwenImageEditPlusPipeline.from_pretrained(
        base,
        revision=None if local else QWEN_EDIT_REVISION,
        torch_dtype=torch.bfloat16,
    )


def _apply_native(pipe: object, shard_root: Path) -> None:
    assert torch is not None
    tensors = _read_shards(shard_root)
    rest, pairs = _fp8_pairs(tensors)
    transformer = pipe.transformer
    if rest:
        transformer.load_state_dict(rest, strict=False)
    _require_native_bindings(bind_fp8_linears(transformer, pairs))


def _load_emulated(path: Path, layout: Fp8Layout):
    assert torch is not None
    assert QwenImageEditPlusPipeline is not None
    if layout == "diffusers":
        return QwenImageEditPlusPipeline.from_pretrained(str(path), torch_dtype=torch.bfloat16)
    if layout == "shards":
        pipe = _base_edit_pipe()
        pipe.transformer.load_state_dict(dequant_state(_read_shards(path)), strict=False)
        return pipe
    return _unexpected_layout(layout)


def _load_native(path: Path, layout: Fp8Layout):
    assert torch is not None
    assert QwenImageEditPlusPipeline is not None
    if layout == "diffusers":
        pipe = QwenImageEditPlusPipeline.from_pretrained(str(path), torch_dtype=torch.bfloat16)
        shard_root = path / "transformer" if (path / "transformer").is_dir() else path
        _apply_native(pipe, shard_root)
        return pipe
    if layout == "shards":
        pipe = _base_edit_pipe()
        _apply_native(pipe, path)
        return pipe
    return _unexpected_layout(layout)


def load_fp8_edit(path: Path, *, mode: Fp8ExecMode = "emulated"):
    """Load the FP8 comparison pipeline.

    ``emulated`` dequantizes canvas shards to BF16 (sm_80 weight-only).
    ``native`` keeps float8 Linear weights and runs ``torch._scaled_mm`` on
    sm_89 or newer. A directory with ``model_index.json`` is a diffusers
    folder. Shards load into ``FORGE_QWEN_EDIT`` or the Hub repo.
    """
    require_edit_runtime()
    assert torch is not None
    assert QwenImageEditPlusPipeline is not None
    layout = fp8_layout(path)
    if mode == "emulated":
        pipe = _load_emulated(path, layout)
    elif mode == "native":
        pipe = _load_native(path, layout)
    else:
        pipe = _unexpected_fp8_mode(mode)
    pipe.enable_model_cpu_offload()
    return pipe


def edit_settings() -> dict[str, float]:
    """Steps and true CFG shared with the canvas FP8 edit eval."""
    return {"steps": float(EDIT_STEPS), "true_cfg": EDIT_TRUE_CFG}
