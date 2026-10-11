"""CPU proof of Qwen-Image-Edit INT4: calibrate, quantize, convert, check keys.

The model is a random-init ``QwenImageTransformer2DModel`` with one block.
Calibration reads an image from ``data/diffusion_calib`` and builds
``modulate_index`` the way diffusers does when ``zero_cond_t`` is true: the
first image shape is the target (index 0) and the rest are the condition
image (index 1). Text embeddings are a stable hash of the prompt, so this
path does not download Qwen2-VL. Fast INT4 uses group size 64 and SVD rank 8.
Nunchaku's low-rank pack tile is 16, so the packed factors are zero-padded
from 8 to 16. No CUDA kernel runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from PIL import Image
from safetensors.torch import save_file

from voxprint_forge.diffusion.int4.calib import image_root, load_prompts
from voxprint_forge.diffusion.int4.pins import SVD_GROUP
from voxprint_forge.diffusion.int4.quantfunc_smooth import install_quantfunc_shim
from voxprint_forge.diffusion.int4.qwen_convert import convert_qwen_block, nunchaku_block_key_report
from voxprint_forge.diffusion.int4.qwen_pack import packed_lowrank_rank
from voxprint_forge.errors import StageError

try:
    from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel
except ImportError:
    NunchakuQwenImageTransformer2DModel = None

TINY_HEADS = 4
TINY_HEAD_DIM = 32
TINY_RANK = 8
TINY_SAMPLES = 2
TARGET_SHAPE = (1, 2, 2)
COND_SHAPE = (1, 2, 2)


def tiny_config() -> dict[str, object]:
    """Constructor kwargs for the one-block random-init Edit transformer."""
    return {
        "patch_size": 2,
        "in_channels": 16,
        "out_channels": 4,
        "num_layers": 1,
        "attention_head_dim": TINY_HEAD_DIM,
        "num_attention_heads": TINY_HEADS,
        "joint_attention_dim": 32,
        "guidance_embeds": False,
        "axes_dims_rope": (8, 12, 12),
        "zero_cond_t": True,
    }


def build_tiny_transformer(*, seed: int = 0) -> QwenImageTransformer2DModel:
    """Random-init Edit transformer with ``zero_cond_t`` enabled."""
    generator = torch.Generator().manual_seed(seed)
    model = QwenImageTransformer2DModel(**tiny_config())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype))
    model.eval()
    return model.to(dtype=torch.bfloat16)


def prove_tiny_edit(repo: Path, *, rank: int = TINY_RANK, device: torch.device | None = None) -> dict[str, object]:
    """Run calib, fast INT4, convert, and the Nunchaku key/shape check.

    ``device`` defaults to CPU. CUDA is used only as a torch device for the
    same math; the nunchaku CUDA kernels are not imported.
    """
    chosen = torch.device("cpu") if device is None else device
    report, _model, _converted, _images, _prompts = _run_tiny(repo, rank=rank, device=chosen)
    return report


def smoke_tiny(
    repo: Path,
    *,
    rank: int = TINY_RANK,
    device: torch.device | None = None,
    dest: Path | None = None,
) -> dict[str, object]:
    """Calibrate, quantize, convert, and produce one transformer latent.

    A cheap GPU (RTX 4090, RTX 5090, or A10) is the intended host. The model
    is one block, so the torch work is the same path as ``prove_tiny_edit``.
    When the vitoom wheel is importable and the device is CUDA, the converted
    checkpoint is loaded with ``NunchakuQwenImageTransformer2DModel`` after the
    QuantFunc smooth shim and one forward is the image. The image is the
    transformer latent. VAE decode is a separate step.
    """
    chosen = _resolve_device(device)
    report, model, converted, images, prompts = _run_tiny(repo, rank=rank, device=chosen)
    report["image"] = "transformer latent"
    report["vae_decode"] = False
    report["nunchaku_loaded"] = False
    if dest is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        save_file(_checkpoint_tensors(model, converted), dest, metadata=_checkpoint_metadata(rank))
        report["checkpoint"] = str(dest)
    if NunchakuQwenImageTransformer2DModel is None:
        report["nunchaku_note"] = "nunchaku is not installed; the latent is the diffusers forward"
        return report
    if chosen.type != "cuda":
        report["nunchaku_note"] = "SVDQW8A16Linear requant of QuantFunc-style mods needs CUDA"
        return report
    if dest is None:
        raise StageError("smoke-tiny nunchaku load needs a checkpoint path", code="smoke_tiny")
    install_quantfunc_shim()
    loaded = NunchakuQwenImageTransformer2DModel.from_pretrained(
        str(dest),
        device="cuda",
        torch_dtype=torch.bfloat16,
    )
    prompt = str(prompts[0].get("prompt", "edit"))
    batch = _edit_batch(loaded, images[0], prompt)
    with torch.no_grad():
        output = loaded(**batch)
    sample = output.sample if hasattr(output, "sample") else output[0]
    report["nunchaku_loaded"] = True
    report["nunchaku_note"] = "loaded the tiny checkpoint and ran one forward"
    report["output_shape"] = list(sample.shape)
    report["cuda_kernels"] = True
    return report


def _run_tiny(
    repo: Path,
    *,
    rank: int,
    device: torch.device,
) -> tuple[
    dict[str, object],
    QwenImageTransformer2DModel,
    dict[str, torch.Tensor],
    list[Path],
    list[dict[str, object]],
]:
    chosen = device
    prompts = load_prompts(repo)
    images = _sample_images(prompts, repo)
    model = build_tiny_transformer().to(chosen)
    block_name = "transformer_blocks.0"
    inputs, captured = _collect_inputs(model, images, prompts)
    quant = _quantize_block(model, captured, rank=rank)
    converted = convert_qwen_block(
        quant["state"],
        quant["scales"],
        quant["smooth"],
        quant["branches"],
        block_name,
    )
    shapes = nunchaku_block_key_report(converted, block_name)
    _check_rank_padding(shapes, block_name, rank)
    _check_mod_has_no_smooth(shapes, block_name)
    one = inputs[-1]["output"]
    report: dict[str, object] = {
        "device": str(chosen),
        "zero_cond_t": True,
        "rank": rank,
        "packed_rank": packed_lowrank_rank(rank),
        "group_size": SVD_GROUP,
        "samples": len(inputs),
        "images": [str(path) for path in images],
        "modulate_index": "target tokens 0, condition tokens 1",
        "output_shape": list(one.shape),
        "keys": len(shapes),
        "shapes": shapes,
        "cuda_kernels": False,
    }
    return report, model, converted, images, prompts


def _resolve_device(device: torch.device | None) -> torch.device:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _checkpoint_metadata(rank: int) -> dict[str, str]:
    config = tiny_config()
    payload = dict(config)
    payload["axes_dims_rope"] = list(config["axes_dims_rope"])  # type: ignore[index]
    return {
        "config": json.dumps(payload),
        "quantization_config": json.dumps(
            {
                "method": "svdquant",
                "rank": packed_lowrank_rank(rank),
                "weight": {"dtype": "int4", "group_size": SVD_GROUP},
                "activation": {"dtype": "int4"},
            }
        ),
        "voxprint_svd_rank": str(rank),
    }


def _checkpoint_tensors(
    model: QwenImageTransformer2DModel,
    converted: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Unquantized tiny weights plus the converted block. Mods stay W4A16 on disk."""
    block_name = "transformer_blocks.0"
    prefix = block_name + "."
    tensors: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        if name.startswith(prefix):
            continue
        tensors[name] = _checkpoint_tensor(value)
    for name, value in model.transformer_blocks[0].state_dict().items():
        full = f"{block_name}.{name}"
        if full in converted:
            continue
        tensors[full] = _checkpoint_tensor(value)
    for key, value in converted.items():
        tensors[key] = value.detach().cpu().contiguous()
    return tensors


def _checkpoint_tensor(value: torch.Tensor) -> torch.Tensor:
    tensor = value.detach().cpu().contiguous()
    if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
        return tensor.to(dtype=torch.bfloat16)
    return tensor


def _sample_images(prompts: list[dict[str, object]], repo: Path) -> list[Path]:
    root = image_root(repo)
    found: list[Path] = []
    for item in prompts:
        name = item.get("image")
        if not isinstance(name, str) or not name:
            continue
        path = root / "images" / name if not Path(name).is_absolute() else Path(name)
        if not path.is_file():
            path = root / name
        if path.is_file() and path not in found:
            found.append(path)
        if len(found) >= TINY_SAMPLES:
            break
    if len(found) < TINY_SAMPLES:
        raise StageError("diffusion_calib needs two images for the tiny Edit proof", code="calib_missing")
    return found


def _collect_inputs(
    model: QwenImageTransformer2DModel,
    images: list[Path],
    prompts: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, list[torch.Tensor]]]:
    captured: dict[str, list[torch.Tensor]] = {}
    hooks = []
    for name, module in model.transformer_blocks[0].named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_pre_hook(_capture(name, captured)))
    records: list[dict[str, object]] = []
    try:
        for index, path in enumerate(images):
            prompt = str(prompts[index].get("prompt", "edit"))
            batch = _edit_batch(model, path, prompt)
            with torch.no_grad():
                output = model(**batch)
            sample = output.sample if hasattr(output, "sample") else output[0]
            records.append({"image": path, "prompt": prompt, "output": sample.detach().cpu()})
    finally:
        for hook in hooks:
            hook.remove()
    if not captured:
        raise StageError("tiny calibration captured no linear inputs", code="calib")
    return records, captured


def _capture(name: str, sink: dict[str, list[torch.Tensor]]):
    def hook(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        if not args:
            return
        sink.setdefault(name, []).append(args[0].detach())

    return hook


def _edit_batch(model: QwenImageTransformer2DModel, image: Path, prompt: str) -> dict[str, object]:
    """One image-conditioned step. Condition tokens use the swatch, target tokens do not."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    target = _prod(TARGET_SHAPE)
    cond = _prod(COND_SHAPE)
    seq = target + cond
    channels = int(model.config.in_channels)
    hidden = torch.zeros((1, seq, channels), dtype=dtype, device=device)
    hidden[:, target:] = _image_latent(image, cond, channels, dtype, device)
    text_dim = int(model.config.joint_attention_dim)
    text = _prompt_latent(prompt, 4, text_dim, dtype, device)
    mask = torch.ones((1, text.shape[1]), dtype=torch.bool, device=device)
    return {
        "hidden_states": hidden,
        "encoder_hidden_states": text,
        "encoder_hidden_states_mask": mask,
        "timestep": torch.tensor([10], device=device),
        "img_shapes": [[TARGET_SHAPE, COND_SHAPE]],
        "return_dict": True,
    }


def _image_latent(path: Path, tokens: int, channels: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    if image.size[0] < 1 or image.size[1] < 1:
        raise StageError(f"{path} has no pixels", code="calib")
    red, green, blue = image.resize((1, 1)).getpixel((0, 0))
    color = torch.tensor((red, green, blue), dtype=torch.float32) / 255.0
    column = color.repeat((channels + 2) // 3)[:channels]
    return column.to(dtype=dtype, device=device).view(1, 1, channels).expand(1, tokens, channels).contiguous()


def _prompt_latent(prompt: str, length: int, dim: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "little")
    generator = torch.Generator().manual_seed(seed)
    return torch.randn((1, length, dim), generator=generator, dtype=torch.float32).to(dtype=dtype, device=device)


def _quantize_block(
    model: QwenImageTransformer2DModel,
    captured: dict[str, list[torch.Tensor]],
    *,
    rank: int,
) -> dict[str, dict[str, torch.Tensor] | dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    """Group-wise INT4 plus a rank-8 SVD branch. Modulation stays W4A16 without SVD."""
    state: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    smooth: dict[str, torch.Tensor] = {}
    branches: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    block = model.transformer_blocks[0]
    prefix = "transformer_blocks.0"
    for name, parameter in block.named_parameters():
        if ".norm" in name and name.endswith(".weight"):
            state[f"{prefix}.{name}"] = parameter.detach().to(dtype=torch.bfloat16).cpu()
    fused: dict[str, list[str]] = {
        f"{prefix}.attn.to_q": [f"{prefix}.attn.to_q", f"{prefix}.attn.to_k", f"{prefix}.attn.to_v"],
        f"{prefix}.attn.add_k_proj": [
            f"{prefix}.attn.add_q_proj",
            f"{prefix}.attn.add_k_proj",
            f"{prefix}.attn.add_v_proj",
        ],
    }
    linears = {name: module for name, module in block.named_modules() if isinstance(module, torch.nn.Linear)}
    for name, module in linears.items():
        full = f"{prefix}.{name}"
        weight = module.weight.detach().to(dtype=torch.bfloat16).cpu()
        bias = None if module.bias is None else module.bias.detach().to(dtype=torch.bfloat16).cpu()
        mod = name in {"img_mod.1", "txt_mod.1"}
        smoothed, factor = (weight, None) if mod else _smooth(weight, captured.get(name, []))
        state[f"{full}.weight"] = smoothed
        if bias is not None:
            state[f"{full}.bias"] = bias
        scale = _group_scale(smoothed, SVD_GROUP)
        scales[f"{full}.weight.scale.0"] = scale
        if factor is not None:
            smooth[full] = factor
        if not mod:
            branches[full] = _low_rank(smoothed, scale, rank)
    for anchor, parts in fused.items():
        residuals = [_residual(state[f"{part}.weight"], scales[f"{part}.weight.scale.0"]) for part in parts]
        branches[anchor] = _svd_pair(torch.cat(residuals, dim=0), rank)
    return {"state": state, "scales": scales, "smooth": smooth, "branches": branches}


def _smooth(weight: torch.Tensor, activations: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    channel = weight.detach().float().abs().amax(dim=0).clamp_min(1e-6)
    if activations:
        flat = torch.cat([item.detach().float().reshape(-1, item.shape[-1]).cpu() for item in activations], dim=0)
        act = flat.abs().amax(dim=0).clamp_min(1e-6)
    else:
        act = torch.ones_like(channel)
    factor = (channel / act).sqrt().clamp(1e-3, 1e3)
    smoothed = weight.float() * factor.unsqueeze(0)
    return smoothed.to(dtype=torch.bfloat16), factor.to(dtype=torch.bfloat16)


def _group_scale(weight: torch.Tensor, group: int) -> torch.Tensor:
    oc, ic = weight.shape
    if ic % group != 0:
        raise StageError(f"in_features {ic} is not divisible by group {group}", code="quant")
    groups = ic // group
    viewed = weight.float().view(oc, groups, group)
    scale = viewed.abs().amax(dim=-1).clamp_min(1e-6) / 7.0
    return scale.view(oc, 1, groups, 1).to(dtype=torch.bfloat16)


def _residual(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    oc, ic = weight.shape
    groups = int(scale.shape[2])
    group = ic // groups
    scale_g = scale.float().reshape(oc, groups, 1)
    quantized = weight.float().view(oc, groups, group).div(scale_g).round().clamp(-8, 7)
    dequant = quantized * scale_g
    return (weight.float() - dequant.reshape(oc, ic)).to(dtype=torch.bfloat16)


def _low_rank(weight: torch.Tensor, scale: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    return _svd_pair(_residual(weight, scale), rank)


def _svd_pair(residual: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    oc, ic = residual.shape
    used = min(rank, oc, ic)
    packed = packed_lowrank_rank(used)
    _u, values, vh = torch.linalg.svd(residual.float(), full_matrices=False)
    take = min(used, int(values.shape[0]))
    up = torch.zeros((oc, packed), dtype=torch.bfloat16)
    down = torch.zeros((packed, ic), dtype=torch.bfloat16)
    if take:
        scale = values[:take].sqrt()
        up[:, :take] = (_u[:, :take] * scale.unsqueeze(0)).to(dtype=torch.bfloat16)
        down[:take] = (scale.unsqueeze(1) * vh[:take]).to(dtype=torch.bfloat16)
    return down, up


def _check_rank_padding(shapes: dict[str, list[int]], block_name: str, rank: int) -> None:
    packed = packed_lowrank_rank(rank)
    up = shapes[f"{block_name}.attn.to_qkv.proj_up"]
    down = shapes[f"{block_name}.attn.to_qkv.proj_down"]
    if up[-1] != packed or down[-1] != packed:
        raise StageError(
            f"packed low-rank axes are {up[-1]} and {down[-1]}, expected {packed} for SVD rank {rank}",
            code="convert",
        )


def _check_mod_has_no_smooth(shapes: dict[str, list[int]], block_name: str) -> None:
    for key in shapes:
        if ".img_mod.1.smooth_factor" in key or ".txt_mod.1.smooth_factor" in key:
            raise StageError(f"official Nunchaku mod layout must not contain {key}", code="convert")
    for prefix in ("img_mod.1", "txt_mod.1"):
        qweight = shapes[f"{block_name}.{prefix}.qweight"]
        if len(qweight) != 2 or qweight[0] < 1 or qweight[1] < 1:
            raise StageError(f"{prefix} qweight shape {qweight} is not a packed matrix", code="convert")


def _prod(shape: tuple[int, int, int]) -> int:
    frames, height, width = shape
    return frames * height * width
