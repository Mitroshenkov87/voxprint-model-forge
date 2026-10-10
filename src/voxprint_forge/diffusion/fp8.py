"""Per-tensor FP8 e4m3fn with a BF16 scale. Format tag `voxprint-fp8-v1`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Never

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from voxprint_forge.errors import StageError

FP8_MAX = 448.0
FORMAT = "voxprint-fp8-v1"
Profile = Literal["diffusion", "controlnet", "edit_transformer", "edit_encoder"]

_FLOATS = frozenset({"F32", "F16", "BF16"})
_DIFFUSION_NEEDLES = (
    "embed",
    "wte",
    "norm",
    "adaln",
    "t_embedder",
    "time_embed",
    "timestep",
    "cap_embedder",
    "caption_embedder",
    "context_embedder",
    "final_layer",
    "proj_out",
    "norm_out",
    "lm_head",
    "rope",
    "pad_token",
)
_EDIT_TRANSFORMER_NEEDLES = ("img_in", "txt_in", "norm", "adaln", "proj_out")
_ZERO_CONV = ("zero_conv", "zeroconv", "zero-conv")


def _unexpected_profile(profile: Never) -> Never:
    raise StageError(f"unhandled fp8 profile {profile}", code="fp8_profile")


def keep_bf16(name: str, profile: Profile) -> bool:
    """True when this tensor must stay BF16 for the given component."""
    lower = name.lower()
    if name.endswith(".bias") or not name.endswith(".weight"):
        return True
    if profile == "diffusion":
        return any(needle in lower for needle in _DIFFUSION_NEEDLES)
    if profile == "controlnet":
        if any(needle in lower for needle in _ZERO_CONV):
            return True
        return any(needle in lower for needle in _DIFFUSION_NEEDLES)
    if profile == "edit_transformer":
        return any(needle in lower for needle in _EDIT_TRANSFORMER_NEEDLES)
    if profile == "edit_encoder":
        if "visual" in lower or "vision" in lower:
            return True
        return "embed" in lower or "norm" in lower
    return _unexpected_profile(profile)


def quantize_this(name: str, tensor: torch.Tensor, profile: Profile) -> bool:
    if not tensor.is_floating_point() or tensor.ndim < 2:
        return False
    return not keep_bf16(name, profile)


def quantize_meta(name: str, shape: tuple[int, ...], dtype_name: str, profile: Profile) -> bool:
    """Same decision as quantize_this, from safetensors metadata, without allocating the tensor."""
    if dtype_name not in _FLOATS or len(shape) < 2:
        return False
    return not keep_bf16(name, profile)


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return float8_e4m3fn values and a 0-dim BF16 scale of amax/448."""
    finite = weight.detach().float()
    amax = torch.amax(finite.abs())
    if not torch.isfinite(amax):
        amax = torch.zeros((), dtype=torch.float32)
    scale = torch.clamp(amax / FP8_MAX, min=torch.finfo(torch.float32).tiny)
    quantized = (finite / scale).to(torch.float8_e4m3fn)
    return quantized, scale.to(dtype=torch.bfloat16)


def _stored(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
        return tensor.detach().to(dtype=torch.bfloat16)
    return tensor.detach()


def quantize_state(tensors: dict[str, torch.Tensor], profile: Profile) -> dict[str, torch.Tensor]:
    """Quantize Linear weights. Scales use the key `<name>.scale_weight`."""
    output: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if quantize_this(name, tensor, profile):
            quantized, scale = quantize_weight(tensor)
            output[name] = quantized
            output[f"{name}.scale_weight"] = scale
        else:
            output[name] = _stored(tensor)
    return output


def save_quantized(tensors: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu = {name: tensor.detach().cpu().contiguous() for name, tensor in tensors.items()}
    save_file(cpu, str(path), metadata={"format": FORMAT})


def dequant_state(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Expand FP8 tensors with their BF16 scales. Other tensors pass through."""
    scales = {name[: -len(".scale_weight")]: value for name, value in tensors.items() if name.endswith(".scale_weight")}
    output: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if name.endswith(".scale_weight"):
            continue
        scale = scales.get(name)
        if tensor.dtype == torch.float8_e4m3fn:
            if scale is None:
                raise StageError(f"{name} is fp8 without a scale", code="fp8_scale")
            output[name] = tensor.to(dtype=torch.bfloat16) * scale.to(dtype=torch.bfloat16)
        elif tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
            output[name] = tensor.to(dtype=torch.bfloat16)
        else:
            output[name] = tensor
    return output


def read_format(path: Path) -> str | None:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        meta = handle.metadata() or {}
    value = meta.get("format")
    return str(value) if value is not None else None


def dequant_file(path: Path) -> dict[str, torch.Tensor]:
    return dequant_state(load_file(str(path)))
