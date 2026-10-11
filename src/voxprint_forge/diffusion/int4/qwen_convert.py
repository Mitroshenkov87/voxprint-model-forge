"""Convert a quantized Qwen-Image transformer block to the Nunchaku key layout.

Key names follow ``convert_to_nunchaku_qwenimage_transformer_block_state_dict``
on the pinned ``syh`` commit. Modulation layers stay W4A16 (``qweight``,
``wscales``, ``wzeros``, ``bias``). The other linears are W4A4 and carry
``smooth_factor``. This is the layout official ``NunchakuQwenImageTransformer2DModel``
loads. The vitoom wheel's ``SVDQW8A16Linear`` mod path is a different contract
and is handled by ``quantfunc_smooth``.
"""

from __future__ import annotations

import torch

from voxprint_forge.diffusion.int4.qwen_pack import (
    convert_to_nunchaku_w4a4_linear,
    convert_to_nunchaku_w4x16_linear_weight,
)

MOD_KEYS = ("qweight", "wscales", "wzeros", "bias")
LINEAR_KEYS = (
    "qweight",
    "bias",
    "smooth_factor",
    "smooth_factor_orig",
    "proj_down",
    "proj_up",
)


def _update(lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor], prefix: str) -> None:
    for key, value in rhs.items():
        full = f"{prefix}.{key}" if prefix else key
        if full in lhs:
            raise KeyError(full)
        lhs[full] = value


def _mod_state(weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None) -> dict[str, torch.Tensor]:
    packed, stored_scale, stored_zero, stored_bias = convert_to_nunchaku_w4x16_linear_weight(
        weight, scale, bias=bias, adanorm_splits=6
    )
    return {"qweight": packed, "wscales": stored_scale, "wzeros": stored_zero, "bias": stored_bias}


def convert_qwen_block(
    state: dict[str, torch.Tensor],
    scales: dict[str, torch.Tensor],
    smooth: dict[str, torch.Tensor],
    branches: dict[str, tuple[torch.Tensor, torch.Tensor]],
    block_name: str,
) -> dict[str, torch.Tensor]:
    """Convert one dual-stream Qwen block to Nunchaku tensors.

    ``state`` holds float weights named ``{block}.{local}.weight``. ``scales``
    holds ``{block}.{local}.weight.scale.0`` as ``[oc, 1, groups, 1]``.
    Modulation scales may be ``[1, 1, 1, 1]``.
    """
    converted: dict[str, torch.Tensor] = {}
    local_map: dict[str, str | tuple[str, ...]] = {
        "img_mod.1": "img_mod.1",
        "txt_mod.1": "txt_mod.1",
        "attn.to_qkv": ("attn.to_q", "attn.to_k", "attn.to_v"),
        "attn.add_qkv_proj": ("attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"),
        "attn.norm_q": "attn.norm_q",
        "attn.norm_k": "attn.norm_k",
        "attn.norm_added_q": "attn.norm_added_q",
        "attn.norm_added_k": "attn.norm_added_k",
        "attn.to_out.0": "attn.to_out.0",
        "attn.to_add_out": "attn.to_add_out",
        "img_mlp.net.0.proj": "img_mlp.net.0.proj",
        "img_mlp.net.2": _down_name(state, block_name, "img_mlp"),
        "txt_mlp.net.0.proj": "txt_mlp.net.0.proj",
        "txt_mlp.net.2": _down_name(state, block_name, "txt_mlp"),
    }
    smooth_map = {
        "attn.to_qkv": "attn.to_q",
        "attn.add_qkv_proj": "attn.add_k_proj",
        "attn.to_out.0": "attn.to_out.0",
        "attn.to_add_out": "attn.to_out.0",
        "img_mlp.net.0.proj": "img_mlp.net.0.proj",
        "img_mlp.net.2": local_map["img_mlp.net.2"],
        "txt_mlp.net.0.proj": "txt_mlp.net.0.proj",
        "txt_mlp.net.2": local_map["txt_mlp.net.2"],
    }
    branch_map = {
        "attn.to_qkv": "attn.to_q",
        "attn.add_qkv_proj": "attn.add_k_proj",
        "attn.to_out.0": "attn.to_out.0",
        "attn.to_add_out": "attn.to_add_out",
        "img_mlp.net.0.proj": "img_mlp.net.0.proj",
        "img_mlp.net.2": local_map["img_mlp.net.2"],
        "txt_mlp.net.0.proj": "txt_mlp.net.0.proj",
        "txt_mlp.net.2": local_map["txt_mlp.net.2"],
    }
    mod_names = {"img_mod.1", "txt_mod.1"}
    for converted_name, sources in local_map.items():
        names = (sources,) if isinstance(sources, str) else sources
        weights = [_weight(state, block_name, name) for name in names]
        biases = [_optional(state, block_name, name, "bias") for name in names]
        scale_rows = [_optional_scale(scales, block_name, name) for name in names]
        if len(weights) > 1:
            weight = torch.cat(weights, dim=0)
            bias = None if any(item is None for item in biases) else torch.cat(biases, dim=0)
            if any(item is None for item in scale_rows):
                scale = None
            elif all(item.numel() == 1 for item in scale_rows):
                scale = torch.cat(
                    [
                        item.reshape(-1)[:1].expand(weights[index].shape[0]).reshape(weights[index].shape[0], 1, 1, 1)
                        for index, item in enumerate(scale_rows)
                    ],
                    dim=0,
                )
            else:
                scale = torch.cat(scale_rows, dim=0)
        else:
            weight, bias, scale = weights[0], biases[0], scale_rows[0]
        if scale is None:
            converted[f"{converted_name}.weight"] = weight.detach().to(dtype=torch.bfloat16).cpu()
            if bias is not None:
                converted[f"{converted_name}.bias"] = bias.detach().to(dtype=torch.bfloat16).cpu()
            continue
        if converted_name in mod_names:
            stored_bias = None if bias is None else bias.to(dtype=torch.bfloat16)
            _update(
                converted,
                _mod_state(weight.to(dtype=torch.bfloat16), scale, stored_bias),
                converted_name,
            )
            continue
        smooth_name = str(smooth_map[converted_name])
        branch_name = str(branch_map[converted_name])
        _update(
            converted,
            convert_to_nunchaku_w4a4_linear(
                weight.to(dtype=torch.bfloat16),
                scale,
                None if bias is None else bias.to(dtype=torch.bfloat16),
                smooth.get(f"{block_name}.{smooth_name}"),
                branches.get(f"{block_name}.{branch_name}"),
            ),
            converted_name,
        )
    return {f"{block_name}.{key}": value for key, value in converted.items()}


def nunchaku_block_key_report(converted: dict[str, torch.Tensor], block_name: str) -> dict[str, list[int]]:
    """Shapes of the Nunchaku keys this block must expose. Missing keys raise."""
    required = []
    for prefix in ("img_mod.1", "txt_mod.1"):
        required.extend(f"{block_name}.{prefix}.{key}" for key in MOD_KEYS)
    for prefix in (
        "attn.to_qkv",
        "attn.add_qkv_proj",
        "attn.to_out.0",
        "attn.to_add_out",
        "img_mlp.net.0.proj",
        "img_mlp.net.2",
        "txt_mlp.net.0.proj",
        "txt_mlp.net.2",
    ):
        required.extend(f"{block_name}.{prefix}.{key}" for key in LINEAR_KEYS)
        scale_keys = [
            key
            for key in converted
            if key.startswith(f"{block_name}.{prefix}.") and key.rsplit(".", 1)[-1] in {"wscales", "wcscales"}
        ]
        if len(scale_keys) != 1:
            raise KeyError(f"{block_name}.{prefix} needs one wscales or wcscales key, found {scale_keys}")
        required.append(scale_keys[0])
    for prefix in ("attn.norm_q", "attn.norm_k", "attn.norm_added_q", "attn.norm_added_k"):
        required.append(f"{block_name}.{prefix}.weight")
    missing = [key for key in required if key not in converted]
    if missing:
        raise KeyError(missing[0])
    return {key: list(converted[key].shape) for key in required}


def _down_name(state: dict[str, torch.Tensor], block_name: str, mlp: str) -> str:
    linear = f"{mlp}.net.2.linear"
    if f"{block_name}.{linear}.weight" in state:
        return linear
    plain = f"{mlp}.net.2"
    if f"{block_name}.{plain}.weight" not in state:
        raise KeyError(f"{block_name}.{plain}.weight")
    return plain


def _weight(state: dict[str, torch.Tensor], block_name: str, local: str) -> torch.Tensor:
    key = f"{block_name}.{local}.weight"
    if key not in state:
        raise KeyError(key)
    return state[key]


def _optional(state: dict[str, torch.Tensor], block_name: str, local: str, suffix: str) -> torch.Tensor | None:
    return state.get(f"{block_name}.{local}.{suffix}")


def _optional_scale(scales: dict[str, torch.Tensor], block_name: str, local: str) -> torch.Tensor | None:
    return scales.get(f"{block_name}.{local}.weight.scale.0")
