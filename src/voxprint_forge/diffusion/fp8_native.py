"""FP8 E4M3 linears that keep the file bytes and run on FP8 tensor cores.

The weight buffer is the ``float8_e4m3fn`` tensor from a ``voxprint-fp8-v1`` file.
It is not requantised. The scale is the stored BF16 ``<name>.scale_weight``
cast to FP32, which is exact for every BF16 value, so a dequant path and a
native path use the same scale.
"""

from __future__ import annotations

import inspect
from typing import Never

import torch
from torch import nn

from voxprint_forge.compare.codes import CompareSetup
from voxprint_forge.diffusion.alias import alias_key

FP8_MAX = 448.0
_ACT_SCALES = ("dynamic", "unit")


def _unexpected_act(scale: Never) -> Never:
    raise CompareSetup(f"unhandled fp8 activation scale {scale}", code="compare_plan")


class Fp8Linear(nn.Module):
    """One frozen FP8 linear. ``weight`` stays ``float8_e4m3fn``."""

    def __init__(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
        act_scale: str,
    ) -> None:
        super().__init__()
        if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
            raise CompareSetup("fp8 native weight must be a float8 matrix", code="compare_weights")
        out_features, in_features = weight.shape
        if in_features % 16 != 0 or out_features % 16 != 0:
            raise CompareSetup("fp8 native K and N must be multiples of 16", code="compare_weights")
        if act_scale not in _ACT_SCALES:
            raise CompareSetup(f"fp8_act_scale must be one of {', '.join(_ACT_SCALES)}", code="compare_plan")
        self.act_scale = act_scale
        self.register_buffer("weight", weight.detach(), persistent=False)
        self.register_buffer(
            "weight_scale",
            scale.detach().to(dtype=torch.float32).reshape(()),
            persistent=False,
        )
        if bias is None:
            self.register_buffer("bias", None, persistent=False)
        else:
            self.register_buffer("bias", bias.detach().to(dtype=torch.bfloat16), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """FP8 matmul. ``M`` is padded to a multiple of 16 and sliced back."""
        leading = inputs.shape[:-1]
        flat = inputs.reshape(-1, inputs.shape[-1])
        rows, _cols = flat.shape
        scale_a = _activation_scale(flat, self.act_scale)
        scaled = (flat.to(dtype=torch.float32) / scale_a).clamp(-FP8_MAX, FP8_MAX)
        activation = scaled.to(dtype=torch.float8_e4m3fn)
        pad = (16 - (rows % 16)) % 16
        if pad:
            padded = torch.zeros((rows + pad, activation.shape[1]), dtype=activation.dtype, device=activation.device)
            padded[:rows] = activation
            activation = padded
        output = fp8_mm(
            activation,
            self.weight.t(),
            scale_a,
            self.weight_scale,
            self.bias,
            out_dtype=torch.bfloat16,
        )
        if pad:
            output = output[:rows]
        return output.reshape(*leading, output.shape[-1])


def fp8_mm(
    activation: torch.Tensor,
    weight_t: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """One scaled FP8 matmul. ``weight_t`` is the column-major mat2 ``_scaled_mm`` expects.

    The public ``scaled_mm`` is used when its signature takes per-tensor scales.
    Torch 2.14's public function takes scale recipes instead, so this wrapper
    calls ``torch._scaled_mm`` with ``use_fast_accum=True``.
    """
    kwargs: dict[str, object] = {
        "scale_a": scale_a,
        "scale_b": scale_b,
        "out_dtype": out_dtype,
        "use_fast_accum": True,
    }
    if bias is not None:
        kwargs["bias"] = bias
    public = getattr(torch.nn.functional, "scaled_mm", None)
    if public is not None and _simple_scaled_mm(public):
        result = public(activation, weight_t, **kwargs)
    else:
        result = torch._scaled_mm(activation, weight_t, **kwargs)
    if isinstance(result, tuple):
        return result[0]
    return result


def apply_fp8_native(
    module: nn.Module,
    fp8_state: dict[str, object],
    *,
    act_scale: str = "dynamic",
    require_full: bool = False,
) -> dict[str, object]:
    """Replace eligible linears and load the other tensors as BF16.

    A linear whose ``K`` or ``N`` is not a multiple of 16 stays an ``nn.Linear``
    with the dequantised BF16 weight. When the module already holds real base
    weights, at least half of its weight keys must map. ``require_full`` is the
    from-config shell: every parameter and persistent buffer must be filled,
    and a shape mismatch is ``compare_weights``.
    """
    if act_scale not in _ACT_SCALES:
        raise CompareSetup(f"fp8_act_scale must be one of {', '.join(_ACT_SCALES)}", code="compare_plan")
    weight_keys = [key for key in module.state_dict() if key.endswith(".weight") and not key.endswith(".scale_weight")]
    expected = {key for key, value in module.state_dict().items() if isinstance(value, torch.Tensor)}
    linears = [(name, child) for name, child in module.named_modules() if isinstance(child, nn.Linear)]
    consumed: set[str] = set()
    filled: set[str] = set()
    mismatched: list[str] = []
    swapped = 0
    fallback = 0
    mapped = 0
    first: Fp8Linear | None = None
    for qual, linear in linears:
        module_key = f"{qual}.weight" if qual else "weight"
        bias_name = f"{qual}.bias" if qual else "bias"
        file_key = _file_key(module_key, fp8_state)
        if file_key is None:
            continue
        value = fp8_state[file_key]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float8_e4m3fn:
            if require_full and isinstance(value, torch.Tensor) and tuple(value.shape) != tuple(linear.weight.shape):
                mismatched.append(module_key)
                consumed.add(file_key)
            continue
        if require_full and tuple(value.shape) != tuple(linear.weight.shape):
            mismatched.append(module_key)
            consumed.add(file_key)
            consumed.add(f"{file_key}.scale_weight")
            continue
        scale = _scale_for(file_key, fp8_state)
        consumed.add(file_key)
        consumed.add(f"{file_key}.scale_weight")
        bias_key = _file_key(bias_name, fp8_state)
        bias = fp8_state.get(bias_key) if bias_key else None
        bias_tensor = bias if isinstance(bias, torch.Tensor) else None
        if (
            require_full
            and bias_key
            and isinstance(bias, torch.Tensor)
            and linear.bias is not None
            and tuple(bias.shape) != tuple(linear.bias.shape)
        ):
            mismatched.append(bias_name)
            consumed.add(bias_key)
            bias_tensor = None
        out_features, in_features = value.shape
        if in_features % 16 == 0 and out_features % 16 == 0:
            layer = Fp8Linear(value, scale, bias_tensor, act_scale)
            _set_submodule(module, qual, layer)
            if first is None:
                first = layer
            swapped += 1
        else:
            weight = value.detach().to(dtype=torch.bfloat16) * scale.detach().to(dtype=torch.bfloat16)
            linear.weight = nn.Parameter(weight, requires_grad=False)
            if bias_tensor is not None and linear.bias is not None:
                linear.bias = nn.Parameter(bias_tensor.detach().to(dtype=torch.bfloat16), requires_grad=False)
            fallback += 1
        if bias_key:
            consumed.add(bias_key)
        mapped += 1
        if require_full:
            filled.add(module_key)
            if bias_tensor is not None and linear.bias is not None:
                filled.add(bias_name)
    mapped += _load_remaining(
        module,
        fp8_state,
        consumed,
        filled=filled if require_full else None,
        mismatched=mismatched if require_full else None,
    )
    if require_full:
        bad = set(mismatched)
        missing = [key for key in expected if key not in filled and key not in bad]
        if missing or mismatched:
            _shell_coverage_error(missing, mismatched)
    elif mapped < max(1, len(weight_keys) // 2):
        raise CompareSetup(f"fp8 native matched {mapped} tensors", code="compare_weights")
    return {
        "layers_swapped": swapped,
        "layers_fallback": fallback,
        "mapped": mapped,
        "first_layer": first,
    }


def self_check(layer: Fp8Linear) -> None:
    """Compare one forward with ``x @ (w.bf16 * scale).T``.

    Relative error above ``3e-2`` is exit 2 (``compare_fp8_native``). Call this
    after the arm is on the GPU so a layout or scale bug stops the job before
    the image loop.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(0)
    sample = torch.randn(16, layer.weight.shape[1], generator=generator, dtype=torch.bfloat16)
    sample = sample.to(device=layer.weight.device)
    with torch.no_grad():
        got = layer(sample)
        weight = layer.weight.detach().to(dtype=torch.float32) * layer.weight_scale.detach().to(dtype=torch.float32)
        reference = sample.to(dtype=torch.float32) @ weight.t()
        if layer.bias is not None:
            reference = reference + layer.bias.detach().to(dtype=torch.float32)
        denom = reference.abs().mean().clamp(min=1e-6)
        relative = float((got.to(dtype=torch.float32) - reference).abs().mean() / denom)
    if relative > 3e-2:
        raise CompareSetup(
            f"fp8 native self-check relative error {relative:.4f}",
            code="compare_fp8_native",
        )


def _activation_scale(flat: torch.Tensor, act_scale: str) -> torch.Tensor:
    if act_scale == "unit":
        return torch.ones((), dtype=torch.float32, device=flat.device)
    if act_scale == "dynamic":
        peak = flat.detach().abs().amax().to(dtype=torch.float32)
        tiny = torch.finfo(torch.float32).tiny
        return torch.clamp(peak / FP8_MAX, min=tiny)
    return _unexpected_act(act_scale)


def _simple_scaled_mm(public: object) -> bool:
    try:
        parameters = inspect.signature(public).parameters
    except (TypeError, ValueError):
        return False
    return "scale_recipe_a" not in parameters and "scale_a" in parameters


def _file_key(module_key: str, state: dict[str, object]) -> str | None:
    probe = {module_key: None}
    for key in state:
        if not isinstance(key, str) or key.endswith(".scale_weight"):
            continue
        if alias_key(key, probe) == module_key:
            return key
    return None


def _scale_for(file_key: str, state: dict[str, object]) -> torch.Tensor:
    scale = state.get(f"{file_key}.scale_weight")
    if not isinstance(scale, torch.Tensor):
        raise CompareSetup(f"{file_key} is fp8 without a scale", code="compare_weights")
    return scale


def _set_submodule(module: nn.Module, qual: str, child: nn.Module) -> None:
    if not qual:
        raise CompareSetup("fp8 native cannot replace the root module", code="compare_weights")
    parent: nn.Module = module
    parts = qual.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], child)


def _shell_coverage_error(missing: list[str], mismatched: list[str]) -> None:
    """Raise ``compare_weights`` naming keys a from-config shell did not fill."""
    missing_keys = sorted(set(missing))
    mismatch_keys = sorted(set(mismatched))
    parts: list[str] = []
    if missing_keys:
        shown = ", ".join(missing_keys[:12])
        suffix = f" ({len(missing_keys)} keys)" if len(missing_keys) > 12 else ""
        parts.append(f"missing {shown}{suffix}")
    if mismatch_keys:
        shown = ", ".join(mismatch_keys[:12])
        suffix = f" ({len(mismatch_keys)} keys)" if len(mismatch_keys) > 12 else ""
        parts.append(f"shape mismatch {shown}{suffix}")
    raise CompareSetup(
        "fp8 shell did not cover the transformer: " + "; ".join(parts),
        code="compare_weights",
    )


def _load_remaining(
    module: nn.Module,
    state: dict[str, object],
    consumed: set[str],
    *,
    filled: set[str] | None = None,
    mismatched: list[str] | None = None,
) -> int:
    current = module.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    weight_hits = 0
    for raw_key, value in state.items():
        if not isinstance(raw_key, str) or raw_key in consumed or raw_key.endswith(".scale_weight"):
            continue
        if not isinstance(value, torch.Tensor) or value.dtype == torch.float8_e4m3fn:
            continue
        key = alias_key(raw_key, current)
        if key is None or key not in current:
            continue
        target = current[key]
        tensor = value.detach()
        if tensor.shape != target.shape:
            if mismatched is not None:
                mismatched.append(key)
            continue
        if tensor.is_floating_point() and tensor.dtype != target.dtype:
            tensor = tensor.to(dtype=target.dtype)
        mapped[key] = tensor
        if filled is not None:
            filled.add(key)
        if key.endswith(".weight"):
            weight_hits += 1
    if mapped:
        module.load_state_dict(mapped, strict=False)
    return weight_hits
