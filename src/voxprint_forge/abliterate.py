"""Difference-of-means refusal-direction ablation.

One direction is the normalised gap between mean residual activations on
harmful prompts and harmless prompts, taken at the layer where that gap is
largest. Writing matrices (attention output and MLP down projection) are
orthogonalised against it: W := W - r r^T W.

On an 80 GB GPU a 24-33B bf16 model can stay resident. The layer path still
moves one decoder block at a time so a 24 GB card can run the same code.
Heretic is not imported or invoked.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from voxprint_forge.errors import StageError

WRITE_LEAVES = frozenset({"o_proj", "out_proj", "down_proj", "wo", "dense_4h_to_h"})
Encode = Callable[[str], torch.Tensor]


@dataclass(frozen=True)
class AblationResult:
    layer: int
    direction_norm: float
    matrices: int
    mode: str


def orthogonalize_weight(weight: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Project the refusal direction out of a [out, in] matrix."""
    if weight.ndim != 2:
        raise StageError("orthogonalize expects a 2D weight", code="abliterate_shape")
    hidden = direction.shape[0]
    if weight.shape[0] != hidden:
        return weight
    unit = direction.to(dtype=weight.dtype, device=weight.device)
    projected = torch.outer(unit, unit @ weight)
    return weight - projected


def difference_of_means(harmful: torch.Tensor, harmless: torch.Tensor) -> torch.Tensor:
    if harmful.ndim != 2 or harmless.ndim != 2:
        raise StageError("activations must be [n, hidden]", code="abliterate_shape")
    delta = harmful.mean(dim=0) - harmless.mean(dim=0)
    norm = torch.linalg.vector_norm(delta)
    if float(norm) == 0.0:
        raise StageError("refusal direction has zero length", code="abliterate_direction")
    return delta / norm


def select_direction(
    harmful_by_layer: dict[int, torch.Tensor],
    harmless_by_layer: dict[int, torch.Tensor],
) -> tuple[int, torch.Tensor]:
    best_layer = -1
    best_norm = -1.0
    best: torch.Tensor | None = None
    for layer in sorted(harmful_by_layer):
        if layer not in harmless_by_layer:
            continue
        delta = harmful_by_layer[layer].mean(dim=0) - harmless_by_layer[layer].mean(dim=0)
        norm = float(torch.linalg.vector_norm(delta))
        if norm > best_norm:
            best_norm = norm
            best_layer = layer
            best = delta / delta.norm()
    if best is None:
        raise StageError("no layer activations to ablate", code="abliterate_empty")
    return best_layer, best


def decoder_layers(model: nn.Module) -> list[nn.Module]:
    layers = getattr(model, "layers", None)
    if isinstance(layers, nn.ModuleList):
        return list(layers)
    inner = getattr(model, "model", None)
    if inner is not None:
        nested = getattr(inner, "layers", None)
        if isinstance(nested, nn.ModuleList):
            return list(nested)
    raise StageError("model has no decoder ModuleList named layers", code="abliterate_arch")


def apply_orthogonalization(model: nn.Module, direction: torch.Tensor) -> int:
    count = 0
    hidden = direction.shape[0]
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            leaf = name.rsplit(".", 1)[-1]
            if leaf not in WRITE_LEAVES:
                continue
            if module.weight.shape[0] != hidden:
                continue
            updated = orthogonalize_weight(module.weight.data, direction)
            module.weight.copy_(updated)
            count += 1
    if count == 0:
        raise StageError("no o_proj/down_proj matrices matched the direction", code="abliterate_nomatch")
    return count


def _move(value: object, device: torch.device) -> object:
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def collect_last_token(
    model: nn.Module,
    encode: Encode,
    prompts: Sequence[str],
    *,
    device: torch.device,
    mode: str,
) -> dict[int, torch.Tensor]:
    layers = decoder_layers(model)
    captured: dict[int, list[torch.Tensor]] = {index: [] for index in range(len(layers))}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def make_layer_hooks(index: int, layer: nn.Module) -> None:
        def pre(
            _module: nn.Module,
            args: tuple[object, ...],
            kwargs: dict[str, object],
        ) -> tuple[tuple[object, ...], dict[str, object]]:
            layer.to(device)
            moved_args = tuple(_move(arg, device) for arg in args)
            moved_kwargs = {key: _move(val, device) for key, val in kwargs.items()}
            return moved_args, moved_kwargs

        def post(_module: nn.Module, _args: tuple[object, ...], output: object) -> object:
            hidden = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(hidden):
                raise StageError("layer output has no hidden state", code="abliterate_output")
            captured[index].append(hidden[:, -1, :].detach().float().cpu())
            layer.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if isinstance(output, tuple):
                return (_move(output[0], torch.device("cpu")),) + output[1:]
            return _move(output, torch.device("cpu"))

        hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        hooks.append(layer.register_forward_hook(post))

    if mode == "layer":
        for index, layer in enumerate(layers):
            make_layer_hooks(index, layer)
    elif mode == "resident":
        model.to(device)

        def remember(_module: nn.Module, _args: tuple[object, ...], output: object, index: int) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(hidden):
                captured[index].append(hidden[:, -1, :].detach().float().cpu())

        for index, layer in enumerate(layers):
            hooks.append(layer.register_forward_hook(lambda _m, _a, out, i=index: remember(_m, _a, out, i)))
    else:
        raise StageError(f"unknown ablation mode {mode}", code="abliterate_mode")

    try:
        for prompt in prompts:
            token_ids = encode(prompt)
            if token_ids.ndim == 1:
                token_ids = token_ids.unsqueeze(0)
            token_ids = token_ids.to(device if mode == "resident" else torch.device("cpu"))
            with torch.inference_mode():
                model(token_ids)
    finally:
        for hook in hooks:
            hook.remove()
        if mode == "resident":
            model.to("cpu")
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return {index: torch.cat(rows, dim=0) for index, rows in captured.items() if rows}


def abliterate_module(
    model: nn.Module,
    encode: Encode,
    harmful: Sequence[str],
    harmless: Sequence[str],
    *,
    device: torch.device | None = None,
    mode: str = "layer",
) -> AblationResult:
    if not harmful or not harmless:
        raise StageError("ablation needs harmful and harmless prompts", code="abliterate_prompts")
    chosen = device or torch.device("cpu")
    harmful_acts = collect_last_token(model, encode, harmful, device=chosen, mode=mode)
    harmless_acts = collect_last_token(model, encode, harmless, device=chosen, mode=mode)
    layer, direction = select_direction(harmful_acts, harmless_acts)
    matrices = apply_orthogonalization(model, direction)
    return AblationResult(
        layer=layer,
        direction_norm=float(torch.linalg.vector_norm(direction)),
        matrices=matrices,
        mode=mode,
    )


def gpu_can_hold(weight_bytes: int, free_vram_bytes: int, margin_bytes: int = 6 * 1024**3) -> bool:
    return weight_bytes + margin_bytes < free_vram_bytes


def choose_mode(weight_bytes: int, free_vram_bytes: int | None) -> str:
    if free_vram_bytes is None:
        return "layer"
    if gpu_can_hold(weight_bytes, free_vram_bytes):
        return "resident"
    return "layer"


def save_ablated(model: nn.Module, tokenizer: object, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    save = getattr(model, "save_pretrained", None)
    if not callable(save):
        raise StageError("model cannot save_pretrained", code="abliterate_save")
    save(dest)
    tok_save = getattr(tokenizer, "save_pretrained", None)
    if callable(tok_save):
        tok_save(dest)


def load_hf(source: Path, *, offload_dir: Path | None, dtype: torch.dtype = torch.bfloat16) -> tuple[nn.Module, object]:
    kwargs: dict[str, object] = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if offload_dir is not None:
        offload_dir.mkdir(parents=True, exist_ok=True)
        kwargs["device_map"] = "cpu"
        kwargs["offload_folder"] = str(offload_dir)
    model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(source)
    return model, tokenizer


def encode_with_tokenizer(tokenizer: object) -> Encode:
    def encode(text: str) -> torch.Tensor:
        encoded = tokenizer(text, return_tensors="pt")  # type: ignore[operator]
        return encoded["input_ids"]

    return encode


def abliterate_directory(
    source: Path,
    dest: Path,
    harmful: Sequence[str],
    harmless: Sequence[str],
    *,
    weight_bytes: int,
    free_vram_bytes: int | None,
    offload_dir: Path | None = None,
) -> AblationResult:
    mode = choose_mode(weight_bytes, free_vram_bytes)
    device = torch.device("cuda" if mode == "resident" and torch.cuda.is_available() else "cpu")
    if mode == "layer" and torch.cuda.is_available():
        device = torch.device("cuda")
    model, tokenizer = load_hf(source, offload_dir=offload_dir if mode == "layer" else None)
    result = abliterate_module(
        model,
        encode_with_tokenizer(tokenizer),
        harmful,
        harmless,
        device=device,
        mode=mode,
    )
    save_ablated(model, tokenizer, dest)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
