"""PEFT-style LoRA layers for ``nn.Linear`` modules of a diffusion transformer."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import torch
from torch import nn


class LoRALinear(nn.Module):
    """``base(x) + scale * B(A(x))``. ``B`` starts at zero, so training starts from the base."""

    def __init__(self, base: nn.Linear, rank: int, alpha: int, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        weight = base.weight
        if dtype is None:
            dtype = weight.dtype if weight.dtype in (torch.bfloat16, torch.float32) else None
            # FP16 and FP8 storage (16 GB tier) keep the adapter in a trainable dtype.
            dtype = dtype or (torch.float32 if weight.dtype == torch.float16 else torch.bfloat16)
        self.lora_A.to(device=weight.device, dtype=dtype)
        self.lora_B.to(device=weight.device, dtype=self.lora_A.weight.dtype)

    @property
    def weight(self) -> torch.Tensor:
        """The frozen base weight. Callers that read ``linear.weight`` keep working."""
        return self.base.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Base output plus the scaled low-rank update."""
        out = self.base(x)
        delta = self.lora_B(self.lora_A(x.to(self.lora_A.weight.dtype))) * self.scale
        return out + delta.to(out.dtype)


def inject(
    model: nn.Module, targets: Iterable[str], rank: int, alpha: int, dtype: torch.dtype | None = None
) -> dict[str, LoRALinear]:
    """Wrap every ``nn.Linear`` whose name ends with a target suffix. Freezes the base weights."""
    suffixes = tuple(targets)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wanted = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name and any(name == s or name.endswith("." + s) for s in suffixes)
    ]
    wrapped: dict[str, LoRALinear] = {}
    for name in wanted:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        layer = LoRALinear(getattr(parent, child), rank, alpha, dtype=dtype)
        setattr(parent, child, layer)
        wrapped[name] = layer
    if not wrapped:
        raise ValueError("no LoRA target modules matched")
    return wrapped


def lora_parameters(layers: Mapping[str, LoRALinear]) -> list[nn.Parameter]:
    """Trainable parameters, in a stable order."""
    params: list[nn.Parameter] = []
    for name in sorted(layers):
        params += [layers[name].lora_A.weight, layers[name].lora_B.weight]
    return params


def lora_state(layers: Mapping[str, LoRALinear]) -> dict[str, torch.Tensor]:
    """``<module>.lora_A.weight`` / ``<module>.lora_B.weight`` tensors on the CPU, contiguous."""
    state: dict[str, torch.Tensor] = {}
    for name in sorted(layers):
        state[f"{name}.lora_A.weight"] = layers[name].lora_A.weight.detach().to("cpu").contiguous()
        state[f"{name}.lora_B.weight"] = layers[name].lora_B.weight.detach().to("cpu").contiguous()
    return state


def load_lora_state(layers: Mapping[str, LoRALinear], state: Mapping[str, torch.Tensor]) -> None:
    """Copy saved tensors into the layers. Missing or misshaped tensors raise ``ValueError``."""
    expected = set(lora_state(layers))
    if set(state) != expected:
        raise ValueError("LoRA tensors do not match this base model")
    for name, layer in layers.items():
        for part in ("lora_A", "lora_B"):
            target = getattr(layer, part).weight
            source = state[f"{name}.{part}.weight"]
            if tuple(source.shape) != tuple(target.shape):
                raise ValueError("LoRA tensors do not match this base model")
            with torch.no_grad():
                target.copy_(source.to(target.dtype))
