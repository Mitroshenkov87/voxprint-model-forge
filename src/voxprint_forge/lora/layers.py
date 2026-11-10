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
        # Frozen de-distillation adapter (Turbo only). Never trained, never saved.
        self.training_A: nn.Linear | None = None
        self.training_B: nn.Linear | None = None
        self.training_lora_scale = 1.0
        self.training_adapter_scale = 0.0

    @property
    def has_training_adapter(self) -> bool:
        """True when a frozen training adapter is attached to this layer."""
        return self.training_A is not None and self.training_B is not None

    @property
    def weight(self) -> torch.Tensor:
        """The frozen base weight. Callers that read ``linear.weight`` keep working."""
        return self.base.weight

    def set_training_adapter(
        self,
        a_weight: torch.Tensor,
        b_weight: torch.Tensor,
        *,
        scale: float = 1.0,
        lora_scale: float = 1.0,
    ) -> None:
        """Attach a frozen LoRA (A, B) used only while training. Gradients stay off."""
        a_weight, b_weight = _orient(a_weight, b_weight, self.base.in_features, self.base.out_features)
        self.training_A = nn.Linear(self.base.in_features, a_weight.shape[0], bias=False)
        self.training_B = nn.Linear(a_weight.shape[0], self.base.out_features, bias=False)
        with torch.no_grad():
            self.training_A.weight.copy_(a_weight.to(dtype=self.training_A.weight.dtype))
            self.training_B.weight.copy_(b_weight.to(dtype=self.training_B.weight.dtype))
        device = self.base.weight.device
        dtype = self.lora_A.weight.dtype
        self.training_A.to(device=device, dtype=dtype)
        self.training_B.to(device=device, dtype=dtype)
        self.training_A.requires_grad_(False)
        self.training_B.requires_grad_(False)
        self.training_lora_scale = float(lora_scale)
        self.training_adapter_scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Base output plus the scaled low-rank update (and a frozen training adapter, if any)."""
        out = self.base(x)
        if self.has_training_adapter and self.training_adapter_scale:
            assert self.training_A is not None and self.training_B is not None
            frozen = self.training_B(self.training_A(x.to(self.training_A.weight.dtype)))
            out = out + (frozen * self.training_lora_scale * self.training_adapter_scale).to(out.dtype)
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



def _orient(
    a_weight: torch.Tensor, b_weight: torch.Tensor, in_features: int, out_features: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(A, B)`` as ``[rank, in]`` and ``[out, rank]``. Some files store A transposed."""
    if a_weight.ndim != 2 or b_weight.ndim != 2:
        raise ValueError("LoRA tensors do not match this base model")
    if a_weight.shape[1] == in_features and b_weight.shape[0] == out_features:
        return a_weight, b_weight
    if a_weight.shape[0] == in_features and b_weight.shape[1] == out_features:
        return a_weight.T.contiguous(), b_weight.T.contiguous()
    raise ValueError("LoRA tensors do not match this base model")
