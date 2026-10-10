"""VRAM and RAM splits tried by the fit stage before a smaller quant is accepted.

Order is the preference order. ``gpu`` keeps weights and KV on the card.
The later splits move something into RAM: KV (``--no-kv-offload``), MoE experts
(``--n-cpu-moe``), a diffusion text encoder, or sequential loading so the peak
is one component rather than the sum. An offload that fits is preferred to a
lower quant whose only advantage is that it passed the on-GPU gate.
"""

from __future__ import annotations

from dataclasses import dataclass

from voxprint_forge.fitmath import GIB, budget_bytes


@dataclass(frozen=True)
class Split:
    """One VRAM/RAM placement: name, device bytes, RAM bytes, and llama-server flags."""
    name: str
    vram_bytes: int
    ram_bytes: int
    flags: tuple[str, ...]


def llm_splits(weights: int, kv: int, *, moe_layers: int = 0, expert_fraction: float = 0.0) -> tuple[Split, ...]:
    """GPU, KV-in-RAM, and (for MoE) ``--n-cpu-moe``."""
    found = [
        Split("gpu", weights + kv, 0, ()),
        Split("kv_ram", weights, kv, ("--no-kv-offload",)),
    ]
    if moe_layers > 0 and expert_fraction > 0:
        moved = int(weights * expert_fraction)
        experts = max(1, int(round(moe_layers * expert_fraction)))
        found.append(
            Split(
                "n_cpu_moe",
                weights - moved + kv,
                moved,
                ("--n-cpu-moe", str(experts)),
            )
        )
    return tuple(found)


def diffusion_splits(transformer: int, encoder: int, vae: int = 0) -> tuple[Split, ...]:
    """Full load, encoder in RAM, and sequential load (peak is the larger resident set)."""
    together = transformer + encoder + vae
    without_encoder = transformer + vae
    peak = max(encoder, without_encoder)
    return (
        Split("gpu", together, 0, ()),
        Split("encoder_offload", without_encoder, encoder, ()),
        Split("sequential", peak, together - peak, ()),
    )


def options_for(
    weights: int,
    kv: int,
    *,
    moe_layers: int = 0,
    expert_fraction: float = 0.0,
    encoder_gib: float = 0.0,
) -> tuple[Split, ...]:
    """Preference order for one quant: GPU, then KV in RAM, MoE experts, encoder, sequential.

    Diffusion splits do not include the KV cache. The cache stays on the card with
    the transformer, so those options gain ``kv`` bytes of VRAM here.
    """
    found = list(llm_splits(weights, kv, moe_layers=moe_layers, expert_fraction=expert_fraction))
    if encoder_gib > 0:
        encoder = int(encoder_gib * GIB)
        transformer = max(0, weights - encoder)
        for split in diffusion_splits(transformer, encoder):
            if split.name == "gpu":
                continue
            found.append(Split(split.name, split.vram_bytes + kv, split.ram_bytes, split.flags))
    return tuple(found)


def split_fits(split: Split, *, budget_gb: float, headroom_gb: float, ram_bytes: int) -> bool:
    """True when device bytes fit the card and RAM bytes fit the machine.

    The card limit is ``budget_bytes``, which already subtracts headroom.
    This is the formula the fit stage uses before it measures a split.
    """
    if split.vram_bytes > budget_bytes(budget_gb, headroom_gb):
        return False
    return split.ram_bytes <= ram_bytes


def first_fitting(
    splits: tuple[Split, ...] | list[Split],
    *,
    budget_gb: float,
    headroom_gb: float,
    ram_bytes: int,
) -> Split | None:
    """Prefer ``gpu``. The next fitting offload wins over trying a smaller quant."""
    for split in splits:
        if split_fits(split, budget_gb=budget_gb, headroom_gb=headroom_gb, ram_bytes=ram_bytes):
            return split
    return None
