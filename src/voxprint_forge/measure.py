"""Pick a quant that fits a card budget, then parse llama-server buffer sizes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from voxprint_forge.errors import StageError
from voxprint_forge.fitmath import Arch, fits_budget, gib, min_vram_gb, quant_rank
from voxprint_forge.plan import Plan, Tier

# b11476 prints "%12s model buffer size" and "%10s KV buffer size" with the backend name.
# CPU_Mapped is often first. CUDA_Host is host memory and is not the card.
_DEVICE_MODEL = re.compile(
    r"(?P<device>[A-Za-z0-9_]+)\s+model buffer size\s*=\s*(?P<mib>[0-9]+(?:\.[0-9]+)?)\s*MiB",
    re.IGNORECASE,
)
_DEVICE_KV = re.compile(
    r"(?P<device>[A-Za-z0-9_]+)\s+KV buffer size\s*=\s*(?P<mib>[0-9]+(?:\.[0-9]+)?)\s*MiB",
    re.IGNORECASE,
)
_MIB = 1024 * 1024


@dataclass(frozen=True)
class Candidate:
    quant: str
    weights: int
    path: Path
    origin: str


@dataclass(frozen=True)
class BufferReport:
    """CUDA device totals. `devices` is one entry per CUDA* buffer, excluding *_Host."""

    weights_bytes: int
    kv_bytes: int
    devices: dict[str, dict[str, float]]


@dataclass(frozen=True)
class Measurement:
    ctx: int
    cache: str
    weights_bytes: int
    kv_bytes: int
    computed_kv_bytes: int
    nvidia_used_bytes: int | None
    fits: bool


def fit_contexts(tier: Tier, plan: Plan) -> tuple[int, ...]:
    """16 GB is gated at the shortest context. 24 GB must fit every context."""
    if not plan.contexts:
        raise StageError("plan has no contexts", code="measure_ctx")
    if tier.budget_gb <= 16:
        return (min(plan.contexts),)
    return plan.contexts


def fitting_candidates(
    candidates: list[Candidate],
    arch: Arch,
    contexts: tuple[int, ...],
    *,
    budget_gb: float,
    headroom_gb: float,
    cache: str,
) -> list[Candidate]:
    ranked = sorted(candidates, key=lambda item: quant_rank(item.quant), reverse=True)
    chosen: list[Candidate] = []
    for cand in ranked:
        ok = all(
            fits_budget(
                cand.weights,
                arch.kv_bytes(ctx, cache),
                budget_gb=budget_gb,
                headroom_gb=headroom_gb,
            )
            for ctx in contexts
        )
        if ok:
            chosen.append(cand)
    return chosen


def choose_cache(
    candidates: list[Candidate],
    arch: Arch,
    contexts: tuple[int, ...],
    *,
    budget_gb: float,
    headroom_gb: float,
) -> tuple[str, list[Candidate]]:
    for cache in ("q8_0", "q4_0"):
        fitting = fitting_candidates(
            candidates,
            arch,
            contexts,
            budget_gb=budget_gb,
            headroom_gb=headroom_gb,
            cache=cache,
        )
        if fitting:
            return cache, fitting
    return "q4_0", []


def _cuda_device(name: str) -> bool:
    folded = name.lower()
    if folded.endswith("_host"):
        return False
    return folded.startswith("cuda")


def _mib_bytes(mib: float) -> int:
    return int(round(mib * _MIB))


def parse_buffers(log: str) -> BufferReport:
    """Sum CUDA* model and KV buffers. CPU_Mapped and *_Host are not the card."""
    weights: dict[str, float] = {}
    kv: dict[str, float] = {}
    for match in _DEVICE_MODEL.finditer(log):
        device = match.group("device")
        if not _cuda_device(device):
            continue
        weights[device] = weights.get(device, 0.0) + float(match.group("mib"))
    for match in _DEVICE_KV.finditer(log):
        device = match.group("device")
        if not _cuda_device(device):
            continue
        kv[device] = kv.get(device, 0.0) + float(match.group("mib"))
    if not weights or not kv:
        raise StageError("llama-server log has no CUDA model/KV buffer sizes", code="measure_log")
    names = sorted(set(weights) | set(kv))
    devices = {
        name: {"weights_mib": weights.get(name, 0.0), "kv_mib": kv.get(name, 0.0)}
        for name in names
    }
    return BufferReport(
        weights_bytes=_mib_bytes(sum(weights.values())),
        kv_bytes=_mib_bytes(sum(kv.values())),
        devices=devices,
    )


def measurement_fits(
    weights_bytes: int,
    kv_bytes: int,
    *,
    budget_gb: float,
    headroom_gb: float,
) -> bool:
    return fits_budget(weights_bytes, kv_bytes, budget_gb=budget_gb, headroom_gb=headroom_gb)


def tier_min_vram(weights_bytes: int, kv_bytes: int, headroom_gb: float) -> int:
    return min_vram_gb(gib(weights_bytes + kv_bytes), headroom_gb)
