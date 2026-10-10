"""Pick a quant that fits a card budget, then parse llama-server buffer sizes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from voxprint_forge.errors import StageError
from voxprint_forge.fitmath import Arch, fits_budget, gib, min_vram_gb, quant_rank
from voxprint_forge.plan import Plan, Tier

_MODEL = re.compile(r"model buffer size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*MiB", re.IGNORECASE)
_KV = re.compile(r"KV buffer size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*MiB", re.IGNORECASE)
_MIB = 1024 * 1024


@dataclass(frozen=True)
class Candidate:
    quant: str
    weights: int
    path: Path
    origin: str


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


def parse_buffers(log: str) -> tuple[int, int]:
    model = _MODEL.search(log)
    kv = _KV.search(log)
    if model is None or kv is None:
        raise StageError("llama-server log has no model/KV buffer sizes", code="measure_log")
    return int(float(model.group(1)) * _MIB), int(float(kv.group(1)) * _MIB)


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
