"""Read llama.cpp b11476 KL statistics from a perplexity log."""

from __future__ import annotations

import re
from dataclasses import dataclass

from voxprint_forge.errors import StageError

_PPL_Q = re.compile(r"Mean PPL\(Q\)\s*:\s*([0-9]+(?:\.[0-9]+)?)")
_PPL_BASE = re.compile(r"Mean PPL\(base\)\s*:\s*([0-9]+(?:\.[0-9]+)?)")
_KL_MEAN = re.compile(r"Mean\s+KLD:\s*([0-9]+(?:\.[0-9]+)?)")
_KL_P99 = re.compile(r"99\.0%\s+KLD:\s*([0-9]+(?:\.[0-9]+)?)")
_RMS = re.compile(r"RMS Δp\s*:\s*([0-9]+(?:\.[0-9]+)?)")
_TOP1 = re.compile(r"Same top p:\s*([0-9]+(?:\.[0-9]+)?)")
_TPS = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s+tokens per second")


@dataclass(frozen=True)
class KlStats:
    """One quant versus the F16 base. Percents are the figures llama.cpp prints."""

    ppl_quant: float
    ppl_base: float
    kl_mean: float
    kl_p99: float
    top1_agreement_pct: float
    delta_p_rms_pct: float
    tokens_per_second: float | None


def tokens_per_second(log: str) -> float | None:
    """First finite tokens/s figure in a llama.cpp perf print, or None."""
    for match in _TPS.finditer(log):
        value = float(match.group(1))
        if value > 0:
            return value
    return None


def parse_kl(log: str) -> KlStats:
    """Parse the KL block llama-perplexity prints when ``--kl-divergence`` is set."""
    ppl_quant = _required(_PPL_Q, log, "Mean PPL(Q)")
    ppl_base = _required(_PPL_BASE, log, "Mean PPL(base)")
    kl_mean = _required(_KL_MEAN, log, "Mean KLD")
    kl_p99 = _required(_KL_P99, log, "99.0% KLD")
    rms = _required(_RMS, log, "RMS delta-p")
    top1 = _required(_TOP1, log, "Same top p")
    return KlStats(
        ppl_quant=ppl_quant,
        ppl_base=ppl_base,
        kl_mean=kl_mean,
        kl_p99=kl_p99,
        top1_agreement_pct=top1,
        delta_p_rms_pct=rms,
        tokens_per_second=tokens_per_second(log),
    )


def _required(pattern: re.Pattern[str], log: str, label: str) -> float:
    match = pattern.search(log)
    if match is None:
        raise StageError(f"KL log has no {label} line", code="kl_log")
    return float(match.group(1))
