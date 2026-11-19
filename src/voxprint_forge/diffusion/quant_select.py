"""Pick a MOSS quant mode for one GPU and one VRAM budget.

This is the Forge-local model/quant selector. It is planned to move to
voxprint-common with the other shared model pieces. Callers should import it
from here until that move, so the import has one place to change.

The selector does not load weights and does not download. ``forge moss-prequant``
measures the candidates and calls :func:`choose`.
"""

from __future__ import annotations

from dataclasses import dataclass


class QuantSelectError(Exception):
    """No quant mode fits this GPU and budget.

    Forge raises ``NoQuantFits`` (``moss_prequant_none_fit``) from this.
    The selector stays free of Forge error types so it can move to voxprint-common.
    """

    code = "moss_prequant_none_fit"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.code = "moss_prequant_none_fit"


@dataclass(frozen=True)
class Gpu:
    """One card: display name, memory in decimal GB, and compute capability times 10 (89 means sm_89)."""

    name: str
    vram_gb: float
    sm: int


@dataclass(frozen=True)
class Mode:
    """A candidate quant. ``min_sm`` is the compute capability times 10 (fp8_e4m3 needs 89)."""

    name: str
    min_sm: int
    backend: str


@dataclass(frozen=True)
class Candidate:
    """One measured mode. ``peak_gb`` is the larger of the generation and train peaks."""

    mode: str
    min_sm: int
    generate_gb: float
    train_gb: float
    lpips: float
    clip: float

    @property
    def peak_gb(self) -> float:
        """The peak that must fit the budget and the card."""
        return max(self.generate_gb, self.train_gb)


# fp8_e4m3 needs Ada (sm_89) or newer. int8 and nf4 are bitsandbytes weight-only
# modes and run on the cards we ship for (Ampere sm_86 and up).
MODES: tuple[Mode, ...] = (
    Mode("fp8_e4m3", 89, "torchao"),
    Mode("int8", 70, "bitsandbytes"),
    Mode("nf4", 70, "bitsandbytes"),
)

# The four cards the selector tests name. Memory is the card's total, in GB.
KNOWN_GPUS: tuple[Gpu, ...] = (
    Gpu("RTX 3060", 12.0, 86),
    Gpu("RTX 3080 Laptop", 16.0, 86),
    Gpu("RTX 4080", 16.0, 89),
    Gpu("RTX 4090", 24.0, 89),
)


def mode_by_name(name: str) -> Mode:
    """The mode named ``name``. Raises ``KeyError`` when it is not a candidate."""
    for mode in MODES:
        if mode.name == name:
            return mode
    raise KeyError(name)


def known_gpu(name: str) -> Gpu:
    """A card from :data:`KNOWN_GPUS`. ``name`` matches the display name."""
    for gpu in KNOWN_GPUS:
        if gpu.name == name:
            return gpu
    raise KeyError(name)


def fits(candidate: Candidate, gpu: Gpu, budget_gb: float) -> bool:
    """True when ``candidate`` is allowed on ``gpu`` and its peak is within the budget and the card."""
    if gpu.sm < candidate.min_sm:
        return False
    return candidate.peak_gb <= min(float(budget_gb), float(gpu.vram_gb))


def skip_reason(candidate: Candidate, gpu: Gpu, budget_gb: float) -> str:
    """Why ``candidate`` does not fit, or ``fits`` when it does."""
    if gpu.sm < candidate.min_sm:
        return f"{candidate.mode} needs sm>={candidate.min_sm}, {gpu.name} is sm {gpu.sm}"
    limit = min(float(budget_gb), float(gpu.vram_gb))
    if candidate.peak_gb > limit:
        return f"{candidate.mode} peak {candidate.peak_gb:.2f} GB is over {limit:.2f} GB"
    return "fits"


def choose(candidates: tuple[Candidate, ...] | list[Candidate], gpu: Gpu, budget_gb: float) -> Candidate:
    """Best quality among modes that fit.

    Lower LPIPS wins, then higher CLIP, then the mode name. Raises
    :class:`QuantSelectError` when none fit. fp8 is absent from ``candidates``
    when the caller already skipped it for ``sm < 89``.
    """
    fitting = [item for item in candidates if fits(item, gpu, budget_gb)]
    if not fitting:
        detail = "; ".join(skip_reason(item, gpu, budget_gb) for item in candidates) or "no candidates"
        raise QuantSelectError(
            f"no quant fits {gpu.name} ({gpu.vram_gb:g} GB, sm {gpu.sm}) at budget {budget_gb:g} GB: {detail}"
        )
    return min(fitting, key=lambda item: (item.lpips, -item.clip, item.mode))
