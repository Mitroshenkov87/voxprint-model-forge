"""Image metrics for one original-versus-candidate pair.

SSIM and PSNR use torchmetrics. LPIPS (alex), DINOv2-base cosine, and CLIP
ViT-B/32 are measured by the scorer. The pass bar is the existing Forge gate:
mean LPIPS <= 0.10 and CLIP-T drop <= 1%. The other metrics are reported only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import torch

from voxprint_forge.compare.codes import CompareSetup
from voxprint_forge.diffusion.policy import clip_drop


class PairScore(TypedDict):
    """Metrics for one pair. PSNR is null when the images match exactly."""

    lpips: float
    ssim: float
    psnr: float | None
    psnr_saturated: bool
    dino_cosine: float
    clip_i: float
    clip_t_reference: float
    clip_t_candidate: float
    clip_t_drop: float


def ssim_psnr(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float | None, bool]:
    """SSIM and PSNR on RGB tensors shaped ``(3, H, W)`` with values in 0..1.

    Identical images have unbounded PSNR. That case returns ``None`` and
    ``saturated=True`` so the JSON stays finite.
    """
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

    if reference.shape != candidate.shape:
        raise CompareSetup(
            f"metric size {tuple(reference.shape)} != {tuple(candidate.shape)}",
            code="compare_metric",
        )
    if reference.ndim != 3 or reference.shape[0] != 3:
        raise CompareSetup("metrics expect RGB CHW tensors", code="compare_metric")
    left = reference.detach().float().clamp(0, 1).unsqueeze(0).cpu()
    right = candidate.detach().float().clamp(0, 1).unsqueeze(0).cpu()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    ssim_value = float(ssim_metric(left, right))
    psnr_value = float(psnr_metric(left, right))
    if not torch.isfinite(torch.tensor(psnr_value)):
        return ssim_value, None, True
    return ssim_value, psnr_value, False


def finish_score(
    *,
    lpips: float,
    ssim: float,
    psnr: float | None,
    psnr_saturated: bool,
    dino_cosine: float,
    clip_i: float,
    clip_t_reference: float,
    clip_t_candidate: float,
) -> PairScore:
    """Attach the relative CLIP-T drop to a scorer's raw numbers."""
    return PairScore(
        lpips=lpips,
        ssim=ssim,
        psnr=psnr,
        psnr_saturated=psnr_saturated,
        dino_cosine=dino_cosine,
        clip_i=clip_i,
        clip_t_reference=clip_t_reference,
        clip_t_candidate=clip_t_candidate,
        clip_t_drop=clip_drop(clip_t_reference, clip_t_candidate),
    )


def mean(values: Sequence[float]) -> float | None:
    """Arithmetic mean, or ``None`` when there are no values."""
    if not values:
        return None
    return sum(values) / len(values)


def passes(lpips_mean: float, clip_drop_mean: float, *, lpips_max: float, clip_drop_max: float) -> bool:
    """True when this candidate meets the Forge quality bar."""
    return lpips_mean <= lpips_max and clip_drop_mean <= clip_drop_max
