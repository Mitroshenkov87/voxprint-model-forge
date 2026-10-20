"""Training presets, likeness and memory tiers (plan §2).

Preset values are [U] until GPU calibration. :data:`MEASURED` holds the per-base, per-tier figures
that a calibration run actually measured; a base or tier that is missing there has not been measured.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields, replace

PRESETS: dict[str, dict[str, object]] = {
    "quick": {"rank": 16, "alpha": 16, "steps": 150, "lr": 1e-4, "resolution": 768},
    "balanced": {"rank": 32, "alpha": 32, "steps": 1500, "lr": 1e-4, "resolution": 1024},
    "best": {"rank": 32, "alpha": 32, "steps": 3000, "lr": 8e-5, "resolution": 1024},
}
LIKENESS: dict[str, tuple[float, float]] = {
    "flexible": (0.7, 0.15),
    "balanced": (1.0, 0.10),
    "close": (1.3, 0.05),
}
VRAM_TIERS = (16, 24)


@dataclass(frozen=True)
class Measured:
    """One measured training run: seconds per optimizer step and peak reserved graphics memory (GiB).

    ``micro_batches`` is batch x grad_accum of that run, so adapters can scale the time to another config.
    ``te_quant`` is the text-encoder mode the tier default resolved to. ``source`` names the run.
    """

    s_per_step: float
    peak_reserved_gb: float
    micro_batches: int
    resolution: int
    te_quant: str
    source: str
    peak_measured: bool = True


# MOSS-Image Base: RTX 4090 with FORGE_VRAM_CAP_GB 16 or 22.5, Balanced, rank 32, 1024 px,
# batch 1 x accum 2, moss-memprobe v2 (18 Oct 2026). Z-Image on the 16 GB RTX 4090 laptop
# tier is 9.7 s/step at 1024; that run did not record a peak, so the card line stays the
# preset tier. The 24 GB Z-Image time stays the unmeasured 1.6 s/step.
MEASURED: dict[str, dict[int, Measured]] = {
    "moss-image-base": {
        16: Measured(4.89, 13.6, 2, 1024, "int8", "moss-memprobe v2 train16"),
        24: Measured(4.90, 18.5, 2, 1024, "none", "moss-memprobe v2 train24"),
    },
    "z-image": {
        16: Measured(9.7, 0.0, 1, 1024, "none", "rtx 4090 16 GB laptop", peak_measured=False),
    },
}


def measured(mdl_id: str, tier: int) -> Measured | None:
    """The measured run for ``mdl_id`` on memory tier ``tier``, or ``None`` when it was not measured."""
    return MEASURED.get(mdl_id, {}).get(int(tier))


def measured_peaks() -> dict[str, dict[int, float]]:
    """Peak reserved GiB for tiers whose peak was measured. A timing-only run is left out."""
    found: dict[str, dict[int, float]] = {}
    for mdl_id, tiers in MEASURED.items():
        peaks = {tier: run.peak_reserved_gb for tier, run in tiers.items() if run.peak_measured}
        if peaks:
            found[mdl_id] = peaks
    return found
SCHEDULERS = ("constant", "cosine")
CAPTION_MODES = ("long", "short", "both")
PRECISIONS = ("bf16", "fp16")


@dataclass(frozen=True)
class TrainConfig:
    """Every training value. Presets fill it; Expert flags override single fields."""

    preset: str = "balanced"
    likeness: str = "balanced"
    vram: int = 24
    rank: int = 32
    alpha: int = 32
    steps: int = 1500
    lr: float = 1e-4
    scheduler: str = "constant"
    warmup: int = 0
    batch: int = 2
    grad_accum: int = 1
    resolution: int = 1024
    precision: str = "bf16"
    grad_checkpointing: bool = False
    base_fp8: bool = False
    train_text_encoder: bool = False
    caption_mode: str = "both"
    caption_dropout: float = 0.10
    seed: int = 0
    save_every: int = 250
    previews: bool = True
    preview_every: int = 250
    replay: float = 0.25
    te_quant: str = "auto"
    te_device: str = "gpu"

    def to_dict(self) -> dict[str, object]:
        """Plain dict for project files."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> TrainConfig:
        """Rebuild from :meth:`to_dict`; unknown keys are ignored."""
        names = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in names})  # type: ignore[arg-type]


def detect_vram_gb() -> int:
    """16 or 24: the tier for the first CUDA card. No card picks 16."""
    try:
        import torch

        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / 2**30
            return 16 if total < 20 else 24
    except Exception:  # noqa: BLE001 - detection is best effort
        pass
    return 16


TE_QUANT_MODES = ("auto", "none", "int8", "nf4")


def build_config(
    preset: str = "balanced",
    likeness: str = "balanced",
    vram: str | int = "auto",
    **overrides: object,
) -> TrainConfig:
    """Preset values, adjusted by likeness and memory tier, then the explicit overrides."""
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}")
    if likeness not in LIKENESS:
        raise ValueError(f"unknown likeness {likeness!r}")
    tier = detect_vram_gb() if vram == "auto" else int(vram)
    if tier not in VRAM_TIERS:
        raise ValueError(f"vram must be auto, 16 or 24, not {vram!r}")
    base = PRESETS[preset]
    factor, dropout = LIKENESS[likeness]
    steps = int(round(int(base["steps"]) * factor))  # type: ignore[call-overload]
    small = tier == 16
    config = TrainConfig(
        preset=preset,
        likeness=likeness,
        vram=tier,
        rank=int(base["rank"]),  # type: ignore[call-overload]
        alpha=int(base["alpha"]),  # type: ignore[call-overload]
        steps=steps,
        lr=float(base["lr"]),  # type: ignore[arg-type]
        resolution=int(base["resolution"]),  # type: ignore[call-overload]
        # Batch 1 on both tiers. The 24 GB tier keeps an effective batch of 2 through
        # accumulation: batch 2 OOMed in its first step under a 23.5 GiB cap (CALIB-R2,
        # peak 24 743 MiB), and accumulation halves the activations at the same 1024 px.
        batch=1,
        grad_accum=1 if preset == "quick" else 2,
        # Both tiers: without checkpointing a 1024 px batch of 2 keeps every block's
        # activations, about 79 GB on Z-Image (synth4 LORA-CALIB), far over 24 GB.
        grad_checkpointing=True,
        base_fp8=small,
        caption_dropout=dropout,
        save_every=max(1, steps // 6),
        preview_every=max(1, steps // 6),
    )
    clean = {key: value for key, value in overrides.items() if value is not None}
    if "te_quant" not in clean and os.environ.get("VOXPRINT_FORGE_TE_QUANT", "").strip():
        clean["te_quant"] = os.environ["VOXPRINT_FORGE_TE_QUANT"].strip().lower()
    if "te_device" not in clean and os.environ.get("VOXPRINT_FORGE_TE_DEVICE", "").strip():
        clean["te_device"] = os.environ["VOXPRINT_FORGE_TE_DEVICE"].strip().lower()
    config = replace(config, **clean)  # type: ignore[arg-type]
    check_config(config)
    return config


def check_config(config: TrainConfig) -> None:
    """Raise ``ValueError`` on values outside the supported range."""
    if config.rank < 1 or config.alpha < 1 or config.steps < 1 or config.batch < 1 or config.grad_accum < 1:
        raise ValueError("rank, alpha, steps, batch and grad_accum must be at least 1")
    if not 0 < config.lr < 1:
        raise ValueError("lr must be between 0 and 1")
    if config.resolution < 64 or config.resolution % 64:
        raise ValueError("resolution must be a multiple of 64")
    if config.scheduler not in SCHEDULERS or config.caption_mode not in CAPTION_MODES:
        raise ValueError("unknown scheduler or caption mode")
    if config.precision not in PRECISIONS:
        raise ValueError("precision must be bf16 or fp16")
    if not 0 <= config.caption_dropout < 1 or not 0 <= config.replay <= 1:
        raise ValueError("caption_dropout must be in [0, 1) and replay in [0, 1]")
    if config.te_quant not in TE_QUANT_MODES:
        raise ValueError("te_quant must be auto, none, int8 or nf4")
    if config.te_device not in ("gpu", "cpu"):
        raise ValueError("te_device must be gpu or cpu")
    if config.save_every < 1 or config.preview_every < 1 or config.warmup < 0:
        raise ValueError("save_every and preview_every must be at least 1")
