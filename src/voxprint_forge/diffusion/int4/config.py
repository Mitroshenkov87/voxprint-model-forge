"""fast_int4 YAML and the deepcompressor command lines."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from voxprint_forge.diffusion.int4.pins import (
    DEFAULT_SVD_RANK,
    EDIT_STEPS,
    EDIT_TRUE_CFG,
    QWEN_EDIT_REPO,
    QWEN_IMAGE_REPO,
    SVD_GROUP,
    TargetName,
)


def fast_int4_yaml(*, rank: int = DEFAULT_SVD_RANK) -> str:
    """Overlay on upstream ``int4.yaml`` and ``fast.yaml``. Later files win."""
    if rank < 1:
        raise ValueError(rank)
    return (
        "quant:\n"
        "  calib:\n"
        "    num_samples: 64\n"
        "  smooth:\n"
        "    proj:\n"
        "      num_grids: 10\n"
        "  wgts:\n"
        "    dtype: sint4\n"
        "    enable_low_rank: true\n"
        "    group_shapes:\n"
        "    - - 1\n"
        f"      - {SVD_GROUP}\n"
        "      - 1\n"
        "      - 1\n"
        "      - 1\n"
        "    low_rank:\n"
        f"      rank: {rank}\n"
        "pipeline:\n"
        "  shift_activations: true\n"
    )


def _local_or(env_name: str, repo: str) -> str:
    value = os.environ.get(env_name, "").strip()
    return value or repo


def model_yaml(target: TargetName) -> str:
    """Model config. ``path`` lets the pinned loader fetch a repo it does not know by name.

    ``FORGE_QWEN_EDIT`` and ``FORGE_QWEN_IMAGE`` replace the Hub ids with a local snapshot.
    """
    if target == "qwen-image-edit-2511":
        name = "qwen-image-edit-2511"
        path = _local_or("FORGE_QWEN_EDIT", QWEN_EDIT_REPO)
        guidance = EDIT_TRUE_CFG
        steps = EDIT_STEPS
    elif target == "qwen-image":
        name = "qwen-image"
        path = _local_or("FORGE_QWEN_IMAGE", QWEN_IMAGE_REPO)
        guidance = 4.0
        steps = 20
    else:
        raise ValueError(target)
    return (
        "pipeline:\n"
        f"  name: {name}\n"
        f"  path: {path}\n"
        "  dtype: torch.bfloat16\n"
        "eval:\n"
        f"  num_steps: {steps}\n"
        f"  guidance_scale: {guidance}\n"
        "  protocol: fmeuler{num_steps}-g{guidance_scale}\n"
        "quant:\n"
        "  calib:\n"
        "    batch_size: 1\n"
    )


def collect_yaml(prompt_file: Path) -> str:
    """Collect config pointing at our prompt yaml."""
    return (
        "collect:\n"
        "  root: datasets\n"
        "  dataset_name: voxprint\n"
        f"  data_path: {prompt_file.resolve().as_posix()}\n"
        "  num_samples: 64\n"
    )


def write_configs(work: Path, target: TargetName, *, rank: int) -> dict[str, Path]:
    """Write the model, fast_int4 overlay, and collect configs."""
    config_dir = work / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    model = config_dir / "model.yaml"
    overlay = config_dir / "fast_int4.yaml"
    collect = config_dir / "collect.yaml"
    prompts = config_dir / "prompts.yaml"
    model.write_text(model_yaml(target), encoding="utf-8")
    overlay.write_text(fast_int4_yaml(rank=rank), encoding="utf-8")
    collect.write_text(collect_yaml(prompts), encoding="utf-8")
    return {"model": model, "overlay": overlay, "collect": collect, "prompts": prompts}


def quant_argv(
    deepcompressor: Path,
    configs: dict[str, Path],
    save_dir: Path,
    *,
    python: str | None = None,
) -> list[str]:
    """``python -m deepcompressor.app.diffusion.ptq`` with int4, fast, and our overlay."""
    root = deepcompressor / "examples" / "diffusion"
    executable = python or sys.executable
    return [
        executable,
        "-m",
        "deepcompressor.app.diffusion.ptq",
        str(configs["model"]),
        str(root / "configs" / "svdquant" / "int4.yaml"),
        str(root / "configs" / "svdquant" / "fast.yaml"),
        str(configs["overlay"]),
        "--save-model",
        str(save_dir),
        "--skip-eval",
        "true",
        "--skip-gen",
        "true",
    ]


def collect_argv(
    configs: dict[str, Path],
    *,
    python: str | None = None,
) -> list[str]:
    """Collect calibration caches with the patched image-edit path."""
    executable = python or sys.executable
    return [
        executable,
        "-m",
        "deepcompressor.app.diffusion.dataset.collect.calib",
        str(configs["model"]),
        str(configs["collect"]),
    ]


def convert_argv(
    quant_dir: Path,
    output_root: Path,
    model_name: str,
    *,
    python: str | None = None,
) -> list[str]:
    """Export a saved checkpoint to the Nunchaku svdquant layout."""
    executable = python or sys.executable
    return [
        executable,
        "-m",
        "deepcompressor.backend.nunchaku.convert",
        "--quant-path",
        str(quant_dir),
        "--output-root",
        str(output_root),
        "--model-name",
        model_name,
    ]
