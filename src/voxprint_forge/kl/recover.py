"""Rebuild the F16 reference of an ablated quant from the stock BF16 and that quant.

Forge's ablation writes ``W := W - r r^T W`` into every attention-output and
MLP-down matrix whose output size is the hidden size. After that step
``r^T W = 0`` for all of them, so ``r`` is the eigenvector with the smallest
eigenvalue of ``sum_i W_i W_i^T`` taken over the published quant's
``attn_output`` and ``ffn_down`` tensors. Quantisation noise only fills that
null direction a little, so the smallest eigenvalue sits far below the next.

Re-running the ablation instead is not exact: the activation pass is not
bit-reproducible, and on Qwen3-0.6B two CPU runs gave updates that differ by
about a quarter of their size. Recovering ``r`` from the file we published
scores the quants against the weights they were really made from.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import gguf
import numpy as np
import torch
from gguf.quants import dequantize
from safetensors.torch import load_file, save_file

from voxprint_forge.abliterate import WRITE_LEAVES, orthogonalize_weight
from voxprint_forge.errors import StageError

GGUF_WRITERS = ("attn_output.weight", "ffn_down.weight")
# Smallest over second-smallest eigenvalue. Ablated Qwen3-0.6B: 0.006 (Q5_K_M), 0.017 (Q4_K_M).
# A quant that was never ablated has no null direction and lands near 1.
MAX_EIGEN_RATIO = 0.1


def hidden_size(source: Path) -> int:
    """``hidden_size`` from ``config.json`` (top level or ``text_config``)."""
    path = source / "config.json"
    if not path.is_file():
        raise StageError("BF16 source has no config.json", code="kl_reference")
    config = json.loads(path.read_text(encoding="utf-8"))
    for block in (config, config.get("text_config") or {}):
        value = block.get("hidden_size") if isinstance(block, dict) else None
        if isinstance(value, int) and value > 0:
            return value
    raise StageError("config.json has no hidden_size", code="kl_reference")


def recover_direction(quant: Path, hidden: int) -> tuple[np.ndarray, dict[str, object]]:
    """Unit refusal direction removed from ``quant``, plus the matrix count and eigenvalue ratio."""
    reader = gguf.GGUFReader(str(quant))
    gram = np.zeros((hidden, hidden), dtype=np.float64)
    count = 0
    for tensor in reader.tensors:
        if not tensor.name.endswith(GGUF_WRITERS):
            continue
        weight = np.asarray(dequantize(tensor.data, tensor.tensor_type), dtype=np.float64)
        if weight.ndim != 2 or weight.shape[0] != hidden:
            continue
        gram += weight @ weight.T
        count += 1
    if count == 0:
        raise StageError(f"{quant.name} has no attn_output/ffn_down of width {hidden}", code="kl_reference")
    values, vectors = np.linalg.eigh(gram)
    ratio = float(values[0] / values[1]) if values[1] > 0 else 1.0
    if ratio > MAX_EIGEN_RATIO:
        raise StageError(
            f"{quant.name} has no ablated direction (eigenvalue ratio {ratio:.3f} > {MAX_EIGEN_RATIO})",
            code="kl_reference",
        )
    direction = vectors[:, 0]
    return direction / np.linalg.norm(direction), {"matrices": count, "eigen_ratio": ratio}


def apply_direction(source: Path, dest: Path, direction: np.ndarray, hidden: int) -> int:
    """Copy ``source`` to ``dest`` with ``W := W - r r^T W`` on the same matrices the ablation step changes."""
    dest.mkdir(parents=True, exist_ok=True)
    unit = torch.from_numpy(direction.astype(np.float32))
    changed = 0
    for item in sorted(source.iterdir()):
        if item.is_dir():
            continue
        if item.suffix != ".safetensors":
            shutil.copy2(item, dest / item.name)
            continue
        tensors = load_file(str(item))
        for name, value in tensors.items():
            parts = name.split(".")
            if len(parts) < 2 or parts[-1] != "weight" or parts[-2] not in WRITE_LEAVES:
                continue
            if value.ndim != 2 or value.shape[0] != hidden:
                continue
            tensors[name] = orthogonalize_weight(value, unit).contiguous()
            changed += 1
        save_file(tensors, str(dest / item.name), metadata={"format": "pt"})
    return changed


def recover_ablation(quant: Path, source: Path, dest: Path) -> dict[str, object]:
    """Write the ablated BF16 tree that ``quant`` was made from into ``dest`` and return the record."""
    hidden = hidden_size(source)
    direction, info = recover_direction(quant, hidden)
    changed = apply_direction(source, dest, direction, hidden)
    if changed != info["matrices"]:
        raise StageError(
            f"BF16 has {changed} writing matrices, {quant.name} has {info['matrices']}",
            code="kl_reference",
        )
    return {"from": quant.name, **info}
