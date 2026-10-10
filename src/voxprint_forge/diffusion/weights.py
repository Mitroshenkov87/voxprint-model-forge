"""BF16 casts and FP8 files. Reads one safetensors tensor at a time."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from voxprint_forge.diffusion.fp8 import (
    FORMAT,
    Profile,
    dequant_state,
    quantize_meta,
    quantize_this,
    quantize_weight,
)
from voxprint_forge.diffusion.names import SHARD_BYTES, shard_name
from voxprint_forge.errors import StageError

_FLOATS = frozenset({"F32", "F16", "BF16"})
_WIDTH = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}


@dataclass(frozen=True)
class TensorRef:
    """Where one tensor lives and whether the profile quantizes it."""
    name: str
    path: Path
    shape: tuple[int, ...]
    dtype_name: str
    bf16_bytes: int
    fp8_bytes: int


def safetensor_files(source: Path) -> list[Path]:
    """Safetensors shards in a model directory, in index order."""
    if source.is_file():
        return [source]
    files = sorted(path for path in source.rglob("*.safetensors") if path.is_file())
    if not files:
        raise StageError(f"no safetensors under {source}", code="weights_missing")
    return files


def _numel(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _bf16_bytes(dtype_name: str, numel: int) -> int:
    if dtype_name in _FLOATS:
        return numel * 2
    width = _WIDTH.get(dtype_name)
    if width is None:
        raise StageError(f"unsupported dtype {dtype_name}", code="dtype")
    return numel * width


def index_tensors(source: Path, profile: Profile) -> list[TensorRef]:
    """List tensors in ``source`` with the profile's quantize decision."""
    rows: list[TensorRef] = []
    seen: set[str] = set()
    for path in safetensor_files(source):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in seen:
                    raise StageError(f"duplicate tensor {name}", code="weights_dup")
                seen.add(name)
                view = handle.get_slice(name)
                shape = tuple(int(dim) for dim in view.get_shape())
                dtype_name = str(view.get_dtype())
                numel = _numel(shape)
                quantize = quantize_meta(name, shape, dtype_name, profile)
                fp8_bytes = numel + 2 if quantize else _bf16_bytes(dtype_name, numel)
                rows.append(
                    TensorRef(
                        name=name,
                        path=path,
                        shape=shape,
                        dtype_name=dtype_name,
                        bf16_bytes=_bf16_bytes(dtype_name, numel),
                        fp8_bytes=fp8_bytes,
                    )
                )
    return rows


def plan_shards(sizes: list[tuple[str, int]], max_bytes: int) -> list[list[str]]:
    """Pack names into groups. A single tensor larger than max_bytes stays alone."""
    groups: list[list[str]] = []
    current: list[str] = []
    used = 0
    for name, size in sizes:
        if current and used + size > max_bytes:
            groups.append(current)
            current = []
            used = 0
        current.append(name)
        used += size
    if current:
        groups.append(current)
    return groups


def _load_named(refs: list[TensorRef]) -> dict[str, torch.Tensor]:
    by_file: dict[Path, list[str]] = {}
    for ref in refs:
        by_file.setdefault(ref.path, []).append(ref.name)
    found: dict[str, torch.Tensor] = {}
    for path, names in by_file.items():
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in names:
                found[name] = handle.get_tensor(name)
    return found


def _as_bf16(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
        return tensor.to(dtype=torch.bfloat16)
    return tensor


def has_fp32(source: Path) -> bool:
    """True when any tensor in ``source`` is stored as FP32."""
    for ref in index_tensors(source, "diffusion"):
        if ref.dtype_name == "F32":
            return True
    return False


def link_or_copy(src: Path, dest: Path) -> None:
    """Hard-link ``src`` to ``dest``, or copy when the link fails."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    try:
        os.link(src, dest)
    except OSError:
        shutil.copy2(src, dest)


def link_tree(src: Path, dest: Path) -> None:
    """Hard-link or copy every file under ``src`` into ``dest``."""
    if not src.is_dir():
        raise StageError(f"missing directory {src}", code="weights_missing")
    for path in sorted(item for item in src.rglob("*") if item.is_file()):
        link_or_copy(path, dest / path.relative_to(src))


def passthrough_or_bf16(source: Path, dest: Path) -> str:
    """Copy a file when it is already BF16. Cast floating tensors when it is FP32."""
    if has_fp32(source):
        refs = index_tensors(source, "diffusion")
        tensors = {name: _as_bf16(value) for name, value in _load_named(refs).items()}
        dest.parent.mkdir(parents=True, exist_ok=True)
        save_file({name: value.contiguous() for name, value in tensors.items()}, str(dest))
        return "cast"
    link_or_copy(source, dest)
    return "passthrough"


def write_bf16_shards(source: Path, dest_dir: Path, prefix: str, *, max_bytes: int = SHARD_BYTES) -> list[Path]:
    """Rewrite ``source`` as BF16 shards no larger than ``max_bytes``."""
    refs = index_tensors(source, "diffusion")
    groups = plan_shards([(ref.name, ref.bf16_bytes) for ref in refs], max_bytes)
    by_name = {ref.name: ref for ref in refs}
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    total = len(groups)
    for index, names in enumerate(groups, start=1):
        tensors = {name: _as_bf16(value) for name, value in _load_named([by_name[name] for name in names]).items()}
        path = dest_dir / shard_name(prefix, index, total)
        save_file({name: value.contiguous() for name, value in tensors.items()}, str(path))
        written.append(path)
    return written


def _quantize_refs(refs: list[TensorRef], profile: Profile) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for name, tensor in _load_named(refs).items():
        if quantize_this(name, tensor, profile):
            quantized, scale = quantize_weight(tensor)
            output[name] = quantized
            output[f"{name}.scale_weight"] = scale
        else:
            output[name] = _as_bf16(tensor)
    return output


def write_fp8_file(source: Path, dest: Path, profile: Profile) -> None:
    """Quantize ``source`` to one FP8 safetensors file."""
    tensors = _quantize_refs(index_tensors(source, profile), profile)
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
        str(dest),
        metadata={"format": FORMAT},
    )


def write_fp8_shards(
    source: Path,
    dest_dir: Path,
    prefix: str,
    profile: Profile,
    *,
    max_bytes: int = SHARD_BYTES,
) -> list[Path]:
    """Quantize ``source`` to FP8 shards no larger than ``max_bytes``."""
    refs = index_tensors(source, profile)
    groups = plan_shards([(ref.name, ref.fp8_bytes) for ref in refs], max_bytes)
    by_name = {ref.name: ref for ref in refs}
    dest_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    total = len(groups)
    for index, names in enumerate(groups, start=1):
        tensors = _quantize_refs([by_name[name] for name in names], profile)
        path = dest_dir / shard_name(prefix, index, total)
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
            str(path),
            metadata={"format": FORMAT},
        )
        written.append(path)
    return written


def load_tensors(paths: list[Path]) -> dict[str, torch.Tensor]:
    """Load every tensor in these safetensors files."""
    merged: dict[str, torch.Tensor] = {}
    for path in paths:
        for name, value in load_file(str(path)).items():
            if name in merged:
                raise StageError(f"duplicate tensor {name}", code="weights_dup")
            merged[name] = value
    return merged


def materialise_bf16(paths: list[Path], dest: Path) -> Path:
    """Write the dequantized BF16 tensors of one or more FP8 shards."""
    state = dequant_state(load_tensors(paths))
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: value.contiguous() for name, value in state.items()}, str(dest))
    return dest


def copy_config(source_dir: Path, dest: Path) -> None:
    """Copy config and tokenizer files from ``source_dir`` to ``dest``."""
    config = source_dir / "config.json"
    if not config.is_file():
        return
    link_or_copy(config, dest)
