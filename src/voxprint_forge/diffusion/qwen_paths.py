"""Stable paths under ~/forge-out/canvas/qwen-image-2512. No weights are copied into a reused encoder."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Literal

from voxprint_forge.diffusion.qwen_spec import INT4_MARKER_NAME, STABLE_DIR
from voxprint_forge.diffusion.sources import tree_hashes
from voxprint_forge.errors import StageError

EncoderStatus = Literal["reuse", "reuse-pending", "convert"]

INT4_MARKER_TEXT = (
    "INT4 deferred\n"
    "No official licence-clean nunchaku-tech SVDQuant build of Qwen-Image-2512 is pinned.\n"
    "This directory contains no weights.\n"
)

ENCODER_PENDING_TEXT = (
    "Encoder reuse pending\n"
    "The text encoder matches the Edit-2511 pin, and the part C FP8 file is not on disk yet.\n"
    "This directory contains no weights.\n"
)

_WEIGHT_SUFFIXES = frozenset({".safetensors", ".gguf", ".bin", ".pt", ".ckpt"})


def stable_root(forge_out: Path) -> Path:
    """Directory of stable names that point at the latest encoder files."""
    return forge_out / "canvas" / STABLE_DIR


def encoder_hashes(text_encoder: Path) -> dict[str, str]:
    """Hash every file under text_encoder and prefix the relative names."""
    return {f"text_encoder/{name}": digest for name, digest in tree_hashes(text_encoder).items()}


def classify_encoder(found: dict[str, str], pins: dict[str, str], edit_fp8: Path) -> EncoderStatus:
    """Match pins, then prefer part C's FP8 file. A mismatch is converted, not linked."""
    if found != pins:
        return "convert"
    if edit_fp8.is_file():
        return "reuse"
    return "reuse-pending"


def edit_encoder_fp8(forge_out: Path, job_id: str, output_name: str, filename: str) -> Path:
    """Path of the edit-part FP8 text encoder inside this job."""
    return forge_out / "work" / job_id / output_name / filename


def _clear_path(dest: Path) -> None:
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    elif dest.is_dir():
        shutil.rmtree(dest)


def _symlink_or_hardlink(dest: Path, target: Path) -> None:
    """Link the existing file. A hardlink is the same inode, not a second copy."""
    try:
        dest.symlink_to(target)
        return
    except OSError:
        pass
    try:
        os.link(target, dest)
    except OSError as exc:
        raise StageError(
            f"cannot link {dest.name} without copying weights: {exc}",
            code="weights_link",
        ) from exc


def _replace_with_link(dest: Path, target: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _clear_path(dest)
    _symlink_or_hardlink(dest, target)


def link_stable_file(forge_out: Path, name: str, target: Path) -> Path:
    """Point a stable name at ``target`` and return the link path."""
    dest = stable_root(forge_out) / name
    _replace_with_link(dest, target)
    return dest


def refresh_fp8_dir(forge_out: Path, shards: list[Path]) -> Path:
    """Point fp8/ at the shards that will actually be published."""
    dest = stable_root(forge_out) / "fp8"
    if dest.is_symlink() or dest.is_file():
        dest.unlink()
    dest.mkdir(parents=True, exist_ok=True)
    keep = {path.name for path in shards}
    for child in dest.iterdir():
        if child.name not in keep and (child.is_symlink() or child.is_file()):
            child.unlink()
    for path in shards:
        link = dest / path.name
        if link.is_symlink() or link.is_file():
            link.unlink()
        _symlink_or_hardlink(link, path.resolve())
    return dest


def assert_no_weight_files(directory: Path) -> None:
    """Raise when ``directory`` still contains weight files."""
    if not directory.exists():
        return
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in _WEIGHT_SUFFIXES:
            raise StageError(f"{directory.name} contains weights {path.name}", code="int4_weights")


def write_int4_marker(forge_out: Path) -> Path:
    """Write the marker that says the int4 encoder was skipped."""
    dest = stable_root(forge_out) / "int4"
    dest.mkdir(parents=True, exist_ok=True)
    assert_no_weight_files(dest)
    marker = dest / INT4_MARKER_NAME
    marker.write_text(INT4_MARKER_TEXT, encoding="utf-8")
    return marker


def write_encoder_pending(forge_out: Path) -> Path:
    """Write the marker that says the FP8 encoder is not ready yet."""
    dest = stable_root(forge_out) / "encoder"
    _clear_path(dest)
    dest.mkdir(parents=True)
    marker = dest / "PENDING.txt"
    marker.write_text(ENCODER_PENDING_TEXT, encoding="utf-8")
    assert_no_weight_files(dest)
    return marker


def reset_encoder_dir(forge_out: Path) -> Path:
    """Remove a previous encoder link so a converted tree can be hardlinked."""
    dest = stable_root(forge_out) / "encoder"
    _clear_path(dest)
    return dest


def format_stable_lines(forge_out: Path, *, bf16_encoder: Path, expected_fp8: Path) -> list[str]:
    """Lines that record the BF16 and expected FP8 encoder paths."""
    root = stable_root(forge_out)
    return [
        f"stable fp8 {root / 'fp8'}",
        f"stable int4 {root / 'int4'}",
        f"stable encoder {root / 'encoder'}",
        f"encoder bf16 {bf16_encoder}",
        f"encoder expected fp8 {expected_fp8}",
    ]


def hardlink_tree(src: Path, dest: Path) -> None:
    """Copy a tree with hardlinks so deleting the download does not drop the stable encoder."""
    if not src.is_dir():
        raise StageError(f"missing directory {src}", code="weights_missing")
    dest.mkdir(parents=True, exist_ok=True)
    for path in sorted(item for item in src.rglob("*") if item.is_file()):
        target = dest / path.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        try:
            os.link(path, target)
        except OSError:
            target.write_bytes(path.read_bytes())
