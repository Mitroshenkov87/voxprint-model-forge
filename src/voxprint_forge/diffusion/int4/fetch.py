"""Download the comparison weights and the Wan experts before a rental run.

``--dry-run`` does not call this module. A licence that is not Apache-2.0
stops the download. QuantFunc files are checked against the pinned sha256.
"""

from __future__ import annotations

import json
import os
import shlex
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from voxprint_forge.diffusion.int4.pins import (
    FP8_EDIT_MIRROR,
    QUANTFUNC_R32,
    QUANTFUNC_R32_SHA256,
    QUANTFUNC_R128,
    QUANTFUNC_R128_SHA256,
    QUANTFUNC_R256,
    QUANTFUNC_R256_SHA256,
    QUANTFUNC_REPO,
    QUANTFUNC_REVISION,
    QWEN_EDIT_REPO,
    QWEN_EDIT_REVISION,
    WAN_WEIGHT_REPO,
    WAN_WEIGHT_REVISION,
)
from voxprint_forge.diffusion.names import EDIT_FP8_PREFIX
from voxprint_forge.diffusion.sources import assert_apache
from voxprint_forge.errors import ChecksumError, DownloadError, ToolMissingError
from voxprint_forge.hashing import sha256_file

LicenceCheck = Callable[[str, str, str | None], str]
_QUANTFUNC = (
    (QUANTFUNC_R128, QUANTFUNC_R128_SHA256),
    (QUANTFUNC_R32, QUANTFUNC_R32_SHA256),
    (QUANTFUNC_R256, QUANTFUNC_R256_SHA256),
)


def find_local_fp8(forge_out: Path) -> Path | None:
    """Return a canvas FP8 shard directory under ``forge_out``, if one exists."""
    if not forge_out.is_dir():
        return None
    pattern = f"{EDIT_FP8_PREFIX}*.safetensors"
    for path in forge_out.rglob(pattern):
        if path.is_file():
            return path.parent
    return None


def _filled(path: Path) -> bool:
    return path.is_dir() and any(path.rglob("*.safetensors"))


def fetch_quantfunc(dest: Path, token: str | None) -> Path:
    """Download the QuantFunc INT4 files and check their sha256."""
    folder = dest / "quantfunc"
    folder.mkdir(parents=True, exist_ok=True)
    for name, digest in _QUANTFUNC:
        path = folder / name
        if path.is_file() and sha256_file(path) == digest:
            continue
        try:
            hf_hub_download(
                repo_id=QUANTFUNC_REPO,
                filename=name,
                revision=QUANTFUNC_REVISION,
                local_dir=str(folder),
                token=token,
            )
        except Exception as exc:
            raise DownloadError(f"QuantFunc {name}: {exc}", code="download") from None
        found = sha256_file(path)
        if found != digest:
            raise ChecksumError(f"{name} sha256 {found} != {digest}", code="checksum")
    return folder


def fetch_wan(dest: Path, token: str | None) -> Path:
    """Snapshot Wan2.2-T2V-A14B at the pinned revision."""
    folder = dest / "wan2.2-t2v-a14b"
    if _filled(folder / "high_noise_model") and _filled(folder / "low_noise_model"):
        return folder
    try:
        snapshot_download(
            repo_id=WAN_WEIGHT_REPO,
            revision=WAN_WEIGHT_REVISION,
            local_dir=str(folder),
            token=token,
        )
    except Exception as exc:
        raise DownloadError(f"Wan2.2-T2V-A14B: {exc}", code="download") from None
    return folder


def fetch_qwen_edit(dest: Path, token: str | None) -> Path:
    """Snapshot Qwen-Image-Edit-2511 so quant and the FP8 arm share one tree."""
    folder = dest / "qwen-image-edit-2511"
    if (folder / "model_index.json").is_file():
        return folder
    try:
        snapshot_download(
            repo_id=QWEN_EDIT_REPO,
            revision=QWEN_EDIT_REVISION,
            local_dir=str(folder),
            token=token,
        )
    except Exception as exc:
        raise DownloadError(f"Qwen-Image-Edit-2511: {exc}", code="download") from None
    return folder


def resolve_fp8(dest: Path, forge_out: Path | None, token: str | None) -> Path:
    """Use ``FORGE_FP8_EDIT``, a local canvas tree, or the public edit mirror."""
    env = os.environ.get("FORGE_FP8_EDIT", "").strip()
    if env:
        path = Path(env)
        if not path.exists():
            raise ToolMissingError(f"FORGE_FP8_EDIT does not exist: {path}", code="tool_missing")
        return path
    if forge_out is not None:
        found = find_local_fp8(forge_out)
        if found is not None:
            return found
    folder = dest / "qwen-edit-fp8"
    if _filled(folder):
        return folder
    try:
        snapshot_download(repo_id=FP8_EDIT_MIRROR, local_dir=str(folder), token=token)
    except Exception as exc:
        raise ToolMissingError(
            "No local Qwen-Image-Edit FP8 tree. Set FORGE_FP8_EDIT to the canvas "
            f"part C output, or publish {FP8_EDIT_MIRROR}. Download said: {exc}",
            code="tool_missing",
        ) from None
    if not _filled(folder) and not (folder / "model_index.json").is_file():
        raise ToolMissingError(f"{FP8_EDIT_MIRROR} has no FP8 weights", code="tool_missing")
    return folder


def fetch_weights(
    dest: Path,
    token: str | None,
    *,
    forge_out: Path | None = None,
    licence: LicenceCheck | None = None,
) -> dict[str, str]:
    """Download every rental input and return the environment paths."""
    check = licence or assert_apache
    check(QUANTFUNC_REPO, QUANTFUNC_REVISION, token)
    check(WAN_WEIGHT_REPO, WAN_WEIGHT_REVISION, token)
    check(QWEN_EDIT_REPO, QWEN_EDIT_REVISION, token)
    fp8 = resolve_fp8(dest, forge_out, token)
    quantfunc = fetch_quantfunc(dest, token)
    qwen = fetch_qwen_edit(dest, token)
    wan = fetch_wan(dest, token)
    paths = {
        "FORGE_FP8_EDIT": str(fp8),
        "FORGE_QUANTFUNC": str(quantfunc),
        "FORGE_QWEN_EDIT": str(qwen),
        "FORGE_WAN_CKPT": str(wan),
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "paths.json").write_text(json.dumps(paths, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"export {key}={shlex.quote(value)}" for key, value in paths.items()]
    (dest / "exports.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths
