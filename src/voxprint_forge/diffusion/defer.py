"""Hold canvas uploads until tokens are available. Downloads stay public."""

from __future__ import annotations

import os
from pathlib import Path

from voxprint_forge.diffusion.publish import collect_files, write_models_md, write_sidecars

UPLOAD_DEFERRED = "upload-deferred"
_TRUE = frozenset({"1", "true", "yes", "on"})


def env_defer_upload() -> bool:
    return os.environ.get("FORGE_DEFER_UPLOAD", "").strip().lower() in _TRUE


def defer_requested(flag: bool) -> bool:
    """True when the flag is set or FORGE_DEFER_UPLOAD=1."""
    return flag or env_defer_upload()


def write_local_files(
    output: Path,
    tiers: dict[str, tuple[str, int | None, str, str]],
    repo: str,
    revision: str,
) -> list[dict[str, object]]:
    """Sha256 sidecars and MODELS.md. No remote calls."""
    files = collect_files(output, tiers, repo, revision)
    write_models_md(output, files)
    write_sidecars(output)
    return [
        {
            "name": item.name,
            "sha256": item.sha256,
            "size": item.size,
            "tier": item.tier,
            "vram_gb": item.vram_gb,
        }
        for item in files
    ]


def drop_marker(path: Path) -> None:
    if path.is_file():
        path.unlink()
