"""Free-space checks. A short disk reclaims leftovers, then stops if it is still short."""

from __future__ import annotations

import shutil
from pathlib import Path

from voxprint_forge.diffusion.planload import CanvasPlan
from voxprint_forge.errors import DiskShortError
from voxprint_forge.fitmath import gib
from voxprint_forge.reclaim import reclaim_for_disk

MARGIN = 5 * 1024**3


def free_bytes(path: Path) -> int:
    """Free bytes on the filesystem that holds ``path``."""
    target = path if path.exists() else path.parent
    target.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(target).free


def ensure_free(
    path: Path,
    need: int,
    what: str,
    *,
    forge_out: Path | None = None,
    cache_dir: Path | None = None,
    plan: CanvasPlan | None = None,
) -> int:
    """Raise when ``path`` is still short after one reclaim pass.

    The pass deletes verified leftovers, finished staging, stale partials, and
    orphaned cache blobs, logs each size, and checks again. It does not wait.
    """
    free = free_bytes(path)
    required = need + MARGIN
    if free < required:
        reclaim_for_disk(forge_out if forge_out is not None else path, cache_dir=cache_dir, plan=plan)
        free = free_bytes(path)
    if free < required:
        raise DiskShortError(
            f"need {gib(required):.1f} GiB free for {what}, have {gib(free):.1f} GiB at {path}",
            code="disk_short",
        )
    return free
