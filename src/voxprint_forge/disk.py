"""Free-space checks. A short disk stops the run and leaves finished quants in place."""

from __future__ import annotations

import shutil
from pathlib import Path

from voxprint_forge.errors import DiskShortError
from voxprint_forge.fitmath import gib

MARGIN = 5 * 1024**3


def free_bytes(path: Path) -> int:
    target = path if path.exists() else path.parent
    target.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(target).free


def ensure_free(path: Path, need: int, what: str) -> int:
    free = free_bytes(path)
    required = need + MARGIN
    if free < required:
        raise DiskShortError(
            f"need {gib(required):.1f} GiB free for {what}, have {gib(free):.1f} GiB at {path}",
            code="disk_short",
        )
    return free
