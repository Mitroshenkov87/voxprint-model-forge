"""15% free-space floor. A step aborts with exit 2 after reclaim still misses it."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

from voxprint_forge.diffusion.int4.markers import load_done
from voxprint_forge.diffusion.int4.pins import DISK_FLOOR
from voxprint_forge.errors import UsageError
from voxprint_forge.reclaim import delete_logged


class DiskFloorError(UsageError):
    """Free space stayed under the floor after reclaim. Exit 2.

    JSON ``error.code`` is ``disk_floor``. Exit 7 remains the general
    "next stage does not fit its byte budget" code.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, code="disk_floor")


def free_ratio(path: Path) -> float:
    """Free bytes divided by total bytes for the filesystem holding ``path``."""
    target = path if path.exists() else path.parent
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target)
    if usage.total <= 0:
        return 0.0
    return usage.free / usage.total


def reclaim_verified(done: Path) -> int:
    """Delete sources listed on verified publish markers. Missing paths free nothing."""
    if not done.is_dir():
        return 0
    freed = 0
    for marker in sorted(done.iterdir()):
        if not marker.is_file():
            continue
        payload = load_done(done, marker.name)
        if not isinstance(payload, dict) or payload.get("verified") is not True:
            continue
        sources = payload.get("sources")
        if not isinstance(sources, list):
            continue
        for item in sources:
            freed += delete_logged(Path(str(item)))
    return freed


def guard_disk(
    path: Path,
    *,
    floor: float = DISK_FLOOR,
    ratio: Callable[[Path], float] | None = None,
    reclaim: Callable[[], int] | None = None,
) -> float:
    """Reclaim once when ``path`` is under ``floor``, then abort if it still is."""
    probe = ratio or free_ratio
    current = probe(path)
    if current >= floor:
        return current
    if reclaim is not None:
        reclaim()
    current = probe(path)
    if current < floor:
        raise DiskFloorError(
            f"free space {current:.1%} is under the {floor:.0%} floor at {path} after reclaim",
        )
    return current
