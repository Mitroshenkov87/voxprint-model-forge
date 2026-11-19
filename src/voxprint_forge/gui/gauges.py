"""Disk, RAM, and VRAM readings. The window only displays what these probes return."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from voxprint_forge import disk


def _gib(amount: int) -> str:
    return f"{amount / (1024**3):.1f} GiB"


def gauge_percent(total: int, used: int) -> int:
    """Used fraction of ``total`` as a percent from 0 to 100."""
    if total <= 0:
        return 0
    return max(0, min(100, round(used * 100 / total)))


def gauge_text(title: str, total: int, used: int) -> str:
    """One gauge line: title, used, total, and percent."""
    free = max(total - used, 0)
    return f"{title}  {_gib(free)} free of {_gib(total)}"


def disk_total_free(path: Path) -> tuple[int, int]:
    """Total and free bytes on the filesystem that holds ``path``."""
    free = disk.free_bytes(path)
    target = path if path.exists() else path.parent
    total = shutil.disk_usage(target).total
    return total, free


def vram_status() -> tuple[int, int] | None:
    """Bytes (total, used) on GPU 0, or None when nvidia-smi cannot see a GPU."""
    try:
        from voxprint_forge.vram_probe import status

        return status()
    except Exception:  # noqa: BLE001 - the gauge is optional
        return None


@dataclass
class GaugeProbes:
    """VRAM, RAM, and disk readings the window paints."""
    disk: Callable[[], tuple[int, int]]
    ram: Callable[[], tuple[int, int]]
    vram: Callable[[], tuple[int, int] | None]
