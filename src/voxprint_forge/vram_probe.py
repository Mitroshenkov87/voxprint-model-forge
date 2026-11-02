"""Graphics-memory readings for the UI.

nvidia-smi through :mod:`voxprint_common.gpu_probe`. This module does not
import torch, so a window can show a gauge without creating a CUDA context.
"""

from __future__ import annotations

import os
import subprocess

_MIB = 1024 * 1024
# The old CUDA probe used total_memory / 2**30 < 20. Keep that cut so a 16 GB
# card stays on 16 and a 24 GB card stays on 24.
_TIER_CUT_GIB = 20.0


def total_bytes() -> int | None:
    """Total memory of GPU 0 in bytes, or None when no NVIDIA GPU is visible."""
    gpu = _first()
    if gpu is None or gpu.memory_mib is None:
        return None
    return int(gpu.memory_mib) * _MIB


def used_bytes(index: int = 0) -> int | None:
    """Memory in use on GPU ``index``, in bytes, or None when nvidia-smi cannot say."""
    from voxprint_common.gpu_probe import nvidia_smi_executable

    command = [
        nvidia_smi_executable(),
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        if os.name == "nt":
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            )
        else:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return _mib_line(completed.stdout or "", index)


def status() -> tuple[int, int] | None:
    """Bytes (total, used) on GPU 0, or None when either reading is missing."""
    gpu = _first()
    if gpu is None or gpu.memory_mib is None:
        return None
    used = used_bytes(gpu.index)
    if used is None:
        return None
    return int(gpu.memory_mib) * _MIB, used


def tier_gb() -> int:
    """16 or 24 for GPU 0. No card, or a card under 20 GiB, is 16."""
    gpu = _first()
    gib = None if gpu is None else gpu.memory_gib
    if gib is None:
        return 16
    return 16 if gib < _TIER_CUT_GIB else 24


def _first():
    """GPU 0 as nvidia-smi reported it, or None."""
    try:
        from voxprint_common.gpu_probe import query_gpus

        found = query_gpus()
    except Exception:  # noqa: BLE001 - a probe miss is "no card"
        return None
    if not found:
        return None
    return found[0]


def _mib_line(text: str, index: int) -> int | None:
    """One ``memory.used`` row, in bytes. ``nounits`` reports mebibytes."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    if index < 0 or index >= len(lines):
        index = 0
    head = lines[index].split()[0]
    try:
        mib = float(head)
    except ValueError:
        return None
    if mib < 0:
        return None
    return int(mib * _MIB)
