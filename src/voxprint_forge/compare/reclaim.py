"""Delete compare job weights after the run. Review images stay on disk."""

from __future__ import annotations

import shutil
from pathlib import Path


def reclaim_weights(path: Path) -> bool:
    """Remove the weight directory. Returns whether it existed.

    Results (``pairs.jsonl``, summaries, and review images) are not inside this
    directory and are left in place.
    """
    if not path.exists():
        return False
    shutil.rmtree(path)
    return True


def component_bytes(root: Path) -> dict[str, int]:
    """Byte size of each weight file the arm folder actually contains.

    The shared cache is not walked. A local FP8 arm does not link the base
    ``transformer/*.safetensors``, so those bytes are not counted.
    """
    if not root.is_dir():
        return {}
    found: dict[str, int] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix not in {".safetensors", ".gguf", ".bin"} and "safetensors" not in path.name:
            continue
        found[path.relative_to(root).as_posix()] = path.stat().st_size
    return found
