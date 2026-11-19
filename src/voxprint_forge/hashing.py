"""SHA-256 sidecars. The digest itself comes from ``voxprint_common.hashing``.

``voxprint_common`` requires Python 3.14. The Windows laptop smoke imports
forge modules from a Python 3.12 venv, so this module must import without
that library. The digest import stays inside the functions that compute it.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["is_sha256", "sha256_file", "write_sha256"]

_SHA256_TEXT = re.compile(r"^[0-9a-f]{64}$")


def is_sha256(value: str) -> bool:
    """True when ``value`` is 64 lowercase hex digits."""
    return _SHA256_TEXT.fullmatch(value) is not None


def sha256_file(path: Path) -> str:
    """Return the sha256 of ``path`` from ``voxprint_common.hashing``."""
    from voxprint_common.hashing import sha256_file as digest

    return digest(path)


def write_sha256(path: Path, digest: str | None = None) -> Path:
    """Write ``<file>.sha256`` next to ``path``. Returns the sidecar path."""
    value = digest if digest is not None else sha256_file(path)
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(f"{value}  {path.name}\n", encoding="utf-8", newline="\n")
    return sidecar
