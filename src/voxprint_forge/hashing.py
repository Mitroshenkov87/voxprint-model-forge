"""SHA-256 sidecars. The digest itself comes from ``voxprint_common.hashing``."""

from __future__ import annotations

from pathlib import Path

from voxprint_common.hashing import sha256_file

__all__ = ["sha256_file", "write_sha256"]


def write_sha256(path: Path, digest: str | None = None) -> Path:
    """Write ``<file>.sha256`` next to ``path``. Returns the sidecar path."""
    value = digest if digest is not None else sha256_file(path)
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(f"{value}  {path.name}\n", encoding="utf-8")
    return sidecar
