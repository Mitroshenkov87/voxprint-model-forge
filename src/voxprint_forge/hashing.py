"""SHA-256 of files we download or publish."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    """sha256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def write_sha256(path: Path, digest: str | None = None) -> Path:
    """Write `<file>.sha256` next to path. Returns the sidecar path."""
    value = digest if digest is not None else sha256_file(path)
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(f"{value}  {path.name}\n", encoding="utf-8")
    return sidecar
