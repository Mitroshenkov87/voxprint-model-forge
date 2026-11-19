"""Resumable markers under ``~/.done/diffusion-int4/<step>``."""

from __future__ import annotations

import os
from pathlib import Path

from voxprint_forge.layout import read_marker, write_marker

DONE_NAMESPACE = "diffusion-int4"


def done_dir(explicit: Path | None = None) -> Path:
    """Marker directory. ``FORGE_DONE_DIR`` overrides the home default."""
    if explicit is not None:
        return explicit
    env = os.environ.get("FORGE_DONE_DIR", "").strip()
    if env:
        return Path(env)
    return Path.home() / ".done" / DONE_NAMESPACE


def marker_path(root: Path, step_id: str) -> Path:
    """Path of one step marker. The file name is the step id."""
    return root / step_id


def load_done(root: Path, step_id: str) -> dict[str, object] | None:
    """Return the marker payload, or None when the step has not finished."""
    return read_marker(marker_path(root, step_id))


def save_done(root: Path, step_id: str, payload: dict[str, object]) -> None:
    """Write one step marker."""
    write_marker(marker_path(root, step_id), payload)
