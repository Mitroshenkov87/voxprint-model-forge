"""Directories under ~/forge-out. Stage markers live beside the work tree."""

from __future__ import annotations

import json
from pathlib import Path


class Layout:
    """Directories and stage markers under one forge-out root."""
    def __init__(self, forge_out: Path) -> None:
        self.root = forge_out

    def model_dir(self, model_id: str) -> Path:
        """Work directory for one model id."""
        return self.root / "work" / model_id

    def stage(self, model_id: str, name: str) -> Path:
        """Marker path for one model stage."""
        return self.root / "stages" / model_id / f"{name}.json"

    def log_dir(self) -> Path:
        """Directory for stage logs."""
        return self.root / "logs"

    def manifest_dir(self) -> Path:
        """Directory for the run manifest and MODELS snippet."""
        return self.root / "manifests"


def read_marker(path: Path) -> dict[str, object] | None:
    """Load a stage marker, or None when the stage has not finished."""
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return None
    return loaded


def write_marker(path: Path, payload: dict[str, object]) -> None:
    """Write a stage marker as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
