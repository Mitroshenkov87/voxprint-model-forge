"""Detect command-line training runs from a project folder's files (no Qt).

The GUI starts trainings through its own runner; ``forge-train`` and
``voxprint-forge-cli lora train`` write the same project folder without the
GUI knowing. This module classifies a folder from its files only, so the app
can offer and follow runs it did not start.
"""

from __future__ import annotations

import time
from pathlib import Path

from voxprint_forge.lora.progress_log import TRAIN_FILE, read_progress
from voxprint_forge.paths import default_projects_dir, legacy_cli_projects_dir

STALE_SECONDS = 600
FINAL_LORA = "lora.safetensors"
PROJECT_FILE = "project.json"


def run_state(folder: Path, *, now: float | None = None, stale_seconds: float = STALE_SECONDS) -> str:
    """Classify a project folder: ``"none"``, ``"running"``, ``"stopped"`` or ``"done"``.

    ``"done"`` when ``train.json``'s stage is ``done`` or the final ``lora.safetensors``
    exists. Otherwise, with a ``train.json`` report, ``"running"`` while the report is
    fresher than ``stale_seconds`` and its stage is not ``failed``, and ``"stopped"``
    when it is older or failed. ``"none"`` when there is nothing to classify.
    """
    root = Path(folder)
    train_file = root / TRAIN_FILE
    final = root / FINAL_LORA
    if train_file.is_file():
        stage = _stage(root)
        if stage == "done" or final.is_file():
            return "done"
        moment = time.time() if now is None else now
        try:
            age = moment - train_file.stat().st_mtime
        except OSError:
            age = stale_seconds + 1.0
        if stage != "failed" and age <= stale_seconds:
            return "running"
        return "stopped"
    return "done" if final.is_file() else "none"


def find_cli_projects(roots: list[Path] | tuple[Path, ...], *, now: float | None = None) -> list[Path]:
    """Project folders (those holding ``project.json``) directly under each existing root.

    Newest activity first: a folder's stamp is the newest ``train.json`` / ``project.json``
    mtime. ``now`` is accepted for symmetry with :func:`run_state`; ordering is by mtime,
    so it is not used.
    """
    found: list[Path] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for folder in root.iterdir():
            if folder.is_dir() and (folder / PROJECT_FILE).is_file():
                found.append(folder)
    found.sort(key=_stamp, reverse=True)
    return found


def default_roots(home: Path) -> list[Path]:
    """The folders scanned for trainings: the app's own projects, the Documents default and the laptop wrapper's.

    Only roots that exist are returned, so a machine without ``~/VOXPrint/projects``
    (the old ``forge-train`` layout, listed only so leftover projects stay visible)
    or without ``<Documents>/VOXPrint Projects/Model Forge`` simply scans what is there.
    """
    roots = [
        Path(home) / "lora",
        default_projects_dir(),
        legacy_cli_projects_dir(),
    ]
    return [root for root in roots if root.is_dir()]


def _stage(root: Path) -> str:
    """The ``stage`` word of ``train.json``, or ``""`` when the file is missing or unreadable."""
    try:
        row = read_progress(root)
    except (OSError, ValueError):
        return ""
    if not row:
        return ""
    return str(row.get("stage") or "")


def _stamp(folder: Path) -> float:
    """The newest mtime among ``train.json`` and ``project.json`` (0 when neither is readable)."""
    stamp = 0.0
    for name in (TRAIN_FILE, PROJECT_FILE):
        try:
            stamp = max(stamp, (folder / name).stat().st_mtime)
        except OSError:
            continue
    return stamp
