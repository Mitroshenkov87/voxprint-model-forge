"""Launch defaults for the app window and the GUI-to-terminal selection state (no Qt).

The window preloads its draft and Expert values from :class:`LaunchDefaults`
(built once by :func:`parse_launch_args`, flags over the optional
``<home>/model-forge/gui-defaults.json``). Whenever the draft's photo folder
changes, the app records it in ``<home>/model-forge/gui-selection.json`` so the
terminal wrapper (``forge-train --photos-from-gui``) can ask for it with
``python -m voxprint_forge.app.launch --print-selection``. A
``release.txt`` sitting directly inside a photo folder is the release
auto-detect rule shared by the GUI and the wrapper; the rule itself lives in
:mod:`voxprint_forge.lora.release_file` and is re-exported here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from voxprint_common.paths import home

from voxprint_forge.lora.release_file import RELEASE_FILE as RELEASE_FILE
from voxprint_forge.lora.release_file import find_release as find_release

SELECTION_SCHEMA = "voxprint-forge-gui-selection/1"
NO_SELECTION_MESSAGE = (
    "No photo folder selected in the Forge window yet. Pick one there "
    "(Add a person's folder / Add a folder), then run this again."
)
CROP_MODES = ("both", "subject", "whole")


def selection_file() -> Path:
    """Where the GUI records the picked photo folder: ``<home>/model-forge/gui-selection.json``."""
    return home() / "model-forge" / "gui-selection.json"


def defaults_file() -> Path:
    """The optional launch config: ``<home>/model-forge/gui-defaults.json`` (the LaunchDefaults keys)."""
    return home() / "model-forge" / "gui-defaults.json"


def write_selection(folder: Path | None) -> None:
    """Record the picked photo folder (or None) atomically (tmp file + ``os.replace``)."""
    target = selection_file()
    payload = {
        "schema": SELECTION_SCHEMA,
        "photos_dir": str(Path(folder).resolve()) if folder is not None else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()


def read_selection() -> Path | None:
    """The folder the GUI recorded, only when the file parses, names a folder and that folder exists."""
    try:
        data = json.loads(selection_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    photos_dir = data.get("photos_dir")
    if not isinstance(photos_dir, str) or not photos_dir:
        return None
    folder = Path(photos_dir)
    return folder if folder.is_dir() else None


@dataclass(frozen=True)
class LaunchDefaults:
    """What the window preloads: the draft crop, the Expert rank/alpha/steps, the base and the 18+ boxes."""

    crop: str = "both"
    rank: int = 64
    steps: int = 1600
    base: str = "z-image"
    adult: bool = True
    photos: Path | None = None


def _config_defaults() -> dict[str, object]:
    """The optional ``gui-defaults.json``; a missing or broken file means no overrides."""
    try:
        data = json.loads(defaults_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return dict(data) if isinstance(data, dict) else {}


def _int_or(value: object, fallback: int) -> int:
    """An int from JSON, or ``fallback`` when the value is not an integer."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return fallback


def parse_launch_args(argv: list[str]) -> LaunchDefaults:
    """Launch flags over ``gui-defaults.json`` over the built-in defaults.

    ``--rank`` and ``--steps`` must be positive integers; ``--adult`` and
    ``--no-adult`` override the config's ``adult`` key in either direction.
    """
    config = _config_defaults()
    parser = argparse.ArgumentParser(prog="voxprint-model-forge", add_help=False)
    parser.add_argument("--crop", choices=CROP_MODES)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--base")
    adult = parser.add_mutually_exclusive_group()
    adult.add_argument("--adult", dest="adult", action="store_true", default=None)
    adult.add_argument("--no-adult", dest="adult", action="store_false", default=None)
    parser.add_argument("--photos", type=Path)
    ns = parser.parse_args(list(argv))
    if ns.rank is not None and ns.rank < 1:
        parser.error("--rank must be a positive integer")
    if ns.steps is not None and ns.steps < 1:
        parser.error("--steps must be a positive integer")
    crop = ns.crop if ns.crop is not None else str(config.get("crop") or LaunchDefaults.crop)
    if crop not in CROP_MODES:
        crop = LaunchDefaults.crop
    rank = ns.rank if ns.rank is not None else _int_or(config.get("rank"), LaunchDefaults.rank)
    if rank < 1:
        rank = LaunchDefaults.rank
    steps = ns.steps if ns.steps is not None else _int_or(config.get("steps"), LaunchDefaults.steps)
    if steps < 1:
        steps = LaunchDefaults.steps
    base = ns.base if ns.base is not None else str(config.get("base") or LaunchDefaults.base)
    if ns.adult is not None:
        grown = ns.adult
    elif "adult" in config:
        grown = bool(config.get("adult"))
    else:
        grown = LaunchDefaults.adult
    photos = ns.photos
    if photos is None:
        configured = config.get("photos")
        if isinstance(configured, str) and configured:
            photos = Path(configured)
    return LaunchDefaults(crop=crop, rank=rank, steps=steps, base=base, adult=grown, photos=photos)


def main(argv: list[str] | None = None) -> int:
    """``python -m voxprint_forge.app.launch --print-selection``: print the picked folder, or exit 3."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--print-selection" not in args:
        print("usage: python -m voxprint_forge.app.launch --print-selection", file=sys.stderr)
        return 2
    folder = read_selection()
    if folder is None:
        print(NO_SELECTION_MESSAGE, file=sys.stderr)
        return 3
    print(folder)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
