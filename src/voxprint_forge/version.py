"""Release identity. build = CI run number + BUILD.json offset. Codename stays with the suite."""

from __future__ import annotations

import json
import os
from pathlib import Path

from voxprint_forge import __version__

CODENAME_FALLBACK = "Charash"


def _build_file() -> Path | None:
    env = os.environ.get("FORGE_ROOT")
    roots = []
    if env:
        roots.append(Path(env))
    roots.append(Path.cwd())
    roots.append(Path(__file__).resolve().parents[2])
    for root in roots:
        path = root / "BUILD.json"
        if path.is_file():
            return path
    return None


def load_identity() -> tuple[int, str]:
    path = _build_file()
    offset = 0
    codename = CODENAME_FALLBACK
    if path is not None:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            offset = int(raw.get("offset", 0))
            name = raw.get("codename")
            if isinstance(name, str) and name.strip():
                codename = name.strip()
    run = os.environ.get("GITHUB_RUN_NUMBER", "0").strip() or "0"
    try:
        number = int(run)
    except ValueError:
        number = 0
    return offset + number, codename


def version_line() -> str:
    """`X.Y.Z · build N "Charash"`."""
    build, codename = load_identity()
    return f'{__version__} · build {build} "{codename}"'
