"""Suite icons vendored under gui/assets. There is no floppy-disk save glyph."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QIcon


def asset_path(name: str) -> Path:
    """Path of a packaged icon name."""
    return Path(__file__).resolve().parent / "assets" / name


def load_icon(name: str) -> QIcon:
    """Load a packaged icon as a ``QIcon``."""
    path = asset_path(name)
    if not path.is_file():
        return QIcon()
    return QIcon(str(path))
