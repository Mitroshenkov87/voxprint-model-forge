"""Offscreen Qt for the GUI tests. Set before PySide6 is imported."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
