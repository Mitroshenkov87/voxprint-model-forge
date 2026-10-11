"""Offscreen Qt for the GUI tests. Set before PySide6 is imported."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _suite_home(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep model components out of the real suite directory."""
    root = tmp_path_factory.mktemp("suite-home")
    monkeypatch.setenv("VOXPRINT_HOME", str(root))
    monkeypatch.delenv("VOXPRINT_MODELS_DIR", raising=False)
    monkeypatch.delenv("VOXPRINT_MODEL_CACHE", raising=False)
