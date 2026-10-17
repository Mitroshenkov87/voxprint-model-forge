"""Where the MiVOLO worker's interpreter is found (the Windows installer layout)."""

from __future__ import annotations

import sys

from voxprint_forge.lora.backends import mivolo_python


def test_env_var_wins(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_MIVOLO_PYTHON", "/opt/x/python")
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    assert mivolo_python() == "/opt/x/python"


def test_sibling_env_beside_the_app_venv(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VOXPRINT_MIVOLO_PYTHON", raising=False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    exe = tmp_path / "venv-mivolo" / "Scripts" / "python.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    assert mivolo_python() == str(exe)


def test_falls_back_to_this_interpreter(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VOXPRINT_MIVOLO_PYTHON", raising=False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    assert mivolo_python() == sys.executable
