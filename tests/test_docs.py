"""The public docstring gate and the generated CLI reference."""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

from voxprint_forge.cli_reference import render
from voxprint_forge.gui.window import DISPLAY_NAME

ROOT = Path(__file__).resolve().parents[1]


def test_public_docstrings_cover_at_least_65_percent() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "interrogate", "src/voxprint_forge"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_cli_reference_matches_the_parser() -> None:
    assert (ROOT / "docs" / "cli-reference.md").read_text(encoding="utf-8") == render()


def test_install_names_stay_in_the_build_spec() -> None:
    spec = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    install = spec["tool"]["voxprint"]["install"]
    assert install["display-name"] == DISPLAY_NAME
    assert install["start-menu-folder"] == "VOXPrint AI Suite"
    assert install["shortcut-name"] == DISPLAY_NAME
    assert install["windows-dir"] == r"C:\Program Files\VOXPrint AI Suite\Model Forge"
    assert install["linux-dir"] == "/opt/voxprint-ai-suite/model-forge"
    assert install["gui-exe"] == "voxprint-model-forge.exe"
    assert install["cli-exe"] == "voxprint-forge-cli.exe"
    assert install["user-data"] == r"%LOCALAPPDATA%\VOXPrint AI Suite"
    assert spec["project"]["name"] == "voxprint-forge"
    assert spec["project"]["scripts"]["voxprint-forge-cli"] == "voxprint_forge.cli:main"
    assert spec["project"]["scripts"]["forge"] == "voxprint_forge.cli:main"
    assert spec["project"]["gui-scripts"]["voxprint-model-forge"] == "voxprint_forge.app.__main__:main"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# voxprint-model-forge\n")
    assert "## Install names" in readme


def test_the_offline_guide_matches_the_manual() -> None:
    """``scripts/build_guide.py`` output is committed and current, so the wheel and setup carry it."""
    import importlib.util

    import pytest

    pytest.importorskip("markdown")
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_guide.py"
    spec = importlib.util.spec_from_file_location("build_guide", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.stale() == []
