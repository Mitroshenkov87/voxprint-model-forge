"""The public docstring gate and the generated CLI reference."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from voxprint_forge.cli_reference import render

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
