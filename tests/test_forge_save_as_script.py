"""``scripts/forge-save-as`` argument assembly: a stub CLI records the argv."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the wrappers are bash scripts with a Unix venv path",
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "forge-save-as"


def _stub_cli(home: Path) -> Path:
    """A fake ``voxprint-forge-cli`` that appends each argv as a JSON line to $STUB_LOG."""
    cli = home / "VOXPrint" / "forge" / "venv" / "bin" / "voxprint-forge-cli"
    cli.parent.mkdir(parents=True)
    cli.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$(printf '%s\\n' \"$@\" | python3 -c "
        "'import json,sys; print(json.dumps(sys.stdin.read().splitlines()))')\" >> \"${STUB_LOG:?}\"\n",
        encoding="utf-8",
    )
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return cli


def _run(tmp_path: Path, home: Path, *args: str) -> tuple[int, str, list[list[str]]]:
    log = tmp_path / "stub.log"
    env = {**os.environ, "HOME": str(home), "STUB_LOG": str(log)}
    import subprocess

    completed = subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.is_file() else []
    return completed.returncode, completed.stderr + completed.stdout, calls


@pytest.fixture
def home(tmp_path: Path) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    _stub_cli(home)
    return home


def test_name_only_calls_lora_save_as(tmp_path, home) -> None:
    code, _text, calls = _run(tmp_path, home, "Tatiana_Model")
    assert code == 0
    assert calls == [["lora", "save-as", "Tatiana_Model"]]


def test_project_and_checkpoint_pass_through(tmp_path, home) -> None:
    project = tmp_path / "proj"
    code, _text, calls = _run(tmp_path, home, "Tatiana_Model", "--project", str(project), "--checkpoint", "1600")
    assert code == 0
    assert calls == [["lora", "save-as", "Tatiana_Model", "--project", str(project), "--checkpoint", "1600"]]


def test_missing_name_prints_usage(tmp_path, home) -> None:
    code, text, calls = _run(tmp_path, home)
    assert code == 2 and not calls
    assert "Usage: forge-save-as" in text
