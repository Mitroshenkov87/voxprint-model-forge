"""ProcessRunner with a real child process: the PAUSE file pauses, Stop ends it (Linux and Windows)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from voxprint_forge.app import engine, runner  # noqa: E402

CHILD = r'''
import json, sys, time
from pathlib import Path
project = Path(sys.argv[sys.argv.index("--project") + 1])
print(json.dumps({"event": {"step": "train", "message_key": "lora.progress.train", "state": "running"}}), flush=True)
deadline = time.time() + 60
while time.time() < deadline:
    if (project / "PAUSE").exists():
        envelope = {"ok": True, "command": "lora train", "exit_code": 0, "result": {"state": "paused"}}
        print(json.dumps(envelope, indent=2))
        sys.exit(0)
    time.sleep(0.05)
sys.exit(3)
'''


@pytest.fixture
def child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    monkeypatch.setattr(runner, "cli_command", lambda: [sys.executable, str(script)])
    project = tmp_path / "proj"
    project.mkdir()
    return project


def _start(qtbot, project: Path):
    job = runner.ProcessRunner()
    events: list[engine.EngineEvent] = []
    done: list[engine.Outcome] = []
    job.start(["--json", "lora", "train", "--project", str(project)], events.append, done.append)
    qtbot.waitUntil(lambda: bool(events), timeout=30000)
    return job, done


def test_the_pause_file_pauses_and_the_engine_exits_cleanly(qtbot, child: Path) -> None:
    job, done = _start(qtbot, child)
    engine.request_pause(child)
    qtbot.waitUntil(lambda: bool(done), timeout=30000)
    assert done[0].ok and done[0].result == {"state": "paused"} and not job.busy
    engine.clear_pause(child)
    assert not engine.pause_file(child).exists()


def test_stop_ends_the_process(qtbot, child: Path) -> None:
    job, done = _start(qtbot, child)
    job.stop()
    qtbot.waitUntil(lambda: bool(done), timeout=30000)
    assert not done[0].ok and not job.busy
