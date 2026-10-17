"""``forge lora train|extend`` on a tiny CPU stand-in."""

from __future__ import annotations

import json

import pytest

from tests.lora_fakes import TinyAdapter, make_project
from voxprint_forge.cli import main
from voxprint_forge.lora import bases


@pytest.fixture(autouse=True)
def tiny_base():
    bases.register(TinyAdapter)
    yield
    bases.unregister("tiny-test")


ARGS = [
    "--base",
    "tiny-test",
    "--vram",
    "24",
    "--steps",
    "4",
    "--resolution",
    "64",
    "--save-every",
    "2",
    "--no-previews",
]


def test_dry_run_prints_the_plan(tmp_path, capsys) -> None:
    make_project(tmp_path)
    code = main(["--json", "lora", "train", "--project", str(tmp_path), "--dry-run", *ARGS])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0 and payload["result"]["state"] == "dry-run" and payload["result"]["steps"] == 4


def test_train_prints_json_events_then_result(tmp_path, capsys) -> None:
    make_project(tmp_path)
    code = main(["--json", "lora", "train", "--project", str(tmp_path), *ARGS])
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith('{"event"')]
    assert code == 0 and lines
    events = [json.loads(line)["event"] for line in lines]
    assert set(events[0]) == {"step", "message_key", "params", "reason_key", "percent", "eta_seconds", "state"}
    assert events[-1]["state"] == "done"


def test_plain_mode_prints_sentences(tmp_path, capsys, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_LANG", "ru")
    make_project(tmp_path)
    assert main(["lora", "train", "--project", str(tmp_path), *ARGS]) == 0
    assert "Обучение: шаг 4 из 4" in capsys.readouterr().out


def test_unprepared_project_is_a_usage_error(tmp_path, capsys) -> None:
    code = main(["--json", "lora", "train", "--project", str(tmp_path / "none"), *ARGS])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2 and payload["error"]["code"] == "lora.err.not_prepared"
