"""A prepare crash keeps the trace rows already written, and training waits for a new prepare.

Photos with no trace row are not checked and are not set aside. Rule 5 is unchanged.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from tests.lora_fakes import TinyAdapter, make_project
from tests.test_lora_gate_trace_file import (
    ADULT,
    FakeCaptioner,
    ScriptedEstimator,
    ScriptedJudge,
    person,
    photos,
)
from tests.test_lora_prepare import photos as plain_photos
from tests.test_lora_prepare import run as prepare_object
from voxprint_forge.lora import prepare as prepare_mod
from voxprint_forge.lora.errors import LoraBlocked, LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import RULE_VERSION, AgeReading, Gate
from voxprint_forge.lora.gate_run import gate_sha
from voxprint_forge.lora.photos import photo_id
from voxprint_forge.lora.prepare import prepare
from voxprint_forge.lora.presets import build_config
from voxprint_forge.lora.project import Project
from voxprint_forge.lora.train import train


def _config():
    return build_config("quick", "balanced", 24, steps=1, resolution=64, save_every=1, batch=1, previews=False)


def _ids(project: Project) -> list[str]:
    if not project.drops.is_file():
        return []
    return [json.loads(line)["id"] for line in project.drops.read_text(encoding="utf-8").splitlines() if line.strip()]


def _marker(project: Project) -> dict:
    return json.loads(project.gate_run.read_text(encoding="utf-8"))


def _prepare(project: Project, files, judge, estimator, **kw):
    return prepare(
        project,
        files,
        person(),
        gate=Gate(judge, estimator, **kw),
        load_captioner=lambda choice: FakeCaptioner(),
        progress=Progress(),
        mode="expert",
    )


def test_crash_after_photo_k_blocks_training_until_prepare_reruns(tmp_path, monkeypatch) -> None:
    """k trace rows survive, training does not start, and a re-run does not duplicate drops."""
    k = 3
    files = photos(tmp_path / "in", 8)
    project = Project(tmp_path / "proj")
    calls = {"n": 0}
    synced = {"n": 0}
    monkeypatch.setattr("voxprint_forge.lora.gate_run.os.fsync", lambda fd: synced.__setitem__("n", synced["n"] + 1))
    real = prepare_mod.append_trace_row

    def crash(proj, row):
        marker = _marker(proj)
        assert set(marker) == {"run_id", "rule_version", "gate_sha", "started", "state"}
        assert marker["state"] == "running" and marker["gate_sha"] == ""
        assert marker["rule_version"] == RULE_VERSION == 5
        if calls["n"] >= k:
            raise RuntimeError("crash")
        calls["n"] += 1
        real(proj, row)

    monkeypatch.setattr(prepare_mod, "append_trace_row", crash)
    dropped = AgeReading(0, ())
    clear = AgeReading(1, (31.0,))
    judge = ScriptedJudge([ADULT] * 8)
    estimator = ScriptedEstimator([dropped, clear, clear, dropped, clear, clear, clear, clear])
    with pytest.raises(RuntimeError, match="crash"):
        _prepare(project, files, judge, estimator)

    trace = project.gate_trace.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in trace.splitlines()]
    assert len(rows) == k
    assert [row["file"] for row in rows] == [files[i].name for i in range(k)]
    assert synced["n"] >= k
    assert _ids(project) == [photo_id(files[0])]
    assert photo_id(files[3]) not in _ids(project)
    assert _marker(project)["state"] == "running"
    run_id = _marker(project)["run_id"]
    drops_before = project.drops.read_text(encoding="utf-8")
    assert not (project.root / "refusals.jsonl").exists()
    with pytest.raises(LoraUsageError) as info:
        train(project, _config(), TinyAdapter(), Progress())
    assert info.value.code == "lora.err.not_prepared"
    assert project.drops.read_text(encoding="utf-8") == drops_before
    assert not (project.root / "refusals.jsonl").exists()
    # The crash is before prepare approves the project. Nothing was marked passed or refused.
    assert not project.file.is_file() or project.read().get("gate_passed") is not True

    monkeypatch.setattr(prepare_mod, "append_trace_row", real)
    summary = _prepare(
        project,
        files,
        ScriptedJudge([ADULT] * 8),
        ScriptedEstimator([dropped, clear, clear, dropped, clear, clear, clear, clear]),
    )
    assert summary["state"] == "done"
    text = project.gate_trace.read_text(encoding="utf-8")
    assert text.startswith(trace)
    assert len(text.splitlines()) == k + len(files)
    assert _ids(project) == [photo_id(files[0]), photo_id(files[3])]
    assert len(_ids(project)) == len(set(_ids(project)))
    marker = _marker(project)
    assert marker["state"] == "complete" and marker["run_id"] != run_id
    assert marker["rule_version"] == RULE_VERSION
    pinned = hashlib.sha256((project.root / "gate.json").read_bytes()).hexdigest()
    assert marker["gate_sha"] == gate_sha(project) == pinned
    assert project.read()["gate_passed"] is True
    for name in ("drops.jsonl.tmp", "gate_trace.jsonl.tmp", "gate_run.json.tmp"):
        assert not (project.root / name).exists()

    clean = Project(tmp_path / "clean")
    _prepare(
        clean,
        files,
        ScriptedJudge([ADULT] * 8),
        ScriptedEstimator([dropped, clear, clear, dropped, clear, clear, clear, clear]),
    )
    assert clean.drops.read_text(encoding="utf-8") == project.drops.read_text(encoding="utf-8")


def test_a_crash_before_the_first_photo_leaves_the_run_open(tmp_path, monkeypatch) -> None:
    files = photos(tmp_path / "in", 6)
    project = Project(tmp_path / "proj")

    def boom(path):
        raise RuntimeError("crash-before-photo")

    monkeypatch.setattr(prepare_mod, "decode_photo", boom)
    with pytest.raises(RuntimeError, match="crash-before-photo"):
        _prepare(project, files, ScriptedJudge([ADULT] * 6), ScriptedEstimator([AgeReading(1, (31.0,))] * 6))
    assert _marker(project)["state"] == "running"
    assert not project.gate_trace.exists()
    assert not project.drops.exists()
    with pytest.raises(LoraUsageError) as info:
        train(project, _config(), TinyAdapter(), Progress())
    assert info.value.code == "lora.err.not_prepared"
    assert not project.drops.exists()
    assert not (project.root / "refusals.jsonl").exists()


@pytest.mark.parametrize("state", ["running", "interrupted", "weird"])
def test_an_open_gate_run_blocks_training_without_dropping_photos(tmp_path, state: str) -> None:
    project = make_project(tmp_path / "p", 10)
    pin_text = (project.root / "gate.json").read_text(encoding="utf-8")
    items = [(item.id, item.file, item.long, item.short) for item in project.items()]
    project.gate_run.write_text(
        json.dumps(
            {
                "run_id": "deadbeef",
                "rule_version": RULE_VERSION,
                "gate_sha": "",
                "started": "2026-11-02T00:00:00+00:00",
                "state": state,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(LoraUsageError) as info:
        train(project, _config(), TinyAdapter(), Progress())
    assert info.value.code == "lora.err.not_prepared"
    assert not project.drops.exists()
    assert not project.gate_trace.exists()
    assert not (project.root / "refusals.jsonl").exists()
    assert (project.root / "gate.json").read_text(encoding="utf-8") == pin_text
    assert project.read()["gate_passed"] is True
    assert [(item.id, item.file, item.long, item.short) for item in project.items()] == items


def test_a_count_refusal_does_not_open_a_gate_run(tmp_path) -> None:
    with pytest.raises(LoraUsageError) as info:
        prepare_object(tmp_path, plain_photos(tmp_path / "in", 9))
    assert info.value.code == "lora.err.too_few"
    assert not (tmp_path / "proj" / "gate_run.json").exists()


def test_cross_photo_average_still_upgrades_the_stored_tier(tmp_path) -> None:
    """The average upgrade is applied to the rows on disk, once each, not as a second copy."""
    verdict = replace(ADULT, min_age=20.0)
    reading = AgeReading(1, (15.0,))
    one = Gate(ScriptedJudge([verdict]), ScriptedEstimator([reading])).check(object(), attested=True)
    assert one.tier == "ambiguous"
    files = photos(tmp_path / "in", 6)
    project = Project(tmp_path / "proj")
    with pytest.raises(LoraBlocked) as info:
        _prepare(
            project,
            files,
            ScriptedJudge([verdict] * 6),
            ScriptedEstimator([reading] * 6),
            identify=lambda image: "same",
        )
    assert info.value.code == "lora.err.blocked"
    rows = [json.loads(line) for line in project.gate_trace.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6
    assert [row["tier"] for row in rows] == ["strong"] * 6
    assert [row["decision"] for row in rows] == ["minor"] * 6
    assert len({row["id"] for row in rows}) == 6
    ids = _ids(project)
    assert ids[-1] == "job" and len(ids) == 7 and len(set(ids)) == 7
    marker = _marker(project)
    assert marker["state"] == "complete" and marker["gate_sha"] == ""
    assert not (project.root / "gate.json").exists()
