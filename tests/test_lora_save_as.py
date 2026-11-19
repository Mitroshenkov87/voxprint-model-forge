"""Named save-as copies write the trigger Canvas reads, and keep existing metadata."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from safetensors.torch import load_file, save_file

from tests.lora_fakes import TinyAdapter, make_project
from tests.test_lora_export import PERSON, header, trained
from voxprint_forge.cli import main
from voxprint_forge.lora import bases
from voxprint_forge.lora.errors import LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.export import latest_trained_project, save_as
from voxprint_forge.lora.presets import build_config
from voxprint_forge.lora.train import train


@pytest.fixture(autouse=True)
def tiny_base():
    bases.register(TinyAdapter)
    yield
    bases.unregister("tiny-test")


def test_train_checkpoints_write_the_trigger(tmp_path) -> None:
    project = trained(tmp_path)
    last = project.checkpoints / "step-000004.safetensors"
    early = project.checkpoints / "step-000002.safetensors"
    assert header(last)["voxprint.subject.trigger"] == "tcube"
    assert header(early)["voxprint.subject.trigger"] == "tcube"
    assert header(last)["voxprint.lora.base"] == "tiny-test"
    assert header(last)["step"] == "4"


def test_save_as_writes_trigger_metadata_and_sidecar(tmp_path) -> None:
    subject = {**PERSON, "release_attached": False, "release_sha256": "", "adult_generation": True}
    project = trained(tmp_path, subject=subject)
    result = save_as(project, "Tatiana_Model")
    target = project.root / "Tatiana_Model.safetensors"
    assert Path(result["file"]) == target and result["source"] == "step-000004.safetensors"
    meta = header(target)
    assert meta["voxprint.subject.trigger"] == "demo_person"
    assert meta["voxprint.lora.adult_generation"] == "true"
    assert meta["step"] == "4"
    assert meta["voxprint.lora.base"] == "tiny-test"
    sidecar = json.loads((project.root / "voxprint-model.json").read_text(encoding="utf-8"))
    assert sidecar["trigger"] == "demo_person"
    assert sidecar["file"] == "Tatiana_Model.safetensors"
    assert sidecar["voxprint.lora.adult_generation"] is True


def test_save_as_restores_the_trigger_on_a_bare_checkpoint_copy(tmp_path) -> None:
    """A forge-save-as copy of an older checkpoint that only stored base and step."""
    project = trained(tmp_path)
    source = project.checkpoints / "step-000004.safetensors"
    tensors = load_file(str(source))
    save_file(tensors, str(source), metadata={"voxprint.lora.base": "tiny-test", "step": "4", "custom.keep": "yes"})
    assert "voxprint.subject.trigger" not in header(source)
    result = save_as(project, "Tatiana_Model.safetensors")
    meta = header(result["file"])
    assert meta["voxprint.subject.trigger"] == "tcube"
    assert meta["custom.keep"] == "yes"
    assert meta["step"] == "4"
    sidecar = json.loads((Path(result["file"]).parent / "voxprint-model.json").read_text(encoding="utf-8"))
    assert sidecar["trigger"] == "tcube"


def test_save_as_can_copy_an_earlier_checkpoint(tmp_path) -> None:
    project = trained(tmp_path)
    result = save_as(project, tmp_path / "out" / "early", checkpoint=2)
    assert Path(result["file"]) == tmp_path / "out" / "early.safetensors"
    assert result["source"] == "step-000002.safetensors"
    assert header(result["file"])["voxprint.lora.steps"] == "2"
    assert header(result["file"])["voxprint.subject.trigger"] == "tcube"


def _trained_at(folder: Path):
    project = make_project(folder)
    config = build_config("quick", "balanced", 24, steps=4, resolution=64, save_every=2, previews=False)
    train(project, config, TinyAdapter(), Progress())
    return project


def test_latest_trained_project_picks_the_newest(tmp_path) -> None:
    older = _trained_at(tmp_path / "root" / "old")
    newer = _trained_at(tmp_path / "root" / "new")
    (older.root / "train.json").write_text("{}", encoding="utf-8")
    (newer.root / "train.json").write_text("{}", encoding="utf-8")
    now = time.time()
    os.utime(older.root / "project.json", (now - 20, now - 20))
    os.utime(older.root / "train.json", (now - 10, now - 10))
    os.utime(newer.root / "project.json", (now - 5, now - 5))
    os.utime(newer.root / "train.json", (now, now))
    found = latest_trained_project(roots=(tmp_path / "root",))
    assert found.root == newer.root
    with pytest.raises(LoraUsageError) as missing:
        latest_trained_project(roots=(tmp_path / "empty",))
    assert missing.value.code == "lora.err.no_trained_project"


def test_cli_save_as_prints_the_wrapper_line(tmp_path, capsys) -> None:
    project = trained(tmp_path)
    capsys.readouterr()
    assert main(["lora", "save-as", "Tatiana_Model", "--project", str(project.root)]) == 0
    line = capsys.readouterr().out.strip()
    assert line == f"LoRA saved as: {project.root / 'Tatiana_Model.safetensors'} (from step-000004.safetensors)"
    capsys.readouterr()
    assert main(["--json", "lora", "save-as", "Tatiana_Model", "--project", str(project.root)]) == 0
    body = capsys.readouterr().out
    start = body.rindex("\n{") if "\n{" in body else body.index("{")
    payload = json.loads(body[start:])
    assert payload["ok"] is True
    assert payload["result"]["source"] == "step-000004.safetensors"
    assert payload["result"]["file"].endswith("Tatiana_Model.safetensors")
