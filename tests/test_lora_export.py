"""LoRA export: typed header metadata, the voxprint-model/1 sidecar and the user-models install."""

from __future__ import annotations

import hashlib
import json

import pytest
from PIL import Image
from safetensors import safe_open

from tests.lora_fakes import TinyAdapter, gradient, make_project
from voxprint_forge.cli import main
from voxprint_forge.lora import bases
from voxprint_forge.lora.errors import LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.export import export, gate_like
from voxprint_forge.lora.presets import build_config
from voxprint_forge.lora.train import train


@pytest.fixture(autouse=True)
def tiny_base():
    bases.register(TinyAdapter)
    yield
    bases.unregister("tiny-test")


def trained(tmp_path, *, preview=False, subject=None):
    project = make_project(tmp_path / "proj")
    data = project.read()
    data["subject"] = subject or {
        "kind": "object",
        "name": "Test cube",
        "stage_name": "",
        "trigger": "tcube",
        "consent_obtained": False,
        "adult_confirmed": False,
        "rights_confirmed": True,
        "provenance_note": "made for a test",
        "release_attached": True,
        "release_sha256": "ab" * 32,
    }
    project.write(data)
    config = build_config("quick", "balanced", 24, steps=4, resolution=64, save_every=2, previews=False)
    train(project, config, TinyAdapter(), Progress())
    if preview:
        project.previews.mkdir(parents=True, exist_ok=True)
        gradient(5).save(project.previews / "step-000004-0.png")
    return project


def header(path):
    with safe_open(str(path), framework="pt") as handle:
        return handle.metadata()


def test_export_writes_typed_metadata_and_sidecar(tmp_path) -> None:
    project = trained(tmp_path, preview=True)
    result = export(project, Progress(), tags=("cube", "test"))
    target = project.root / "export" / f"{result['id']}.safetensors"
    meta = header(target)
    assert meta["voxprint.lora.base"] == "tiny-test" and meta["voxprint.lora.steps"] == "4"
    assert meta["voxprint.subject.rights_confirmed"] == "true"
    assert meta["voxprint.subject.release_sha256"] == "ab" * 32
    assert all(key.startswith(("voxprint.subject.", "voxprint.lora.")) for key in meta)
    assert not any(gate_like(key) for key in meta)
    assert str(tmp_path) not in json.dumps(meta)
    sidecar = json.loads((target.parent / "voxprint-model.json").read_text())
    assert sidecar["schema"] == "voxprint-model/1" and sidecar["kind"] == "object"
    assert sidecar["base"] == {"mdl_id": "tiny-test", "revision": "0" * 40}
    assert sidecar["default_strength"] == 0.8 and sidecar["strength_range"] == [0.0, 1.5]
    assert sidecar["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert sidecar["size"] == target.stat().st_size and sidecar["tags"] == ["cube", "test"]
    assert sidecar["preview"] == "preview.webp"
    assert sidecar["preview_sha256"] == hashlib.sha256((target.parent / "preview.webp").read_bytes()).hexdigest()
    assert Image.open(target.parent / "preview.webp").format == "WEBP"
    assert (target.parent / "LICENSE").is_file() and (target.parent / "NOTICE").is_file()
    assert not any(gate_like(key) for key in sidecar)
    assert meta["voxprint.lora.adult_content"] == "false" and sidecar["voxprint.lora.adult_content"] is False
    assert "voxprint.subject.consent_method" not in meta and "voxprint.subject.consent_method" not in sidecar


PERSON = {
    "kind": "person",
    "name": "Demo Person",
    "stage_name": "",
    "trigger": "demo_person",
    "consent_obtained": True,
    "adult_confirmed": True,
    "rights_confirmed": False,
    "provenance_note": "",
}


@pytest.mark.parametrize(
    ("extra", "method", "digest", "adult"),
    [
        ({"release_attached": False, "release_sha256": "", "consent_method": "verbal", "adult_content": True},
         "verbal", "", "true"),
        ({"release_attached": True, "release_sha256": "cd" * 32, "consent_method": "file", "adult_content": False},
         "file", "cd" * 32, "false"),
        ({"release_attached": False, "release_sha256": ""}, "verbal", "", "false"),  # project from before the field
    ],
)
def test_person_export_records_consent_method_and_adult_content(tmp_path, extra, method, digest, adult) -> None:
    project = trained(tmp_path, subject={**PERSON, **extra})
    result = export(project, Progress())
    meta = header(result["file"])
    assert meta["voxprint.subject.consent_method"] == method
    assert meta["voxprint.subject.release_sha256"] == digest
    assert meta["voxprint.lora.adult_content"] == adult
    assert all(key.startswith(("voxprint.subject.", "voxprint.lora.")) for key in meta)
    assert not any(gate_like(key) for key in meta)
    sidecar = json.loads((project.root / "export" / "voxprint-model.json").read_text())
    assert sidecar["voxprint.subject.consent_method"] == method
    assert sidecar["voxprint.lora.adult_content"] is (adult == "true")


def test_install_goes_to_user_models_lora_id(tmp_path, monkeypatch) -> None:
    project = trained(tmp_path)
    monkeypatch.setenv("VOXPRINT_USER_MODELS_DIR", str(tmp_path / "um"))
    result = export(project, Progress(), out=tmp_path / "keep" / "mine.safetensors", install_to_mdl=True)
    folder = tmp_path / "um" / "lora" / result["id"]
    sidecar = json.loads((folder / "voxprint-model.json").read_text())
    assert sidecar["file"] == f"{result['id']}.safetensors"
    assert hashlib.sha256((folder / sidecar["file"]).read_bytes()).hexdigest() == sidecar["sha256"]
    assert json.loads((tmp_path / "keep" / "voxprint-model.json").read_text())["file"] == "mine.safetensors"


def test_earlier_checkpoint_and_stable_id(tmp_path) -> None:
    project = trained(tmp_path)
    first = export(project, Progress(), checkpoint=2)
    again = export(project, Progress(), checkpoint=2)
    assert first["id"] == again["id"] and first["id"].startswith("tcube-")
    assert header(first["file"])["voxprint.lora.steps"] == "2"
    with pytest.raises(LoraUsageError):
        export(project, Progress(), checkpoint=3)


def test_untrained_project_cannot_export(tmp_path) -> None:
    with pytest.raises(LoraUsageError):
        export(make_project(tmp_path), Progress())


def test_gate_like_keys() -> None:
    assert gate_like("voxprint.safety.min_age") and gate_like("voxprint.gate.threshold")
    assert not gate_like("voxprint.subject.stage_name") and not gate_like("voxprint.subject.adult_confirmed")


def test_cli_export(tmp_path, capsys) -> None:
    project = trained(tmp_path)
    capsys.readouterr()
    assert main(["--json", "lora", "export", "--project", str(project.root)]) == 0
    out = capsys.readouterr().out
    body = out[out.index("\n{\n") + 1 :] if "\n{\n" in out else out[out.index("{\n") :]
    assert json.loads(body)["result"]["state"] == "done"
