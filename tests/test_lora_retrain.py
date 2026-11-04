"""``forge lora retrain --base moss|z-image PROJECT`` trains another base on the same dataset.

The prepared photos and captions stay base-free. A legacy project keeps its
current base (Z-Image when none was stored) and gains the new one. Stand-in
adapters only: no weights are downloaded.
"""

from __future__ import annotations

import json
from pathlib import Path

from tests.lora_fakes import TinyAdapter, make_project
from voxprint_forge.cli import main
from voxprint_forge.lora import bases
from voxprint_forge.lora.bases import BaseInfo
from voxprint_forge.lora.project import Project, add_base


def _adapter(mdl_id: str, name: str) -> type[TinyAdapter]:
    class Adapter(TinyAdapter):
        info = BaseInfo(mdl_id, f"example/{mdl_id}", "0" * 40, "Apache-2.0", name)

    Adapter.__name__ = f"StandIn_{mdl_id.replace('-', '_')}"
    return Adapter


def _install():
    saved = {mdl: bases._REGISTRY[mdl] for mdl in ("moss-image-base", "z-image") if mdl in bases._REGISTRY}
    bases.register(_adapter("moss-image-base", "MOSS stand-in"))
    bases.register(_adapter("z-image", "Z-Image stand-in"))
    return saved


def _restore(saved) -> None:
    for mdl in ("moss-image-base", "z-image"):
        bases._REGISTRY.pop(mdl, None)
    bases._REGISTRY.update(saved)


def _envelope(text: str) -> dict[str, object]:
    start = text.rindex("\n{") if "\n{" in text else text.index("{")
    return json.loads(text[start:])


def _dataset(root: Path) -> dict[str, bytes]:
    found = {"captions.jsonl": (root / "captions.jsonl").read_bytes()}
    for path in sorted((root / "prepared").iterdir()):
        if path.is_file():
            found[f"prepared/{path.name}"] = path.read_bytes()
    return found


def _retrain_args(root: Path, base: str, *extra: str) -> list[str]:
    return [
        "--json", "lora", "retrain", "--base", base, str(root),
        "--no-fetch", "--no-previews", "--vram", "24", "--resolution", "64",
        "--batch", "1", "--save-every", "1", "--review-rounds", "0", "--steps", "2",
        *extra,
    ]


def test_add_base_on_a_legacy_project_keeps_the_old_files_and_the_dataset(tmp_path: Path) -> None:
    project = make_project(tmp_path)
    before = _dataset(tmp_path)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "step-000001.safetensors").write_bytes(b"old-lora")
    (tmp_path / "export").mkdir()
    (tmp_path / "export" / "old.safetensors").write_bytes(b"old-export")
    (tmp_path / "review").mkdir()
    (tmp_path / "review" / "state.json").write_text(
        json.dumps({"schema": "voxprint-lora-review/1", "checkpoint": "checkpoints/step-000001.safetensors"}) + "\n",
        encoding="utf-8",
    )
    assert "bases" not in project.read()
    assert add_base(project, "moss-image-base") == ["z-image", "moss-image-base"]
    assert add_base(project, "moss-image-base") == ["z-image", "moss-image-base"]
    assert project.read()["bases"] == ["z-image", "moss-image-base"]
    assert _dataset(tmp_path) == before
    for line in (tmp_path / "captions.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            assert "base" not in json.loads(line)
    assert (tmp_path / "checkpoints" / "z-image" / "step-000001.safetensors").read_bytes() == b"old-lora"
    assert (tmp_path / "export" / "z-image" / "old.safetensors").read_bytes() == b"old-export"
    assert not (tmp_path / "checkpoints" / "step-000001.safetensors").exists()
    state = json.loads((tmp_path / "review" / "state.json").read_text(encoding="utf-8"))
    assert state["checkpoint"] == "checkpoints/z-image/step-000001.safetensors"
    # The photos are still the legacy Z-Image set until someone trains the new base.
    assert project.legacy_layout() is False
    project.activate_base("z-image")
    assert project.checkpoints == tmp_path / "checkpoints" / "z-image"


def test_retrain_trains_moss_from_a_legacy_dataset(tmp_path: Path, capsys) -> None:
    saved = _install()
    try:
        project = make_project(tmp_path)
        before = _dataset(tmp_path)
        (tmp_path / "checkpoints").mkdir()
        (tmp_path / "checkpoints" / "note.txt").write_text("kept", encoding="utf-8")
        code = main(_retrain_args(project.root, "moss"))
        payload = _envelope(capsys.readouterr().out)
    finally:
        _restore(saved)
    assert code == 0
    assert payload["result"]["state"] == "done"
    data = project.read()
    assert data["bases"] == ["z-image", "moss-image-base"]
    assert data["train"]["base"] == "moss-image-base"
    assert _dataset(tmp_path) == before
    assert (tmp_path / "checkpoints" / "z-image" / "note.txt").read_text(encoding="utf-8") == "kept"
    assert list((tmp_path / "checkpoints" / "moss-image-base").glob("step-*.safetensors"))
    assert list((tmp_path / "checkpoints").glob("step-*.safetensors")) == []


def test_retrain_appends_z_image_without_touching_the_moss_folder(tmp_path: Path, capsys) -> None:
    saved = _install()
    try:
        project = make_project(tmp_path)
        data = project.read()
        data["bases"] = ["moss-image-base"]
        project.write(data)
        moss = tmp_path / "checkpoints" / "moss-image-base"
        moss.mkdir(parents=True)
        (moss / "keep.txt").write_text("moss", encoding="utf-8")
        before = _dataset(tmp_path)
        code = main(_retrain_args(project.root, "z-image"))
        payload = _envelope(capsys.readouterr().out)
    finally:
        _restore(saved)
    assert code == 0 and payload["result"]["state"] == "done"
    assert project.read()["bases"] == ["moss-image-base", "z-image"]
    assert project.read()["train"]["base"] == "z-image"
    assert (moss / "keep.txt").read_text(encoding="utf-8") == "moss"
    assert list((tmp_path / "checkpoints" / "z-image").glob("step-*.safetensors"))
    assert _dataset(tmp_path) == before


def test_dry_run_leaves_a_legacy_project_unchanged(tmp_path: Path, capsys) -> None:
    saved = _install()
    try:
        project = make_project(tmp_path)
        code = main(_retrain_args(project.root, "moss", "--dry-run"))
        payload = _envelope(capsys.readouterr().out)
    finally:
        _restore(saved)
    assert code == 0
    result = payload["result"]
    assert result["state"] == "dry-run" and result["base"] == "moss-image-base"
    stored = project.read()
    assert "bases" not in stored
    assert "train" not in stored
    assert not (tmp_path / "checkpoints").exists()


def test_unprepared_project_is_a_usage_error(tmp_path: Path, capsys) -> None:
    code = main(["--json", "lora", "retrain", "--base", "moss", str(tmp_path / "none")])
    payload = json.loads(capsys.readouterr().out)
    assert code == 2 and payload["error"]["code"] == "lora.err.not_prepared"


def test_base_must_be_moss_or_z_image(tmp_path: Path, capsys) -> None:
    project = make_project(tmp_path)
    code = main(["lora", "retrain", "--base", "both", str(project.root)])
    capsys.readouterr()
    assert code == 2
    assert "bases" not in Project(project.root).read()
