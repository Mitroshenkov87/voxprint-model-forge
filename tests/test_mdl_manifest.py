"""Schema 1 mdl.json, model-forge views, and ``models fetch``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from voxprint_common.mdl.catalog import VramTier, load_manifest, plan_files
from voxprint_common.paths import models_dir

from tests.test_lora_prepare import FakeCaptioner, FakeEstimator, FakeJudge, photos
from voxprint_forge.app_models import load
from voxprint_forge.cli import main
from voxprint_forge.lora import cli as lora_cli
from voxprint_forge.lora.backends import ESTIMATOR_ID, FACE_DETECTOR_ID, JUDGE_ID
from voxprint_forge.lora.bases import available
from voxprint_forge.lora.gate import Gate
from voxprint_forge.lora.refset import EMBEDDER_ID, TEACHER_ID
from voxprint_forge.mdl_app import fetch, missing_ids
from voxprint_forge.model_views import MARKER, ensure_view, files_for

ROOT = Path(__file__).resolve().parents[1]
MDL = ROOT / "src" / "voxprint_forge" / "data" / "mdl.json"
YUNET = "yunet-2023mar"


def _planned_ids(tier: VramTier, *, memory_gib: float | None = None, only: list[str] | None = None) -> set[str]:
    manifest = load_manifest(MDL)
    planned = plan_files([manifest], tier, models_root=Path("/tmp/voxprint-mdl-plan"), memory_gib=memory_gib, only=only)
    found: set[str] = set()
    for item in planned:
        found.update(item.model_ids)
    return found


def _plant(root: Path, spec: dict[str, tuple[str, int]]) -> None:
    blobs = root / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    for digest, _size in spec.values():
        (blobs / digest).write_bytes(b"x")


def test_load_manifest_reads_mdl_json() -> None:
    manifest = load_manifest(MDL)
    assert manifest.app == "model-forge"
    assert {model.id for model in manifest.models}


def test_plan_files_tiers_cover_required_and_optional_ids() -> None:
    models = load()
    required = {model.id for model in models if model.required}
    optional = [model.id for model in models if not model.required]
    assert _planned_ids(VramTier.GB16, memory_gib=16) == required
    assert _planned_ids(VramTier.GB24) == required
    assert _planned_ids(VramTier.GB8) == required - {"z-image"}
    assert _planned_ids(VramTier.GB24, only=optional) == set(optional)


def test_generator_check_and_ids_match_app_models_and_code() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/gen_app_models.py", "--check"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    document = json.loads(MDL.read_text(encoding="utf-8"))
    mdl_ids = {model["id"] for model in document["models"]}
    app_ids = {model.id for model in load()}
    code_ids = {JUDGE_ID, ESTIMATOR_ID, FACE_DETECTOR_ID, EMBEDDER_ID, TEACHER_ID}
    code_ids.update(info.mdl_id for info in available())
    assert mdl_ids == app_ids
    assert code_ids <= mdl_ids


def test_view_hardlink_and_copy_and_blobs_stay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec = files_for(YUNET)
    root = tmp_path / "linked"
    _plant(root, spec)
    folder = ensure_view(YUNET, root=root)
    assert folder is not None
    rel = next(iter(spec))
    blob = root / "blobs" / spec[rel][0]
    assert os.path.samefile(folder / rel, blob)
    marker = json.loads((folder / MARKER).read_text(encoding="utf-8"))
    assert {"path": rel, "sha256": spec[rel][0]} in marker["files"]

    copied = tmp_path / "copied"
    _plant(copied, spec)

    def _refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("cross device")

    monkeypatch.setattr(os, "link", _refuse)
    copy_folder = ensure_view(YUNET, root=copied)
    assert copy_folder is not None
    copy_blob = copied / "blobs" / spec[rel][0]
    assert (copy_folder / rel).read_bytes() == copy_blob.read_bytes()
    assert not os.path.samefile(copy_folder / rel, copy_blob)

    names = {path.name for path in (root / "blobs").iterdir()}
    (folder / rel).unlink()
    (folder / rel).write_bytes(b"other")
    (folder / MARKER).write_text("{}\n", encoding="utf-8")
    monkeypatch.undo()
    again = ensure_view(YUNET, root=root)
    assert again == folder
    assert {path.name for path in (root / "blobs").iterdir()} == names
    assert blob.read_bytes() == b"x"
    assert os.path.samefile(folder / rel, blob)


def test_prepare_no_fetch_exits_6(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    photos(tmp_path / "in", 10)
    missing = missing_ids()
    assert missing
    code = main(
        [
            "--json",
            "lora",
            "prepare",
            "--project",
            str(tmp_path / "proj"),
            "--photos",
            str(tmp_path / "in"),
            "--kind",
            "object",
            "--name",
            "Blue mug",
            "--rights-confirmed",
            "--no-fetch",
        ]
    )
    out = capsys.readouterr().out
    assert code == 6
    assert "lora.err.gate_missing" in out
    assert "voxprint-forge-cli models fetch --required" in out
    for mdl_id in missing:
        assert mdl_id in out


def test_fetch_stub_builds_views_and_prepare_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    wanted = missing_ids()
    assert wanted

    def _stub(ids: list[str], gui: bool = False) -> dict[str, object]:
        root = models_dir()
        for mdl_id in ids:
            spec = files_for(mdl_id)
            _plant(root, spec)
            ensure_view(mdl_id, root=root)
        return {"ok": True, "ready": True, "ids": list(ids)}

    monkeypatch.setattr(lora_cli, "fetch", _stub)
    monkeypatch.setattr(lora_cli, "_load_gate", lambda: Gate(FakeJudge(), FakeEstimator()))
    monkeypatch.setattr(lora_cli, "_load_captioner", lambda choice: FakeCaptioner())
    photos(tmp_path / "in", 10)
    code = main(
        [
            "--json",
            "lora",
            "prepare",
            "--project",
            str(tmp_path / "proj"),
            "--photos",
            str(tmp_path / "in"),
            "--kind",
            "object",
            "--name",
            "Blue mug",
            "--rights-confirmed",
            "--fetch-missing",
        ]
    )
    assert code == 0, capsys.readouterr().out
    root = tmp_path / "home" / "models" / "model-forge"
    for mdl_id in wanted:
        folder = root / mdl_id
        assert folder.is_dir()
        assert (folder / MARKER).is_file()
        assert any(path.name != MARKER for path in folder.iterdir())


def test_fetch_builds_views_from_mdl_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    spec = files_for(YUNET)

    def _run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert "--manifest" in command
        assert "--json" in command
        assert YUNET in command
        root = models_dir()
        _plant(root, spec)
        line = json.dumps({"type": "result", "ok": True, "ready": True})
        return subprocess.CompletedProcess(command, 0, stdout=line + "\n", stderr="yunet-2023mar\n")

    monkeypatch.setattr("voxprint_forge.mdl_app.subprocess.run", _run)
    result = fetch([YUNET])
    assert result["ok"] is True
    folder = tmp_path / "home" / "models" / "model-forge" / YUNET
    assert folder.is_dir()
    assert (folder / MARKER).is_file()
