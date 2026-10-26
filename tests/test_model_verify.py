"""Check and repair calls the voxprint-common 0.5.0 names.

The progress window stays in that release. These tests substitute the
functions so a headless run does not open it.
"""

from __future__ import annotations

import inspect
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from voxprint_common.mdl import run_verify_repair as upstream_verify
from voxprint_common.model_park import ModelPark

from voxprint_forge.cli import main
from voxprint_forge.mdl_adapter import (
    MdlAdapterError,
    execute_verify,
    installed_model_ids,
    run_verify_repair,
    stage,
)

USER_LORA = b"do-not-touch-lora"
PROJECT_NOTE = b"project-note"


def _plugin_document(model_id: str, filename: str, payload: bytes, url: str) -> dict[str, object]:
    return {
        "schema": "voxprint-model/1",
        "id": model_id,
        "display_name": model_id,
        "app_ids": ["canvas"],
        "task": "text-to-image",
        "license": "mit",
        "files": [{"path": filename, "size": len(payload), "sha256": "ab" * 32}],
        "mirrors": [{"kind": "url", "ref": url, "revision": ""}],
        "tier": {"min_vram_gib": 8, "recommended_vram_gib": 16, "gpu_min_cc": 8},
        "runtime": {"engine": "diffusers", "version_req": ">=0.37.1,<0.38"},
        "base_model": "",
        "card_url": "",
        "flags": [],
        "on_demand": True,
        "build": {"forge_version": "0.1.0", "tag": "", "date": "2026-10-25"},
    }


def _write_plugin(folder: Path, model_id: str, filename: str, payload: bytes, url: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    document = _plugin_document(model_id, filename, payload, url)
    (folder / "voxprint-model.json").write_text(json.dumps(document), encoding="utf-8")


def _install(root: Path) -> tuple[Path, Path, Path]:
    """An installed view, a blob-only model, a plug-in, and user data that must stay put."""
    toy = root / "model-forge" / "toy"
    toy.mkdir(parents=True)
    (toy / "keep.bin").write_bytes(b"0123456789abcdef")
    digest = "cd" * 32
    blob = root / "blobs" / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"blob-only-weights")
    manifest = root.parent / "mdl.json"
    document = {
        "models": [
            {
                "id": "toy",
                "display_name": "Toy",
                "files": [{"path": "keep.bin", "size": 16, "sha256": "aa" * 32}],
            },
            {
                "id": "blob-only",
                "display_name": "Blob",
                "files": [{"path": "weights.bin", "size": 17, "sha256": digest}],
            },
            {
                "id": "never-installed",
                "display_name": "Absent",
                "files": [{"path": "absent.bin", "size": 4, "sha256": "ee" * 32}],
            },
        ]
    }
    manifest.write_text(json.dumps(document), encoding="utf-8")
    _write_plugin(
        root / "plugins" / "plug-toy", "plug-toy", "weights.bin", b"plugin", "https://example.com/models/toy/"
    )
    user = root / "user-models" / "my-lora"
    _write_plugin(user, "secret-lora", "weights.safetensors", b"official-lora", "https://example.com/lora/")
    weights = user / "weights.safetensors"
    weights.write_bytes(USER_LORA)
    project = root / "projects" / "demo"
    project.mkdir(parents=True)
    note = project / "note.txt"
    note.write_bytes(PROJECT_NOTE)
    return manifest, weights, note


def test_installed_ids_skip_user_data_and_absent_catalog_rows(tmp_path: Path) -> None:
    manifest, weights, note = _install(tmp_path)
    assert installed_model_ids(manifest, root=tmp_path) == ["toy", "blob-only", "plug-toy"]
    assert weights.read_bytes() == USER_LORA
    assert note.read_bytes() == PROJECT_NOTE


def test_gui_calls_run_verify_repair_even_when_every_model_is_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, _, _ = _install(tmp_path)
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr("voxprint_forge.mdl_app.manifest_path", lambda: manifest)
    seen: dict[str, object] = {}

    def fake(model_ids: list[str], repair: bool = False, **kwargs: object) -> int:
        seen["ids"] = list(model_ids)
        seen["repair"] = repair
        seen["show"] = kwargs.get("show")
        seen["manifests"] = kwargs.get("manifests")
        return 0

    monkeypatch.setattr("voxprint_common.mdl.run_verify_repair", fake, raising=False)

    def fail_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the progress window is run_verify_repair, not a voxprint-mdl subprocess")

    monkeypatch.setattr("voxprint_forge.mdl_adapter.subprocess.run", fail_run)
    code = main(["models", "verify", "--gui"])
    assert code == 0
    assert seen["ids"] == ["toy", "blob-only", "plug-toy"]
    assert seen["repair"] is False
    assert seen["show"] is True
    assert seen["manifests"] == [manifest]
    assert capsys.readouterr().out == ""


def test_gui_repair_forwards_repair_and_a_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest, _, _ = _install(tmp_path)
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr("voxprint_forge.mdl_app.manifest_path", lambda: manifest)
    seen: dict[str, object] = {}

    def fake(model_ids: list[str], repair: bool = False, **_kwargs: object) -> dict[str, object]:
        seen["ids"] = list(model_ids)
        seen["repair"] = repair
        return {
            "exit_code": 0,
            "text": "toy OK\n",
            "models": [{"id": "toy", "status": "ok"}, {"id": "plug-toy", "status": "ok"}],
        }

    monkeypatch.setattr("voxprint_common.mdl.run_verify_repair", fake, raising=False)
    code = main(["models", "verify", "--gui", "--repair"])
    assert code == 0
    assert seen["repair"] is True
    assert seen["ids"] == ["toy", "blob-only", "plug-toy"]


def test_summary_without_a_code_exits_when_a_model_is_corrupt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest, _, _ = _install(tmp_path)

    def fake(model_ids: list[str], repair: bool = False, **_kwargs: object) -> dict[str, object]:
        assert repair is False
        assert "secret-lora" not in model_ids
        return {"models": [{"id": "toy", "status": "corrupt"}, {"id": "plug-toy", "status": "OK"}]}

    monkeypatch.setattr("voxprint_common.mdl.run_verify_repair", fake, raising=False)
    result = execute_verify(manifest=manifest, repair=False, gui=True, root=tmp_path)
    assert result["ok"] is False
    assert result["exit_code"] == 4
    assert result["models"] == [{"id": "toy", "status": "corrupt"}, {"id": "plug-toy", "status": "OK"}]


def test_cli_prints_the_mdl_table_and_exits_when_broken(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, weights, note = _install(tmp_path)
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr("voxprint_forge.mdl_app.manifest_path", lambda: manifest)
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool, capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
        assert check is False
        assert capture_output is True
        assert text is True
        calls.append(list(command))
        return subprocess.CompletedProcess(
            command, 4, stdout="toy  missing\nplug-toy  corrupt\n", stderr="hashing toy\n"
        )

    monkeypatch.setattr("voxprint_forge.mdl_adapter.subprocess.run", fake_run)
    code = main(["models", "verify"])
    captured = capsys.readouterr()
    assert code == 4
    assert "missing" in captured.out
    assert "corrupt" in captured.out
    assert captured.err == "hashing toy\n"
    command = calls[0]
    assert command[command.index("verify") + 1 : command.index("verify") + 3] == ["--app", "model-forge"]
    assert "--repair" not in command
    assert "--gui" not in command
    assert "toy" in command
    assert "blob-only" in command
    assert "plug-toy" in command
    assert "secret-lora" not in command
    assert "never-installed" not in command
    assert str(manifest) in command
    assert weights.read_bytes() == USER_LORA
    assert note.read_bytes() == PROJECT_NOTE


def test_cli_repair_passes_repair_to_voxprint_mdl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, _, _ = _install(tmp_path)
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr("voxprint_forge.mdl_app.manifest_path", lambda: manifest)
    calls: list[list[str]] = []

    def fake_run(command: list[str], check: bool, capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="toy OK\nplug-toy OK\n", stderr="")

    monkeypatch.setattr("voxprint_forge.mdl_adapter.subprocess.run", fake_run)
    code = main(["models", "verify", "--repair"])
    captured = capsys.readouterr()
    assert code == 0
    assert calls[0][-1] == "--repair"
    assert "verify" in calls[0]
    assert "missing" not in captured.out
    assert "corrupt" not in captured.out
    assert "OK" in captured.out


def test_cli_with_nothing_installed_is_ok_and_does_not_plan_the_catalog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "mdl.json"
    manifest.write_text(json.dumps({"models": [{"id": "absent", "files": []}]}), encoding="utf-8")
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(root))
    monkeypatch.setattr("voxprint_forge.mdl_app.manifest_path", lambda: manifest)

    def fail_run(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("an empty install must not ask voxprint-mdl to plan the catalog")

    monkeypatch.setattr("voxprint_forge.mdl_adapter.subprocess.run", fail_run)
    code = main(["models", "verify"])
    captured = capsys.readouterr()
    assert code == 0
    assert "(none)" in captured.out
    assert "OK" in captured.out


def test_stage_forwards_to_model_park(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake(name: str, models: list[str], *, device: object = "cuda") -> str:
        seen["name"] = name
        seen["models"] = list(models)
        seen["device"] = device
        return "hot"

    monkeypatch.setattr(
        "voxprint_forge.mdl_adapter.for_process",
        lambda *args, **kwargs: SimpleNamespace(stage=fake),
    )
    assert stage("load", ["z-image"], device="cpu") == "hot"
    assert seen == {"name": "load", "models": ["z-image"], "device": "cpu"}


def test_installed_stage_is_the_050_context_manager() -> None:
    stage_params = inspect.signature(ModelPark.stage).parameters
    assert [name for name in stage_params if name != "self"] == ["name", "models", "device"]
    repair_params = inspect.signature(upstream_verify).parameters
    assert "model_ids" in repair_params and "repair" in repair_params and "show" in repair_params


def test_missing_names_are_an_error_and_not_the_download_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr("voxprint_common.mdl.run_verify_repair", raising=False)
    with pytest.raises(MdlAdapterError, match="run_verify_repair"):
        run_verify_repair(["toy"], repair=True)
    monkeypatch.setattr("voxprint_forge.mdl_adapter.for_process", lambda *args, **kwargs: SimpleNamespace())
    with pytest.raises(MdlAdapterError, match=r"model_park\.stage"):
        stage("load", ["toy"])
    adapter = Path(__file__).resolve().parents[1] / "src" / "voxprint_forge" / "mdl_adapter.py"
    source = adapter.read_text(encoding="utf-8")
    assert "voxprint_common.mdl.progress_ui" not in source
    assert "run_window" not in source
    assert "run_verify_window" not in source
    root = Path(__file__).resolve().parents[1] / "src" / "voxprint_forge"
    assert not (root / "mdl_progress.py").exists()
    assert not (root / "model_verify.py").exists()
