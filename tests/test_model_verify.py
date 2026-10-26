"""Check and repair calls the voxprint-common 0.5.2 names.

``--gui`` launches ``voxprint-mdl verify --gui``. These tests substitute that
process so a headless run does not open the window. A terminal check substitutes
``verify_repair`` so the suite does not hash the fixture.
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
    format_verify_counts,
    installed_model_ids,
    mdl_script,
    spawn,
    stage,
    verify_window_argv,
)

USER_LORA = b"do-not-touch-lora"
PROJECT_NOTE = b"project-note"
# CPython defines this as 0x08000000 and only exposes the name on Windows.
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)


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


def _gui(monkeypatch: pytest.MonkeyPatch, exe: str) -> dict[str, object]:
    """Record the ``voxprint-mdl verify --gui`` launch and return what the fake saw."""
    seen: dict[str, object] = {}
    monkeypatch.setattr("voxprint_forge.mdl_adapter.mdl_script", lambda: [exe])

    def fake_run(command: list[str], check: bool = False, **kwargs: object) -> subprocess.CompletedProcess[str]:
        seen["command"] = list(command)
        seen["check"] = check
        seen["kwargs"] = kwargs
        seen["run"] = True
        return subprocess.CompletedProcess(command, 0)

    def fake_popen(command: list[str], **kwargs: object) -> SimpleNamespace:
        seen["command"] = list(command)
        seen["kwargs"] = kwargs
        seen["popen"] = True
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("voxprint_forge.mdl_adapter.subprocess.run", fake_run)
    monkeypatch.setattr("voxprint_forge.mdl_adapter.subprocess.Popen", fake_popen)
    return seen


def _assert_verify_gui(command: object, exe: str, *, repair: bool) -> None:
    assert isinstance(command, list)
    assert command[0] == exe
    assert Path(command[0]).name.lower().removesuffix(".exe") != "voxprint-mdl-gui"
    assert command[1:4] == ["verify", "--app", "model-forge"]
    assert "--gui" in command
    assert ("--repair" in command) is repair
    assert "plug-toy" not in command
    assert "secret-lora" not in command
    assert "never-installed" not in command
    assert "toy" in command
    assert "blob-only" in command


def test_gui_launches_verify_even_when_every_model_is_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, _, _ = _install(tmp_path)
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr("voxprint_forge.mdl_app.manifest_path", lambda: manifest)
    monkeypatch.setattr("voxprint_forge.mdl_adapter._is_windows", lambda: True)
    exe = r"C:\venv\Scripts\voxprint-mdl.exe"
    seen = _gui(monkeypatch, exe)
    code = main(["models", "verify", "--gui"])
    assert code == 0
    _assert_verify_gui(seen["command"], exe, repair=False)
    assert seen["run"] is True
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["creationflags"] == CREATE_NO_WINDOW
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert kwargs["start_new_session"] is True
    assert capsys.readouterr().out == ""


def test_gui_repair_launches_verify_with_repair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest, _, _ = _install(tmp_path)
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr("voxprint_forge.mdl_app.manifest_path", lambda: manifest)
    monkeypatch.setattr("voxprint_forge.mdl_adapter._is_windows", lambda: False)
    exe = "/opt/venv/bin/voxprint-mdl"
    seen = _gui(monkeypatch, exe)
    code = main(["models", "verify", "--gui", "--repair"])
    assert code == 0
    _assert_verify_gui(seen["command"], exe, repair=True)
    command = seen["command"]
    assert isinstance(command, list)
    assert command[-2:] == ["--gui", "--repair"]
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["creationflags"] == 0


def test_gui_with_nothing_installed_does_not_plan_the_catalog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "mdl.json"
    manifest.write_text(json.dumps({"models": [{"id": "absent", "files": []}]}), encoding="utf-8")
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(root))
    monkeypatch.setattr("voxprint_forge.mdl_app.manifest_path", lambda: manifest)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("an empty install must not launch voxprint-mdl")

    monkeypatch.setattr("voxprint_forge.mdl_adapter.subprocess.run", fail)
    monkeypatch.setattr("voxprint_forge.mdl_adapter.subprocess.Popen", fail)
    code = main(["models", "verify", "--gui", "--repair"])
    assert code == 0
    assert capsys.readouterr().out == ""


def test_spawn_passes_create_no_window_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("voxprint_forge.mdl_adapter._is_windows", lambda: True)
    seen: dict[str, object] = {}

    def fake_popen(command: list[str], **kwargs: object) -> SimpleNamespace:
        seen["command"] = list(command)
        seen["kwargs"] = kwargs
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("voxprint_forge.mdl_adapter.subprocess.Popen", fake_popen)
    argv = [r"C:\venv\Scripts\voxprint-mdl.exe", "verify", "--app", "model-forge", "--gui", "--repair"]
    assert spawn(argv) == 0
    assert seen["command"] == argv
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["creationflags"] == CREATE_NO_WINDOW
    assert kwargs["start_new_session"] is True


def test_gui_returns_the_window_exit_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest, _, _ = _install(tmp_path)
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr("voxprint_forge.mdl_app.manifest_path", lambda: manifest)
    monkeypatch.setattr("voxprint_forge.mdl_adapter.mdl_script", lambda: ["voxprint-mdl"])

    def fake_run(command: list[str], check: bool = False, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        assert check is False
        return subprocess.CompletedProcess(command, 2)

    monkeypatch.setattr("voxprint_forge.mdl_adapter.subprocess.run", fake_run)
    assert main(["models", "verify", "--gui"]) == 2


def _outcome(
    code: int,
    counts: dict[str, int],
    steps: list[tuple[str, str]],
    message: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        code=code,
        message=message,
        status=SimpleNamespace(
            extra={"counts": counts},
            steps=[SimpleNamespace(item_id=model_id, state=state) for model_id, state in steps],
        ),
    )


def test_cli_prints_per_status_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, weights, note = _install(tmp_path)
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr("voxprint_forge.mdl_app.manifest_path", lambda: manifest)
    seen: dict[str, object] = {}

    def fake(model_ids: list[str], repair: bool = True, manifests: list[Path] | None = None) -> SimpleNamespace:
        seen["ids"] = list(model_ids)
        seen["repair"] = repair
        seen["manifests"] = list(manifests or [])
        return _outcome(
            2,
            {"ok": 0, "missing": 1, "corrupt": 1, "repaired": 0, "failed": 0},
            [("toy", "missing"), ("blob-only", "corrupt")],
        )

    monkeypatch.setattr("voxprint_forge.mdl_adapter.verify_repair", fake)
    code = main(["models", "verify"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == "2 models: 0 OK, 1 damaged, 1 missing\n"
    assert seen["repair"] is False
    assert seen["ids"] == ["toy", "blob-only"]
    assert seen["manifests"] == [manifest]
    assert "plug-toy" not in seen["ids"]
    assert "secret-lora" not in seen["ids"]
    assert "never-installed" not in seen["ids"]
    assert weights.read_bytes() == USER_LORA
    assert note.read_bytes() == PROJECT_NOTE


def test_cli_counts_repaired_models_as_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, _, _ = _install(tmp_path)
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr("voxprint_forge.mdl_app.manifest_path", lambda: manifest)

    def fake(model_ids: list[str], repair: bool = True, manifests: list[Path] | None = None) -> SimpleNamespace:
        del model_ids, manifests
        assert repair is False
        return _outcome(0, {"ok": 11, "missing": 0, "corrupt": 0, "repaired": 0, "failed": 0}, [])

    monkeypatch.setattr("voxprint_forge.mdl_adapter.verify_repair", fake)
    assert main(["models", "verify"]) == 0
    assert capsys.readouterr().out == "11 models: 11 OK, 0 damaged, 0 missing\n"


def test_cli_repair_passes_repair_and_counts_repaired_files_as_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, _, _ = _install(tmp_path)
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr("voxprint_forge.mdl_app.manifest_path", lambda: manifest)
    seen: dict[str, object] = {}

    def fake(model_ids: list[str], repair: bool = True, manifests: list[Path] | None = None) -> SimpleNamespace:
        del manifests
        seen["repair"] = repair
        seen["ids"] = list(model_ids)
        return _outcome(0, {"ok": 9, "missing": 0, "corrupt": 0, "repaired": 2, "failed": 0}, [])

    monkeypatch.setattr("voxprint_forge.mdl_adapter.verify_repair", fake)
    code = main(["models", "verify", "--repair"])
    captured = capsys.readouterr()
    assert code == 0
    assert seen["repair"] is True
    assert captured.out == "11 models: 11 OK, 0 damaged, 0 missing\n"
    assert format_verify_counts(11, 0, 0) == captured.out


def test_cli_prints_the_upstream_error_when_the_check_has_no_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest, _, _ = _install(tmp_path)
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr("voxprint_forge.mdl_app.manifest_path", lambda: manifest)

    def fake(model_ids: list[str], repair: bool = True, manifests: list[Path] | None = None) -> SimpleNamespace:
        del model_ids, repair, manifests
        return SimpleNamespace(code=2, message="model toy is not available for this card", status=SimpleNamespace())

    monkeypatch.setattr("voxprint_forge.mdl_adapter.verify_repair", fake)
    code = main(["models", "verify"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == "model toy is not available for this card\n"


def test_cli_with_nothing_installed_is_ok_and_does_not_plan_the_catalog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "mdl.json"
    manifest.write_text(json.dumps({"models": [{"id": "absent", "files": []}]}), encoding="utf-8")
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(root))
    monkeypatch.setattr("voxprint_forge.mdl_app.manifest_path", lambda: manifest)

    def fail_repair(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("an empty install must not ask voxprint-mdl to plan the catalog")

    monkeypatch.setattr("voxprint_forge.mdl_adapter.verify_repair", fail_repair)
    code = main(["models", "verify"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == "0 models: 0 OK, 0 damaged, 0 missing\n"


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


def test_missing_mdl_script_is_an_error_and_does_not_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("voxprint_forge.mdl_adapter.sys.executable", str(tmp_path / "python"))
    monkeypatch.setattr("voxprint_forge.mdl_adapter.shutil.which", lambda _name: None)
    monkeypatch.setattr("voxprint_forge.mdl_adapter._is_windows", lambda: False)

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a missing voxprint-mdl must not spawn")

    monkeypatch.setattr("voxprint_forge.mdl_adapter.subprocess.Popen", fail)
    monkeypatch.setattr("voxprint_forge.mdl_adapter.subprocess.run", fail)
    manifest = tmp_path / "mdl.json"
    manifest.write_text("{}", encoding="utf-8")
    assert mdl_script() is None
    with pytest.raises(MdlAdapterError, match=r"voxprint-mdl verify --gui.*does not provide it"):
        verify_window_argv(repair=True, manifest=manifest)
    monkeypatch.setattr("voxprint_forge.mdl_adapter.for_process", lambda *args, **kwargs: SimpleNamespace())
    with pytest.raises(MdlAdapterError, match=r"model_park\.stage"):
        stage("load", ["toy"])


def test_windows_sibling_executable_uses_the_exe_suffix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    exe = tmp_path / "python.exe"
    mdl = tmp_path / "voxprint-mdl.exe"
    exe.write_bytes(b"")
    mdl.write_bytes(b"")
    manifest = tmp_path / "mdl.json"
    manifest.write_text(json.dumps({"models": []}), encoding="utf-8")
    monkeypatch.setattr("voxprint_forge.mdl_adapter.sys.executable", str(exe))
    monkeypatch.setattr("voxprint_forge.mdl_adapter._is_windows", lambda: True)
    argv = verify_window_argv(repair=True, manifest=manifest, root=tmp_path)
    assert argv == [str(mdl), "verify", "--app", "model-forge", "--manifest", str(manifest), "--gui", "--repair"]


def test_missing_names_are_an_error_and_not_the_download_window(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = Path(__file__).resolve().parents[1] / "src" / "voxprint_forge" / "mdl_adapter.py"
    source = adapter.read_text(encoding="utf-8")
    assert "voxprint_common.mdl.progress_ui" not in source
    assert "run_window" not in source
    assert "run_verify_window" not in source
    root = Path(__file__).resolve().parents[1] / "src" / "voxprint_forge"
    assert not (root / "mdl_progress.py").exists()
    assert not (root / "model_verify.py").exists()
