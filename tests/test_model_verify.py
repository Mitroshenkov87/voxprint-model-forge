"""Check and repair installed models: hash, missing, corrupt, and the progress window."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402
from voxprint_common.i18n import set_language  # noqa: E402
from voxprint_common.mdl.descriptor import parse_descriptor_text  # noqa: E402
from voxprint_common.mdl.status import Status  # noqa: E402

from voxprint_forge.cli import main  # noqa: E402
from voxprint_forge.mdl_progress import WindowResult, open_verify_window, run_verify_window  # noqa: E402
from voxprint_forge.model_verify import (  # noqa: E402
    FileReport,
    ModelReport,
    VerifyCancelled,
    discover,
    download_file,
    hash_file,
    repair_models,
    snapshot,
    source_urls,
    verify_models,
)

KEEP = b"0123456789abcdefXYZ"
GONE = b"gone-file-bytes!!"
GOOD = b"0123456789abcdef"
BAD_VIEW = b"ffffffffffffffff"
PLUGIN = b"plugin-weights!!"
USER_LORA = b"do-not-touch-lora"
PROJECT_NOTE = b"project-note"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _plugin_document(model_id: str, filename: str, payload: bytes, url: str) -> dict[str, object]:
    return {
        "schema": "voxprint-model/1",
        "id": model_id,
        "display_name": model_id,
        "app_ids": ["canvas"],
        "task": "text-to-image",
        "license": "mit",
        "files": [{"path": filename, "size": len(payload), "sha256": _sha(payload)}],
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


def _app_row(name: str, payload: bytes, url: str) -> dict[str, object]:
    return {"path": name, "size": len(payload), "sha256": _sha(payload), "mirrors": [url]}


def _install_broken(root: Path) -> tuple[Path, Path, Path, Path]:
    """One installed model with an OK file, a deleted file, and a same-size corrupt view."""
    model = root / "model-forge" / "toy"
    model.mkdir(parents=True)
    (model / "keep.bin").write_bytes(KEEP)
    (model / "bad.bin").write_bytes(BAD_VIEW)
    blob = root / "blobs" / _sha(GOOD)
    blob.parent.mkdir(parents=True)
    blob.write_bytes(GOOD)
    partial = root / "blobs" / (_sha(GONE) + ".partial")
    partial.write_bytes(b"resume")
    manifest = root.parent / "mdl.json"
    document = {
        "models": [
            {
                "id": "toy",
                "display_name": "Toy",
                "files": [
                    _app_row("keep.bin", KEEP, "https://example.com/keep.bin"),
                    _app_row("gone.bin", GONE, "https://example.com/gone.bin"),
                    _app_row("bad.bin", GOOD, "https://example.com/bad.bin"),
                ],
            },
            {
                "id": "never-installed",
                "display_name": "Absent",
                "files": [_app_row("absent.bin", b"absent", "https://example.com/absent.bin")],
            },
        ]
    }
    manifest.write_text(json.dumps(document), encoding="utf-8")
    _write_plugin(root / "plugins" / "plug-toy", "plug-toy", "weights.bin", PLUGIN, "https://example.com/models/toy/")
    user = root / "user-models" / "my-lora"
    _write_plugin(user, "secret-lora", "weights.safetensors", b"official-lora", "https://example.com/lora/")
    (user / "weights.safetensors").write_bytes(USER_LORA)
    project = root / "projects" / "demo"
    project.mkdir(parents=True)
    (project / "note.txt").write_bytes(PROJECT_NOTE)
    return manifest, partial, user / "weights.safetensors", project / "note.txt"


def test_hash_is_chunked_and_cancellable(tmp_path: Path) -> None:
    path = tmp_path / "weights.bin"
    path.write_bytes(b"0123456789abcdef")
    seen: list[int] = []
    cancel = threading.Event()

    def on_chunk(done: int) -> None:
        seen.append(done)
        if done >= 8:
            cancel.set()

    with pytest.raises(VerifyCancelled):
        hash_file(path, chunk=8, cancel=cancel, on_chunk=on_chunk)
    assert seen == [8]

    seen.clear()
    digest = hash_file(path, chunk=8, on_chunk=seen.append)
    assert seen == [8, 16]
    assert digest == _sha(path.read_bytes())


def test_verify_reports_ok_missing_and_corrupt_and_skips_user_data(tmp_path: Path) -> None:
    root = tmp_path / "models"
    manifest, _partial, lora, note = _install_broken(root)
    progress: list[int] = []

    def on_progress(reports: list[ModelReport], phase: str) -> None:
        assert phase == "verify"
        progress.append(reports[0].bytes_hashed)

    found = discover(manifest, root=root)
    assert [item.id for item in found] == ["toy", "plug-toy"]
    reports = verify_models(found, chunk=8, on_progress=on_progress)
    by_id = {item.id: item for item in reports}
    assert by_id["toy"].status == "corrupt"
    files = {item.path: item.status for item in by_id["toy"].files}
    assert files == {"keep.bin": "ok", "gone.bin": "missing", "bad.bin": "corrupt"}
    assert by_id["plug-toy"].status == "missing"
    assert "secret-lora" not in by_id
    assert "never-installed" not in by_id
    assert any(0 < hashed < len(KEEP) for hashed in progress)
    assert lora.read_bytes() == USER_LORA
    assert note.read_bytes() == PROJECT_NOTE


def test_repair_downloads_missing_files_and_relinks_a_good_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "models"
    manifest, partial, lora, note = _install_broken(root)
    bodies = {_sha(GONE): GONE, _sha(PLUGIN): PLUGIN}
    calls: list[tuple[Path, tuple[str, ...], str]] = []

    def fake_download(
        destination: Path,
        urls: tuple[str, ...] | list[str],
        *,
        sha256: str,
        size: int,
        cancel: threading.Event | None = None,
        on_bytes: object = None,
    ) -> None:
        del cancel, size
        partial_now = destination.with_name(destination.name + ".partial")
        if destination.name == _sha(GONE):
            assert partial_now.read_bytes() == b"resume"
        calls.append((destination, tuple(urls), sha256))
        destination.write_bytes(bodies[sha256])
        if callable(on_bytes):
            on_bytes(len(bodies[sha256]), len(bodies[sha256]))

    monkeypatch.setattr("voxprint_forge.model_verify.download_file", fake_download)
    models = discover(manifest, root=root)
    before = verify_models(models, chunk=8)
    repaired = repair_models(models, before, chunk=8)
    downloaded = {item[2] for item in calls}
    assert _sha(GOOD) not in downloaded
    assert _sha(GONE) in downloaded
    assert _sha(PLUGIN) in downloaded
    plugin_call = next(item for item in calls if item[2] == _sha(PLUGIN))
    assert plugin_call[1] == ("https://example.com/models/toy/weights.bin",)
    assert partial.is_file()
    assert (root / "model-forge" / "toy" / "bad.bin").read_bytes() == GOOD
    assert (root / "model-forge" / "toy" / "gone.bin").read_bytes() == GONE
    assert (root / "model-forge" / "toy" / "keep.bin").read_bytes() == KEEP
    assert (root / "plugins" / "plug-toy" / "weights.bin").read_bytes() == PLUGIN
    assert {item.status for item in repaired} == {"ok"}
    assert lora.read_bytes() == USER_LORA
    assert note.read_bytes() == PROJECT_NOTE
    text = (root / "plugins" / "plug-toy" / "voxprint-model.json").read_text(encoding="utf-8")
    assert source_urls(parse_descriptor_text(text), "weights.bin") == ("https://example.com/models/toy/weights.bin",)


def test_download_file_pins_the_checksum_and_keeps_a_partial(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    destination = tmp_path / "blobs" / _sha(GONE)
    destination.parent.mkdir()
    partial = destination.with_name(destination.name + ".partial")
    partial.write_bytes(b"resume")
    seen: dict[str, object] = {}

    def fake_fetch(dest: Path, urls: list[str], **kwargs: object) -> Path:
        assert partial.read_bytes() == b"resume"
        seen["dest"] = dest
        seen["urls"] = urls
        seen["kwargs"] = kwargs
        dest.write_bytes(GONE)
        return dest

    monkeypatch.setattr("voxprint_forge.model_verify.fetch_file", fake_fetch)
    download_file(destination, ["https://example.com/gone.bin"], sha256=_sha(GONE), size=len(GONE))
    kwargs = seen["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["sha256"] == _sha(GONE)
    assert kwargs["size"] == len(GONE)
    assert kwargs["probe_bytes"] == 0
    assert kwargs["mirror_policy"] == "strict-priority"
    assert partial.read_bytes() == b"resume"
    assert destination.read_bytes() == GONE


def test_cli_prints_a_table_and_repair_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "home" / "models"
    manifest, _partial, _lora, _note = _install_broken(root)
    bodies = {_sha(GONE): GONE, _sha(PLUGIN): PLUGIN}

    def fake_download(
        destination: Path,
        urls: tuple[str, ...] | list[str],
        *,
        sha256: str,
        size: int,
        cancel: threading.Event | None = None,
        on_bytes: object = None,
    ) -> None:
        del urls, size, cancel
        destination.write_bytes(bodies[sha256])
        if callable(on_bytes):
            on_bytes(len(bodies[sha256]), len(bodies[sha256]))

    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(root))
    monkeypatch.setattr("voxprint_forge.mdl_app.manifest_path", lambda: manifest)
    monkeypatch.setattr("voxprint_forge.model_verify.download_file", fake_download)
    assert main(["models", "verify"]) == 4
    broken = capsys.readouterr().out
    assert "missing" in broken
    assert "corrupt" in broken
    assert "never-installed" not in broken
    assert main(["models", "verify", "--repair"]) == 0
    fixed = capsys.readouterr().out
    assert "OK" in fixed
    assert "missing" not in fixed
    assert "corrupt" not in fixed


def test_gui_opens_when_every_installed_model_is_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "home" / "models"
    model = root / "model-forge" / "toy"
    model.mkdir(parents=True)
    (model / "keep.bin").write_bytes(KEEP)
    manifest = tmp_path / "mdl.json"
    manifest.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "id": "toy",
                        "display_name": "Toy",
                        "files": [_app_row("keep.bin", KEEP, "https://example.com/keep.bin")],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def fake_window(job: object, *, repair: bool = False, on_settled: object = None) -> int:
        del on_settled
        seen["repair"] = repair
        result = job(threading.Event(), lambda _status: None, repair)  # type: ignore[operator]
        seen["code"] = result.code
        return int(result.code)

    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("VOXPRINT_MODELS_DIR", str(root))
    monkeypatch.setattr("voxprint_forge.mdl_app.manifest_path", lambda: manifest)
    monkeypatch.setattr("voxprint_forge.model_verify.run_verify_window", fake_window)
    assert main(["models", "verify", "--gui"]) == 0
    assert seen["repair"] is False
    assert seen["code"] == 0
    assert capsys.readouterr().out == ""


def _layout_status() -> Status:
    long_name = "YuNet face detector " + ("with a very long display name " * 6)
    reports = [
        ModelReport(
            "yunet", long_name, "ok", 100, 100, "huggingface.co", files=[FileReport("yunet.onnx", "ok", 100, 100)]
        ),
        ModelReport(
            "gone", "Gone model", "missing", 0, 50, "example.com", files=[FileReport("weights.bin", "missing", 50, 0)]
        ),
        ModelReport(
            "bad", "Bad model", "corrupt", 16, 16, "example.com", files=[FileReport("bad.bin", "corrupt", 16, 16)]
        ),
    ]
    return snapshot(reports, phase="verify", done=True)


def test_verify_window_fits_at_1280(qapp: QtWidgets.QApplication) -> None:
    set_language("en")
    window = open_verify_window(_layout_status())
    window.resize(1280, 720)
    qapp.processEvents()
    assert window.width() == 1280
    table = window.findChild(QtWidgets.QTableWidget, "mdlVerifyTable")
    header = table.horizontalHeader()
    assert header.length() <= table.viewport().width()
    words = [table.item(row, 2).text() for row in range(table.rowCount())]
    assert words == ["OK", "missing", "corrupt"]
    for name in ("repair", "cancel"):
        button = window.findChild(QtWidgets.QPushButton, name)
        assert button.isVisible()
        origin = button.mapTo(window, QtCore.QPoint(0, 0))
        assert origin.x() >= 0
        assert origin.y() >= 0
        assert origin.x() + button.width() <= window.width()
        assert origin.y() + button.height() <= window.height()
    window.close()


def test_repair_button_reruns_and_close_returns_zero(qapp: QtWidgets.QApplication) -> None:
    calls: list[bool] = []

    def job(
        cancel: threading.Event,
        on_status: object,
        do_repair: bool,
    ) -> WindowResult:
        del cancel
        calls.append(do_repair)
        if do_repair:
            report = ModelReport("toy", "Toy", "ok", 4, 4, "example.com", files=[FileReport("a.bin", "ok", 4, 4)])
            code = 0
        else:
            report = ModelReport(
                "toy", "Toy", "missing", 0, 4, "example.com", files=[FileReport("a.bin", "missing", 4, 0)]
            )
            code = 4
        done = snapshot([report], phase="verify", done=True)
        on_status(done)  # type: ignore[operator]
        return WindowResult(code, done)

    abort = QtCore.QTimer()
    abort.setSingleShot(True)
    abort.timeout.connect(lambda: QApplication.instance().exit(99))
    abort.start(5000)

    def settled(window: object) -> None:
        qapp.processEvents()
        name = "repair" if len(calls) == 1 else "cancel"
        button = window.findChild(QtWidgets.QPushButton, name)  # type: ignore[attr-defined]
        button.click()

    try:
        code = run_verify_window(job, repair=False, on_settled=settled)
    finally:
        abort.stop()
    assert calls == [False, True]
    assert code == 0
