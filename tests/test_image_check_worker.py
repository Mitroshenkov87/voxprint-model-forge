"""The Photos image check runs in a worker. The window keeps only the result cache."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402
from voxprint_common.i18n import set_language  # noqa: E402
from voxprint_common.worker import EventKind, Handlers, JobCancelled, WorkerEvent, WorkerHost  # noqa: E402

from tests.forge_app_stub import ALL_COMMANDS, StubRunner, make_photos  # noqa: E402
from voxprint_forge.app.draft import photo_key  # noqa: E402
from voxprint_forge.app.image_checks import CHECK_STOPPED, ImageChecks  # noqa: E402
from voxprint_forge.lora.image_check import ImageChecker  # noqa: E402
from voxprint_forge.lora.image_job import check as run_check  # noqa: E402
from voxprint_forge.ui.check_gate import GateState  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def make_app(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    from voxprint_forge.app.window import ForgeApp

    made: list[ForgeApp] = []

    def build(*, image_checks: ImageChecks):
        set_language("en")
        app = ForgeApp(
            runner=StubRunner(),
            home=tmp_path / "state",
            commands=ALL_COMMANDS,
            open_path=lambda _path: None,
            launch=lambda _argv: None,
            image_checks=image_checks,
        )
        app.window.show()
        qapp.processEvents()
        made.append(app)
        return app

    yield build
    for app in made:
        app.expert.set_enabled(False)
        app.window.close()
    set_language("en")


def _fake_checks(tmp_path: Path) -> ImageChecks:
    host = WorkerHost(
        Handlers(
            run="tests.image_worker_fake:run",
            load_model="tests.image_worker_fake:load_model",
            unload_model="tests.image_worker_fake:unload_model",
            check="tests.image_worker_fake:check",
        ),
        state_dir=tmp_path / "image-check-worker",
    )
    return ImageChecks(host)


def _pump(app: object, predicate, timeout: float = 8.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QtWidgets.QApplication.processEvents()
        if predicate():
            return
        time.sleep(0.02)
    QtWidgets.QApplication.processEvents()
    gate = app.photo_gate  # type: ignore[attr-defined]
    raise AssertionError(gate.state)


def _start(app: object, tmp_path: Path) -> None:
    photos = make_photos(tmp_path / "photos", 2)
    app.draft.add_photos(photos)  # type: ignore[attr-defined]
    app.draft.kind = "object"  # type: ignore[attr-defined]
    app.repaint()  # type: ignore[attr-defined]
    QtWidgets.QApplication.processEvents()
    button = app.window.findChild(QtWidgets.QPushButton, "forge.photos.check")  # type: ignore[attr-defined]
    assert button is not None
    button.click()


def _settled(app: object) -> bool:
    return app.photo_gate.state is not GateState.RUNNING  # type: ignore[attr-defined]


def test_window_source_does_not_import_the_checker() -> None:
    source = (ROOT / "src/voxprint_forge/app/window.py").read_text(encoding="utf-8")
    assert "lora.image_check" not in source
    assert "lora.image_job" not in source
    assert "import torch" not in source


def test_importing_the_window_does_not_load_the_checker() -> None:
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "offscreen"
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import sys\n"
        "import voxprint_forge.app.window\n"
        "banned = ('voxprint_forge.lora.image_check', 'voxprint_forge.lora.image_job', 'torch')\n"
        "print('\\n'.join(name for name in banned if name in sys.modules))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "", done.stdout + done.stderr


def test_malformed_result_is_not_stored(tmp_path: Path) -> None:
    checks = _fake_checks(tmp_path)
    try:
        checks.observe(
            WorkerEvent(
                kind=EventKind.FINISHED,
                value={"results": [{"id": "photo", "ok": 0, "message": "lora.drop.collage"}]},
            )
        )
        assert checks.cache == {}
        assert checks.failure == CHECK_STOPPED
    finally:
        checks.host.shutdown(timeout=0.2)


def test_check_handler_lets_an_error_escape(monkeypatch: pytest.MonkeyPatch) -> None:
    class Boom(ImageChecker):
        def classify(self, path: Path, token: object) -> tuple[bool, str]:
            del path, token
            raise RuntimeError("detector down")

    monkeypatch.setattr("voxprint_forge.lora.image_job.ImageChecker", Boom)
    ctx = SimpleNamespace(
        payload={"items": [{"id": "a", "path": "a.png"}]},
        model_object=None,
        cancelled=lambda: False,
        progress=lambda **_kwargs: None,
    )
    with pytest.raises(RuntimeError, match="detector down"):
        run_check(ctx)  # type: ignore[arg-type]


def test_cancelled_check_returns_nothing() -> None:
    seen: list[Path] = []

    class Quiet(ImageChecker):
        def classify(self, path: Path, token: object) -> tuple[bool, str]:
            del token
            seen.append(path)
            return False, "lora.drop.collage"

    ctx = SimpleNamespace(
        payload={"items": [{"id": "a", "path": "a.png"}]},
        model_object=Quiet(),
        cancelled=lambda: True,
        progress=lambda **_kwargs: None,
    )
    with pytest.raises(JobCancelled):
        run_check(ctx)  # type: ignore[arg-type]
    assert seen == []


def test_worker_failure_leaves_the_window_up(make_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_TEST_IMAGE_MODE", "fail")
    app = make_app(image_checks=_fake_checks(tmp_path))
    _start(app, tmp_path)
    _pump(app, lambda: _settled(app))
    assert app.window.isVisible()
    assert app.photo_gate.state is GateState.FAILED
    assert app.draft.inactive == set()
    assert app.draft.photo_notes == {}
    assert app.image_checks.cache == {}
    bar = app.window.findChild(QtWidgets.QWidget, "InfoBar")
    assert bar is not None and bar.isVisible()
    assert app.window.findChild(QtWidgets.QLabel, "forgePhotoRefusal") is None


def test_worker_crash_leaves_the_window_up(make_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VOXPRINT_TEST_IMAGE_MODE", "crash")
    app = make_app(image_checks=_fake_checks(tmp_path))
    _start(app, tmp_path)
    _pump(app, lambda: _settled(app))
    assert app.window.isVisible()
    assert app.photo_gate.state is GateState.FAILED
    assert app.draft.inactive == set()
    assert app.draft.photo_notes == {}
    assert app.image_checks.cache == {}
    bar = app.window.findChild(QtWidgets.QWidget, "InfoBar")
    assert bar is not None and bar.isVisible()
    assert app.window.findChild(QtWidgets.QLabel, "forgePhotoRefusal") is None


def test_malformed_worker_rows_are_not_photo_refusals(
    make_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VOXPRINT_TEST_IMAGE_MODE", "bad")
    app = make_app(image_checks=_fake_checks(tmp_path))
    _start(app, tmp_path)
    _pump(app, lambda: _settled(app))
    assert app.window.isVisible()
    assert app.photo_gate.state is GateState.FAILED
    assert app.draft.inactive == set()
    assert app.draft.photo_notes == {}
    assert app.image_checks.cache == {}


def test_worker_quality_result_sets_one_photo_aside(
    make_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    photos = make_photos(tmp_path / "photos", 2)
    refused = photo_key(photos[0])
    marker = tmp_path / "unloaded"
    monkeypatch.setenv("VOXPRINT_TEST_IMAGE_REFUSE", refused)
    monkeypatch.setenv("VOXPRINT_TEST_UNLOAD_MARKER", str(marker))
    app = make_app(image_checks=_fake_checks(tmp_path))
    app.draft.add_photos(photos)
    app.draft.kind = "object"
    app.repaint()
    button = app.window.findChild(QtWidgets.QPushButton, "forge.photos.check")
    assert button is not None
    button.click()
    _pump(app, lambda: _settled(app))
    assert app.photo_gate.state is GateState.PASSED
    assert refused in app.draft.inactive
    assert photo_key(photos[1]) not in app.draft.inactive
    assert app.draft.photo_notes[refused] == "lora.drop.collage"
    assert app.image_checks.cache[refused] == (False, "lora.drop.collage")
    assert app.image_checks.cache[photo_key(photos[1])] == (True, "")
    _pump(app, marker.is_file, timeout=3.0)
    assert marker.read_text(encoding="utf-8") == "unloaded"


def test_cancelling_the_worker_does_not_set_photos_aside(
    make_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    photos = make_photos(tmp_path / "photos", 2)
    monkeypatch.setenv("VOXPRINT_TEST_IMAGE_MODE", "wait")
    monkeypatch.setenv("VOXPRINT_TEST_IMAGE_REFUSE", photo_key(photos[0]))
    app = make_app(image_checks=_fake_checks(tmp_path))
    app.draft.add_photos(photos)
    app.draft.kind = "object"
    app.repaint()
    button = app.window.findChild(QtWidgets.QPushButton, "forge.photos.check")
    assert button is not None
    button.click()
    _pump(app, lambda: app.photo_gate.state is GateState.RUNNING)
    button.click()
    _pump(app, lambda: _settled(app))
    assert app.window.isVisible()
    assert app.photo_gate.state is GateState.CANCELLED
    assert app.draft.inactive == set()
    assert app.draft.photo_notes == {}
    assert app.image_checks.cache == {}
    bar = app.window.findChild(QtWidgets.QWidget, "InfoBar")
    assert bar is None or not bar.isVisible()
