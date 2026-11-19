"""The finished window against the demo fixture: photos cards, live progress and the result card.

The fixture is synthetic end to end: generated abstract pictures, a fictional
adult "Demo Model" with a complete release.txt, a mid-training project folder
and a finished one. Nothing starts a training job — the running project is
*followed* exactly like a terminal run.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from forge_gui_demo import (  # noqa: E402
    DEFAULT_STRENGTH,
    LORA_BYTES,
    TRIGGER,
    build_demo,
)

try:
    from PySide6 import QtWidgets
    from voxprint_common.i18n import set_language

    from tests.forge_app_stub import ALL_COMMANDS, StubRunner, fake_image_checks, make_photos
except ImportError:  # pragma: no cover - the GUI tests below are skipped without PySide6
    set_language = None
    QtWidgets = None

from voxprint_forge.app.draft import STEPS  # noqa: E402
from voxprint_forge.app.text import tr  # noqa: E402
from voxprint_forge.lora.project import Project  # noqa: E402

if TYPE_CHECKING:
    from voxprint_forge.app.window import ForgeApp

requires_qt = pytest.mark.skipif(QtWidgets is None, reason="PySide6 is not installed")


@pytest.fixture
def make_app(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    from voxprint_forge.app.window import ForgeApp

    made = []

    def build(projects_root: Path | None = None) -> ForgeApp:
        set_language("en")
        app = ForgeApp(
            runner=StubRunner(),
            home=tmp_path / "state",
            projects_root=projects_root if projects_root is not None else tmp_path / "state" / "lora",
            commands=ALL_COMMANDS,
            open_path=lambda _p: None,
            detached=lambda _a: None,
            image_checks=fake_image_checks(tmp_path / f"worker-{len(made)}"),
        )
        app.window.show()
        qapp.processEvents()
        made.append(app)
        return app

    yield build
    for app in made:
        app._stop_cli_watch()
        app.expert.set_enabled(False)
        app.window.close()
    set_language("en")


@pytest.fixture
def demo(tmp_path) -> dict[str, Path]:
    return build_demo(tmp_path / "demo")


def _page(app: ForgeApp) -> QtWidgets.QWidget:
    page = app.window.findChild(QtWidgets.QWidget, "forgeStepPage")
    assert page is not None
    return page



@requires_qt
def test_followed_running_project_paints_step_loss_eta_vram_and_previews(make_app, demo, qapp) -> None:
    app = make_app(projects_root=demo["root"] / "projects")
    app.open_project(demo["running"])
    qapp.processEvents()
    assert app.layout.sections.step.step_id == "train"
    assert app._followed_cli_run()

    page = _page(app)
    assert page.findChild(QtWidgets.QLabel, "progressText").text() == tr(
        "lora.progress.status", step=640, steps=1600,
        elapsed=tr("lora.progress.span.minutes", count=25),
        eta=tr("lora.progress.span.minutes", count=38),
    )
    assert page.findChild(QtWidgets.QLabel, "forge.progress.stage").text() == tr(
        "forge.progress.stage", stage=tr("forge.app.stage.train")
    )
    assert page.findChild(QtWidgets.QLabel, "forge.progress.step").text() == tr(
        "forge.progress.step", step=640, total=1600
    )
    assert page.findChild(QtWidgets.QLabel, "forge.train.loss").text() == tr(
        "lora.progress.loss", loss="0.1847"
    )
    assert page.findChild(QtWidgets.QLabel, "forge.progress.elapsed").text() == tr(
        "forge.progress.elapsed", elapsed=tr("lora.progress.span.minutes", count=25)
    )
    assert page.findChild(QtWidgets.QLabel, "forge.progress.eta").text() == tr(
        "forge.progress.eta", eta=tr("lora.progress.span.minutes", count=38)
    )
    assert page.findChild(QtWidgets.QLabel, "forge.train.vram").text() == tr(
        "forge.app.train.vram", allocated="9.5", reserved="10.2"
    )
    bar = page.findChild(QtWidgets.QProgressBar, "progress")
    assert bar.isVisible() and bar.value() == 400
    assert page.findChild(QtWidgets.QLabel, "progressPercent").text() == "40%"
    strip = page.findChild(QtWidgets.QListWidget, "forge.progress.previews")
    assert strip is not None and strip.count() == 2



@requires_qt
def test_done_project_paints_path_trigger_strength_and_sha256(make_app, demo, qapp) -> None:
    app = make_app(projects_root=demo["root"] / "projects")
    app.open_project(demo["done"])
    qapp.processEvents()
    assert app.layout.sections.step.step_id == "result"

    page = _page(app)
    lora = demo["done"] / "lora.safetensors"
    assert page.findChild(QtWidgets.QLabel, "forge.result.path").text() == str(lora)
    assert page.findChild(QtWidgets.QLabel, "forge.result.trigger").text() == TRIGGER
    assert page.findChild(QtWidgets.QLabel, "forge.result.strength").text() == f"{DEFAULT_STRENGTH}"
    assert page.findChild(QtWidgets.QLabel, "forge.result.sha256").text() == hashlib.sha256(LORA_BYTES).hexdigest()

    copy = next(b for b in page.findChildren(QtWidgets.QPushButton) if b.text() == tr("forge.app.result.copy_path"))
    copy.click()
    qapp.processEvents()
    assert QtWidgets.QApplication.clipboard().text() == str(lora)
    assert app._status_label.text() == tr("forge.app.result.path_copied")
    names = [b.text() for b in page.findChildren(QtWidgets.QPushButton)]
    assert tr("forge.app.result.open_folder") in names
    assert tr("forge.app.result.save") in names



@requires_qt
def test_photos_step_shows_the_release_fields_and_training_note(make_app, demo, qapp) -> None:
    app = make_app()
    app._dropped([demo["photos"]])
    qapp.processEvents()
    page = _page(app)

    count = page.findChild(QtWidgets.QLabel, "forge.photos.count")
    assert count is not None and count.text() == tr("forge.app.photos.count", count=12)
    card = page.findChild(QtWidgets.QFrame, "forge.photos.release")
    assert card is not None and card.property("checkGate") == "go"
    status = card.findChild(QtWidgets.QLabel, "forge.photos.release.status")
    assert status.text() == tr("forge.app.photos.release.complete")
    fields = {label.text() for label in card.findChildren(QtWidgets.QLabel, "forge.photos.release.field")}
    assert fields == {
        tr("forge.app.photos.release.field.name"),
        tr("forge.app.photos.release.field.adult"),
        tr("forge.app.photos.release.field.consent"),
        tr("forge.app.photos.release.field.no_minors"),
        tr("forge.app.photos.release.field.date"),
    }
    taken = card.findChild(QtWidgets.QLabel, "forge.photos.release.taken")
    assert taken is not None and tr("forge.app.photos.release.field.consent") in taken.text()

    training = page.findChild(QtWidgets.QFrame, "forge.photos.training")
    assert training is not None
    assert training.findChild(QtWidgets.QLabel, "forge.photos.training.note").text() == tr(
        "forge.app.photos.training.missing"
    )
    # The complete release answered the consent questions by itself.
    assert app.draft.name == "Demo Model" and app.draft.consent and app.draft.adult



@requires_qt
def test_the_release_template_button_writes_release_txt(make_app, tmp_path, qapp) -> None:
    app = make_app()
    folder = tmp_path / "photos"
    make_photos(folder, 12)
    app._dropped([folder])
    qapp.processEvents()
    page = _page(app)
    card = page.findChild(QtWidgets.QFrame, "forge.photos.release")
    assert card is not None and card.property("checkGate") == "run"

    card.findChild(QtWidgets.QPushButton, "forge.photos.release.template").click()
    qapp.processEvents()
    written = folder / "release.txt"
    assert written.is_file()
    assert 'Model name: "Tatiana Model"' in written.read_text(encoding="utf-8")
    page = _page(app)
    card = page.findChild(QtWidgets.QFrame, "forge.photos.release")
    assert card is not None and card.property("checkGate") == "go"



@requires_qt
def test_training_txt_card_shows_its_first_lines(make_app, tmp_path, qapp) -> None:
    app = make_app()
    folder = tmp_path / "photos"
    make_photos(folder, 12)
    (folder / "training.txt").write_text("light: soft window light\nlens: 85mm\nbackground: plain\n", encoding="utf-8")
    app._dropped([folder])
    qapp.processEvents()
    card = _page(app).findChild(QtWidgets.QFrame, "forge.photos.training")
    assert card is not None and card.property("checkGate") == "go"
    body = card.findChild(QtWidgets.QLabel, "forge.photos.training.body")
    assert "light: soft window light" in body.text()



@requires_qt
def test_train_step_edits_trigger_and_strength_and_records_them(make_app, tmp_path, qapp) -> None:
    from voxprint_forge.lora.export import recorded_strength

    app = make_app()
    photos = make_photos(tmp_path / "photos", 12)
    app.draft.add_photos(photos)
    app.draft.kind = "person"
    app.draft.set_names("Demo Model", "")
    app.draft.consent = app.draft.adult = True
    app.draft.set_verbal(True)
    Project(app.project).write({"kind": "person", "subject": {"name": "Demo Model", "trigger": app.draft.trigger}})
    app.layout.sections._index = STEPS.index("train")
    app.repaint()
    qapp.processEvents()
    page = _page(app)

    trigger = page.findChild(QtWidgets.QLineEdit, "forge.train.trigger")
    strength = page.findChild(QtWidgets.QDoubleSpinBox, "forge.train.strength")
    assert trigger is not None and trigger.text() == app.draft.trigger
    assert strength is not None and strength.value() == pytest.approx(DEFAULT_STRENGTH)
    assert strength.minimum() == 0.0 and strength.maximum() == 1.5

    trigger.setText("vxpmanual")
    strength.setValue(1.45)
    qapp.processEvents()
    data = json.loads((app.project / "project.json").read_text(encoding="utf-8"))
    assert data["subject"]["trigger"] == "vxpmanual"
    assert data["train"]["strength"] == pytest.approx(1.45)
    assert recorded_strength(data) == pytest.approx(1.45)
    # An out-of-range stored value keeps the built-in default.
    assert recorded_strength({"train": {"strength": 2.0}}) == pytest.approx(DEFAULT_STRENGTH)
