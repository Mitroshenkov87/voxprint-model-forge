"""Photos step: empty-state hint, one add action, icon remove buttons, sample-set toggle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402
from voxprint_common.i18n import set_language  # noqa: E402

from tests.forge_app_stub import ALL_COMMANDS, StubRunner, make_photos  # noqa: E402
from voxprint_forge.app import engine  # noqa: E402
from voxprint_forge.app.text import tr  # noqa: E402


@pytest.fixture
def make_app(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    from voxprint_forge.app.window import ForgeApp

    made = []

    def build(lang="en"):
        set_language(lang)
        app = ForgeApp(runner=StubRunner(), home=tmp_path / "state", commands=ALL_COMMANDS,
                       open_path=lambda _p: None, launch=lambda _a: None)
        app.window.show()
        qapp.processEvents()
        made.append(app)
        return app

    yield build
    for app in made:
        app.expert.set_enabled(False)
        app.window.close()
    set_language("en")


def _page(app):
    return app.window.findChild(QtWidgets.QWidget, "forgeStepPage")


def _drop_paths(widget, paths: list[Path]) -> None:
    """Deliver a file drop the way the desktop does."""
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(path)) for path in paths])
    pos = QtCore.QPointF(widget.rect().center())
    actions = QtCore.Qt.DropAction.CopyAction
    buttons = QtCore.Qt.MouseButton.LeftButton
    widget.dragEnterEvent(QtGui.QDragEnterEvent(pos, actions, mime, buttons, QtCore.Qt.KeyboardModifier.NoModifier))
    widget.dropEvent(QtGui.QDropEvent(pos, actions, mime, buttons, QtCore.Qt.KeyboardModifier.NoModifier))
    QtWidgets.QApplication.processEvents()


def _label_texts(app) -> list[str]:
    page = _page(app)
    return [w.text() for w in page.findChildren(QtWidgets.QLabel) if w.isVisibleTo(app.window)]


def test_empty_hint_is_visible_only_while_the_list_is_empty(make_app, tmp_path) -> None:
    from voxprint_forge.app.details import Details

    app = make_app()
    app.window.resize(1280, 900)
    QtWidgets.QApplication.processEvents()
    page = _page(app)
    hint = page.findChild(QtWidgets.QLabel, "forge.photos.empty_hint")
    title = page.findChild(QtWidgets.QLabel, "forge.photos.empty_title")
    icon = page.findChild(QtWidgets.QLabel, "forge.photos.empty_icon")
    board = page.findChild(QtWidgets.QWidget, "sourceDrop")
    assert hint is not None and hint.isVisibleTo(app.window)
    assert hint.text() == tr("forge.app.photos.empty_hint")
    assert hint.property("muted") == "true"
    assert title is not None and title.text() == tr("forge.app.photos.drop")
    assert title.text() not in hint.text()
    assert tr("forge.app.step.photos.short") not in hint.text()
    assert tr("lora.step.photos.desc") not in hint.text()
    texts = _label_texts(app)
    assert texts.count(tr("forge.app.photos.drop")) == 1
    assert texts.count(tr("forge.app.step.photos.short")) == 1
    assert page.findChild(QtWidgets.QLabel, "presetLabel") is None
    assert icon is not None and icon.pixmap() is not None and not icon.pixmap().isNull()
    assert board is not None and board.testAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground)
    assert "dashed" in app.window.styleSheet()
    assert 160 <= board.height() <= 360
    assert app.window.findChild(Details, "forge.details.step.photos") is None
    assert tr("forge.app.details") not in [b.text() for b in page.findChildren(QtWidgets.QPushButton)]
    app.draft.add_photos(make_photos(tmp_path / "p", 2))
    app.repaint()
    QtWidgets.QApplication.processEvents()
    page = _page(app)
    assert page.findChild(QtWidgets.QLabel, "forge.photos.empty_hint") is None
    assert page.findChild(QtWidgets.QLabel, "forge.photos.empty_title") is None
    assert page.findChild(QtWidgets.QLabel, "forge.photos.empty_icon") is None
    board = page.findChild(QtWidgets.QWidget, "sourceDrop")
    add = page.findChild(QtWidgets.QPushButton, "forge.photos.add")
    gap = add.mapTo(page, QtCore.QPoint(0, 0)).y() - board.mapTo(page, QtCore.QPoint(0, board.height())).y()
    assert 0 <= gap <= 24


def test_one_add_button_accepts_files_and_a_folder(make_app, tmp_path, monkeypatch) -> None:
    app = make_app()
    page = _page(app)
    adds = [b for b in page.findChildren(QtWidgets.QPushButton) if b.isVisibleTo(app.window) and b.menu() is not None]
    assert len(adds) == 1
    add = adds[0]
    assert add.text() == tr("forge.app.photos.add")
    assert add.objectName() == "forge.photos.add"
    labels = [action.text() for action in add.menu().actions()]
    assert labels == [tr("forge.app.photos.add_files"), tr("forge.app.photos.add_folder")]
    visible = [b.text() for b in page.findChildren(QtWidgets.QAbstractButton) if b.isVisibleTo(app.window)]
    assert tr("forge.app.photos.add_folder") not in visible
    files = make_photos(tmp_path / "files", 2)
    folder = tmp_path / "folder"
    packed = make_photos(folder, 3)

    def open_files(*_args, **_kwargs):
        return [str(path) for path in files], ""

    def open_folder(*_args, **_kwargs):
        return str(folder)

    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileNames", open_files)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", open_folder)
    add.menu().actions()[0].trigger()
    QtWidgets.QApplication.processEvents()
    assert set(app.draft.photos) == set(files)
    add = _page(app).findChild(QtWidgets.QPushButton, "forge.photos.add")
    add.menu().actions()[1].trigger()
    QtWidgets.QApplication.processEvents()
    assert set(app.draft.photos) == set(files + packed)
    extra = make_photos(tmp_path / "dropped", 1)
    nested = tmp_path / "nested"
    inside = make_photos(nested, 1)
    board = _page(app).findChild(QtWidgets.QWidget, "sourceDrop")
    _drop_paths(board, extra + [nested])
    assert set(app.draft.photos) == set(files + packed + extra + inside)


@pytest.mark.parametrize("lang", ["en", "ru"])
def test_remove_controls_are_icon_buttons(make_app, tmp_path, lang) -> None:
    from PySide6.QtTest import QTest

    app = make_app(lang)
    photos = make_photos(tmp_path / "p", 3)
    app.draft.add_photos(photos)
    app.repaint()
    QtWidgets.QApplication.processEvents()
    page = _page(app)
    buttons = [b for b in page.findChildren(QtWidgets.QToolButton, "forge.photos.remove") if b.isVisibleTo(app.window)]
    assert len(buttons) == 3
    assert tr("forge.app.photos.remove") not in [b.text() for b in page.findChildren(QtWidgets.QAbstractButton)]
    for button in buttons:
        assert button.text() == ""
        assert button.icon().isNull()  # hidden until hover or keyboard focus; no bar above the tile
        assert button.property("shown") == "false"
        assert button.width() == button.height()
        thumb = button.parentWidget()
        assert thumb.objectName() == "forge.photos.thumb"
        assert thumb.geometry().contains(button.geometry().center())
        assert button.y() < thumb.height() / 2
        button.setFocus()
        QtWidgets.QApplication.processEvents()
        assert not button.icon().isNull()
        assert button.property("shown") == "true"
        assert button.accessibleName() == tr("forge.app.photos.remove")
        assert button.toolTip() == tr("lora.photos.remove.tip")
    QTest.mouseClick(buttons[1], QtCore.Qt.MouseButton.LeftButton)
    QtWidgets.QApplication.processEvents()
    assert app.draft.photos == [photos[0], photos[2]]
    page = _page(app)
    tiles = [t for t in page.findChildren(QtWidgets.QFrame, "forge.photos.item") if t.isVisibleTo(app.window)]
    assert len(tiles) == 2
    selected = page.findChild(QtWidgets.QToolButton, "forge.photos.remove_selected")
    assert not selected.isVisibleTo(app.window)
    QTest.mouseClick(tiles[0], QtCore.Qt.MouseButton.LeftButton)
    QtWidgets.QApplication.processEvents()
    selected = _page(app).findChild(QtWidgets.QToolButton, "forge.photos.remove_selected")
    assert selected.isVisibleTo(app.window)
    assert selected.text() == ""
    assert not selected.icon().isNull()
    assert selected.property("selectionRemove") == "true"
    assert selected.accessibleName() == tr("forge.app.photos.remove_selected")
    assert selected.toolTip()
    QTest.mouseClick(selected, QtCore.Qt.MouseButton.LeftButton)
    QtWidgets.QApplication.processEvents()
    assert app.draft.photos == [photos[2]]


def test_kind_options_have_a_one_line_note(make_app) -> None:
    app = make_app()
    page = _page(app)
    label = next(w for w in page.findChildren(QtWidgets.QLabel) if w.text() == tr("forge.app.kind.label"))
    board = page.findChild(QtWidgets.QWidget, "sourceDrop")
    assert label.mapTo(page, QtCore.QPoint(0, 0)).y() < board.mapTo(page, QtCore.QPoint(0, 0)).y()
    segments = page.findChild(QtWidgets.QFrame, "kindSegments")
    assert segments is not None and segments.isVisibleTo(app.window)
    assert segments.mapTo(page, QtCore.QPoint(0, 0)).y() < board.mapTo(page, QtCore.QPoint(0, 0)).y()
    expected = {
        "person": "One specific person from the photos.",
        "object": "One specific thing or product.",
        "style": (
            "The look shared by the photos (lighting, colours, rendering, mood) "
            "rather than who or what is in them."
        ),
    }
    for kind, text in expected.items():
        note = page.findChild(QtWidgets.QLabel, f"forge.kind.note.{kind}")
        assert note is not None and note.isVisibleTo(app.window)
        assert note.text() == text
        chip = page.findChild(QtWidgets.QPushButton, f"forge.chip.{kind}")
        assert chip is not None and chip.property("segment") == "true"
        assert note.mapTo(page, QtCore.QPoint(0, 0)).y() > chip.mapTo(page, QtCore.QPoint(0, 0)).y()
        assert segments.isAncestorOf(chip)


def test_count_line_follows_the_subject_and_the_toggle(make_app) -> None:
    app = make_app("ru")
    assert tr("forge.app.step.photos.short") in _label_texts(app)
    for kind in ("person", "object", "style"):
        app.draft.kind = kind
        app.repaint()
        QtWidgets.QApplication.processEvents()
        texts = _label_texts(app)
        assert texts.count(tr(f"forge.app.step.photos.short.{kind}")) == 1
        assert tr("forge.app.step.photos.short") not in texts
        assert "одного объекта" not in " ".join(texts)
    app.window.findChild(QtWidgets.QCheckBox, "forge.photos.extend").click()
    QtWidgets.QApplication.processEvents()
    texts = _label_texts(app)
    assert texts.count(tr("forge.app.step.photos.short_refs.style")) == 1
    assert tr("forge.app.step.photos.short.style") not in texts


def test_extend_toggle_maps_onto_the_refs_pipeline(make_app, tmp_path) -> None:
    app = make_app()
    toggle = app.window.findChild(QtWidgets.QCheckBox, "forge.photos.extend")
    hint = app.window.findChild(QtWidgets.QLabel, "forge.photos.extend.hint")
    assert toggle is not None and not toggle.isChecked()
    assert toggle.property("switch") == "true"
    assert type(toggle).__name__ == "_Switch"
    assert toggle.text() == tr("forge.app.photos.extend")
    assert hint is not None and hint.isVisibleTo(app.window)
    assert hint.text() == tr("forge.app.photos.extend.hint")
    assert hint.property("muted") == "true"
    assert hint.mapTo(app.window, QtCore.QPoint(0, 0)).y() > toggle.mapTo(app.window, QtCore.QPoint(0, 0)).y()
    assert app.draft.source == "photos"
    assert app.window.findChild(QtWidgets.QPushButton, "forge.chip.refs") is None
    assert app.window.findChild(QtWidgets.QPushButton, "forge.chip.photos") is None
    app.draft.kind = "object"
    app.draft.add_photos(make_photos(tmp_path / "one", 1))
    app.repaint()
    assert not app.draft.check_photos().ok
    app.window.findChild(QtWidgets.QCheckBox, "forge.photos.extend").click()
    QtWidgets.QApplication.processEvents()
    assert app.draft.source == "refs"
    assert app.window.findChild(QtWidgets.QCheckBox, "forge.photos.extend").isChecked()
    assert app.draft.check_photos().ok
    approved = make_photos(tmp_path / "kept", 2)
    app.draft.approved = list(approved)
    assert app.draft.training_photos() == approved
    args = engine.prepare_args(app.draft, app.project)
    flagged = [Path(args[i + 1]) for i, flag in enumerate(args) if flag == "--photo"]
    assert flagged == approved
    assert "--extend" not in args
    app.window.findChild(QtWidgets.QCheckBox, "forge.photos.extend").click()
    QtWidgets.QApplication.processEvents()
    assert app.draft.source == "photos"
    assert not app.window.findChild(QtWidgets.QCheckBox, "forge.photos.extend").isChecked()
    assert app.draft.training_photos() == app.draft.photos
    folder = tmp_path / "old-project"
    (folder / "photos").mkdir(parents=True)
    make_photos(folder / "photos", 2)
    (folder / "project.json").write_text(json.dumps({
        "schema": "voxprint-lora-project/1",
        "kind": "style",
        "subject": {"name": "Soft light", "stage_name": "", "trigger": "soft_light"},
    }), encoding="utf-8")
    app.open_project(folder)
    QtWidgets.QApplication.processEvents()
    assert app.draft.kind == "style"
    assert app.draft.name == "Soft light"
    assert app.draft.source == "photos"
    assert len(app.draft.photos) == 2
    assert not app.window.findChild(QtWidgets.QCheckBox, "forge.photos.extend").isChecked()
