"""Review screen, the sample-set switch, and the shared button radius."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtGui, QtTest, QtWidgets  # noqa: E402
from voxprint_common.i18n import set_language  # noqa: E402

from tests.forge_app_stub import ALL_COMMANDS, StubRunner, make_photos  # noqa: E402
from voxprint_forge.app.details import Details  # noqa: E402
from voxprint_forge.app.draft import STEPS  # noqa: E402
from voxprint_forge.app.text import tr  # noqa: E402
from voxprint_forge.app.window import (  # noqa: E402
    FORGE_STYLE,
    _control_width,
    button_radius_px,
    format_span,
    menu_popup_point,
    popup_menu,
    register_catalogs,
)
from voxprint_forge.lora.project import SCHEMA, Project  # noqa: E402
from voxprint_forge.lora.review import save_state, transition  # noqa: E402


def _saturation(color: QtGui.QColor) -> float:
    """How far the colour is from grey, from 0 to 1."""
    hi = max(color.red(), color.green(), color.blue())
    lo = min(color.red(), color.green(), color.blue())
    if hi == 0:
        return 0.0
    return (hi - lo) / hi


def _has_light_pixel(widget: QtWidgets.QWidget) -> bool:
    """True when the widget paints a light pixel, so the text is visible on the theme."""
    image = widget.grab().toImage()
    for y in range(0, image.height(), 2):
        for x in range(0, image.width(), 2):
            color = image.pixelColor(x, y)
            if color.red() > 180 and color.green() > 180 and color.blue() > 170:
                return True
    return False


@pytest.fixture
def make_app(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    from voxprint_forge.app.window import ForgeApp

    made = []

    def build(lang="en"):
        set_language(lang)
        app = ForgeApp(
            runner=StubRunner(), home=tmp_path / "state", commands=ALL_COMMANDS,
            open_path=lambda _p: None, launch=lambda _a: None,
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


def test_square_buttons_use_the_shared_radius() -> None:
    radius = button_radius_px()
    assert radius == 4
    assert "border-radius: 11px" not in FORGE_STYLE
    assert f"border-radius: {radius}px" in FORGE_STYLE
    assert 'QPushButton[keepMark="true"]' in FORGE_STYLE
    assert 'QPushButton[reviewMark="true"]' in FORGE_STYLE
    assert 'QLabel[reviewBadge="like"]' in FORGE_STYLE
    assert 'QLabel[reviewBadge="dislike"]' in FORGE_STYLE
    assert "border: 2px solid #c45c5c" in FORGE_STYLE
    assert 'QToolButton[photoRemove="true"]' in FORGE_STYLE
    assert 'QToolButton[selectionRemove="true"]' in FORGE_STYLE
    assert "QFrame#card" in FORGE_STYLE and "border-radius: 6px" in FORGE_STYLE


def test_the_sample_switch_adds_review_rounds_and_off_keeps_a_straight_train(make_app, tmp_path) -> None:
    app = make_app()
    photos = make_photos(tmp_path / "photos", 2)
    app.draft.kind = "object"
    app.draft.add_photos(photos)
    app.draft.set_source("refs")
    flagged = app._train_args()
    assert flagged[flagged.index("--review-rounds") + 1] == "3"
    assert flagged[flagged.index("--samples-per-round") + 1] == "8"
    assert flagged[flagged.index("--first-round-steps") + 1] == "120"
    assert flagged[flagged.index("--first-round-samples") + 1] == "6"
    assert flagged[flagged.index("--sample-steps") + 1] == "28"
    app.draft.set_source("photos")
    plain = app._train_args()
    assert plain[plain.index("--review-rounds") + 1] == "0"
    assert "--samples-per-round" not in plain
    dry = app._train_args(dry_run=True)
    assert "--review-rounds" not in dry


def _seed_review(project: Path, photos: list[Path]) -> None:
    root = Project(project)
    root.prepared.mkdir(parents=True, exist_ok=True)
    items = []
    for index, photo in enumerate(photos):
        name = f"prepared/p{index:02d}.png"
        target = project / name
        target.write_bytes(photo.read_bytes())
        items.append({"id": f"p{index:02d}", "file": name, "long": "a gradient", "short": "gradient", "round": 0})
    (project / "captions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in items), encoding="utf-8")
    root.write({
        "schema": SCHEMA,
        "kind": "object",
        "gate_passed": True,
        "subject": {"name": "Test cube", "trigger": "tcube", "rights_confirmed": True, "kind": "object"},
    })
    state = transition(
        {}, "start", rounds=3, samples_per_round=3, prompts=["a cube"], seed=1, base="draw-test", config={"steps": 1},
    )
    samples = []
    for index, verdict in enumerate(("like", "dislike", "unmarked")):
        folder = project / "review" / "round-0001"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"s{index:02d}.png"
        image = QtGui.QImage(64, 64, QtGui.QImage.Format.Format_RGB32)
        image.fill(QtGui.qRgb(40 + index * 70, 80, 160))
        image.save(str(path))
        samples.append({
            "id": f"r1-s{index:02d}",
            "file": f"review/round-0001/s{index:02d}.png",
            "seed": index,
            "prompt": "a cube",
            "round": 1,
            "verdict": verdict,
        })
    state = transition(state, "pause", checkpoint="checkpoints/step-000001.safetensors", samples=samples)
    save_state(root, state)


def test_the_sample_switch_hint_describes_pictures_during_training() -> None:
    set_language("en")
    assert "before training" not in tr("forge.app.photos.extend.hint").lower()
    assert tr("forge.app.photos.extend.hint").startswith("During training")
    set_language("ru")
    hint = tr("forge.app.photos.extend.hint")
    assert "перед обучением" not in hint
    assert hint == (
        "Во время обучения приложение покажет новые картинки — вы отметите удачные, и они войдут в набор."
    )
    set_language("en")


def _assert_review_fits(app, qapp) -> None:
    """At 1280×800 the heading sits fully in the scroll and the continue row sits in the window."""
    app.window.setFixedSize(1280, 800)
    qapp.processEvents()
    heading = app.window.findChild(QtWidgets.QLabel, "forge.review.heading")
    assert heading is not None
    assert heading.height() >= heading.fontMetrics().height()
    area = heading.parentWidget()
    while area is not None and not isinstance(area, QtWidgets.QScrollArea):
        area = area.parentWidget()
    assert isinstance(area, QtWidgets.QScrollArea)
    area.verticalScrollBar().setValue(0)
    qapp.processEvents()
    viewport = area.viewport()
    top = heading.mapTo(viewport, QtCore.QPoint(0, 0))
    assert top.y() >= 8
    assert top.y() + heading.height() <= viewport.height()
    host = app.window.findChild(QtWidgets.QPushButton, "forge.review.continue").parentWidget()
    assert host.width() >= host.sizeHint().width()
    origin = host.mapTo(app.window, QtCore.QPoint(0, 0))
    assert origin.x() >= 0 and origin.y() >= 0
    assert origin.x() + host.width() <= app.window.width()
    assert origin.y() + host.height() <= app.window.height()
    view_bottom = viewport.mapTo(app.window, QtCore.QPoint(0, viewport.height())).y()
    assert origin.y() >= view_bottom - 1
    note = app.window.findChild(QtWidgets.QLabel, "forge.review.note")
    note_at = note.mapTo(app.window, QtCore.QPoint(0, note.height()))
    assert note_at.y() <= app.window.height()


def _assert_menu_anchored_to_button(
    menu: QtWidgets.QMenu, button: QtWidgets.QAbstractButton, window: QtWidgets.QWidget,
) -> None:
    """The menu's right edge meets the button's, and it touches the button from below or from above."""
    origin = window.mapFromGlobal(menu.pos())
    top_left = button.mapTo(window, QtCore.QPoint(0, 0))
    bottom_right = button.mapTo(window, QtCore.QPoint(button.width(), button.height()))
    menu_right = origin.x() + menu.width()
    menu_bottom = origin.y() + menu.height()
    assert abs(menu_right - bottom_right.x()) <= 2
    gap_below = origin.y() - bottom_right.y()
    gap_above = top_left.y() - menu_bottom
    if bottom_right.y() + menu.height() <= window.height():
        assert 0 <= gap_below <= 2
    else:
        assert 0 <= gap_above <= 2


def test_the_finish_menu_hangs_off_the_finish_button(make_app, tmp_path, qapp) -> None:
    """At 1280×800 the menu's top meets the button's bottom and their right edges line up."""
    app = make_app("ru")
    photos = make_photos(tmp_path / "src", 2)
    _seed_review(app.project, photos)
    app.layout.sections._index = STEPS.index("train")
    app.window.setFixedSize(1280, 800)
    app.repaint()
    qapp.processEvents()
    finish = app.window.findChild(QtWidgets.QPushButton, "forge.review.finish")
    assert finish is not None and finish.text() == "Завершить"
    menu = finish.menu()
    assert menu is not None
    QtTest.QTest.mouseClick(finish, QtCore.Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert menu.isVisible()
    _assert_menu_anchored_to_button(menu, finish, app.window)
    image = menu.grab().toImage()
    edge = image.pixelColor(0, image.height() // 2)
    fill = image.pixelColor(12, image.height() // 2)
    assert edge.lightness() > fill.lightness() + 40
    discard = next(item for item in menu.actions() if item.objectName() == "forge.review.finish.discard")
    assert _has_danger_text(image, menu.actionGeometry(discard))
    assert not _has_danger_text(image, menu.actionGeometry(menu.actions()[0]))


def test_the_finish_menu_flips_above_the_button_when_nothing_fits_below(qapp) -> None:
    """A button on the bottom edge opens the menu upward, still sharing its right edge."""
    window = QtWidgets.QWidget()
    window.setFixedSize(1280, 220)
    window.show()
    button = QtWidgets.QPushButton("Завершить", window)
    button.setGeometry(1080, 176, 140, 32)
    menu = QtWidgets.QMenu(button)
    menu.setObjectName("forgeReviewMenu")
    save = menu.addAction("Сохранить")
    save.setObjectName("forge.review.finish.save")
    discard = menu.addAction("Принудительно, без сохранения")
    discard.setObjectName("forge.review.finish.discard")
    button.setMenu(menu)
    button.show()
    qapp.processEvents()
    popup_menu(button)
    qapp.processEvents()
    assert menu.isVisible()
    _assert_menu_anchored_to_button(menu, button, window)
    origin = window.mapFromGlobal(menu.pos())
    top = button.mapTo(window, QtCore.QPoint(0, 0)).y()
    assert 0 <= top - (origin.y() + menu.height()) <= 2
    menu.close()
    window.close()


def _assert_review_tooltips(window: QtWidgets.QWidget) -> None:
    """The main button, ▾, Finish, and every finish-menu item explain themselves."""
    cont = window.findChild(QtWidgets.QPushButton, "forge.review.continue")
    chevron = window.findChild(QtWidgets.QPushButton, "forge.review.continue.chevron")
    finish = window.findChild(QtWidgets.QPushButton, "forge.review.finish")
    assert cont is not None and chevron is not None and finish is not None
    assert cont.toolTip().strip()
    assert chevron.toolTip().strip()
    assert finish.toolTip().strip()
    menu = finish.menu()
    assert menu is not None and menu.toolTipsVisible()
    actions = [item for item in menu.actions() if not item.isSeparator()]
    assert actions
    assert all(item.toolTip().strip() for item in actions)


def test_the_review_note_follows_the_button_and_the_tooltips_explain_it(make_app, tmp_path, qapp) -> None:
    """One muted line names the current mode, and the long explanations live on the controls."""
    app = make_app("ru")
    photos = make_photos(tmp_path / "src", 2)
    _seed_review(app.project, photos)
    app.layout.sections._index = STEPS.index("train")
    app.repaint()
    qapp.processEvents()
    note = app.window.findChild(QtWidgets.QLabel, "forge.review.note")
    cont = app.window.findChild(QtWidgets.QPushButton, "forge.review.continue")
    chevron = app.window.findChild(QtWidgets.QPushButton, "forge.review.continue.chevron")
    finish = app.window.findChild(QtWidgets.QPushButton, "forge.review.finish")
    assert note is not None and cont is not None and chevron is not None and finish is not None
    assert note.property("muted") == "true"
    assert note.text() == "Следующий раунд и снова проверка"
    assert "\n" not in note.text()
    assert cont.toolTip() == tr("forge.app.review.continue.tip")
    assert chevron.toolTip() == tr("forge.app.review.unattended.tip")
    assert finish.toolTip() == tr("forge.app.review.finish.tip")
    _assert_review_tooltips(app.window)
    QtTest.QTest.mouseClick(chevron, QtCore.Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert note.text() == "Дообучить без проверок"
    assert cont.toolTip() == tr("forge.app.review.unattended.tip")
    assert chevron.toolTip() == tr("forge.app.review.continue.tip")
    _assert_review_tooltips(app.window)
    app.window.close()
    english = make_app("en")
    _seed_review(english.project, photos)
    english.layout.sections._index = STEPS.index("train")
    english.repaint()
    qapp.processEvents()
    en_note = english.window.findChild(QtWidgets.QLabel, "forge.review.note")
    en_chevron = english.window.findChild(QtWidgets.QPushButton, "forge.review.continue.chevron")
    assert en_note is not None and en_chevron is not None
    assert en_note.text() == "Next round, then another review"
    assert "\n" not in en_note.text()
    _assert_review_tooltips(english.window)
    QtTest.QTest.mouseClick(en_chevron, QtCore.Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert en_note.text() == "Train on without reviews"
    _assert_review_tooltips(english.window)


def _has_danger_text(image: QtGui.QImage, rect: QtCore.QRect) -> bool:
    """True when ``rect`` paints the danger text colour, a light red."""
    inset = rect.adjusted(8, 4, -8, -4)
    for y in range(inset.top(), inset.bottom(), 2):
        for x in range(inset.left(), inset.right(), 2):
            color = image.pixelColor(x, y)
            if color.red() > 210 and color.red() > color.green() + 20 and color.blue() > 170:
                return True
    return False


def _card(button: QtWidgets.QPushButton) -> QtWidgets.QFrame:
    card = button.parentWidget()
    assert isinstance(card, QtWidgets.QFrame)
    return card


def test_the_review_screen_counts_marks_and_offers_continue_and_finish(make_app, tmp_path, qapp) -> None:
    app = make_app("ru")
    photos = make_photos(tmp_path / "src", 2)
    _seed_review(app.project, photos)
    app.layout.sections._index = STEPS.index("train")
    app.repaint()
    qapp.processEvents()
    assert app.window.findChild(QtWidgets.QLabel, "forge.review.heading").text() == "Проверьте новые картинки"
    assert app.window.findChild(QtWidgets.QLabel, "forge.review.lead").text() == (
        "Отметьте, какие подходят: 👍 — добавить в набор, 👎 — не брать."
    )
    assert app.window.findChild(Details, "forge.details.step.train") is None
    page = app.window.findChild(QtWidgets.QWidget, "forgeStepPage")
    labels = [w.text() for w in page.findChildren(QtWidgets.QLabel)]
    assert tr("forge.app.step.train.short") not in labels
    assert page.findChildren(QtWidgets.QPushButton, "disclosure") == []
    back = app.window.findChild(QtWidgets.QPushButton, "wizardBack")
    nxt = app.window.findChild(QtWidgets.QPushButton, "wizardNext")
    assert back is not None and nxt is not None
    assert not back.isVisible() and not nxt.isVisible()
    assert app._status_label.text() == tr("forge.app.review.footer", liked=1, disliked=1, unmarked=1)
    assert app.window.findChild(QtWidgets.QLabel, "forge.review.round").text() == "Раунд 1 из 3"
    finish = app.window.findChild(QtWidgets.QPushButton, "forge.review.finish")
    assert finish is not None and finish.text() == "Завершить"
    save_action = next(item for item in finish.menu().actions() if item.objectName() == "forge.review.finish.save")
    discard_action = next(
        item for item in finish.menu().actions() if item.objectName() == "forge.review.finish.discard"
    )
    assert save_action.text() == "Сохранить"
    assert discard_action.text() == "Принудительно, без сохранения"
    menu = finish.menu()
    assert menu is not None
    QtTest.QTest.mouseClick(finish, QtCore.Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert menu.isVisible()
    assert abs(menu.pos().x() - menu_popup_point(finish).x()) <= 2
    window_origin = app.window.mapToGlobal(QtCore.QPoint(0, 0))
    menu_right = menu.pos().x() + menu.width()
    button_right = finish.mapToGlobal(QtCore.QPoint(finish.width(), 0)).x()
    assert menu.pos().x() >= window_origin.x()
    assert menu_right <= window_origin.x() + app.window.width()
    assert menu_right <= button_right + 2
    menu_pixel = menu.grab().toImage().pixelColor(6, 6)
    assert menu_pixel.red() < 90 and menu_pixel.green() < 90 and menu_pixel.blue() < 110
    menu.close()
    qapp.processEvents()
    cont = app.window.findChild(QtWidgets.QPushButton, "forge.review.continue")
    chevron = app.window.findChild(QtWidgets.QPushButton, "forge.review.continue.chevron")
    host = cont.parentWidget()
    assert cont is not None and chevron is not None and host is not None
    assert cont.text() == "Продолжить с подкреплением" and cont.isEnabled() and cont.menu() is None
    text_width = host.fontMetrics().horizontalAdvance(cont.text())
    plain_width = host.fontMetrics().horizontalAdvance("Просто продолжить")
    assert host.sizeHint().width() >= text_width
    assert host.sizeHint().width() >= plain_width
    assert host.width() >= host.sizeHint().width()
    assert host.minimumSizeHint().width() == host.sizeHint().width()
    assert finish.x() > host.x() and finish.width() < host.width() and finish.height() <= host.height()
    face_image = host.grab().toImage()
    face = face_image.pixelColor(12, host.height() // 2)
    assert face.red() > 200 and face.green() > 180 and face.blue() < 160
    corner = face_image.pixelColor(0, 0)
    assert 10 < corner.red() < 80 and corner.green() < 80 and corner.blue() < 90
    _assert_review_fits(app, qapp)
    note = app.window.findChild(QtWidgets.QLabel, "forge.review.note")
    assert note is not None and note.property("muted") == "true"
    assert note.text() == tr("forge.app.review.note.reinforced")
    assert "\n" not in note.text()
    counts = app.window.findChild(QtWidgets.QLabel, "forge.review.counts")
    assert counts.text() == tr("forge.app.review.counts", liked=1, disliked=1, unmarked=1)
    assert len(app.window.findChildren(QtWidgets.QLabel, "forge.review.original")) == 2
    like = app.window.findChild(QtWidgets.QPushButton, "forge.review.like.r1-s02")
    dislike = app.window.findChild(QtWidgets.QPushButton, "forge.review.dislike.r1-s00")
    assert like is not None and dislike is not None
    assert like.width() >= 32 and like.height() == like.width()
    assert like.toolTip() == tr("forge.app.review.like.tip")
    assert dislike.toolTip() == tr("forge.app.review.dislike.tip")
    liked = _card(app.window.findChild(QtWidgets.QPushButton, "forge.review.like.r1-s00"))
    assert liked.property("verdict") == "like"
    liked_badge = liked.findChild(QtWidgets.QLabel, "forge.review.badge")
    assert liked_badge.isVisible() and liked_badge.text() == "✓"
    assert liked_badge.property("reviewBadge") == "like"
    assert liked.findChild(QtWidgets.QLabel, "forge.review.picture").graphicsEffect() is None
    left = _card(app.window.findChild(QtWidgets.QPushButton, "forge.review.dislike.r1-s01"))
    assert left.property("verdict") == "dislike"
    left_badge = left.findChild(QtWidgets.QLabel, "forge.review.badge")
    assert left_badge.isVisible() and left_badge.text() == "×"
    effect = left.findChild(QtWidgets.QLabel, "forge.review.picture").graphicsEffect()
    assert isinstance(effect, QtWidgets.QGraphicsOpacityEffect) and effect.opacity() == 0.5
    plain = _card(like)
    assert plain.property("verdict") == "unmarked"
    assert not plain.findChild(QtWidgets.QLabel, "forge.review.badge").isVisible()
    assert plain.findChild(QtWidgets.QLabel, "forge.review.picture").graphicsEffect() is None
    like.click()
    qapp.processEvents()
    assert counts.text() == tr("forge.app.review.counts", liked=2, disliked=1, unmarked=0)
    assert app._status_label.text() == tr("forge.app.review.footer", liked=2, disliked=1, unmarked=0)
    dislike.click()
    qapp.processEvents()
    assert counts.text() == tr("forge.app.review.counts", liked=1, disliked=2, unmarked=0)
    saved = json.loads((app.project / "review" / "state.json").read_text(encoding="utf-8"))
    verdicts = {sample["id"]: sample["verdict"] for sample in saved["samples"]}
    assert verdicts == {"r1-s00": "dislike", "r1-s01": "dislike", "r1-s02": "like"}
    cont.click()
    qapp.processEvents()
    assert app.runner.calls[-1][:3] == ["--json", "lora", "continue"]
    assert "--unattended" not in app.runner.calls[-1]
    assert app.runner.calls[-1][-1] == str(app.project)

    def accept_finish() -> None:
        box = QtWidgets.QApplication.activeModalWidget()
        assert isinstance(box, QtWidgets.QDialog)
        assert box.objectName() == "forge.review.finish.confirm"
        assert box.text() == tr("forge.app.review.finish.confirm", done=1, planned=1)
        background = box.grab().toImage().pixelColor(8, box.height() - 8)
        assert background.red() < 80 and background.green() < 80 and background.blue() < 100
        label = box.findChild(QtWidgets.QLabel, "qt_msgbox_label")
        assert label is not None and _has_light_pixel(label)
        button = next(item for item in box.buttons() if item.text() == tr("forge.app.review.finish.save"))
        button.click()

    QtCore.QTimer.singleShot(0, accept_finish)
    save_action.trigger()
    qapp.processEvents()
    assert app.runner.calls[-1][:3] == ["--json", "lora", "finish"]


def test_opening_a_paused_review_returns_to_the_training_step(make_app, tmp_path, qapp) -> None:
    app = make_app()
    folder = tmp_path / "paused"
    photos = make_photos(folder / "photos", 2)
    _seed_review(folder, photos)
    app.open_project(folder)
    qapp.processEvents()
    assert app.layout.sections.step.step_id == "train"
    assert app.window.findChild(QtWidgets.QLabel, "forge.review.counts") is not None
    assert app.draft.rights is True
    assert app.draft.prepared is not None
    assert not app.window.findChild(QtWidgets.QPushButton, "wizardNext").isVisible()


def test_unmarked_pictures_ask_before_continue_and_marks_can_be_set_in_bulk(make_app, tmp_path, qapp) -> None:
    app = make_app("en")
    photos = make_photos(tmp_path / "src", 2)
    _seed_review(app.project, photos)
    app.layout.sections._index = STEPS.index("train")
    app.repaint()
    qapp.processEvents()
    assert app.window.findChild(QtWidgets.QLabel, "forge.review.round").text() == "Round 1 of 3"
    finish = app.window.findChild(QtWidgets.QPushButton, "forge.review.finish")
    assert finish is not None and finish.text() == "Finish"
    assert next(
        item for item in finish.menu().actions() if item.objectName() == "forge.review.finish.save"
    ).text() == "Save"
    cont = app.window.findChild(QtWidgets.QPushButton, "forge.review.continue")
    assert cont is not None and cont.text() == "Continue with reinforcement"
    seen: list[str] = []

    def click() -> None:
        box = QtWidgets.QApplication.activeModalWidget()
        assert isinstance(box, QtWidgets.QDialog)
        assert box.text() == tr("forge.app.review.unmarked")
        seen.append(box.text())
        button = next(item for item in box.buttons() if item.text() == tr("forge.app.review.unmarked.cancel"))
        button.click()

    QtCore.QTimer.singleShot(0, click)
    cont.click()
    qapp.processEvents()
    assert seen == [tr("forge.app.review.unmarked")]
    assert app.runner.calls == []
    app.window.findChild(QtWidgets.QPushButton, "forge.review.like_all").click()
    qapp.processEvents()
    saved = json.loads((app.project / "review" / "state.json").read_text(encoding="utf-8"))
    assert {sample["verdict"] for sample in saved["samples"]} == {"like"}
    assert app._status_label.text() == tr("forge.app.review.footer", liked=3, disliked=0, unmarked=0)
    cont.click()
    qapp.processEvents()
    assert app.runner.calls[-1][:3] == ["--json", "lora", "continue"]
    app.window.findChild(QtWidgets.QPushButton, "forge.review.reset").click()
    qapp.processEvents()
    saved = json.loads((app.project / "review" / "state.json").read_text(encoding="utf-8"))
    assert {sample["verdict"] for sample in saved["samples"]} == {"unmarked"}
    thumb = app.window.findChild(QtWidgets.QPushButton, "forge.review.like.r1-s00")
    assert not thumb.isChecked()
    QtTest.QTest.keyClick(thumb, QtCore.Qt.Key.Key_Space)
    qapp.processEvents()
    assert thumb.isChecked()
    saved = json.loads((app.project / "review" / "state.json").read_text(encoding="utf-8"))
    verdicts = {sample["id"]: sample["verdict"] for sample in saved["samples"]}
    assert verdicts["r1-s00"] == "like"


def test_progress_spans_use_hours_and_minutes() -> None:
    register_catalogs()
    set_language("ru")
    assert format_span(120) == "2 мин"
    assert format_span(900) == "15 мин"
    assert format_span(3600) == "1 ч"
    assert format_span(5400) == "1 ч 30 мин"
    assert format_span(20) == "меньше 1 мин"
    set_language("en")
    assert format_span(120) == "2 min"
    assert format_span(3600) == "1 h"
    assert format_span(5400) == "1 h 30 min"
    assert format_span(20) == "under 1 min"


def test_a_running_stage_locks_the_form_and_shows_the_train_json_bar(make_app, qapp) -> None:
    app = make_app("ru")
    app.runner.busy = True
    app.project.mkdir(parents=True, exist_ok=True)
    save_state(Project(app.project), transition(
        {}, "start", rounds=3, samples_per_round=8, prompts=["shape"], seed=1, base="draw-test", config={"steps": 1500},
    ))
    (app.project / "train.json").write_text(json.dumps({
        "step": 450, "total": 1500, "loss": 0.18, "elapsed_seconds": 120, "eta_seconds": 900, "stage": "step",
    }), encoding="utf-8")
    app.layout.sections._index = STEPS.index("train")
    app.repaint()
    qapp.processEvents()
    page = app.window.findChild(QtWidgets.QWidget, "forgeStepPage")
    labels = [label.text() for label in page.findChildren(QtWidgets.QLabel)]
    assert "Идёт обучение — раунд 1 из 3" in labels
    assert tr("forge.app.step.train.short") not in labels
    text = page.findChild(QtWidgets.QLabel, "progressText")
    bar = page.findChild(QtWidgets.QProgressBar, "progress")
    percent = page.findChild(QtWidgets.QLabel, "progressPercent")
    assert text.text() == "Шаг 450 из 1500 · прошло 2 мин · осталось около 15 мин"
    assert "0.1800" not in text.text()
    assert bar.isVisible() and bar.value() == 300
    assert percent.text() == "30%" and percent.isVisible()
    assert text.mapTo(page, QtCore.QPoint(0, 0)).y() < bar.mapTo(page, QtCore.QPoint(0, 0)).y()
    assert bar.mapTo(page, QtCore.QPoint(0, 0)).x() < percent.mapTo(page, QtCore.QPoint(0, 0)).x()
    loss = page.findChild(QtWidgets.QLabel, "forge.train.loss")
    assert loss.text() == tr("lora.progress.loss", loss="0.1800")
    assert not app.window.findChild(QtWidgets.QPushButton, "forge.chip.balanced").isEnabled()
    assert not app.window.findChild(QtWidgets.QComboBox, "forge.train.base").isEnabled()
    previews = app.window.findChild(QtWidgets.QCheckBox, "forge.train.previews")
    assert previews is not None and previews.property("switch") == "true" and not previews.isEnabled()
    buttons = page.findChildren(QtWidgets.QPushButton)
    start = next(button for button in buttons if button.text() == tr("forge.app.train.start"))
    estimate = next(button for button in buttons if button.text() == tr("forge.app.train.estimate"))
    pause = next(button for button in buttons if button.text() == tr("forge.app.train.pause"))
    stop = next(button for button in buttons if button.text() == tr("forge.app.train.stop"))
    assert not start.isEnabled() and not estimate.isEnabled()
    assert pause.isEnabled() and stop.isEnabled()
    muted = estimate.grab().toImage().pixelColor(8, estimate.height() // 2)
    assert muted.red() < 80 and muted.green() < 80


def test_stop_asks_before_it_ends_the_run(make_app, qapp) -> None:
    app = make_app("ru")
    app.runner.busy = True
    app.layout.sections._index = STEPS.index("train")
    app.repaint()
    qapp.processEvents()
    stop = next(
        button for button in app.window.findChildren(QtWidgets.QPushButton)
        if button.text() == tr("forge.app.train.stop")
    )

    def cancel() -> None:
        box = QtWidgets.QApplication.activeModalWidget()
        assert isinstance(box, QtWidgets.QDialog)
        assert box.text() == tr("forge.app.train.stop.ask")
        button = next(item for item in box.buttons() if item.text() == tr("forge.app.train.stop.cancel"))
        button.click()

    QtCore.QTimer.singleShot(0, cancel)
    stop.click()
    qapp.processEvents()
    assert app.runner.stopped == 0

    def confirm() -> None:
        box = QtWidgets.QApplication.activeModalWidget()
        button = next(item for item in box.buttons() if item.text() == tr("forge.app.train.stop.confirm"))
        button.click()

    QtCore.QTimer.singleShot(0, confirm)
    stop.click()
    qapp.processEvents()
    assert app.runner.stopped == 1


def test_choice_buttons_fit_their_labels(make_app, qapp) -> None:
    app = make_app("ru")
    app.layout.sections._index = STEPS.index("train")
    app.repaint()
    qapp.processEvents()
    radius = button_radius_px()
    shared = (
        'QPushButton[chip="true"], QPushButton#overflow, QPushButton#expertToggle '
        f"{{ border-radius: {radius}px; }}"
    )
    assert shared in FORGE_STYLE
    assert "border-radius: 15px" not in FORGE_STYLE
    assert "border-radius: 14px" not in FORGE_STYLE
    expected = {"balanced": "Сбалансированная", "best": "Лучшее качество", "close": "Близко к фото"}
    for name, label in expected.items():
        button = app.window.findChild(QtWidgets.QPushButton, f"forge.chip.{name}")
        assert button.text() == label
        assert button.width() == _control_width(button)
        assert button.width() >= button.fontMetrics().horizontalAdvance(button.text())
    likeness = [
        button.text() for button in app.window.findChildren(QtWidgets.QPushButton, "forge.chip.balanced")
    ]
    assert "Умеренно" in likeness
    app.layout.sections._index = STEPS.index("photos")
    app.repaint()
    qapp.processEvents()
    for kind in ("person", "object", "style"):
        button = app.window.findChild(QtWidgets.QPushButton, f"forge.chip.{kind}")
        assert button.width() >= button.fontMetrics().horizontalAdvance(button.text())
    for name in ("overflow", "expertToggle"):
        button = app.window.findChild(QtWidgets.QPushButton, name)
        assert button is not None
        assert button.width() >= button.fontMetrics().horizontalAdvance(button.text())


def test_just_continue_sends_unattended(make_app, tmp_path, qapp) -> None:
    app = make_app("en")
    photos = make_photos(tmp_path / "src", 2)
    _seed_review(app.project, photos)
    app.layout.sections._index = STEPS.index("train")
    app.repaint()
    qapp.processEvents()
    button = app.window.findChild(QtWidgets.QPushButton, "forge.review.continue")
    chevron = app.window.findChild(QtWidgets.QPushButton, "forge.review.continue.chevron")
    host = button.parentWidget()
    assert button.text() == "Continue with reinforcement"
    assert host._anim.duration() == 150
    gold = host.grab().toImage().pixelColor(12, host.height() // 2)
    chevron.setFocus()
    QtTest.QTest.keyClick(chevron, QtCore.Qt.Key.Key_Right)
    qapp.processEvents()
    assert button.text() == "Just continue"
    saved = json.loads((app.project / "project.json").read_text(encoding="utf-8"))
    assert saved["continue_mode"] == "plain"
    QtTest.QTest.qWait(200)
    qapp.processEvents()
    plain = host.grab().toImage().pixelColor(12, host.height() // 2)
    assert _saturation(plain) < _saturation(gold)
    assert plain.red() > 140 and plain.green() > 120
    app.repaint()
    qapp.processEvents()
    button = app.window.findChild(QtWidgets.QPushButton, "forge.review.continue")
    chevron = app.window.findChild(QtWidgets.QPushButton, "forge.review.continue.chevron")
    assert button.text() == "Just continue"
    chevron.setFocus()
    QtTest.QTest.keyClick(chevron, QtCore.Qt.Key.Key_Left)
    qapp.processEvents()
    assert button.text() == "Continue with reinforcement"
    QtTest.QTest.keyClick(chevron, QtCore.Qt.Key.Key_Down)
    qapp.processEvents()
    assert button.text() == "Just continue"
    app.window.findChild(QtWidgets.QPushButton, "forge.review.like_all").click()
    qapp.processEvents()
    button = app.window.findChild(QtWidgets.QPushButton, "forge.review.continue")
    assert button.text() == "Just continue"
    button.setFocus()
    QtTest.QTest.keyClick(button, QtCore.Qt.Key.Key_Return)
    qapp.processEvents()
    assert app.runner.calls[-1][:4] == ["--json", "lora", "continue", str(app.project)]
    assert app.runner.calls[-1][-1] == "--unattended"


def test_finish_cancel_leaves_the_review_open(make_app, tmp_path, qapp) -> None:
    app = make_app("ru")
    photos = make_photos(tmp_path / "src", 2)
    _seed_review(app.project, photos)
    app.layout.sections._index = STEPS.index("train")
    app.repaint()
    qapp.processEvents()

    def cancel() -> None:
        box = QtWidgets.QApplication.activeModalWidget()
        assert isinstance(box, QtWidgets.QDialog)
        assert box.text() == tr("forge.app.review.finish.confirm", done=1, planned=1)
        button = next(item for item in box.buttons() if item.text() == tr("forge.app.review.unmarked.cancel"))
        button.click()

    QtCore.QTimer.singleShot(0, cancel)
    finish = app.window.findChild(QtWidgets.QPushButton, "forge.review.finish")
    save = next(item for item in finish.menu().actions() if item.objectName() == "forge.review.finish.save")
    save.trigger()
    qapp.processEvents()
    assert app.runner.calls == []
    assert json.loads((app.project / "review" / "state.json").read_text(encoding="utf-8"))["phase"] == "review"


def test_the_result_names_the_steps_the_lora_got(make_app, qapp, monkeypatch) -> None:
    app = make_app("ru")
    app.project.mkdir(parents=True, exist_ok=True)
    Project(app.project).write({
        "schema": SCHEMA,
        "kind": "object",
        "gate_passed": True,
        "subject": {"name": "Test cube", "trigger": "tcube", "rights_confirmed": True, "kind": "object"},
        "train": {
            "state": "done",
            "step": 450,
            "steps_planned": 1500,
            "checkpoint": "checkpoints/step-000001.safetensors",
        },
    })
    app.draft.trained = True
    app.layout.sections._index = STEPS.index("result")
    app.repaint()
    qapp.processEvents()
    label = app.window.findChild(QtWidgets.QLabel, "forge.result.steps")
    assert label is not None
    assert label.text() == "Завершено на 450 из 1500 шагов"
    chosen = app.project / "shape.voxlora"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *_args, **_kwargs: (str(chosen), ""))
    export = next(
        button for button in app.window.findChildren(QtWidgets.QPushButton) if button.text() == "Экспорт LoRA"
    )
    assert export.objectName() == "primary"
    export.click()
    qapp.processEvents()
    sent = app.runner.calls[-1]
    assert sent[:4] == ["--json", "lora", "export", str(app.project)]
    assert sent[sent.index("-o") + 1] == str(chosen)


def test_review_rounds_live_under_details(make_app, qapp) -> None:
    app = make_app("ru")
    app.draft.set_source("refs")
    app.layout.sections._index = STEPS.index("train")
    app.repaint()
    qapp.processEvents()
    details = app.window.findChild(Details, "forge.details.step.train")
    assert details is not None
    details.button.click()
    qapp.processEvents()
    spin = app.window.findChild(QtWidgets.QSpinBox, "forge.train.review_rounds")
    assert spin is not None
    assert spin.minimum() == 1 and spin.maximum() == 6 and spin.value() == 3
    assert spin.toolTip() == tr("forge.app.train.review_rounds.tip")
    label = app.window.findChild(QtWidgets.QLabel, "forge.train.review_rounds.label")
    assert label is not None and label.text() == "Число раундов"
    spin.setValue(4)
    args = app._train_args()
    assert args[args.index("--review-rounds") + 1] == "4"
    app.runner.busy = True
    app.repaint()
    qapp.processEvents()
    locked = app.window.findChild(QtWidgets.QSpinBox, "forge.train.review_rounds")
    assert locked is not None and not locked.isEnabled()
    app.runner.busy = False
    app.draft.set_source("photos")
    app.repaint()
    qapp.processEvents()
    assert app.window.findChild(QtWidgets.QSpinBox, "forge.train.review_rounds") is None


def test_first_round_steps_live_under_details(make_app, qapp) -> None:
    app = make_app("ru")
    app.draft.set_source("refs")
    app.layout.sections._index = STEPS.index("train")
    app.repaint()
    qapp.processEvents()
    details = app.window.findChild(Details, "forge.details.step.train")
    assert details is not None
    details.button.click()
    qapp.processEvents()
    spin = app.window.findChild(QtWidgets.QSpinBox, "forge.train.first_round_steps")
    assert spin is not None
    assert spin.minimum() == 100 and spin.maximum() == 500 and spin.value() == 120
    assert spin.toolTip() == tr("forge.app.train.first_round_steps.tip")
    spin.setValue(250)
    args = app._train_args()
    assert args[args.index("--first-round-steps") + 1] == "250"


def test_round_one_shows_the_first_pictures_line_and_a_grey_start_button(make_app, qapp) -> None:
    app = make_app("ru")
    app.project.mkdir(parents=True, exist_ok=True)
    save_state(Project(app.project), transition(
        {}, "start", rounds=3, samples_per_round=8, first_round_steps=200, prompts=["shape"],
        seed=1, base="draw-test", config={"steps": 1500}, total_steps=1500,
    ))
    (app.project / "train.json").write_text(json.dumps({
        "elapsed_seconds": 90, "eta_seconds": 400, "loss": 0.21, "stage": "step",
        "step": 40, "total": 200, "round": 1, "first_pictures_seconds": 480,
        "seconds_per_sample": 18, "sample_seconds_estimated": False,
    }), encoding="utf-8")
    app.runner.busy = True
    app.layout.sections._index = STEPS.index("train")
    app.repaint()
    qapp.processEvents()
    page = app.window.findChild(QtWidgets.QWidget, "forgeStepPage")
    text = page.findChild(QtWidgets.QLabel, "progressText")
    elapsed = format_span(90)
    eta = format_span(480)
    assert text.text() == f"Шаг 40 из 200 · прошло {elapsed} · первые картинки примерно через {eta} · 18 с на картинку"
    start = app.window.findChild(QtWidgets.QPushButton, "primary")
    assert start is not None and not start.isEnabled()
    assert start.text() == "Начать обучение"
    image = start.grab().toImage()
    color = image.pixelColor(6, image.height() // 2)
    assert color.red() < 160 and color.green() < 160


def test_discard_asks_then_sends_finish_discard(make_app, tmp_path, qapp) -> None:
    app = make_app("ru")
    photos = make_photos(tmp_path / "src", 2)
    _seed_review(app.project, photos)
    app.layout.sections._index = STEPS.index("train")
    app.repaint()
    qapp.processEvents()
    finish = app.window.findChild(QtWidgets.QPushButton, "forge.review.finish")
    discard = next(item for item in finish.menu().actions() if item.objectName() == "forge.review.finish.discard")

    def cancel() -> None:
        box = QtWidgets.QApplication.activeModalWidget()
        assert isinstance(box, QtWidgets.QDialog)
        assert box.text() == tr("forge.app.review.finish.discard.confirm")
        button = next(item for item in box.buttons() if item.text() == tr("forge.app.review.unmarked.cancel"))
        button.click()

    QtCore.QTimer.singleShot(0, cancel)
    discard.trigger()
    qapp.processEvents()
    assert app.runner.calls == []

    def confirm() -> None:
        box = QtWidgets.QApplication.activeModalWidget()
        assert isinstance(box, QtWidgets.QDialog)
        assert box.windowTitle()
        title = box.findChild(QtWidgets.QLabel, "forgeDialogTitle")
        assert title is not None and title.text() == box.windowTitle()
        card = box.findChild(QtWidgets.QFrame, "forgeDialogCard")
        assert card is not None
        edge = card.grab().toImage().pixelColor(0, card.height() // 2)
        fill = card.grab().toImage().pixelColor(card.width() // 2, card.height() // 2)
        assert edge.lightness() > fill.lightness() + 40
        shadow_at = card.mapTo(box, QtCore.QPoint(card.width() // 2, card.height() + 6))
        shadow = box.grab().toImage().pixelColor(shadow_at)
        assert shadow.alpha() > 40
        button = next(item for item in box.buttons() if item.text() == tr("forge.app.review.finish.discard"))
        assert button.property("danger") == "true"
        face = button.grab().toImage().pixelColor(10, button.height() // 2)
        assert face.red() > face.green() + 20 and face.red() > face.blue()
        button.click()

    QtCore.QTimer.singleShot(0, confirm)
    discard.trigger()
    qapp.processEvents()
    assert app.runner.calls[-1] == ["--json", "lora", "finish", str(app.project), "--discard", "--yes"]
