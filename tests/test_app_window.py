"""Model Forge window on the Yitro shell: steps, gates, tooltips, help, Expert values (offscreen)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets  # noqa: E402
from voxprint_common.i18n import set_language  # noqa: E402

from tests.forge_app_stub import ALL_COMMANDS, StubRunner, make_photos, write_prepared  # noqa: E402
from voxprint_forge.app.draft import STEPS, Prepared  # noqa: E402
from voxprint_forge.app.engine import EngineEvent, Outcome  # noqa: E402
from voxprint_forge.app.text import tr  # noqa: E402
from voxprint_forge.app.window import _WrappingLabel  # noqa: E402
from voxprint_forge.mdl_app import missing_id_text  # noqa: E402

CONTROLS = (
    QtWidgets.QAbstractButton,
    QtWidgets.QLineEdit,
    QtWidgets.QComboBox,
    QtWidgets.QAbstractSpinBox,
    QtWidgets.QSlider,
    QtWidgets.QAbstractItemView,
)


@pytest.fixture
def make_app(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    from voxprint_forge.app.window import ForgeApp

    made = []

    def build(runner=None, commands=ALL_COMMANDS, lang="en"):
        set_language(lang)
        app = ForgeApp(runner=runner or StubRunner(), home=tmp_path / "state", commands=commands,
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


def _runner(photos, *, blocked=False):
    drops = [(photos[-1], "lora.drop.safety")]
    if blocked:
        prepare = Outcome(False, None, "lora.err.blocked", "blocked", 6)
    else:
        prepare = Outcome(True, {"state": "done", "prepared": len(photos) - 1})
    train_events = [EngineEvent("train", "lora.progress.train", {"step": 5, "steps": 10}, "lora.reason.train",
                                50.0, 120)]
    runner = StubRunner({"prepare": ([], prepare), "train": (train_events, Outcome(True, {"state": "done"}))})
    runner.effects["prepare"] = lambda project: write_prepared(project, [] if blocked else photos[:-1], drops)
    return runner


def _next(app):
    return app.widget._parts()["next"]


def _status(app):
    return app._status_label.text()


def _walk_to(app, step, photos):
    app.draft.add_photos(photos)
    app.draft.kind = "person"
    app.repaint()
    order = list(STEPS)
    for name in order[: order.index(step)]:
        if name == "subject":
            app.draft.set_names("Shape Demo", "")
            app.draft.consent = app.draft.adult = True
            app.draft.set_verbal(True)
            app.repaint()
        if name == "prepare":
            app.prepare()
        if name == "train":
            app.train()
        assert _next(app).isEnabled(), (name, _status(app))
        _next(app).click()
    assert app.layout.sections.step.step_id == step


def test_shell_parts_this_app_extends_exist(make_app) -> None:
    app = make_app()
    for name, part in app.widget._parts().items():
        assert part is not None, name
    assert app.surface.app_id == "model-forge"
    assert hasattr(app.widget, "_centre_split")
    assert callable(getattr(super(type(app.widget), app.widget), "_fit_expert_panel", None))


@pytest.mark.parametrize(("lang", "count", "reset"), [
    ("en", "Step 1 of 6", "Reset to the “Balanced” preset"),
    ("ru", "Шаг 1 из 6", "Сбросить к пресету «Сбалансированная»"),
])
def test_review_wording(make_app, lang, count, reset) -> None:
    make_app()
    set_language(lang)
    try:
        assert tr("wizard.step_count", current=1, total=len(STEPS)) == count
        name = "Balanced" if lang == "en" else "Сбалансированная"
        assert tr("simple.preset.reset", name=name) == reset
    finally:
        set_language("en")


def test_next_is_gated_with_a_plain_reason(make_app, tmp_path) -> None:
    app = make_app()
    assert not _next(app).isEnabled()
    assert _status(app) == tr("lora.err.too_few", need=10)
    app.draft.add_photos(make_photos(tmp_path / "p", 10))
    app.repaint()
    assert _status(app) == tr("forge.app.err.no_kind")
    app.window.findChild(QtWidgets.QPushButton, "forge.chip.object").click()
    assert _next(app).isEnabled()
    _next(app).click()
    assert _status(app) == tr("lora.err.no_name")
    assert app.window.findChild(QtWidgets.QCheckBox, "forge.subject.rights") is not None
    assert app.window.findChild(QtWidgets.QCheckBox, "forge.subject.consent") is None


def _subject_step(app, tmp_path):
    app.draft.add_photos(make_photos(tmp_path / "p", 12))
    app.draft.kind = "person"
    app.repaint()
    _next(app).click()
    app.draft.set_names("Shape Demo", "")
    app.repaint()
    QtWidgets.QApplication.processEvents()  # Qt shows widgets added to a visible window on the next pass


def _box(app, name):
    return app.window.findChild(QtWidgets.QCheckBox, f"forge.subject.{name}")


def test_verbal_sits_by_consent_and_lifts_the_release(make_app, tmp_path) -> None:
    app = make_app()
    _subject_step(app, tmp_path)
    consent, verbal = _box(app, "consent"), _box(app, "verbal")
    assert consent.parent() is verbal.parent() and consent.parent().objectName() == "forge.subject.consent_row"
    consent.click()
    _box(app, "adult").click()
    assert not _next(app).isEnabled() and _status(app) == tr("lora.err.no_release")
    app.draft.release = tmp_path / "release.pdf"
    app.repaint()
    assert _next(app).isEnabled()
    _box(app, "verbal").click()
    QtWidgets.QApplication.processEvents()
    assert app.draft.verbal and app.draft.release is None and _next(app).isEnabled()
    assert not app.window.findChild(QtWidgets.QPushButton, "forge.subject.release").isEnabled()


def test_adult_content_shows_right_of_18_only_when_ticked(make_app, tmp_path) -> None:
    app = make_app()
    _subject_step(app, tmp_path)
    adult, content = _box(app, "adult"), _box(app, "adult_content")
    assert adult.parent() is content.parent() and adult.parent().objectName() == "forge.subject.adult_row"
    assert not content.isVisibleTo(app.window) and not app.draft.adult_content
    adult.click()
    assert content.isVisibleTo(app.window)
    content.click()
    assert app.draft.adult_content
    QtWidgets.QApplication.processEvents()
    assert content.geometry().left() > adult.geometry().right() and content.geometry().top() == adult.geometry().top()
    adult.click()
    assert not content.isVisibleTo(app.window) and not content.isChecked() and not app.draft.adult_content


def test_subject_attr_is_disabled_and_cleared_when_18_is_unticked(make_app, tmp_path) -> None:
    app = make_app()
    _subject_step(app, tmp_path)
    field = app.window.findChild(QtWidgets.QLineEdit, "forge.subject.attr")
    adult = _box(app, "adult")
    assert field is not None and not field.isEnabled() and field.text() == "" and app.draft.attr is None
    adult.click()
    assert field.isEnabled()
    field.setText("42")
    QtWidgets.QApplication.processEvents()
    assert app.draft.attr == 42 and field.text() == "42"
    adult.click()
    QtWidgets.QApplication.processEvents()
    assert not field.isEnabled() and field.text() == "" and app.draft.attr is None
    assert not app.draft.adult_content


@pytest.mark.parametrize("lang", ["en", "ru"])
def test_subject_rows_wrap_without_clipping_in_a_narrow_window(make_app, tmp_path, lang) -> None:
    app = make_app(lang=lang)
    _subject_step(app, tmp_path)
    _box(app, "consent").click()
    _box(app, "verbal").click()
    QtWidgets.QApplication.processEvents()
    _box(app, "adult").click()
    app.window.resize(560, app.window.height())
    QtWidgets.QApplication.processEvents()
    for name in ("consent", "verbal", "adult", "adult_content"):
        box = _box(app, name)
        assert box.isVisibleTo(app.window), name
        assert box.width() >= box.sizeHint().width(), name
        row = box.parentWidget()
        assert row.rect().contains(box.geometry()), name


def test_full_flow_runs_the_cli_commands(make_app, tmp_path) -> None:
    photos = make_photos(tmp_path / "p", 12)
    runner = _runner(photos)
    app = make_app(runner)
    _walk_to(app, "result", photos)
    commands = [call[2] for call in runner.calls]
    assert commands == ["prepare", "train"]
    assert "--consent-obtained" in runner.calls[0] and "--subject-adult" in runner.calls[0]
    assert runner.calls[0][runner.calls[0].index("--consent-method") + 1] == "verbal"
    assert "--release" not in runner.calls[0] and "--adult-content" not in runner.calls[0]
    app.install()
    assert runner.calls[-1][2] == "export" and "--install-to-mdl" in runner.calls[-1]
    assert _status(app) == tr("forge.app.result.installed")


def test_progress_shows_the_engine_sentence(make_app, tmp_path) -> None:
    photos = make_photos(tmp_path / "p", 12)
    runner = _runner(photos)
    runner.hold = False
    app = make_app(runner)
    _walk_to(app, "train", photos)
    runner.hold = True
    app.train()
    text = app._progress_text.text()
    assert tr("lora.progress.train", step=5, steps=10) in text and tr("lora.reason.train") in text
    assert tr("lora.progress.eta", minutes=2) in text
    assert not _next(app).isEnabled()
    app.pause()
    assert (app.project / "PAUSE").is_file()
    runner.release(Outcome(True, {"state": "paused"}))
    assert app.paused
    app.train()
    assert "--resume" in runner.calls[-1] and runner.pause_seen_at_start[-1] is False


def test_a_leftover_pause_file_is_removed_before_start(make_app, tmp_path) -> None:
    photos = make_photos(tmp_path / "p", 12)
    runner = _runner(photos)
    app = make_app(runner)
    _walk_to(app, "train", photos)
    (app.project / "PAUSE").write_text("")
    app.train()
    assert runner.pause_seen_at_start[-1] is False


def test_stop_ends_the_process_and_leaves_no_pause_file(make_app, tmp_path) -> None:
    photos = make_photos(tmp_path / "p", 12)
    runner = _runner(photos)
    app = make_app(runner)
    _walk_to(app, "train", photos)
    runner.hold = True
    app.train()
    app.stop()
    assert runner.stopped == 1 and not (app.project / "PAUSE").exists() and not app.paused


def test_a_safety_block_cannot_be_bypassed(make_app, tmp_path) -> None:
    photos = make_photos(tmp_path / "p", 12)
    app = make_app(_runner(photos, blocked=True))
    _walk_to(app, "prepare", photos)
    app.prepare()
    assert not _next(app).isEnabled()
    assert _status(app) == tr("lora.err.blocked")
    texts = " ".join(button.text() for button in app.window.findChildren(QtWidgets.QAbstractButton))
    assert "anyway" not in texts.lower()
    app.layout.sections.advance()  # even a forced advance cannot start training
    app.repaint()
    assert not app.draft.check("train").ok
    assert app.draft.prepared.blocked and app.draft.prepared.drops[0]["reason"] == "lora.drop.safety"


def test_manage_models_passes_the_manifest(make_app) -> None:
    app = make_app()
    launched: list[list[str]] = []
    app._launch = launched.append
    app.manage_models()
    argv = launched[0]
    assert argv[argv.index("--gui") + 1 : argv.index("--gui") + 3] == ["--app", "model-forge"]
    manifest = argv[argv.index("--manifest") + 1]
    assert manifest.endswith("mdl.json")
    assert "download" not in argv


@pytest.mark.parametrize(
    ("lang", "label"),
    [
        ("en", "Check and repair models"),
        ("ru", "Проверить и восстановить модели"),
    ],
)
def test_check_and_repair_opens_the_verify_window(make_app, qapp, lang, label) -> None:
    app = make_app(lang=lang)
    launched: list[list[str]] = []
    app._launch = launched.append
    app.layout.sections.select("check")
    app.widget.refresh()
    qapp.processEvents()
    button = app.window.findChild(QtWidgets.QPushButton, "forge.app.check.repair")
    assert button.text() == label
    button.click()
    assert launched[0][-3:] == ["models", "verify", "--gui"]


def test_gate_missing_blocks_with_a_plain_sentence(make_app, tmp_path) -> None:
    photos = make_photos(tmp_path / "p", 12)
    runner = StubRunner({"prepare": ([], Outcome(False, None, "lora.err.gate_missing", "missing", 6))})
    app = make_app(runner)
    _walk_to(app, "prepare", photos)
    app.prepare()
    assert app.draft.prepared is None
    assert _status(app) == tr("lora.err.gate_missing", ids=missing_id_text())
    assert not _next(app).isEnabled()


def test_prepare_lists_kept_photos_and_drops(make_app, tmp_path) -> None:
    photos = make_photos(tmp_path / "p", 12)
    app = make_app(_runner(photos))
    _walk_to(app, "train", photos)
    assert len(app.draft.prepared.kept) == 11
    assert app.draft.prepared.drops[0]["reason"] == "lora.drop.safety"


def test_a_command_missing_from_this_build_says_so(make_app, tmp_path) -> None:
    photos = make_photos(tmp_path / "p", 12)
    app = make_app(StubRunner(), commands=frozenset({"train", "extend"}))
    _walk_to(app, "prepare", photos)
    app.prepare()
    assert app.runner.calls == []
    assert _status(app) == tr("forge.app.err.missing_command")


@pytest.mark.parametrize("lang", ["en", "ru"])
@pytest.mark.parametrize("expert", [False, True])
def test_every_control_has_a_tooltip_and_every_step_has_help(make_app, tmp_path, lang, expert) -> None:
    photos = make_photos(tmp_path / "p", 12)
    app = make_app(_runner(photos), lang=lang)
    app.expert.set_enabled(expert)
    app.draft.add_photos(photos)
    app.draft.kind = "person"
    for step in STEPS:
        for key in (f"lora.step.{step}.title", f"lora.step.{step}.desc", f"lora.step.{step}.help"):
            assert tr(key) != key, key
        if app.layout.sections.step.step_id != step:
            if step == "refset":
                app.draft.set_names("Shape Demo", "")
                app.draft.consent = app.draft.adult = True
                app.draft.set_verbal(True)
            if step == "train":
                app.prepare()
            if step == "result":
                app.train()
            app.repaint()
            _next(app).click()
        app.repaint()
        missing = []
        checked = 0
        for widget in app.window.findChildren(QtWidgets.QWidget):
            if isinstance(widget, CONTROLS) and widget.isVisibleTo(app.window) and widget.isEnabled():
                if isinstance(widget, QtWidgets.QAbstractItemView) and widget.parent() is not None and isinstance(
                    widget.parent(), QtWidgets.QComboBox
                ):
                    continue
                checked += 1
                tip = widget.toolTip()
                if not tip or tip.startswith(("lora.", "forge.app.")):
                    missing.append((step, type(widget).__name__, widget.objectName(), getattr(widget, "text",
                                                                                              lambda: "")()))
        assert not missing, missing
        assert checked >= 6, (step, checked)
        assert app.layout.sections.step.step_id == step


def test_simple_choices_write_live_expert_values(make_app, tmp_path) -> None:
    photos = make_photos(tmp_path / "p", 12)
    app = make_app(_runner(photos))
    app.expert.set_enabled(True)
    _walk_to(app, "train", photos)
    state = app.layout.state
    app.window.findChild(QtWidgets.QPushButton, "forge.chip.quick").click()
    assert state.get("lora.rank") == 16
    app.window.findChild(QtWidgets.QPushButton, "forge.chip.24").click()
    app.window.findChild(QtWidgets.QPushButton, "forge.chip.close").click()
    assert state.get("lora.steps") == 195
    assert not app.custom()
    app.layout.set_value("lora.rank", 64)
    assert app.custom()
    app.repaint()
    assert app.window.findChild(QtWidgets.QPushButton, "forge.app.preset.reset") is not None
    app.apply_simple()
    assert state.get("lora.rank") == 16 and not app.custom()
    app.train()
    sent = app.runner.calls[-1]
    assert sent[sent.index("--rank") + 1] == "16" and sent[sent.index("--preset") + 1] == "quick"


def test_entry_point_builds_the_window(qapp, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    from voxprint_forge.app import __main__ as entry

    monkeypatch.setattr(QtWidgets.QApplication, "exec", lambda _self: 0)
    assert entry.main([]) == 0


@pytest.mark.parametrize("lang", ["en", "ru"])
def test_open_guide_opens_the_offline_page_in_the_app_language(make_app, lang) -> None:
    app = make_app(lang=lang)
    opened = []
    app._open_path = opened.append
    app.open_guide()
    assert opened and opened[0].name == f"{lang}.html" and opened[0].is_file()
    assert 'src="http' not in opened[0].read_text(encoding="utf-8")


@pytest.mark.parametrize("lang", ["en", "ru"])
@pytest.mark.parametrize("expert", [False, True])
def test_base_picker_is_short_with_a_hint_and_only_trainable_bases(make_app, tmp_path, lang, expert) -> None:
    from voxprint_forge.app import basepicker
    from voxprint_forge.lora import bases

    photos = make_photos(tmp_path / "p", 12)
    app = make_app(_runner(photos), lang=lang)
    app.expert.set_enabled(expert)
    _walk_to(app, "train", photos)
    QtWidgets.QApplication.processEvents()
    combo = app.window.findChild(QtWidgets.QComboBox, "forge.train.base")
    ids = [combo.itemData(i) for i in range(combo.count())]
    assert ids == [info.mdl_id for info in bases.trainable()]
    assert ids[0] == "z-image" and "moss-image-base" in ids
    assert combo.itemText(0) == "Z-Image" and combo.width() <= basepicker.MAX_WIDTH
    moss = combo.itemData(ids.index("moss-image-base"), basepicker.DETAIL_ROLE)
    assert tr("forge.app.base.moss-image-base.note") in moss
    detail = combo.itemData(0, basepicker.DETAIL_ROLE)
    assert detail == f"{tr('forge.app.base.z-image.note')} · {tr('forge.app.base.vram_tier', gb=16)}"
    tip = combo.itemData(0, QtCore.Qt.ItemDataRole.ToolTipRole)
    assert "16" in detail and tip.endswith(detail)  # no video memory line until it is measured
    hint = app.window.findChild(QtWidgets.QLabel, "forge.train.base.hint")
    assert hint is not None and hint.text() == tr("forge.app.base.hint") and hint.isVisibleTo(app.window)


def test_base_vram_uses_measured_peaks_only_when_present(monkeypatch) -> None:
    from voxprint_forge.app import basepicker

    # Z-Image has no measured run yet: the preset tier, no memory sentence in its tooltip.
    assert basepicker.vram_text("z-image") == tr("forge.app.base.vram_tier", gb=16)
    assert basepicker.tooltip("z-image", "Z-Image", "repo", "Apache-2.0").endswith(basepicker.detail_line("z-image"))
    # MOSS-Image Base: moss-memprobe v2 measured 13.6 GiB (16 GB tier) and 18.5 GiB (24 GB tier) reserved.
    assert basepicker.MEASURED_PEAK_GB["moss-image-base"] == {16: 13.6, 24: 18.5}
    moss = tr("forge.app.base.vram_measured", gb="13.6")
    assert basepicker.vram_text("moss-image-base") == moss
    assert basepicker.memory_details() == tr("forge.app.base.vram_measured.tip", vram=f"MOSS-Image Base {moss}")
    monkeypatch.setattr(basepicker, "MEASURED_PEAK_GB", {})
    assert basepicker.memory_details() is None
    monkeypatch.setitem(basepicker.MEASURED_PEAK_GB, "z-image", {16: 14.5, 24: 19.0})
    measured = tr("forge.app.base.vram_measured", gb="14.5")
    assert basepicker.vram_text("z-image") == measured
    assert basepicker.tooltip("z-image", "Z-Image", "repo", "Apache-2.0").endswith(
        tr("forge.app.base.vram_measured.tip", vram=measured))


@pytest.mark.parametrize("lang", ["en", "ru"])
def test_base_details_name_only_measured_memory(make_app, tmp_path, lang) -> None:
    from voxprint_forge.app import basepicker
    from voxprint_forge.app.details import Details

    app = _train(make_app, tmp_path, lang, False)
    box = app.window.findChild(Details, "forge.details.train.base")
    texts = [label.text() for label in box.body.findChildren(QtWidgets.QLabel)]
    assert texts == [tr("forge.app.base.details"), basepicker.memory_details()]
    assert "MOSS-Image Base" in texts[1] and "Z-Image" not in texts[1]
    assert all(word not in text for text in texts for word in ("Not measured", "Ещё не измерено"))


def _train(make_app, tmp_path, lang, expert):
    photos = make_photos(tmp_path / "p", 12)
    app = make_app(_runner(photos), lang=lang)
    app.expert.set_enabled(expert)
    _walk_to(app, "train", photos)
    QtWidgets.QApplication.processEvents()
    return app


def _labels_fit(page):
    """Every visible label is inside the page width and tall enough for its wrapped text."""
    bad = []
    for label in page.findChildren(QtWidgets.QLabel):
        if not label.isVisibleTo(page) or not label.text():
            continue
        corner = label.mapTo(page, QtCore.QPoint(label.width(), label.height()))
        if corner.x() > page.width() + 1:
            bad.append(("wide", label.text()[:40]))
        if label.wordWrap() and label.height() + 1 < label.heightForWidth(label.width()):
            bad.append(("short", label.text()[:40]))
    return bad


@pytest.mark.parametrize("lang", ["en", "ru"])
@pytest.mark.parametrize("expert", [False, True])
def test_train_step_text_is_never_cut_with_details_closed_or_open(make_app, tmp_path, lang, expert) -> None:
    from voxprint_forge.app.details import Details

    app = _train(make_app, tmp_path, lang, expert)
    page = app.window.findChild(QtWidgets.QWidget, "forgeStepPage")
    boxes = app.window.findChildren(Details)
    assert {box.objectName() for box in boxes} == {"forge.details.step.train", "forge.details.train.base"}
    assert all(not box.opened and not box.body.isVisibleTo(app.window) for box in boxes)
    assert _labels_fit(page) == []
    for box in boxes:
        box.button.click()
    QtWidgets.QApplication.processEvents()
    QtWidgets.QApplication.processEvents()
    assert all(box.body.isVisibleTo(app.window) for box in boxes)
    assert _labels_fit(page) == []
    app.repaint()
    QtWidgets.QApplication.processEvents()
    assert all(box.opened for box in app.window.findChildren(Details))  # survives a repaint


@pytest.mark.parametrize("lang", ["en", "ru"])
@pytest.mark.parametrize("expert", [False, True])
def test_base_list_opens_below_the_field(make_app, tmp_path, lang, expert) -> None:
    app = _train(make_app, tmp_path, lang, expert)
    combo = app.window.findChild(QtWidgets.QComboBox, "forge.train.base")
    combo.showPopup()
    QtWidgets.QApplication.processEvents()
    popup = combo.view().window()
    field = combo.mapToGlobal(QtCore.QPoint(0, combo.height()))
    top_left = popup.mapToGlobal(QtCore.QPoint(0, 0))
    combo.hidePopup()
    assert top_left.y() >= field.y() and top_left.x() == field.x()


def test_every_step_starts_with_a_short_line_and_closed_details(make_app, tmp_path) -> None:
    from voxprint_forge.app.details import Details

    photos = make_photos(tmp_path / "p", 12)
    app = make_app(_runner(photos))
    app.draft.add_photos(photos)
    app.draft.kind = "person"
    app.repaint()
    for step in STEPS:
        if app.layout.sections.step.step_id != step:
            if step == "refset":
                app.draft.set_names("Shape Demo", "")
                app.draft.consent = app.draft.adult = True
                app.draft.set_verbal(True)
            if step == "train":
                app.prepare()
            if step == "result":
                app.train()
            app.repaint()
            _next(app).click()
        QtWidgets.QApplication.processEvents()
        intro = app.window.findChild(Details, f"forge.details.step.{step}")
        if step == "photos":
            assert intro is None  # the explanation sits in the empty drop zone
        else:
            assert intro is not None and not intro.opened, step
        assert len(tr(f"forge.app.step.{step}.short")) <= 60, step


def _scroll_problems(app, step):
    """Open every Details on the step at 1024x768 and scroll to each control: it must show fully inside
    the scroll viewport, which ends above the pinned footer (status line, Back/Next)."""
    from voxprint_forge.app.details import Details

    for box in app.window.findChildren(Details):
        if not box.opened:
            box.button.click()
    for _ in range(3):
        QtWidgets.QApplication.processEvents()
    page = app.window.findChild(QtWidgets.QWidget, "forgeStepPage")
    scroll = page.parentWidget().parentWidget()
    assert isinstance(scroll, QtWidgets.QScrollArea)
    viewport = scroll.viewport()
    view_top = viewport.mapToGlobal(QtCore.QPoint(0, 0)).y()
    view_bottom = view_top + viewport.height()
    footer_top = app._status_label.mapToGlobal(QtCore.QPoint(0, 0)).y()
    next_top = _next(app).mapToGlobal(QtCore.QPoint(0, 0)).y()
    bad = []
    if not view_bottom <= footer_top <= next_top:
        bad.append(("footer overlaps the body", view_bottom, footer_top, next_top))
    layout = page.layout()
    need = layout.totalHeightForWidth(page.width()) if layout.hasHeightForWidth() else layout.totalSizeHint().height()
    if page.height() + 1 < need:
        bad.append(("page squeezed", page.height(), need))
    controls = [w for kind in CONTROLS for w in page.findChildren(kind) if w.isVisibleTo(page)]
    disclosures = [c for c in controls if c.objectName() == "disclosure"]
    if step == "photos":
        assert not disclosures  # no Details button; the note is the empty-state hint
    else:
        assert disclosures
    bar = scroll.verticalScrollBar()
    if bar.maximum() < page.height() - viewport.height():
        bad.append(("scroll range too short", bar.maximum(), page.height() - viewport.height()))
    for control in controls:
        # Scroll the bar itself: ensureWidgetVisible only reveals a line edit's cursor, not the whole field.
        y = control.mapTo(page, QtCore.QPoint(0, 0)).y()
        bar.setValue(y if control.height() > viewport.height() else max(0, y + control.height() - viewport.height()))
        QtWidgets.QApplication.processEvents()
        top = control.mapToGlobal(QtCore.QPoint(0, 0)).y()
        if top < view_top or top + min(control.height(), viewport.height()) > view_bottom + 1:
            bad.append((step, type(control).__name__, control.objectName() or getattr(control, "text", str)()))
    return bad


@pytest.mark.parametrize("lang", ["en", "ru"])
@pytest.mark.parametrize("expert", [False, True])
def test_every_control_scrolls_into_view_above_the_footer_with_all_details_open(
        make_app, tmp_path, lang, expert) -> None:
    photos = make_photos(tmp_path / "p", 12)
    app = make_app(_runner(photos), lang=lang)
    app.expert.set_enabled(expert)
    app.window.resize(1024, 768)
    app.draft.add_photos(photos)
    app.draft.kind = "person"
    app.repaint()
    problems = []
    for step in STEPS:
        if app.layout.sections.step.step_id != step:
            if step == "refset":
                app.draft.set_names("Shape Demo", "")
                app.draft.consent = app.draft.adult = True
                app.draft.set_verbal(True)
            if step == "train":
                app.prepare()
            if step == "result":
                app.train()
            app.repaint()
            _next(app).click()
        QtWidgets.QApplication.processEvents()
        if step == "train":
            from voxprint_forge.app.details import Details

            names = {box.objectName() for box in app.window.findChildren(Details)}
            assert names == {"forge.details.step.train", "forge.details.train.base"}
            start = next(b for b in app.window.findChildren(QtWidgets.QPushButton)
                         if b.text() == tr("forge.app.train.start"))
            page = app.window.findChild(QtWidgets.QWidget, "forgeStepPage")
            assert page.isAncestorOf(start)  # the Start/Pause row is part of the scrolling body
        problems += _scroll_problems(app, step)
    assert app.window.size() == QtCore.QSize(1024, 768)
    assert problems == []


def test_a_dictionary_trigger_warns_in_the_footer_and_next_stays_on(make_app, tmp_path) -> None:
    app = make_app()
    _subject_step(app, tmp_path)
    _box(app, "consent").click()
    _box(app, "adult").click()
    _box(app, "verbal").click()
    app.draft.trigger = "synthetic"
    app.draft.trigger_edited = True
    app.repaint()
    QtWidgets.QApplication.processEvents()
    assert _next(app).isEnabled()
    assert _status(app) == tr("lora.warn.trigger_word")
    label = app._status_label
    assert label.wordWrap()
    assert label.height() + 1 >= label.heightForWidth(max(label.width(), 1))


def test_wrapping_label_grows_after_a_wide_measure(qapp) -> None:
    """A floor taken at a wide size hint must not keep the label short once the column narrows."""
    text = "\n".join(
        [
            "3 of 9 photos were set aside. 6 can be used.",
            "grid-00.png: This file is a collage or a grid of several pictures. Use one photo per file.",
            "grid-01.png: This file is a collage or a grid of several pictures. Use one photo per file.",
            "grid-02.png: This file is a collage or a grid of several pictures. Use one photo per file.",
            "Use one photo per file.",
        ]
    )
    label = _WrappingLabel(text, "forgePhotoRefusal")
    label.setMinimumHeight(label.heightForWidth(1600))
    host = QtWidgets.QWidget()
    column = QtWidgets.QVBoxLayout(host)
    column.addWidget(label)
    column.addStretch(1)
    host.resize(420, 700)
    host.show()
    qapp.processEvents()
    assert label.width() < 500
    assert label.height() + 1 >= label.heightForWidth(label.width())


def test_simple_refusal_names_each_photo_and_offers_expert_when_it_would_accept(make_app, tmp_path) -> None:
    from tests.test_lora_refusal import _collage, _plain

    app = make_app()
    app.draft.add_photos([_plain(tmp_path / "ok", index) for index in range(6)])
    app.draft.add_photos([_collage(tmp_path / "ok", index) for index in range(3)])
    app.draft.kind = "object"
    app.repaint()
    QtWidgets.QApplication.processEvents()
    frame = app.window.findChild(QtWidgets.QLabel, "forgePhotoRefusal")
    assert frame is not None and frame.isVisibleTo(app.window)
    assert _status(app) == tr("lora.refuse.line", dropped=3, total=9, kept=6)
    assert not _next(app).isEnabled()
    assert tr("lora.drop.collage") in frame.text()
    assert tr("lora.refuse.fix.collage") in frame.text()
    button = app.window.findChild(QtWidgets.QPushButton, "forgePhotoExpert")
    assert button is not None and button.text() == tr("forge.app.refuse.expert")
    page = app.window.findChild(QtWidgets.QWidget, "forgeStepPage")
    assert _labels_fit(page) == []
    app.window.resize(640, 720)
    QtWidgets.QApplication.processEvents()
    page = app.window.findChild(QtWidgets.QWidget, "forgeStepPage")
    assert _labels_fit(page) == []
    assert "forgePhotoRefusal" in app.window.styleSheet()
    assert "#2a2a33" in app.window.styleSheet() and "#f4f1ea" in app.window.styleSheet()
    button.click()
    QtWidgets.QApplication.processEvents()
    assert app.draft.expert
    assert app.window.findChild(QtWidgets.QLabel, "forgePhotoRefusal") is None
    app.expert.set_enabled(False)

    grids = make_app()
    grids.draft.add_photos([_collage(tmp_path / "grids", index) for index in range(8)])
    grids.draft.add_photos([_plain(tmp_path / "grids", 0)])
    grids.repaint()
    QtWidgets.QApplication.processEvents()
    named = grids.window.findChild(QtWidgets.QLabel, "forgePhotoRefusal")
    assert named is not None
    assert grids.window.findChild(QtWidgets.QPushButton, "forgePhotoExpert") is None
    assert tr("lora.drop.collage") in named.text() and tr("lora.refuse.fix.collage") in named.text()
    assert "Expert" not in named.text()

    prepared = make_app()
    kept_photos = [_plain(tmp_path / "prep", index) for index in range(12)]
    _walk_to(prepared, "prepare", kept_photos)
    prepared.draft.prepared = Prepared(
        [{"id": str(index), "file": str(kept_photos[index]), "short": "a mug"} for index in range(6)],
        [{"name": f"grid-{index}.png", "reason": "lora.drop.collage"} for index in range(3)],
        False,
    )
    prepared.repaint()
    QtWidgets.QApplication.processEvents()
    way = prepared.window.findChild(QtWidgets.QLabel, "forgePrepareWayOut")
    assert way is not None and prepared.window.findChild(QtWidgets.QPushButton, "forgePrepareExpert") is not None
    reason = prepared.window.findChild(QtWidgets.QLabel, "forgePrepareReason")
    assert reason is not None and tr("lora.drop.collage") in reason.text()
    prepare_page = prepared.window.findChild(QtWidgets.QWidget, "forgeStepPage")
    assert _labels_fit(prepare_page) == []

    blocked = make_app()
    blocked.draft.prepared = Prepared(
        [],
        [{"name": "a.png", "reason": "lora.drop.safety"}],
        True,
    )
    blocked.draft.expert = False
    _walk_to(blocked, "prepare", [_plain(tmp_path / "safe", index) for index in range(10)])
    blocked.draft.prepared = Prepared([], [{"name": "a.png", "reason": "lora.drop.safety"}], True)
    blocked.repaint()
    QtWidgets.QApplication.processEvents()
    assert blocked.window.findChild(QtWidgets.QPushButton, "forgePrepareExpert") is None
    assert blocked.window.findChild(QtWidgets.QLabel, "forgePrepareWayOut") is None
    assert _status(blocked) == tr("lora.err.blocked")


def test_collage_drops_show_a_wrapping_notice(make_app, tmp_path) -> None:
    from voxprint_forge.app.draft import Prepared

    photos = make_photos(tmp_path / "p", 12)
    app = make_app(_runner(photos))
    _walk_to(app, "prepare", photos)
    app.draft.prepared = Prepared(
        [{"id": str(index), "file": str(photos[index]), "short": "a woman"} for index in range(10)],
        [{"name": "grid.png", "file": "grid.png", "reason": "lora.drop.collage"}] * 8,
    )
    app.repaint()
    QtWidgets.QApplication.processEvents()
    notice = app.window.findChild(QtWidgets.QLabel, "forgePrepareCollage")
    assert notice is not None and notice.isVisibleTo(app.window)
    assert notice.wordWrap() and notice.text() == tr("forge.app.prepare.collage", count=8)
    page = app.window.findChild(QtWidgets.QWidget, "forgeStepPage")
    assert _labels_fit(page) == []
    assert "forgePrepareCollage" in app.window.styleSheet()
    assert "#2a2a33" in app.window.styleSheet() and "#f4f1ea" in app.window.styleSheet()
