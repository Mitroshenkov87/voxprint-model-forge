"""Model Forge window on the Yitro shell: steps, gates, tooltips, help, Expert values (offscreen)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtWidgets  # noqa: E402
from voxprint_common.i18n import set_language, tr  # noqa: E402

from tests.forge_app_stub import ALL_COMMANDS, StubRunner, make_photos, write_prepared  # noqa: E402
from voxprint_forge.app.draft import STEPS  # noqa: E402
from voxprint_forge.app.engine import EngineEvent, Outcome  # noqa: E402

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
    ("en", "Step 1 of 5", "Reset to the “Balanced” preset"),
    ("ru", "Шаг 1 из 5", "Сбросить к пресету «Сбалансированно»"),
])
def test_review_wording(make_app, lang, count, reset) -> None:
    make_app()
    set_language(lang)
    try:
        assert tr("wizard.step_count", current=1, total=5) == count
        name = "Balanced" if lang == "en" else "Сбалансированно"
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


def test_gate_missing_blocks_with_a_plain_sentence(make_app, tmp_path) -> None:
    photos = make_photos(tmp_path / "p", 12)
    runner = StubRunner({"prepare": ([], Outcome(False, None, "lora.err.gate_missing", "missing", 6))})
    app = make_app(runner)
    _walk_to(app, "prepare", photos)
    app.prepare()
    assert app.draft.prepared is None
    assert _status(app) == tr("lora.err.gate_missing")
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
            if step == "prepare":
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
    assert state.get("lora.steps") == 780
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
