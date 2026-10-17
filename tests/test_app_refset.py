"""The "Build a set" step: references, rounds, Keep/Reject, This is it, layout at 1024x768 (offscreen, stubs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6 import QtCore, QtWidgets  # noqa: E402
from voxprint_common.i18n import set_language, tr  # noqa: E402

from tests.forge_app_stub import ALL_COMMANDS, StubRunner  # noqa: E402
from tests.test_app_window import CONTROLS, _next, _scroll_problems, _status  # noqa: E402
from tests.test_lora_refset import SUBJECT, Loaders, _refs  # noqa: E402
from voxprint_forge.app import engine  # noqa: E402
from voxprint_forge.app.engine import Outcome  # noqa: E402
from voxprint_forge.lora import refset  # noqa: E402
from voxprint_forge.lora.errors import LoraBlocked  # noqa: E402
from voxprint_forge.lora.events import Progress  # noqa: E402


@pytest.fixture
def make_app(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
    from voxprint_forge.app.window import ForgeApp

    made = []

    def build(runner=None, lang="en"):
        set_language(lang)
        app = ForgeApp(runner=runner or StubRunner(), home=tmp_path / "state", commands=ALL_COMMANDS,
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


def _engine(app_ref: list, *, loaders_factory=Loaders) -> StubRunner:
    """A runner whose refset commands run the real loop on stub models."""
    runner = StubRunner()

    def round_(project: Path) -> None:
        args = runner.calls[-1]
        count = int(args[args.index("--candidates") + 1])
        strength = float(args[args.index("--strength") + 1])
        steps = int(args[args.index("--steps") + 1])
        refs = [Path(args[i + 1]) for i, a in enumerate(args) if a == "--ref"]
        loaders = loaders_factory()
        try:
            refset.run_round(project, refs, SUBJECT, refset.check_settings(count, strength, steps),
                             load_gate=loaders.gate, load_generator=loaders.generator,
                             load_embedder=loaders.embedder, progress=Progress(), describe=lambda _i: "a test pattern")
        except LoraBlocked:
            pass  # the scripted outcome says blocked

    def finish(project: Path) -> None:
        args = runner.calls[-1]
        refset.finish(project, Progress(), mode=args[args.index("--mode") + 1])

    runner.effects["refset.round"] = round_
    runner.effects["refset.finish"] = finish
    return runner


def _to_refset(app, refs) -> None:
    app.draft.add_photos(refs)
    app.draft.kind = "person"
    app.repaint()
    app.window.findChild(QtWidgets.QPushButton, "forge.chip.refs").click()
    assert app.draft.source == "refs"
    assert _next(app).isEnabled(), _status(app)
    _next(app).click()
    app.draft.set_names("Shape Demo", "")
    app.draft.consent = app.draft.adult = True
    app.draft.set_verbal(True)
    app.repaint()
    assert _next(app).isEnabled(), _status(app)
    _next(app).click()
    assert app.layout.sections.step.step_id == "refset"


def _buttons(app, prefix):
    QtWidgets.QApplication.processEvents()
    return [b for b in app.window.findChildren(QtWidgets.QPushButton) if b.objectName().startswith(prefix)
            and b.isVisibleTo(app.window)]


def _make(app):
    QtWidgets.QApplication.processEvents()
    return next(b for b in app.window.findChildren(QtWidgets.QPushButton) if b.isVisibleTo(app.window)
                and b.text() in (tr("forge.app.refset.make"), tr("forge.app.refset.next_round")))


def _find(app, kind, name):
    QtWidgets.QApplication.processEvents()
    found = [w for w in app.window.findChildren(kind, name) if w.isVisibleTo(app.window)]
    assert len(found) == 1, (name, len(found))
    return found[0]


def test_photos_step_takes_1_to_10_references(make_app, tmp_path) -> None:
    app = make_app()
    app.draft.kind = "person"
    app.repaint()
    app.window.findChild(QtWidgets.QPushButton, "forge.chip.refs").click()
    assert not _next(app).isEnabled()
    assert _status(app) == tr("forge.app.err.refs_count", low=1, high=10)
    app.draft.add_photos(_refs(tmp_path / "r", 11))
    app.repaint()
    assert not _next(app).isEnabled()
    app.draft.remove_photo(app.draft.photos[-1])
    app.repaint()
    assert _next(app).isEnabled()
    assert tr("forge.app.step.photos.short_refs") in [w.text() for w in app.window.findChildren(QtWidgets.QLabel)]


def test_own_photos_skip_the_step(make_app, tmp_path) -> None:
    from tests.forge_app_stub import make_photos

    app = make_app()
    app.draft.add_photos(make_photos(tmp_path / "p", 12))
    app.draft.kind = "object"
    app.repaint()
    _next(app).click()
    app.draft.set_names("Cube", "")
    app.draft.rights = True
    app.repaint()
    _next(app).click()
    assert app.layout.sections.step.step_id == "refset"
    assert _next(app).isEnabled()
    labels = [w.text() for w in app.window.findChildren(QtWidgets.QLabel)]
    assert tr("forge.app.refset.skip") in labels


def test_round_keep_and_this_is_it_hand_the_set_to_prepare(make_app, tmp_path) -> None:
    refs = _refs(tmp_path / "refs", 3)
    holder: list = []
    app = make_app(_engine(holder))
    _to_refset(app, refs)
    assert not _next(app).isEnabled()
    assert _status(app) == tr("forge.app.err.refset_not_done")
    finish = _find(app, QtWidgets.QPushButton, "forge.refset.finish")
    assert not finish.isEnabled()
    _make(app).click()
    call = app.runner.calls[-1]
    assert call[:5] == ["--json", "lora", "refset", "round", "--project"]
    assert call[call.index("--candidates") + 1] == "24" and call[call.index("--strength") + 1] == "0.50"
    assert "--consent-obtained" in call and "--subject-adult" in call and call.count("--ref") == 3
    keeps = _buttons(app, "forge.refset.keep.")
    rejects = _buttons(app, "forge.refset.reject.")
    assert len(keeps) == 24 and len(rejects) == 24
    for button in keeps[:7]:
        button.click()
    rejects[7].click()
    keeps[7].click()  # Keep after Reject: only one stays ticked
    assert not rejects[7].isChecked()
    state = refset.load_state(app.project)
    assert len(refset.kept(state)) == 8
    label = _find(app, QtWidgets.QLabel, "presetLabel")
    assert label.text() == tr("forge.app.refset.kept", kept=8, total=11, low=10, high=30)
    _find(app, QtWidgets.QPushButton, "forge.refset.finish").click()
    assert app.runner.calls[-1][2:4] == ["refset", "finish"]
    assert len(app.draft.approved) == 11
    assert _status(app) == tr("forge.app.refset.finished", count=11)
    assert _next(app).isEnabled()
    _next(app).click()
    assert app.layout.sections.step.step_id == "prepare"
    args = engine.prepare_args(app.draft, app.project)
    photos = [Path(args[i + 1]) for i, a in enumerate(args) if a == "--photo"]
    assert photos == app.draft.approved and not set(photos) & set(refs)
    metadata = json.loads((refset.folder(app.project) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["base"] == app.simple["base"] and metadata["teacher"]["licence"] == "Apache-2.0"


def test_a_second_round_starts_from_kept_pictures(make_app, tmp_path) -> None:
    app = make_app(_engine([]))
    _to_refset(app, _refs(tmp_path / "refs", 2))
    _make(app).click()
    _buttons(app, "forge.refset.keep.")[0].click()
    _make(app).click()
    assert _make(app).text() == tr("forge.app.refset.next_round")
    state = refset.load_state(app.project)
    assert len(state["rounds"]) == 2
    assert len(_buttons(app, "forge.refset.keep.")) == 24


def test_simple_has_no_round_settings_and_expert_has_them(make_app, tmp_path) -> None:
    app = make_app(_engine([]))
    _to_refset(app, _refs(tmp_path / "refs", 2))
    page = _find(app, QtWidgets.QWidget, "forgeStepPage")
    assert not page.findChildren(QtWidgets.QAbstractSpinBox)
    assert tr("forge.app.refset.simple", count=24) in [w.text() for w in page.findChildren(QtWidgets.QLabel)]
    app.expert.set_enabled(True)
    page = _find(app, QtWidgets.QWidget, "forgeStepPage")
    count = page.findChild(QtWidgets.QSpinBox, "forge.refset.count")
    strength = page.findChild(QtWidgets.QDoubleSpinBox, "forge.refset.strength")
    steps = page.findChild(QtWidgets.QSpinBox, "forge.refset.steps")
    assert (count.minimum(), count.maximum()) == (8, 64)
    assert (strength.minimum(), strength.maximum()) == (0.25, 0.9)
    count.setValue(12)
    strength.setValue(0.4)
    steps.setValue(6)
    assert page.findChild(QtWidgets.QTableWidget, "forge.refset.prompts") is None
    _make(app).click()
    call = app.runner.calls[-1]
    assert call[call.index("--candidates") + 1] == "12"
    assert call[call.index("--strength") + 1] == "0.40" and call[call.index("--steps") + 1] == "6"
    assert len(_buttons(app, "forge.refset.keep.")) == 12
    table = _find(app, QtWidgets.QTableWidget, "forge.refset.prompts")
    assert table is not None and table.rowCount() == 2
    table.item(0, 1).setText("an edited description")
    rid = table.item(0, 1).data(QtCore.Qt.ItemDataRole.UserRole)
    assert refset.load_prompts(app.project)[rid] == "an edited description"


def test_a_blocked_set_cannot_continue(make_app, tmp_path) -> None:
    runner = _engine([], loaders_factory=lambda: Loaders(fail_refs=True))
    runner.script["refset.round"] = ([], Outcome(False, None, "lora.err.refset_blocked", "", 6))
    app = make_app(runner)
    _to_refset(app, _refs(tmp_path / "refs", 2))
    _make(app).click()
    assert _status(app) == tr("lora.err.refset_blocked")
    assert not _make(app).isEnabled()
    assert not _find(app, QtWidgets.QPushButton, "forge.refset.restart").isEnabled()
    assert not _find(app, QtWidgets.QPushButton, "forge.refset.finish").isEnabled()
    assert not _next(app).isEnabled()
    assert not _buttons(app, "forge.refset.keep.")
    app.restart_set()
    assert refset.load_state(app.project)["blocked"]


@pytest.mark.parametrize("lang", ["en", "ru"])
@pytest.mark.parametrize("expert", [False, True])
def test_refset_step_scrolls_above_the_footer_at_1024x768(make_app, tmp_path, lang, expert) -> None:
    app = make_app(_engine([]), lang=lang)
    app.expert.set_enabled(expert)
    app.window.resize(1024, 768)
    _to_refset(app, _refs(tmp_path / "refs", 3))
    problems = _scroll_problems(app, "refset-empty")
    _make(app).click()
    QtWidgets.QApplication.processEvents()
    problems += _scroll_problems(app, "refset-review")
    page = _find(app, QtWidgets.QWidget, "forgeStepPage")
    assert page.isAncestorOf(_find(app, QtWidgets.QPushButton, "forge.refset.finish"))
    grid = _find(app, QtWidgets.QWidget, "forge.refset.grid")
    assert grid is not None and grid.height() >= grid.flow.heightForWidth(grid.width())
    assert app.window.size() == QtCore.QSize(1024, 768)
    assert problems == []


@pytest.mark.parametrize("lang", ["en", "ru"])
@pytest.mark.parametrize("expert", [False, True])
def test_every_refset_control_has_a_tooltip(make_app, tmp_path, lang, expert) -> None:
    app = make_app(_engine([]), lang=lang)
    app.expert.set_enabled(expert)
    _to_refset(app, _refs(tmp_path / "refs", 2))
    _make(app).click()
    missing = []
    for widget in app.window.findChildren(QtWidgets.QWidget):
        if isinstance(widget, CONTROLS) and widget.isVisibleTo(app.window) and widget.isEnabled():
            if isinstance(widget.parent(), QtWidgets.QComboBox):
                continue
            tip = widget.toolTip()
            if not tip or tip.startswith(("lora.", "forge.app.")):
                missing.append((type(widget).__name__, widget.objectName()))
    assert not missing
    for key in ("forge.app.step.refset.short",):
        assert len(tr(key)) <= 60
