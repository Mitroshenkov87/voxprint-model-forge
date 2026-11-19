"""Command-line training runs in the app: folder classification, discovery and GUI follow.

The pure helper (:mod:`voxprint_forge.app.cli_runs`) reads project folders only; the GUI
tests open a fake ``forge-train`` project and watch it finish without a real training job.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from voxprint_forge import paths
from voxprint_forge.app.cli_runs import default_roots, find_cli_projects, run_state
from voxprint_forge.app.text import tr
from voxprint_forge.lora.progress_log import write_row
from voxprint_forge.lora.project import Project

try:
    from PySide6 import QtWidgets
    from voxprint_common.i18n import set_language

    from tests.forge_app_stub import ALL_COMMANDS, StubRunner, fake_image_checks, make_photos, write_prepared
except ImportError:  # pragma: no cover - the GUI tests below are skipped without PySide6
    set_language = None
    QtWidgets = None

requires_qt = pytest.mark.skipif(QtWidgets is None, reason="PySide6 is not installed")

NOW = 1_800_000_000.0


def _touch_train(folder: Path, stage: str, *, mtime: float | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    write_row(folder, {"step": 5, "total": 10, "loss": 0.2, "elapsed_seconds": 300,
                       "eta_seconds": 300, "stage": stage})
    if mtime is not None:
        os.utime(folder / "train.json", (mtime, mtime))


def test_run_state_is_none_without_training_files(tmp_path: Path) -> None:
    assert run_state(tmp_path, now=NOW) == "none"
    (tmp_path / "project.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "captions.jsonl").write_text("\n", encoding="utf-8")
    assert run_state(tmp_path, now=NOW) == "none"


def test_run_state_running_stopped_done_from_train_json(tmp_path: Path) -> None:
    _touch_train(tmp_path, "step", mtime=NOW - 30)
    assert run_state(tmp_path, now=NOW) == "running"
    _touch_train(tmp_path, "step", mtime=NOW - 601)
    assert run_state(tmp_path, now=NOW) == "stopped"
    _touch_train(tmp_path, "failed", mtime=NOW - 10)
    assert run_state(tmp_path, now=NOW) == "stopped"
    _touch_train(tmp_path, "done", mtime=NOW - 9_999_999)
    assert run_state(tmp_path, now=NOW) == "done"


def test_run_state_done_ignores_staleness_and_may_use_only_the_final_lora(tmp_path: Path) -> None:
    _touch_train(tmp_path, "step", mtime=NOW - 90_000)
    (tmp_path / "lora.safetensors").write_bytes(b"final")
    assert run_state(tmp_path, now=NOW) == "done"
    other = tmp_path / "other"
    other.mkdir()
    (other / "lora.safetensors").write_bytes(b"final")
    assert run_state(other, now=NOW) == "done"


def test_run_state_stale_seconds_is_configurable(tmp_path: Path) -> None:
    _touch_train(tmp_path, "step", mtime=NOW - 700)
    assert run_state(tmp_path, now=NOW, stale_seconds=800) == "running"
    assert run_state(tmp_path, now=NOW, stale_seconds=600) == "stopped"


def _write_project(folder: Path, *, train_mtime: float | None = None,
                   project_mtime: float | None = None) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    Project(folder).write({"kind": "object", "subject": {"name": "Cube"}})
    if train_mtime is not None:
        _touch_train(folder, "step", mtime=train_mtime)
    if project_mtime is not None:
        os.utime(folder / "project.json", (project_mtime, project_mtime))


def test_find_cli_projects_orders_by_newest_activity_and_filters_roots(tmp_path: Path) -> None:
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    old = root_a / "old-project"
    new = root_a / "new-project"
    other = root_b / "other-project"
    _write_project(old, project_mtime=NOW - 5_000)
    _write_project(new, train_mtime=NOW - 10, project_mtime=NOW - 20)
    _write_project(other, project_mtime=NOW - 100)
    (root_a / "not-a-project").mkdir()  # no project.json: skipped
    found = find_cli_projects([root_a, root_b, tmp_path / "missing"], now=NOW)
    assert found == [new, other, old]


def test_default_roots_lists_only_existing_roots(tmp_path: Path, monkeypatch) -> None:
    from voxprint_forge.app import cli_runs

    home = tmp_path / "home"
    (home / "lora").mkdir(parents=True)
    projects = tmp_path / "profile" / "VOXPrint" / "projects"
    projects.mkdir(parents=True)
    # Pin the discovery roots in cli_runs itself: the real Documents projects folder
    # exists on this machine and is resolved outside Path.home (the xdg documents
    # dir), so a Path.home monkeypatch alone leaks it into the result.
    monkeypatch.setattr(cli_runs, "default_projects_dir", lambda: tmp_path / "missing" / "documents-projects")
    monkeypatch.setattr(cli_runs, "legacy_cli_projects_dir", lambda: projects)
    assert default_roots(home) == [home / "lora", projects]
    (home / "lora").rmdir()
    assert default_roots(home) == [projects]


def test_default_roots_includes_the_documents_default_and_the_legacy_root(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("XDG_DOCUMENTS_DIR", raising=False)
    home = tmp_path / "home"
    (home / "lora").mkdir(parents=True)
    profile = tmp_path / "profile"
    documents_projects = profile / "Documents" / "VOXPrint Projects" / "Model Forge"
    documents_projects.mkdir(parents=True)
    legacy = profile / "VOXPrint" / "projects"
    legacy.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: profile))
    monkeypatch.setattr(paths, "_xdg_user_dir_documents", lambda: None)
    monkeypatch.setattr(paths, "_windows_documents_dir", lambda: None)
    assert default_roots(home) == [home / "lora", documents_projects, legacy]


def test_an_explicit_project_path_outside_every_default_root_is_still_found(tmp_path: Path) -> None:
    """Projects passed by an explicit path keep working: discovery never forces a relocation."""
    explicit_root = tmp_path / "custom" / "projects"
    folder = explicit_root / "cube-20261114-1900"
    _write_project(folder, project_mtime=NOW - 50)
    assert find_cli_projects([explicit_root], now=NOW) == [folder]


if QtWidgets is not None:

    @pytest.fixture
    def make_app(qapp, tmp_path, monkeypatch):
        monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
        from voxprint_forge.app.window import ForgeApp

        made = []

        def build(runner=None, projects_root=None):
            set_language("en")
            app = ForgeApp(
                runner=runner or StubRunner(),
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


def _write_cli_project(folder: Path, photos: list[Path], *, stage: str = "step", step: int = 5,
                       total: int = 10) -> None:
    """A folder ``forge-train`` leaves behind: subject, prepared photos and a train.json row."""
    Project(folder).write({
        "kind": "object",
        "gate_passed": True,
        "subject": {"name": "CLI Cube", "trigger": "clicube", "rights_confirmed": True, "kind": "object"},
        "train": {"step": step, "steps_planned": total},
    })
    write_prepared(folder, photos, [])
    write_row(folder, {"step": step, "total": total, "loss": 0.2, "elapsed_seconds": 300,
                       "eta_seconds": 300, "stage": stage})


@requires_qt
def test_opening_a_running_cli_project_follows_it_to_the_result_step(make_app, tmp_path, qapp) -> None:
    app = make_app()
    folder = tmp_path / "projects" / "cube-20261114-2203"
    photos = make_photos(folder / "photos", 12)
    _write_cli_project(folder, photos)
    app.open_project(folder)
    qapp.processEvents()
    assert app.project == folder
    assert app.layout.sections.step.step_id == "train"
    page = app.window.findChild(QtWidgets.QWidget, "forgeStepPage")
    text = page.findChild(QtWidgets.QLabel, "progressText")
    bar = page.findChild(QtWidgets.QProgressBar, "progress")
    assert text.text() == tr("lora.progress.status", step=5, steps=10,
                             elapsed=tr("lora.progress.span.minutes", count=5),
                             eta=tr("lora.progress.span.minutes", count=5))
    assert "Step 5 of 10" in text.text()
    assert bar.isVisible() and bar.value() == 500
    assert app.draft.prepared is not None and len(app.draft.prepared.kept) == 12

    write_row(folder, {"step": 10, "total": 10, "loss": 0.1, "elapsed_seconds": 600,
                       "eta_seconds": 0, "stage": "done"})
    checkpoints = Project(folder).checkpoints
    checkpoints.mkdir(parents=True, exist_ok=True)
    (checkpoints / "step-000005.safetensors").write_bytes(b"checkpoint")
    (folder / "lora.safetensors").write_bytes(b"final")
    app._read_cli_progress()  # one watcher tick, straight from the test
    qapp.processEvents()
    assert app.layout.sections.step.step_id == "result"
    assert app.draft.trained
    pick = app.window.findChild(QtWidgets.QComboBox, "forge.result.checkpoint")
    assert [pick.itemText(i) for i in range(pick.count())] == [
        tr("forge.app.result.final"), tr("forge.app.result.step", step=5),
    ]


def _button(app, name):
    return app.window.findChild(QtWidgets.QPushButton, name)


@requires_qt
def test_a_followed_cli_run_draws_the_running_train_page_and_locks_it(make_app, tmp_path, qapp) -> None:
    """A terminal run shows the same progress block as a GUI run; Start stays off."""
    from voxprint_forge.lora.progress_log import VRAM_FILE

    app = make_app()
    folder = tmp_path / "projects" / "cube-20261114-2300"
    photos = make_photos(folder / "photos", 12)
    _write_cli_project(folder, photos, stage="train", step=400, total=1600)
    (folder / VRAM_FILE).write_text(
        "elapsed_seconds,allocated_gb,reserved_gb\n600.000,8.375,9.125\n1200.000,9.500,10.250\n",
        encoding="utf-8",
    )
    app.open_project(folder)
    qapp.processEvents()
    assert app._followed_cli_run()

    page = app.window.findChild(QtWidgets.QWidget, "forgeStepPage")
    text = page.findChild(QtWidgets.QLabel, "progressText")
    bar = page.findChild(QtWidgets.QProgressBar, "progress")
    percent = page.findChild(QtWidgets.QLabel, "progressPercent")
    assert text.text() == tr("lora.progress.status", step=400, steps=1600,
                             elapsed=tr("lora.progress.span.minutes", count=5),
                             eta=tr("lora.progress.span.minutes", count=5))
    assert "Step 400 of 1600" in text.text()
    assert bar.isVisible() and bar.value() == 250
    assert percent.isVisible() and percent.text() == "25%"
    titles = [label.text() for label in page.findChildren(QtWidgets.QLabel, "panel")]
    assert tr("forge.app.train.running") in titles
    loss = page.findChild(QtWidgets.QLabel, "forge.train.loss")
    assert loss is not None and loss.text() == tr("lora.progress.loss", loss="0.2000")
    vram = page.findChild(QtWidgets.QLabel, "forge.train.vram")
    assert vram is not None and vram.text() == tr("forge.app.train.vram", allocated="9.5", reserved="10.2")
    notice = page.findChild(QtWidgets.QLabel, "forge.train.cli_run")
    assert notice is not None and notice.text() == tr("forge.app.train.cli_run")

    start = next(b for b in page.findChildren(QtWidgets.QPushButton) if b.text() == tr("forge.app.train.start"))
    assert not start.isEnabled()
    assert not _button(app, "forge.app.train.estimate").isEnabled()
    assert _button(app, "forge.app.train.stop").isEnabled()
    assert not _button(app, "forge.app.train.pause").isEnabled()
    base = page.findChild(QtWidgets.QComboBox, "forge.base.picker") or page.findChildren(QtWidgets.QComboBox)[-1]
    assert not base.isEnabled()
    previews = page.findChild(QtWidgets.QAbstractButton, "forge.train.previews")
    assert previews is not None and not previews.isEnabled()

    # live: the next watcher tick repaints from the new train.json
    write_row(folder, {"step": 800, "total": 1600, "loss": 0.15, "elapsed_seconds": 2400,
                       "eta_seconds": 2400, "stage": "train"})
    (folder / VRAM_FILE).write_text(
        "elapsed_seconds,allocated_gb,reserved_gb\n2400.000,11.000,11.500\n", encoding="utf-8",
    )
    app._read_cli_progress()
    qapp.processEvents()
    text = page.findChild(QtWidgets.QLabel, "progressText")
    assert "Step 800 of 1600" in text.text()
    bar = page.findChild(QtWidgets.QProgressBar, "progress")
    assert bar.value() == 500
    loss = page.findChild(QtWidgets.QLabel, "forge.train.loss")
    assert loss.text() == tr("lora.progress.loss", loss="0.1500")
    vram = page.findChild(QtWidgets.QLabel, "forge.train.vram")
    assert vram.text() == tr("forge.app.train.vram", allocated="11.0", reserved="11.5")


@requires_qt
def test_a_followed_cli_run_shows_stopped_when_train_json_goes_stale(make_app, tmp_path, qapp) -> None:
    app = make_app()
    folder = tmp_path / "projects" / "cube-20261114-2301"
    photos = make_photos(folder / "photos", 12)
    _write_cli_project(folder, photos, stage="train", step=400, total=1600)
    app.open_project(folder)
    qapp.processEvents()
    assert app._followed_cli_run()

    stale = os.path.getmtime(folder / "train.json") - 700
    os.utime(folder / "train.json", (stale, stale))
    app._read_cli_progress()
    qapp.processEvents()
    assert not app._followed_cli_run()
    assert app._status_label.text() == tr("forge.app.train.cli_run.stopped")
    page = app.window.findChild(QtWidgets.QWidget, "forgeStepPage")
    text = page.findChild(QtWidgets.QLabel, "progressText")
    assert text.text() == tr("forge.app.train.idle")


@requires_qt
def test_a_followed_cli_run_paused_in_the_terminal_offers_continue(make_app, tmp_path, qapp) -> None:
    """A fresh train.json stage "paused" means the terminal run saved a version and exited."""
    app = make_app()
    folder = tmp_path / "projects" / "cube-20261114-2302"
    photos = make_photos(folder / "photos", 12)
    _write_cli_project(folder, photos, stage="train", step=400, total=1600)
    checkpoints = Project(folder).checkpoints
    checkpoints.mkdir(parents=True, exist_ok=True)
    (checkpoints / "step-000400.safetensors").write_bytes(b"checkpoint")
    app.open_project(folder)
    qapp.processEvents()
    write_row(folder, {"step": 400, "total": 1600, "loss": 0.2, "elapsed_seconds": 1200,
                       "eta_seconds": None, "stage": "paused"})
    app._read_cli_progress()
    qapp.processEvents()
    assert not getattr(app, "_cli_watch", False)
    assert app.paused
    assert app._status_label.text() == tr("forge.app.train.cli_run.paused")
    page = app.window.findChild(QtWidgets.QWidget, "forgeStepPage")
    start = next(b for b in page.findChildren(QtWidgets.QPushButton)
                 if b.text() == tr("forge.app.train.continue"))
    assert start.isEnabled()


@requires_qt
def test_stop_for_a_followed_cli_run_requests_a_pause_file(make_app, tmp_path, qapp, monkeypatch) -> None:
    """No pid guessing: Stop asks the terminal run to save a version and exit via PAUSE."""
    app = make_app()
    folder = tmp_path / "projects" / "cube-20261114-2303"
    photos = make_photos(folder / "photos", 12)
    _write_cli_project(folder, photos, stage="train", step=400, total=1600)
    app.open_project(folder)
    qapp.processEvents()
    monkeypatch.setattr(app, "_confirm_stop", lambda: True)
    app._stop_clicked()
    qapp.processEvents()
    assert (folder / "PAUSE").is_file()
    assert app._status_label.text() == tr("forge.app.train.cli_run.stop_requested")
    # the run is still followed until it exits (the PAUSE file is not cleared)
    assert getattr(app, "_cli_watch", False)


@requires_qt
def test_a_finished_cli_project_opens_on_the_result_step(make_app, tmp_path, qapp) -> None:
    app = make_app()
    folder = tmp_path / "projects" / "cube-20261114-2100"
    photos = make_photos(folder / "photos", 12)
    _write_cli_project(folder, photos, stage="done", step=10, total=10)
    (folder / "lora.safetensors").write_bytes(b"final")
    app.open_project(folder)
    qapp.processEvents()
    assert app.layout.sections.step.step_id == "result"
    assert app.draft.trained
    steps = app.window.findChild(QtWidgets.QLabel, "forge.result.steps")
    assert steps is not None and steps.text() == tr("forge.app.result.steps", done=10, planned=10)


@requires_qt
def test_the_cli_runs_action_sits_right_after_open_in_the_overflow_menu(make_app) -> None:
    app = make_app()
    actions = app.widget._parts()["overflow"].menu().actions()
    order = [action.text() for action in actions]
    assert order.index(tr("forge.app.menu.cli_runs")) == order.index(tr("forge.app.menu.open")) + 1
    action = next(item for item in actions if item.text() == tr("forge.app.menu.cli_runs"))
    assert action.toolTip() == tr("forge.app.menu.cli_runs.desc")


@requires_qt
def test_the_cli_runs_menu_lists_projects_and_opens_the_picked_one(make_app, tmp_path, qapp) -> None:
    app = make_app()
    folder = tmp_path / "projects" / "cube-20261114-2203"
    photos = make_photos(folder / "photos", 12)
    _write_cli_project(folder, photos)
    menu = app._cli_runs_menu(find_cli_projects([tmp_path / "projects"]))
    actions = menu.actions()
    assert len(actions) == 1
    assert actions[0].text() == f"{folder.name} — {tr('forge.app.cli_runs.state.running')}"
    assert actions[0].toolTip() == tr("forge.app.menu.cli_runs.desc")
    actions[0].trigger()
    qapp.processEvents()
    assert app.project == folder
    assert app.layout.sections.step.step_id == "train"


@requires_qt
def test_choose_cli_run_says_so_when_nothing_is_found(make_app, qapp, monkeypatch) -> None:
    from voxprint_forge.app import window as forge_window

    app = make_app()
    monkeypatch.setattr(forge_window, "find_cli_projects", lambda roots, *, now=None: [])
    app.choose_cli_run()
    qapp.processEvents()
    assert app._status_label.text() == tr("forge.app.cli_runs.none")


@requires_qt
def test_new_projects_are_created_under_the_configured_projects_root(make_app, tmp_path) -> None:
    app = make_app(projects_root=tmp_path / "documents-projects")
    app.new_model()
    assert app.project.parent == tmp_path / "documents-projects"
    assert app.project.name == "model-1"
    app.project.mkdir(parents=True)  # the first run creates it, so the next name skips it
    app.new_model()
    assert app.project.name == "model-2"


@requires_qt
def test_the_photos_menu_lists_person_folders_and_browse(make_app, tmp_path, monkeypatch, qapp) -> None:
    from voxprint_forge.app import window as forge_window

    app = make_app()
    app.window.resize(1280, 900)
    QtWidgets.QApplication.processEvents()
    photos = tmp_path / "Photos"
    person = photos / "Ada Example"
    person.mkdir(parents=True)
    (person / "ada-01.jpg").write_bytes(b"fake photo")
    monkeypatch.setattr(forge_window, "default_photos_root", lambda: photos)
    add = app.window.findChild(QtWidgets.QPushButton, "forge.photos.add")
    actions = [action.text() for action in add.menu().actions()]
    assert tr("forge.app.photos.add_person") in actions

    picked: list[Path] = []

    def fake_choice(root, folders, parent, browse):
        picked.extend(folders)
        return person

    monkeypatch.setattr(forge_window, "_person_folder_choice", fake_choice)
    app.choose_person_folder()
    qapp.processEvents()
    assert picked == [person]
    assert app.draft.photos == [person / "ada-01.jpg"]
