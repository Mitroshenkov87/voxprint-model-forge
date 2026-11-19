"""Launch defaults, the GUI-to-terminal selection state and the release auto-detect (no models).

The pure parts (:mod:`voxprint_forge.app.launch`) need no Qt; the GUI tests build
the offscreen window with a stub runner, like the other app tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from voxprint_forge.app.launch import (
    NO_SELECTION_MESSAGE,
    RELEASE_FILE,
    LaunchDefaults,
    find_release,
    parse_launch_args,
    read_selection,
    selection_file,
    write_selection,
)

try:
    from PySide6 import QtWidgets
    from voxprint_common.i18n import set_language

    from tests.forge_app_stub import ALL_COMMANDS, StubRunner, fake_image_checks, make_photos
except ImportError:  # pragma: no cover - the GUI tests below are skipped without PySide6
    set_language = None
    QtWidgets = None

requires_qt = pytest.mark.skipif(QtWidgets is None, reason="PySide6 is not installed")


@pytest.fixture()
def suite_home(tmp_path, monkeypatch):
    """An isolated suite home, so the selection and defaults files never touch the real one."""
    home = tmp_path / "suite-home"
    monkeypatch.setenv("VOXPRINT_HOME", str(home))
    return home


# ----- release auto-detect -------------------------------------------------


def test_find_release_returns_the_file_directly_inside_the_folder(tmp_path) -> None:
    folder = tmp_path / "photos"
    folder.mkdir()
    assert find_release(folder) is None
    release = folder / RELEASE_FILE
    release.write_text("synthetic release text", encoding="utf-8")
    assert find_release(folder) == release
    variant = folder / "Release.txt"
    variant.write_text("synthetic release text", encoding="utf-8")
    assert find_release(folder) == release  # the exact name wins over a case variant


def test_find_release_ignores_nested_files_directories_and_unreadable_files(tmp_path) -> None:
    folder = tmp_path / "photos"
    nested = folder / "inner"
    nested.mkdir(parents=True)
    (nested / RELEASE_FILE).write_text("nested", encoding="utf-8")
    assert find_release(folder) is None
    (folder / RELEASE_FILE).mkdir()
    assert find_release(folder) is None
    (folder / RELEASE_FILE).rmdir()
    release = folder / RELEASE_FILE
    release.write_text("text", encoding="utf-8")
    if os.name != "nt":  # chmod(0) does not drop the owner's read bit on Windows
        release.chmod(0)
        try:
            assert find_release(folder) is None
        finally:
            release.chmod(0o644)
    assert find_release(folder) == release
    assert find_release(tmp_path / "missing") is None


# ----- selection state ------------------------------------------------------


def test_selection_round_trip_is_atomic_and_absolute(suite_home) -> None:
    folder = suite_home / "Photos" / "Ada Example"
    folder.mkdir(parents=True)
    write_selection(folder)
    assert read_selection() == folder
    target = selection_file()
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["schema"] == "voxprint-forge-gui-selection/1"
    assert payload["photos_dir"] == str(folder.resolve())
    assert payload["updated_at"].endswith("+00:00")
    # tmp + os.replace: no partial file survives next to the real one
    assert [p.name for p in target.parent.iterdir()] == [target.name]


def test_selection_clears_with_none_and_ignores_a_missing_folder(suite_home) -> None:
    folder = suite_home / "photos"
    folder.mkdir(parents=True)
    write_selection(folder)
    write_selection(None)
    assert read_selection() is None
    assert json.loads(selection_file().read_text(encoding="utf-8"))["photos_dir"] is None
    write_selection(folder)
    folder.rmdir()
    assert read_selection() is None


def test_selection_ignores_broken_files_and_non_directory_values(suite_home) -> None:
    target = selection_file()
    target.parent.mkdir(parents=True)
    target.write_text("not json", encoding="utf-8")
    assert read_selection() is None
    target.write_text(json.dumps({"photos_dir": str(suite_home / "gone")}), encoding="utf-8")
    assert read_selection() is None
    target.write_text(json.dumps({"photos_dir": 42}), encoding="utf-8")
    assert read_selection() is None


# ----- parse_launch_args ----------------------------------------------------


def test_parse_launch_args_defaults(suite_home) -> None:
    defaults = parse_launch_args([])
    assert defaults == LaunchDefaults(crop="both", rank=64, steps=1600, base="z-image", adult=True, photos=None)


def test_parse_launch_args_flags_beat_the_config_file(suite_home) -> None:
    config = suite_home / "model-forge" / "gui-defaults.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps(
            {
                "crop": "whole",
                "rank": 32,
                "steps": 900,
                "base": "z-image-turbo",
                "adult": False,
                "photos": str(suite_home / "from-config"),
            }
        ),
        encoding="utf-8",
    )
    loaded = parse_launch_args([])
    assert loaded.crop == "whole"
    assert loaded.rank == 32
    assert loaded.steps == 900
    assert loaded.base == "z-image-turbo"
    assert loaded.adult is False
    assert loaded.photos == suite_home / "from-config"
    overridden = parse_launch_args(
        [
            "--crop",
            "subject",
            "--rank",
            "64",
            "--steps",
            "1600",
            "--base",
            "z-image",
            "--adult",
            "--photos",
            str(suite_home / "cli"),
        ]
    )
    assert overridden == LaunchDefaults(
        crop="subject", rank=64, steps=1600, base="z-image", adult=True, photos=suite_home / "cli"
    )
    assert parse_launch_args(["--no-adult"]).adult is False


def test_parse_launch_args_rejects_non_positive_rank_and_steps(suite_home) -> None:
    with pytest.raises(SystemExit):
        parse_launch_args(["--rank", "0"])
    with pytest.raises(SystemExit):
        parse_launch_args(["--steps", "-5"])


# ----- console entry ---------------------------------------------------------


def test_app_help_lists_the_launch_flags(capsys) -> None:
    from voxprint_forge.app.__main__ import main

    assert main(["voxprint-model-forge", "--help"]) == 0
    out = capsys.readouterr().out
    for flag in ("--crop", "--rank", "--steps", "--base", "--adult", "--no-adult", "--photos", "--print-selection"):
        assert flag in out


def test_print_selection_module_cli_round_trip(suite_home) -> None:
    env = {**os.environ, "VOXPRINT_HOME": str(suite_home)}
    command = [sys.executable, "-m", "voxprint_forge.app.launch", "--print-selection"]
    missing = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert missing.returncode == 3
    assert NO_SELECTION_MESSAGE in missing.stderr
    folder = suite_home / "Photos" / "Ada Example"
    folder.mkdir(parents=True)
    write_selection(folder)
    found = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert found.returncode == 0
    assert found.stdout.strip() == str(folder)


# ----- GUI preload (offscreen) -----------------------------------------------

if QtWidgets is not None:

    @pytest.fixture
    def make_app(qapp, tmp_path, monkeypatch):
        monkeypatch.setenv("VOXPRINT_HOME", str(tmp_path / "home"))
        from voxprint_forge.app.window import ForgeApp

        made = []

        def build(runner=None, *, launch=None):
            set_language("en")
            app = ForgeApp(
                runner=runner or StubRunner(),
                home=tmp_path / "state",
                projects_root=tmp_path / "state" / "lora",
                commands=ALL_COMMANDS,
                open_path=lambda _p: None,
                detached=lambda _a: None,
                image_checks=fake_image_checks(tmp_path / f"worker-{len(made)}"),
                launch=launch,
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


def _expert_value(app, key):
    return app.expert_values()[key]


@requires_qt
def test_launch_defaults_preload_the_draft_and_the_expert_values(make_app) -> None:
    app = make_app(launch=LaunchDefaults())
    assert app.draft.crop == "both"
    assert app.draft.adult and app.draft.adult_content
    assert app.draft.photos == []
    assert app.simple["base"] == "z-image"
    assert _expert_value(app, "lora.rank") == 64
    assert _expert_value(app, "lora.alpha") == 64
    assert _expert_value(app, "lora.steps") == 1600


@requires_qt
def test_launch_flags_override_the_preload(make_app, tmp_path) -> None:
    photos = tmp_path / "photos"
    photos.mkdir()
    app = make_app(
        launch=LaunchDefaults(crop="subject", rank=32, steps=900, base="z-image-turbo", adult=False, photos=photos)
    )
    assert app.draft.crop == "subject"
    assert not app.draft.adult and not app.draft.adult_content
    assert app.simple["base"] == "z-image-turbo"
    assert _expert_value(app, "lora.rank") == 32
    assert _expert_value(app, "lora.alpha") == 32
    assert _expert_value(app, "lora.steps") == 900
    assert len(app.draft.photos) == 0  # an empty folder adds no photos
    assert read_selection() == photos  # the preloaded folder is recorded for the terminal


COMPLETE_RELEASE = (
    'Model name: "Ada Example"\n'
    "Adult (18+): yes\n"
    "No minors in photos: yes\n"
    "Consent to training: yes\n"
    "Consent to generation: yes\n"
    "LoRA training allowed: yes\n"
    "Adult content in training: yes\n"
    "Date: 2026-11-16\n"
)


@requires_qt
def test_a_release_file_in_the_photo_folder_sets_release_consent_and_adult(make_app, tmp_path, qapp) -> None:
    from voxprint_forge.app.text import tr

    folder = tmp_path / "photos" / "Ada Example"
    photos = make_photos(folder, 12)
    release = folder / RELEASE_FILE
    release.write_text(COMPLETE_RELEASE, encoding="utf-8")
    app = make_app()
    app.draft.set_verbal(True)  # even a ticked verbal box is unticked by complete consent
    app._dropped([folder])
    qapp.processEvents()
    assert app.draft.release == release
    assert app.draft.kind == "person"  # preselected by the release file
    assert app.draft.name == "Ada Example"  # ... and the model name fills an empty name
    assert not app.draft.verbal
    assert app.draft.consent
    assert app.draft.adult and app.draft.adult_content
    assert app.status == ("forge.app.release.found", {"name": RELEASE_FILE}, False)
    label = app._status_label
    assert label is not None and label.text() == tr("forge.app.release.found", name=RELEASE_FILE)
    # the .txt is never listed as a photo; the folder is the recorded selection
    assert sorted(app.draft.photos) == sorted(photos)
    assert read_selection() == folder
    app._remove_paths(list(app.draft.photos))
    qapp.processEvents()
    assert read_selection() is None


@requires_qt
def test_an_incomplete_release_file_leaves_the_consent_fields_untouched(make_app, tmp_path, qapp) -> None:
    folder = tmp_path / "photos" / "Ada Example"
    make_photos(folder, 12)
    release = folder / RELEASE_FILE
    release.write_text('Model name: "Ada Example"\nAdult (18+): no\n', encoding="utf-8")
    app = make_app()
    app._dropped([folder])
    qapp.processEvents()
    assert app.draft.release == release  # the file is still attached for the record
    assert app.draft.kind == "person" and app.draft.name == "Ada Example"
    assert not app.draft.consent and not app.draft.adult and not app.draft.adult_content
    assert app.status == ("forge.app.release.found", {"name": RELEASE_FILE}, False)


@requires_qt
def test_release_file_without_adult_content_leaves_the_label_off(make_app, tmp_path, qapp) -> None:
    folder = tmp_path / "photos" / "Ada Example"
    make_photos(folder, 12)
    release = folder / RELEASE_FILE
    release.write_text(COMPLETE_RELEASE.replace("Adult content in training: yes\n", ""), encoding="utf-8")
    app = make_app()
    app._dropped([folder])
    qapp.processEvents()
    assert app.draft.release == release
    assert app.draft.consent and app.draft.adult
    assert not app.draft.adult_content


@requires_qt
def test_the_window_opens_a_cli_run_started_after_it(make_app, tmp_path, qapp, monkeypatch) -> None:
    from voxprint_forge.app import window as forge_window
    from voxprint_forge.lora.progress_log import write_row
    from voxprint_forge.lora.project import Project

    app = make_app()
    roots = tmp_path / "roots"
    monkeypatch.setattr(forge_window, "default_roots", lambda _home: [roots])
    running = roots / "ada-20261115-0900"
    make_photos(running / "photos", 12)
    Project(running).write({"kind": "object", "subject": {"name": "CLI Cube", "rights_confirmed": True}})
    write_row(
        running, {"step": 3, "total": 10, "loss": 0.2, "elapsed_seconds": 60, "eta_seconds": 300, "stage": "step"}
    )
    os.utime(running / "project.json", (app._started_at + 5, app._started_at + 5))
    old = roots / "old-20261114-0900"
    Project(old).write({"kind": "object", "subject": {"name": "Old"}})
    write_row(old, {"step": 3, "total": 10, "loss": 0.2, "elapsed_seconds": 60, "eta_seconds": 300, "stage": "step"})
    os.utime(old / "project.json", (app._started_at - 100, app._started_at - 100))
    app._follow_cli_runs()
    qapp.processEvents()
    assert app.project == running
    assert app.layout.sections.step.step_id == "train"
    assert getattr(app, "_cli_watch", False)  # _open_cli_run follows it live
    # the same run is never opened twice
    app._follow_cli_runs()
    qapp.processEvents()
    assert app.project == running
