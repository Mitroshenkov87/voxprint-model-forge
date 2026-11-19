"""``scripts/forge-train`` argument assembly: a stub CLI records the argv of both steps.

Synthetic text and pixel files only; nothing is trained and nothing is downloaded.
The interactive consent/age prompts read /dev/tty, so a run that reaches the stub CLI
proves the prompts were skipped.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import pytest
from PIL import Image

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="the wrappers are bash scripts with a Unix venv path",
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "forge-train"


def _photos(folder: Path) -> Path:
    folder.mkdir(parents=True)
    Image.new("RGB", (64, 64), (120, 80, 40)).save(folder / "synthetic.png")
    return folder


def _stub_cli(home: Path) -> Path:
    """A fake ``voxprint-forge-cli`` that appends each argv as a JSON line to $STUB_LOG."""
    cli = home / "VOXPrint" / "forge" / "venv" / "bin" / "voxprint-forge-cli"
    cli.parent.mkdir(parents=True)
    cli.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$(printf '%s\\n' \"$@\" | python3 -c "
        "'import json,sys; print(json.dumps(sys.stdin.read().splitlines()))')\" >> \"${STUB_LOG:?}\"\n",
        encoding="utf-8",
    )
    cli.chmod(cli.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return cli


def _fake_python(home: Path) -> Path:
    """A ``python`` next to the stub CLI: exec the real interpreter (for ``-m voxprint_forge...``)."""
    python = home / "VOXPrint" / "forge" / "venv" / "bin" / "python"
    python.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    python.chmod(python.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return python


def _run_train(tmp_path: Path, home: Path, *args: str) -> tuple[int, str, list[list[str]]]:
    log = tmp_path / "stub.log"
    env = {
        **os.environ,
        "HOME": str(home),
        "STUB_LOG": str(log),
        "XDG_DOCUMENTS_DIR": str(home / "Documents"),
        "VOXPRINT_HOME": str(home / "suite-home"),
    }
    import subprocess

    completed = subprocess.run(
        ["bash", str(SCRIPT), *args],
        env=env,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=120,
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.is_file() else []
    return completed.returncode, completed.stderr + completed.stdout, calls


def _prepare(calls: list[list[str]]) -> list[str]:
    for call in calls:
        if call[:2] == ["lora", "prepare"]:
            return call
    raise AssertionError(f"no lora prepare call in {calls}")


def _train(calls: list[list[str]]) -> list[str]:
    for call in calls:
        if call[:2] == ["lora", "train"]:
            return call
    raise AssertionError(f"no lora train call in {calls}")


@pytest.fixture()
def home(tmp_path: Path) -> Path:
    folder = tmp_path / "home"
    _stub_cli(folder)
    return folder


def test_yes_skips_prompts_and_records_the_verbal_self_declaration(tmp_path, home) -> None:
    code, out, calls = _run_train(
        tmp_path, home, "Ann", "--photos", str(_photos(tmp_path / "photos")), "--yes", "--dry-run"
    )
    assert code == 0, out
    prepare = _prepare(calls)
    # The prompts were skipped (no /dev/tty in the test) and the self-declaration fields
    # are still passed and recorded by lora prepare.
    assert "--consent-obtained" in prepare
    assert "--subject-adult" in prepare
    assert "--consent-method" in prepare
    assert prepare[prepare.index("--consent-method") + 1] == "verbal"
    assert "--release" not in prepare


def test_release_passes_the_file_and_uses_the_file_consent_method(tmp_path, home) -> None:
    release = tmp_path / "release.txt"
    release.write_text("synthetic release text, not a picture", encoding="utf-8")
    code, out, calls = _run_train(
        tmp_path,
        home,
        "Ann",
        "--photos",
        str(_photos(tmp_path / "photos")),
        "--release",
        str(release),
        "--dry-run",
    )
    assert code == 0, out
    prepare = _prepare(calls)
    assert "--consent-method" in prepare
    assert prepare[prepare.index("--consent-method") + 1] == "file"
    assert "--release" in prepare
    assert prepare[prepare.index("--release") + 1] == str(release)
    assert "--consent-obtained" in prepare and "--subject-adult" in prepare
    # lora prepare copies the release into the project and records its sha256 on the
    # subject; the recording itself is covered by the prepare-level release tests.


def test_release_and_yes_together(tmp_path, home) -> None:
    release = tmp_path / "release.txt"
    release.write_text("synthetic release text", encoding="utf-8")
    code, out, calls = _run_train(
        tmp_path, home, "Ann", "--photos", str(_photos(tmp_path / "photos")), "--release", str(release), "--yes"
    )
    assert code == 0, out
    prepare = _prepare(calls)
    assert prepare[prepare.index("--consent-method") + 1] == "file"


def test_missing_release_file_is_a_clear_error(tmp_path, home) -> None:
    code, out, calls = _run_train(
        tmp_path, home, "Ann", "--photos", str(_photos(tmp_path / "photos")), "--release", str(tmp_path / "none.txt")
    )
    assert code != 0
    assert "release file not found or not readable" in out
    assert calls == []


def test_adult_content_is_the_default_and_can_be_turned_off(tmp_path, home) -> None:
    code, out, calls = _run_train(
        tmp_path / "on", home, "Ann", "--photos", str(_photos(tmp_path / "on" / "photos")), "--yes", "--dry-run"
    )
    assert code == 0, out
    assert "--adult-content" in _prepare(calls)
    code, out, calls = _run_train(
        tmp_path / "off",
        home,
        "Ann2",
        "--photos",
        str(_photos(tmp_path / "off" / "photos")),
        "--yes",
        "--dry-run",
        "--no-adult-content",
    )
    assert code == 0, out
    assert "--adult-content" not in _prepare(calls)


def test_crop_defaults_to_both_and_rank_to_64(tmp_path, home) -> None:
    code, out, calls = _run_train(
        tmp_path, home, "Ann", "--photos", str(_photos(tmp_path / "photos")), "--yes", "--dry-run"
    )
    assert code == 0, out
    prepare = _prepare(calls)
    assert prepare[prepare.index("--crop") + 1] == "both"
    train = _train(calls)
    assert train[train.index("--rank") + 1] == "64"
    assert train[train.index("--alpha") + 1] == "64"


def test_crop_and_base_pass_through(tmp_path, home) -> None:
    code, out, calls = _run_train(
        tmp_path,
        home,
        "Ann",
        "--photos",
        str(_photos(tmp_path / "photos")),
        "--yes",
        "--dry-run",
        "--crop",
        "whole",
        "--base",
        "z-image-turbo",
        "--rank",
        "32",
    )
    assert code == 0, out
    prepare = _prepare(calls)
    assert prepare[prepare.index("--crop") + 1] == "whole"
    assert prepare[prepare.index("--base") + 1] == "z-image-turbo"
    train = _train(calls)
    assert train[train.index("--rank") + 1] == "32"
    assert train[train.index("--alpha") + 1] == "32"


def _gui_selection(home: Path, photos: Path) -> Path:
    """Write the selection file the app would have written, into the isolated suite home."""
    selection = home / "suite-home" / "model-forge" / "gui-selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps({
        "schema": "voxprint-forge-gui-selection/1",
        "photos_dir": str(photos),
        "updated_at": "2026-11-15T09:00:00+00:00",
    }), encoding="utf-8")
    return selection


def test_photos_from_gui_uses_the_selection_and_defaults_the_name(tmp_path, home) -> None:
    _fake_python(home)
    folder = _photos(tmp_path / "photos" / "Ada Example")
    _gui_selection(home, folder)
    code, out, calls = _run_train(tmp_path, home, "--photos-from-gui", "--yes", "--dry-run")
    assert code == 0, out
    prepare = _prepare(calls)
    assert prepare[prepare.index("--photos") + 1] == str(folder)
    # NAME was omitted: it defaults to the selected folder's name ("Ada Example" -> ada-example-)
    project = Path(prepare[prepare.index("--project") + 1])
    assert project.name.startswith("ada-example-")


def test_photos_from_gui_stops_with_the_helper_message_when_nothing_is_selected(tmp_path, home) -> None:
    _fake_python(home)
    code, out, calls = _run_train(tmp_path, home, "--photos-from-gui")
    assert code == 1
    assert "No photo folder selected in the Forge window yet." in out
    assert calls == []


RELEASE_TXT = (
    'Model name: "Tatiana Model"\n'
    "Adult (18+): yes\n"
    "No minors in photos: yes\n"
    "Consent to training: yes\n"
    "Consent to generation: yes\n"
    "LoRA training allowed: yes\n"
    "Adult content in training: yes\n"
    "Date: 2026-11-16\n"
)


def _found_line(text: str, *, name: str = "Tatiana Model", adult: str = "on",
                attested: str = "yes", prompts: str = "skipped") -> str:
    digest = hashlib.sha256(text.encode()).hexdigest()
    return (
        f"forge-train: release.txt found in the photo folder (sha256 {digest}); "
        f"name='{name}'; adult {adult}; no minors attested: {attested}; consent prompts {prompts}"
    )


def test_release_file_in_the_photo_folder_is_used_and_skips_the_prompts(tmp_path, home) -> None:
    _fake_python(home)
    folder = _photos(tmp_path / "photos")
    folder.joinpath("release.txt").write_text(RELEASE_TXT, encoding="utf-8")
    # No --yes: reaching the stub CLI proves the consent/age prompts were skipped by the release file.
    code, out, calls = _run_train(tmp_path, home, "Ann", "--photos", str(folder), "--dry-run")
    assert code == 0, out
    assert _found_line(RELEASE_TXT) in out
    prepare = _prepare(calls)
    assert prepare[prepare.index("--consent-method") + 1] == "file"
    assert prepare[prepare.index("--release") + 1] == str(folder / "release.txt")
    assert "--adult-content" in prepare
    # the .txt is never handed to lora prepare as a photo
    assert str(folder / "release.txt") not in [arg for arg in prepare if arg.startswith("--photo")]


def test_release_file_name_is_the_default_when_name_is_omitted(tmp_path, home) -> None:
    _fake_python(home)
    folder = _photos(tmp_path / "photos")
    folder.joinpath("release.txt").write_text(RELEASE_TXT, encoding="utf-8")
    code, out, calls = _run_train(tmp_path, home, "--photos", str(folder), "--dry-run")
    assert code == 0, out
    prepare = _prepare(calls)
    assert prepare[prepare.index("--name") + 1] == "Tatiana Model"
    project = Path(prepare[prepare.index("--project") + 1])
    assert project.name.startswith("tatiana-model-")


def test_cli_name_beats_the_release_file_name(tmp_path, home) -> None:
    _fake_python(home)
    folder = _photos(tmp_path / "photos")
    folder.joinpath("release.txt").write_text(RELEASE_TXT, encoding="utf-8")
    code, out, calls = _run_train(tmp_path, home, "Ann", "--photos", str(folder), "--dry-run")
    assert code == 0, out
    prepare = _prepare(calls)
    assert prepare[prepare.index("--name") + 1] == "Ann"
    project = Path(prepare[prepare.index("--project") + 1])
    assert project.name.startswith("ann-")


def test_incomplete_release_file_falls_back_to_the_prompts(tmp_path, home) -> None:
    _fake_python(home)
    folder = _photos(tmp_path / "photos")
    text = 'Model name: "Tatiana Model"\nAdult (18+): no\n'
    folder.joinpath("release.txt").write_text(text, encoding="utf-8")
    # No --yes and no /dev/tty in the test: the asked prompts abort the run.
    code, out, calls = _run_train(tmp_path, home, "Ann", "--photos", str(folder), "--dry-run")
    assert code != 0
    assert _found_line(text, attested="no", prompts="asked") in out
    assert "Aborted: consent not confirmed." in out
    assert calls == []


def test_incomplete_release_file_with_yes_records_the_verbal_declaration(tmp_path, home) -> None:
    _fake_python(home)
    folder = _photos(tmp_path / "photos")
    text = 'Model name: "Tatiana Model"\nAdult (18+): no\n'
    folder.joinpath("release.txt").write_text(text, encoding="utf-8")
    code, out, calls = _run_train(tmp_path, home, "Ann", "--photos", str(folder), "--yes", "--dry-run")
    assert code == 0, out
    assert _found_line(text, attested="no") in out  # --yes also skips the prompts
    prepare = _prepare(calls)
    assert "--release" not in prepare  # only a complete consent file is recorded
    assert prepare[prepare.index("--consent-method") + 1] == "verbal"


def test_explicit_no_adult_content_wins_over_the_folder_release_file(tmp_path, home) -> None:
    _fake_python(home)
    folder = _photos(tmp_path / "photos")
    folder.joinpath("release.txt").write_text(RELEASE_TXT, encoding="utf-8")
    code, out, calls = _run_train(
        tmp_path, home, "Ann", "--photos", str(folder), "--no-adult-content", "--dry-run"
    )
    assert code == 0, out
    assert _found_line(RELEASE_TXT, adult="off") in out
    prepare = _prepare(calls)
    assert "--release" in prepare
    assert "--adult-content" not in prepare


def test_an_explicit_release_file_wins_over_the_folder_one(tmp_path, home) -> None:
    _fake_python(home)
    folder = _photos(tmp_path / "photos")
    folder.joinpath("release.txt").write_text(RELEASE_TXT, encoding="utf-8")
    explicit = tmp_path / "signed.txt"
    explicit.write_text("explicit release", encoding="utf-8")
    code, out, calls = _run_train(
        tmp_path, home, "Ann", "--photos", str(folder), "--release", str(explicit), "--yes", "--dry-run"
    )
    assert code == 0, out
    assert "release.txt found" not in out
    prepare = _prepare(calls)
    assert prepare[prepare.index("--release") + 1] == str(explicit)


def test_release_name_beats_the_gui_folder_name(tmp_path, home) -> None:
    _fake_python(home)
    folder = _photos(tmp_path / "photos" / "Ada Example")
    folder.joinpath("release.txt").write_text(RELEASE_TXT, encoding="utf-8")
    _gui_selection(home, folder)
    code, out, calls = _run_train(tmp_path, home, "--photos-from-gui", "--dry-run")
    assert code == 0, out
    prepare = _prepare(calls)
    assert prepare[prepare.index("--name") + 1] == "Tatiana Model"
    project = Path(prepare[prepare.index("--project") + 1])
    assert project.name.startswith("tatiana-model-")
