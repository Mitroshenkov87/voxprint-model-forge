"""``lora prepare``: near-duplicates are kept with a warning unless asked; byte-identical copies are skipped.

Synthetic images only.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from tests.test_lora_prepare import noise, photos, run
from voxprint_forge.lora.events import Progress, sentence
from voxprint_forge.lora.project import Project


def _drops(tmp_path: Path) -> dict[str, str]:
    path = tmp_path / "proj" / "drops.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    return {row["name"]: row["reason"] for row in map(json.loads, lines)}


def _twin(folder: Path) -> Path:
    """``img03.png`` saved again as a JPEG: the same picture, different bytes."""
    path = folder / "twin.jpg"
    noise(3).save(path, quality=98)
    return path


def _warnings(progress: Progress) -> list:
    return [event for event in progress.events if event.message_key == "lora.warn.near_duplicate"]


def test_a_near_duplicate_is_kept_with_a_warning_naming_both_files(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    twin = _twin(tmp_path / "in")
    progress = Progress()
    result = run(tmp_path, [*files, twin], progress=progress)
    assert result["prepared"] == 11 and result["set_aside"] == 0
    assert _drops(tmp_path) == {}
    warned = _warnings(progress)
    assert len(warned) == 1
    assert warned[0].params == {"file": "twin.jpg", "other": "img03.png"}
    line = sentence(warned[0])
    assert "twin.jpg" in line and "img03.png" in line and "--drop-near-duplicates" in line
    assert len(Project(tmp_path / "proj").items()) == 11


def test_drop_near_duplicates_sets_the_near_duplicate_aside(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    twin = _twin(tmp_path / "in")
    progress = Progress()
    result = run(tmp_path, [*files, twin], progress=progress, drop_near_duplicates=True)
    assert result["prepared"] == 10 and result["set_aside"] == 1
    assert _drops(tmp_path) == {"twin.jpg": "lora.drop.duplicate"}
    assert not _warnings(progress)


def test_keep_still_overrides_drop_near_duplicates(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    twin = _twin(tmp_path / "in")
    result = run(tmp_path, [*files, twin], keep=[twin], drop_near_duplicates=True)
    assert result["prepared"] == 11 and result["set_aside"] == 0


def test_a_byte_identical_copy_is_still_skipped_by_default(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    copy = tmp_path / "in" / "copy.png"
    shutil.copyfile(files[4], copy)
    progress = Progress()
    result = run(tmp_path, [*files, copy], progress=progress)
    assert result["prepared"] == 10
    items = Project(tmp_path / "proj").items()
    assert len(items) == 10 and len({item.id for item in items}) == 10
    assert not _warnings(progress)


def test_the_cli_flag_is_off_by_default_and_documented() -> None:
    import argparse

    from voxprint_forge.lora import cli as lora_cli

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="cmd")
    lora_cli._prepare_flags(commands.add_parser("prepare"))
    assert parser.parse_args(["prepare", "--project", "p"]).drop_near_duplicates is False
    assert parser.parse_args(["prepare", "--project", "p", "--drop-near-duplicates"]).drop_near_duplicates is True
    assert "--drop-near-duplicates" in commands.choices["prepare"].format_help()
