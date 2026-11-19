"""Simple mode says why each photo was refused, and when Expert mode would accept the set."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from voxprint_forge.lora.errors import LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.i18n import text
from voxprint_forge.lora.refusal import (
    Note,
    Report,
    cli_lines,
    report_from_drops,
    scan_files,
    visible_on_photos_step,
)


def _plain(folder: Path, seed: int) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"plain-{seed:02d}.png"
    image = np.random.default_rng(seed).integers(0, 256, (640, 640, 3), dtype=np.uint8)
    Image.fromarray(image).save(path)
    return path


def _collage(folder: Path, seed: int) -> Path:
    """A grid larger than the minimum side, so the reason is collage and not too small."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"grid-{seed:02d}.png"
    size, panels, gutter = 630, 3, 12
    image = Image.new("RGB", (size, size), (250, 250, 250))
    cell = (size - gutter * (panels - 1)) // panels
    colors = (
        (20, 40, 180), (180, 40, 20), (20, 160, 40),
        (200, 180, 20), (80, 20, 160), (20, 180, 180),
        (160, 80, 40), (40, 40, 40), (220, 100, 160),
    )
    index = 0
    for row in range(panels):
        for col in range(panels):
            tile = Image.new("RGB", (cell, cell), colors[(index + seed) % len(colors)])
            image.paste(tile, (col * (cell + gutter), row * (cell + gutter)))
            index += 1
    image.save(path)
    return path


def test_nine_usable_photos_name_the_count_and_offer_expert(tmp_path: Path) -> None:
    report = scan_files([_plain(tmp_path, index) for index in range(9)])
    assert report.notes == () and report.kept == 9 and report.show_expert()
    assert visible_on_photos_step(report, 9, expert=False)
    assert not visible_on_photos_step(report, 9, expert=True)
    summary, hint = cli_lines(report)
    assert summary == text("lora.refuse.short", kept=9)
    assert hint == text("lora.refuse.expert", kept=9)
    assert "Expert" not in summary


def test_collages_are_named_and_expert_is_offered_only_for_a_set_it_would_keep(tmp_path: Path) -> None:
    mixed = [_plain(tmp_path / "mixed", index) for index in range(6)]
    mixed += [_collage(tmp_path / "mixed", index) for index in range(3)]
    report = scan_files(mixed)
    assert [note.reason for note in report.notes] == ["lora.drop.collage"] * 3
    assert report.kept == 6 and report.show_expert()
    summary, hint = cli_lines(report)
    assert summary == text("lora.refuse.line", dropped=3, total=9, kept=6)
    assert text("lora.refuse.fix.collage") in hint
    assert text("lora.refuse.expert", kept=6) in hint
    assert visible_on_photos_step(report, 9, expert=False)

    only = [_collage(tmp_path / "grids", index) for index in range(8)]
    only.append(_plain(tmp_path / "grids", 0))
    refused = scan_files(only)
    assert refused.kept == 1 and not refused.show_expert()
    spoken = " ".join(cli_lines(refused))
    assert text("lora.refuse.fix.collage") in spoken
    assert "Expert" not in spoken and "эксперт" not in spoken.lower()


def test_a_set_simple_mode_still_accepts_is_not_a_photos_step_refusal() -> None:
    report = Report((Note("tiny.png", "lora.drop.small"),) * 2, (), 10, 12)
    assert not visible_on_photos_step(report, 12, expert=False)
    summary, hint = cli_lines(report)
    assert "2 of 12" in summary
    assert hint == text("lora.refuse.fix.small")
    assert not report.show_expert()


def test_age_and_minors_stay_unchanged_and_carry_no_expert_hint() -> None:
    safety = report_from_drops(
        [{"name": "a.png", "reason": "lora.drop.safety"}, {"name": "b.png", "reason": "lora.drop.collage"}],
        6,
        8,
    )
    assert not safety.show_expert()
    spoken = " ".join(cli_lines(safety))
    assert text("lora.refuse.fix.collage") in spoken
    assert "Expert" not in spoken
    blocked = report_from_drops([{"name": "a.png", "reason": "lora.drop.safety"}], 0, 9, blocked=True)
    assert cli_lines(blocked) == ()
    assert not blocked.show_expert()
    pure = report_from_drops([{"name": "a.png", "reason": "lora.drop.safety"}] * 9, 0, 9)
    assert cli_lines(pure) == ()


def test_simple_prepare_explains_a_short_set_before_the_safety_gate(tmp_path: Path) -> None:
    from voxprint_forge.lora.consent import check_subject
    from voxprint_forge.lora.prepare import prepare
    from voxprint_forge.lora.project import Project

    class Gate:
        def check(self, image: object, *, attested: bool = False) -> None:
            raise AssertionError("safety gate ran")

    progress = Progress()
    files = [_plain(tmp_path / "in", index) for index in range(9)]
    with pytest.raises(LoraUsageError) as info:
        prepare(
            Project(tmp_path / "proj"),
            files,
            check_subject("object", name="Mug", rights_confirmed=True),
            gate=Gate(),  # type: ignore[arg-type]
            load_captioner=lambda choice: None,
            progress=progress,
        )
    assert info.value.code == "lora.err.too_few"
    assert info.value.params == {"need": 1}
    keys = [event.message_key for event in progress.events]
    assert keys == ["lora.refuse.short", "lora.refuse.hint"]
    assert progress.events[1].params["hint"] == text("lora.refuse.expert", kept=9)
    assert not (tmp_path / "proj" / "drops.jsonl").exists()
