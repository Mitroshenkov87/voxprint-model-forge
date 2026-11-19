"""Rare trigger tokens, caption binding, sample prompts, collages and AVIF photos."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from voxprint_forge.lora.consent import check_subject, default_trigger
from voxprint_forge.lora.errors import LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.photos import (
    EXTENSIONS,
    avif_ready,
    caption_marks_collage,
    collect,
    filter_reason,
    looks_like_collage,
    open_photo,
)
from voxprint_forge.lora.project import Project
from voxprint_forge.lora.sample_prompts import build_sample_prompts
from voxprint_forge.lora.trigger import (
    is_rare_trigger,
    looks_like_words,
    prepend_trigger,
    rare_trigger,
    strip_trigger,
)
from voxprint_forge.version import version_line


def test_the_trigger_is_a_rare_token_and_not_the_project_name() -> None:
    token = default_trigger("PG Synthetic A", "PG Synthetic A")
    again = rare_trigger()
    assert is_rare_trigger(token) and is_rare_trigger(again)
    assert token != again
    assert "synthetic" not in token and "pg_" not in token
    assert looks_like_words("pg_synthetic_a") and looks_like_words("synthetic")
    assert not looks_like_words(token)


def test_the_trigger_is_prepended_to_every_caption_once() -> None:
    token = "vxpbcfghj"
    first = prepend_trigger(token, "A woman models various stylish outfits")
    assert first == "vxpbcfghj, A woman models various stylish outfits"
    assert prepend_trigger(token, first) == first
    assert strip_trigger(token, first) == "A woman models various stylish outfits"


def test_the_preview_prompt_uses_the_subject_from_the_short_caption() -> None:
    captions = [{
        "long": "A woman models various stylish outfits",
        "short": "A woman models various stylish outfits",
    }]
    prompts = build_sample_prompts(
        "vxpbcfghj", count=1, seed=2, round_no=1, mode="clothed", captions=captions,
    )
    assert prompts == ["vxpbcfghj, a woman, close-up portrait, wearing near-skin-tone underwear"]
    assert "PG Synthetic" not in prompts[0] and "pg_synthetic" not in prompts[0]
    assert prompts != ["vxpbcfghj"]


def test_panel_seams_flag_a_collage_and_plain_noise_does_not() -> None:
    assert caption_marks_collage("a 3x3 grid of nine photos")
    assert caption_marks_collage("a photo collage")
    assert not caption_marks_collage("a woman models various stylish outfits")
    panels = _panels()
    assert looks_like_collage(panels)
    assert filter_reason(panels, [], min_side=64) == "lora.drop.collage"
    plain = Image.fromarray(np.random.default_rng(1).integers(0, 256, (600, 600, 3), dtype=np.uint8))
    assert not looks_like_collage(plain)
    assert filter_reason(plain, []) != "lora.drop.collage"


def test_a_grid_caption_is_dropped(tmp_path: Path) -> None:
    from tests.test_lora_prepare import FakeCaptioner, photos, run

    class Grid(FakeCaptioner):
        def __init__(self) -> None:
            super().__init__()
            self.seen = 0

        def caption(self, image: Image.Image) -> dict[str, object]:
            self.seen += 1
            if self.seen == 1:
                return {"long": "a 3x3 grid of nine photos", "short": "a 3x3 grid"}
            return super().caption(image)

    files = photos(tmp_path / "in", 11)
    run(tmp_path, files, captioner=Grid())
    project = Project(tmp_path / "proj")
    reasons = [json.loads(line)["reason"] for line in project.drops.read_text(encoding="utf-8").splitlines()]
    assert reasons == ["lora.drop.collage"]
    assert all("grid" not in item.short and "grid" not in item.long for item in project.items())
    assert len(project.items()) == 10


def test_avif_photos_open_and_join_the_set(tmp_path: Path) -> None:
    assert avif_ready()
    assert ".avif" in EXTENSIONS
    path = tmp_path / "shot.avif"
    Image.new("RGB", (32, 24), (10, 20, 30)).save(path)
    image = open_photo(path)
    assert image is not None and image.size == (32, 24)
    pixel = image.getpixel((0, 0))
    assert all(abs(int(got) - want) <= 2 for got, want in zip(pixel, (10, 20, 30), strict=True))
    assert collect(tmp_path, [], []) == [path]


def test_gui_version_and_help_print_and_skip_the_window(capsys: pytest.CaptureFixture[str]) -> None:
    from voxprint_forge.app import __main__ as entry

    assert entry.main(["voxprint-model-forge", "--version"]) == 0
    assert capsys.readouterr().out.strip() == version_line()
    assert entry.main(["voxprint-model-forge", "--help"]) == 0
    help_text = capsys.readouterr().out
    assert "--version" in help_text and "--help" in help_text
    assert entry.main(["voxprint-model-forge", "-V"]) == 0
    assert capsys.readouterr().out.strip() == version_line()


def test_a_typed_trigger_warns_on_the_cli_and_a_blank_one_does_not() -> None:
    from voxprint_forge.lora.cli import _warn_typed_trigger

    warned = Progress()
    _warn_typed_trigger(type("A", (), {"trigger": "synthetic"})(), warned)
    assert warned.events and warned.events[0].message_key == "lora.warn.trigger_word"
    quiet = Progress()
    _warn_typed_trigger(type("A", (), {"trigger": ""})(), quiet)
    assert quiet.events == []
    person = check_subject("person", name="PG Synthetic A", consent_obtained=True, subject_adult=True)
    assert is_rare_trigger(person.trigger)


def _panels() -> Image.Image:
    """A 3x3 collage with light gutters and a different flat colour in each cell."""
    size, panels, gutter = 270, 3, 8
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
            tile = Image.new("RGB", (cell, cell), colors[index])
            image.paste(tile, (col * (cell + gutter), row * (cell + gutter)))
            index += 1
    return image


def test_prepare_refuses_when_every_photo_is_a_collage(tmp_path: Path) -> None:
    from tests.test_lora_prepare import FakeCaptioner, photos, run

    class Grid(FakeCaptioner):
        def caption(self, image: Image.Image) -> dict[str, object]:
            return {"long": "a 3x3 grid of nine photos", "short": "grid"}

    with pytest.raises(LoraUsageError) as info:
        run(tmp_path, photos(tmp_path / "in", 10), captioner=Grid())
    assert info.value.code == "lora.err.too_few"
    reasons = [json.loads(line)["reason"] for line in (tmp_path / "proj" / "drops.jsonl").read_text().splitlines()]
    assert reasons == ["lora.drop.collage"] * 10
