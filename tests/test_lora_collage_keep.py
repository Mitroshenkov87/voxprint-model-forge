"""Collage detector on large single photos, and ``lora prepare --keep``. Synthetic images only."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from tests.test_lora_prepare import ADULT_FACE, FakeEstimator, FakeJudge, noise, obj, photos, run
from voxprint_forge.lora.errors import LoraBlocked
from voxprint_forge.lora.gate import INDICATORS, AgeReading, Gate
from voxprint_forge.lora.photos import COLLAGE_SCORE, KEEPABLE, collage_score, filter_reason, looks_like_collage
from voxprint_forge.lora.project import Project


def _studio_portrait(width: int = 4500, height: int = 3000, seed: int = 7) -> Image.Image:
    """One person in a studio: dark backdrop, a floor line and a lit figure with straight sides.

    The figure's sides and the floor line are long straight edges. The old detector read
    them as panel seams and set such photos aside as collages.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :].repeat(height, axis=0)
    base = 40 + 30 * x
    image = Image.fromarray(np.stack([base, base * 0.9, base * 0.8], axis=2).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, int(height * 0.74), width, height], fill=(150, 130, 110))
    draw.rectangle([int(width * 0.43), int(height * 0.35), int(width * 0.60), int(height * 0.95)], fill=(210, 190, 175))
    draw.ellipse([int(width * 0.47), int(height * 0.08), int(width * 0.56), int(height * 0.36)], fill=(225, 200, 185))
    image = image.filter(ImageFilter.GaussianBlur(3))
    grain = rng.normal(128, 30, (height // 8, width // 8)).clip(0, 255).astype(np.uint8)
    grain_full = np.asarray(Image.fromarray(grain).resize((width, height)), dtype=np.float32)[..., None] - 128
    return Image.fromarray(np.clip(np.asarray(image, np.float32) + grain_full, 0, 255).astype(np.uint8))


def _scene(width: int, height: int, seed: int) -> Image.Image:
    """A photo-like panel: a soft colour gradient, blurred shapes and sensor noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    base, grad = rng.uniform(40, 200, 3), rng.uniform(-60, 60, 3)
    rgb = base + grad * (x / width)[..., None] * 0.5 + grad[::-1] * (y / height)[..., None] * 0.5
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    for _ in range(12):
        cx, cy = rng.uniform(0, width), rng.uniform(0, height)
        r = rng.uniform(0.05, 0.3) * min(width, height)
        draw.ellipse([cx - r, cy - r * 1.3, cx + r, cy + r * 1.3], fill=tuple(int(v) for v in rng.integers(0, 256, 3)))
    image = image.filter(ImageFilter.GaussianBlur(min(width, height) / 300))
    grain = rng.normal(0, 14, (height, width, 3))
    return Image.fromarray(np.clip(np.asarray(image, np.float32) + grain, 0, 255).astype(np.uint8))


def _collage(cols: int, rows: int, width: int, height: int, seed: int, gutter: int = 0) -> Image.Image:
    out = Image.new("RGB", (width, height), (245, 245, 245))
    cell_w = (width - gutter * (cols - 1)) // cols
    cell_h = (height - gutter * (rows - 1)) // rows
    for row in range(rows):
        for col in range(cols):
            panel = _scene(cell_w, cell_h, seed + row * cols + col)
            out.paste(panel, (col * (cell_w + gutter), row * (cell_h + gutter)))
    return out


def test_a_large_single_photo_with_long_straight_edges_is_kept() -> None:
    portrait = _studio_portrait()
    assert portrait.size == (4500, 3000)
    assert collage_score(portrait) < COLLAGE_SCORE / 2
    assert not looks_like_collage(portrait)
    assert filter_reason(portrait, []) is None
    upright = portrait.transpose(Image.Transpose.ROTATE_90)
    assert not looks_like_collage(upright)


@pytest.mark.parametrize("seed", [0, 10, 20, 30, 40])
@pytest.mark.parametrize(
    ("cols", "rows", "size"),
    [(2, 1, (1500, 1000)), (1, 2, (1000, 1500)), (2, 2, (1500, 1000)), (3, 1, (1800, 800))],
)
def test_side_by_side_and_grid_collages_are_still_dropped(cols, rows, size, seed) -> None:
    collage = _collage(cols, rows, *size, seed)
    assert collage_score(collage) >= COLLAGE_SCORE
    assert filter_reason(collage, []) == "lora.drop.collage"


def test_collages_with_gutters_are_still_dropped() -> None:
    assert filter_reason(_collage(2, 1, 1500, 1000, 3, gutter=20), []) == "lora.drop.collage"
    assert filter_reason(_collage(2, 2, 1500, 1000, 5, gutter=16), []) == "lora.drop.collage"
    assert filter_reason(_collage(3, 3, 1200, 1200, 7, gutter=12), []) == "lora.drop.collage"


def test_single_scenes_are_not_collages() -> None:
    for seed in range(100, 106):
        assert not looks_like_collage(_scene(1500, 1000, seed))
        assert not looks_like_collage(_scene(1000, 1500, seed + 50))


def test_filter_reason_only_allows_collage_and_duplicate() -> None:
    assert KEEPABLE == {"lora.drop.collage", "lora.drop.duplicate"}
    collage = _collage(2, 1, 1500, 1000, 0)
    assert filter_reason(collage, [], allow=KEEPABLE) is None
    for reason in ("lora.drop.safety", "lora.drop.blurry", "lora.drop.small"):
        with pytest.raises(ValueError):
            filter_reason(collage, [], allow=frozenset({reason}))
    assert filter_reason(Image.new("RGB", (800, 800), (90, 90, 90)), [], allow=KEEPABLE) == "lora.drop.blurry"


def _drop_reasons(tmp_path: Path) -> dict[str, str]:
    lines = (tmp_path / "proj" / "drops.jsonl").read_text(encoding="utf-8").splitlines()
    return {row["name"]: row["reason"] for row in map(json.loads, lines)}


def _save_collage(folder: Path, name: str = "pair.png") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    _collage(2, 1, 1200, 800, 0).save(path)
    return path


def test_keep_overrides_a_collage_drop(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    pair = _save_collage(tmp_path / "extra")
    first = run(tmp_path / "a", [*files, pair])
    assert first["prepared"] == 10 and _drop_reasons(tmp_path / "a") == {"pair.png": "lora.drop.collage"}
    # --keep adds the file when it is not among the photos yet.
    kept = run(tmp_path / "b", files, keep=[pair])
    assert kept["prepared"] == 11 and kept["set_aside"] == 0
    assert (tmp_path / "b" / "proj" / "drops.jsonl").read_text(encoding="utf-8").strip() == ""


def test_keep_overrides_a_caption_collage_drop(tmp_path) -> None:
    from tests.test_lora_prepare import FakeCaptioner

    class Grid(FakeCaptioner):
        def caption(self, image):
            return {"long": "a photo collage", "short": "collage"}

    files = photos(tmp_path / "in", 11)
    result = run(tmp_path, files, captioner=Grid(), keep=files)
    assert result["prepared"] == 11


def test_keep_overrides_a_duplicate_drop(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    twin = tmp_path / "in" / "twin.jpg"
    noise(3).save(twin, quality=98)
    first = run(tmp_path / "a", [*files, twin], drop_near_duplicates=True)
    assert first["prepared"] == 10 and _drop_reasons(tmp_path / "a") == {"twin.jpg": "lora.drop.duplicate"}
    kept = run(tmp_path / "b", [*files, twin], keep=[twin], drop_near_duplicates=True)
    assert kept["prepared"] == 11 and kept["set_aside"] == 0


class _PickyJudge(FakeJudge):
    """A clean verdict for every photo except the wide collage, which gets ``flagged``."""

    def __init__(self, flagged):
        super().__init__()
        self.flagged = flagged

    def judge(self, image):
        return self.flagged if image.size[0] > image.size[1] else self.verdict


def test_keep_does_not_override_a_safety_drop(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    pair = _save_collage(tmp_path / "extra")
    unsure = replace(ADULT_FACE, indicators={k: "absent" for k in INDICATORS} | {"school_context": "unsure"})
    judge = _PickyJudge(unsure)
    result = run(tmp_path, files, gate=Gate(judge, FakeEstimator(AgeReading(0, ()))), keep=[pair])
    assert result["prepared"] == 10
    assert _drop_reasons(tmp_path) == {"pair.png": "lora.drop.safety"}
    assert all("pair" not in item.file for item in Project(tmp_path / "proj").items())


def test_keep_does_not_prevent_a_keyword_refusal(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    flagged = _save_collage(tmp_path / "extra", "schoolgirl_set.png")
    with pytest.raises(LoraBlocked) as info:
        run(tmp_path, files, keep=[flagged], subject=obj())
    assert info.value.code == "lora.err.blocked"
    project = Project(tmp_path / "proj")
    assert project.read()["gate_passed"] is False
    assert not project.items()


def test_keep_on_the_cli_is_repeatable_and_documented(tmp_path, monkeypatch) -> None:
    import argparse

    from voxprint_forge.lora import cli as lora_cli

    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="cmd")
    lora_cli._prepare_flags(commands.add_parser("prepare"))
    args = parser.parse_args(["prepare", "--project", "p", "--keep", "a.png", "--keep", "b.png"])
    assert args.keep == [Path("a.png"), Path("b.png")]
    help_text = commands.choices["prepare"].format_help()
    assert "--keep FILE" in help_text
