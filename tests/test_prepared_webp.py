"""Lossless WebP prepared crops: pixel-exact writes, legacy PNG fallback, removal."""

from __future__ import annotations

import hashlib
import io

import numpy as np
import pytest
from PIL import Image

from voxprint_forge.imageio_save import WEBP_MAX_SIDE, ImageSaveError, save_lossless_webp
from voxprint_forge.lora.prepare import _read_crop, _save_crop, _unlink_prepared, prepared_path
from voxprint_forge.lora.project import Project


def _random_rgb(seed: int, size: tuple[int, int] = (257, 193)) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (*size, 3), dtype=np.uint8))


def _decoded(payload: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(payload))
    image.load()
    return image


def test_save_lossless_webp_round_trips_pixels_exactly(tmp_path) -> None:
    source = _random_rgb(7)
    target = tmp_path / "crop.webp"
    save_lossless_webp(source, target)
    with Image.open(target) as opened:
        opened.load()
        decoded = opened.tobytes()
        decoded_array = np.asarray(opened)
    assert np.array_equal(decoded_array, np.asarray(source))
    assert decoded == source.tobytes()
    assert hashlib.sha256(decoded).digest() == hashlib.sha256(source.tobytes()).digest()


def test_save_lossless_webp_writes_a_binary_buffer() -> None:
    source = _random_rgb(8)
    buffer = io.BytesIO()
    save_lossless_webp(source, buffer)
    assert _decoded(buffer.getvalue()).tobytes() == source.tobytes()


def test_save_crop_writes_webp_and_prepared_path_reads_it_back(tmp_path) -> None:
    project = Project(tmp_path / "proj")
    project.prepared.mkdir(parents=True)
    source = _random_rgb(9)
    assert _save_crop(project, ("abc-full", source)) == "abc-full"
    assert (project.prepared / "abc-full.webp").is_file()
    assert prepared_path(project, "abc-full") == project.prepared / "abc-full.webp"
    assert _read_crop(prepared_path(project, "abc-full")).tobytes() == source.tobytes()


def test_save_crop_replaces_a_legacy_png_of_the_same_id(tmp_path) -> None:
    project = Project(tmp_path / "proj")
    project.prepared.mkdir(parents=True)
    legacy = project.prepared / "abc-full.png"
    _random_rgb(10).save(legacy, format="PNG")
    source = _random_rgb(11)
    _save_crop(project, ("abc-full", source))
    assert not legacy.exists()
    assert (project.prepared / "abc-full.webp").is_file()
    assert prepared_path(project, "abc-full") == project.prepared / "abc-full.webp"
    assert _read_crop(prepared_path(project, "abc-full")).tobytes() == source.tobytes()


def test_prepared_path_prefers_webp_and_falls_back_to_legacy_png(tmp_path) -> None:
    project = Project(tmp_path / "proj")
    project.prepared.mkdir(parents=True)
    webp = project.prepared / "id1.webp"
    png = project.prepared / "id1.png"
    png.write_bytes(b"legacy")
    assert prepared_path(project, "id1") == png
    webp.write_bytes(b"current")
    assert prepared_path(project, "id1") == webp
    # A missing id names its WebP: what a fresh crop would be written to.
    assert prepared_path(project, "missing") == project.prepared / "missing.webp"


def test_unlink_prepared_removes_webp_and_legacy_png(tmp_path) -> None:
    project = Project(tmp_path / "proj")
    project.prepared.mkdir(parents=True)
    for name in ("id1.webp", "id1.png", "id2.webp"):
        (project.prepared / name).write_bytes(b"x")
    _unlink_prepared(project, "id1")
    assert not (project.prepared / "id1.webp").exists()
    assert not (project.prepared / "id1.png").exists()
    assert (project.prepared / "id2.webp").is_file()
    _unlink_prepared(project, "missing")  # missing_ok: nothing to remove


def test_save_lossless_webp_refuses_a_side_over_the_webp_limit(tmp_path) -> None:
    image = Image.new("RGB", (WEBP_MAX_SIDE + 1, 1))
    with pytest.raises(ImageSaveError):
        save_lossless_webp(image, tmp_path / "wide.webp")
    assert not (tmp_path / "wide.webp").exists()
