"""``lora prepare`` photo I/O: one read per file, worker threads, and the same output as before."""

from __future__ import annotations

import hashlib
import io
import threading

import numpy as np
import pytest
from PIL import Image, ImageOps

from voxprint_forge.lora import prepare as prepare_mod
from voxprint_forge.lora.consent import check_subject
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import INDICATORS, AgeReading, Gate, Verdict
from voxprint_forge.lora.photos import (
    collage_score,
    crop,
    dhash,
    filter_reason,
    measure,
    open_photo,
    photo_id,
    read_photo,
    sharpness,
)
from voxprint_forge.lora.prepare import _ordered, prepare
from voxprint_forge.lora.project import Project

CLEAN = Verdict("no", "no", None, 0.9, "no_person", False, {k: "absent" for k in INDICATORS})


def _old_open(path):
    """``open_photo`` as it was before the single-read decode."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


class RecordingJudge:
    name = "recording-judge"

    def __init__(self):
        self.seen: list[bytes] = []
        self.threads: set[str] = set()

    def judge(self, image):
        self.seen.append(hashlib.sha256(image.tobytes()).digest())
        self.threads.add(threading.current_thread().name)
        return CLEAN

    def unload(self):
        pass


class Estimator:
    name = "estimator"

    def estimate(self, image):
        return AgeReading(0, ())


class RecordingCaptioner:
    name = "recording-captioner"

    def __init__(self):
        self.seen: list[bytes] = []

    def caption(self, image):
        self.seen.append(hashlib.sha256(image.tobytes()).digest())
        return {"long": "a textured square on a plain background", "short": "textured square"}

    def unload(self):
        pass


def _photo(seed: int, width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 160, width, dtype=np.float32)[None, :, None] + rng.uniform(0, 60, 3)
    grain = rng.integers(-50, 51, (height, width, 3))
    return Image.fromarray(np.clip(base + grain, 0, 255).astype(np.uint8))


def _photos(folder):
    """JPEG, PNG, WebP; RGB, RGBA, L and P; with and without an EXIF rotation."""
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    specs = [
        ("jpg", "RGB", 1),
        ("jpg", "RGB", 6),
        ("jpg", "L", 8),
        ("png", "RGBA", 3),
        ("png", "P", 1),
        ("png", "RGB", 5),
        ("webp", "RGB", 1),
        ("jpg", "RGB", 2),
        ("png", "RGB", 1),
        ("jpg", "RGB", 7),
        ("png", "L", 1),
        ("jpg", "RGB", 4),
    ]
    for index, (ext, mode, orientation) in enumerate(specs):
        image = _photo(index, 700 + 13 * index, 560 + 7 * index).convert(mode)
        exif = Image.Exif()
        exif[0x0112] = orientation
        path = folder / f"img{index:02d}.{ext}"
        image.save(path, exif=exif)
        paths.append(path)
    # A byte-identical copy and an unreadable file keep their old outcomes too.
    (folder / "img90.png").write_bytes(paths[4].read_bytes())
    (folder / "img91.jpg").write_bytes(b"not a picture")
    return [*paths, folder / "img90.png", folder / "img91.jpg"]


def _run(tmp_path, files, name, crop_mode="subject"):
    judge, captioner = RecordingJudge(), RecordingCaptioner()
    project = Project(tmp_path / name)
    summary = prepare(
        project,
        files,
        check_subject("object", name="Blue mug", trigger="vxpmugq", rights_confirmed=True),
        gate=Gate(judge, Estimator()),
        load_captioner=lambda choice: captioner,
        progress=Progress(),
        mode="expert",
        crop_mode=crop_mode,
    )
    return project, summary, judge, captioner


@pytest.mark.parametrize("crop_mode", ["subject", "whole"])
def test_prepared_files_are_the_bytes_the_old_pipeline_wrote(tmp_path, crop_mode) -> None:
    files = _photos(tmp_path / "photos")
    project, summary, judge, captioner = _run(tmp_path, files, "proj", crop_mode)
    readable = files[:-2]
    assert summary["new"] == len(readable)
    expected_ids = [photo_id(p) for p in readable]
    assert [item.id for item in project.items()] == expected_ids
    old_crops = [crop(_old_open(p), crop_mode) for p in readable]
    for pid, old in zip(expected_ids, old_crops, strict=True):
        buffer = io.BytesIO()
        old.save(buffer, format="PNG")
        assert (project.prepared / f"{pid}.png").read_bytes() == buffer.getvalue()
    # The gate saw the full decoded photo, the captioner the crop, in input order, on the caller's thread.
    assert judge.seen == [hashlib.sha256(_old_open(p).tobytes()).digest() for p in readable]
    assert captioner.seen == [hashlib.sha256(c.tobytes()).digest() for c in old_crops]
    assert judge.threads == {threading.current_thread().name}
    drops = project.drops.read_text(encoding="utf-8")
    assert "lora.drop.unreadable" in drops and "img91.jpg" in drops


def test_one_thread_and_many_threads_write_the_same_project(tmp_path, monkeypatch) -> None:
    files = _photos(tmp_path / "photos")
    outputs = []
    for workers in (1, 4):
        monkeypatch.setattr(prepare_mod, "WORKERS", workers)
        project, _, _, _ = _run(tmp_path, files, f"proj{workers}")
        files_out = {p.name: p.read_bytes() for p in sorted(project.prepared.iterdir())}
        texts = [project.captions.read_bytes(), project.drops.read_bytes(), project.gate_trace.read_bytes()]
        outputs.append((files_out, texts))
    assert outputs[0] == outputs[1]


def test_read_photo_matches_photo_id_and_the_old_open(tmp_path) -> None:
    for path in _photos(tmp_path / "photos"):
        pid, image = read_photo(path)
        assert pid == photo_id(path)
        old = None if path.name == "img91.jpg" else _old_open(path)
        for new in (image, open_photo(path)):
            if old is None:
                assert new is None
                continue
            assert (new.mode, new.size, new.tobytes()) == (old.mode, old.size, old.tobytes())
            assert type(new) is type(old)
            assert new.info == old.info


def test_measures_match_the_single_checks(tmp_path) -> None:
    for path in _photos(tmp_path / "photos")[:-2]:
        image = open_photo(path)
        found = measure(image)
        assert found.size == image.size
        assert found.collage == collage_score(image)
        assert found.sharp == sharpness(image)
        assert found.hash == dhash(image)
        seen_a: list[int] = []
        seen_b: list[int] = []
        for threshold in (0.0, 60.0, 1e9):
            with_measures = filter_reason(image, seen_a, blur_threshold=threshold, measures=found)
            alone = filter_reason(image, seen_b, blur_threshold=threshold)
            assert with_measures == alone
        assert seen_a == seen_b
    small = _photo(0, 300, 200)
    assert measure(small) == measure(small, 512)
    assert measure(small).collage is None
    assert filter_reason(small, [], measures=measure(small)) == "lora.drop.small"
    assert measure(None) is None


def test_ordered_keeps_order_and_stops_early() -> None:
    started: list[int] = []
    lock = threading.Lock()

    def work(n: int) -> int:
        with lock:
            started.append(n)
        return n * n

    assert list(_ordered(work, range(50), workers=4)) == [n * n for n in range(50)]
    started.clear()
    results = _ordered(work, range(1000), workers=2)
    assert [next(results) for _ in range(3)] == [0, 1, 4]
    results.close()
    # Only a bounded window ran ahead of what was taken.
    assert len(started) <= 3 + 2 * 2 + 2
    assert list(_ordered(work, [], workers=4)) == []
    assert list(_ordered(work, range(5), workers=1)) == [0, 1, 4, 9, 16]
