"""MivoloEstimator.estimate sends the worker a lossless WebP frame (PNG over the WebP side limit)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from voxprint_forge.imageio_save import WEBP_MAX_SIDE
from voxprint_forge.lora.backends import MivoloEstimator

REPLY = json.dumps({"n_faces": 1, "ages": [30.0], "faces": [{"box": [1, 2, 3, 4], "score": 0.9}]}) + "\n"


class _FakeProc:
    def __init__(self) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(REPLY)

    def poll(self) -> int | None:
        return None


def _estimate(tmp_path: Path, image: Image.Image) -> Path:
    est = object.__new__(MivoloEstimator)
    est.tmp = tmp_path
    est.proc = _FakeProc()  # type: ignore[assignment]
    reading = est.estimate(image)
    assert reading.n_faces == 1
    sent = json.loads(est.proc.stdin.getvalue().splitlines()[0])  # type: ignore[union-attr]
    path = Path(sent["path"])
    assert path.is_file()
    return path


def test_frame_is_lossless_webp(tmp_path: Path) -> None:
    pixels = np.random.default_rng(0).integers(0, 256, (200, 300, 3), dtype=np.uint8)
    path = _estimate(tmp_path, Image.fromarray(pixels, "RGB"))
    assert path.suffix == ".webp"
    with Image.open(path) as decoded:
        assert np.array_equal(np.asarray(decoded.convert("RGB")), pixels)
    try:
        import cv2
    except ImportError:
        pytest.skip("cv2 not installed")
    bgr = cv2.imread(str(path))
    assert bgr is not None
    assert np.array_equal(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), pixels)


def test_frame_over_webp_limit_is_png(tmp_path: Path) -> None:
    path = _estimate(tmp_path, Image.new("RGB", (WEBP_MAX_SIDE + 1, 1)))
    assert path.suffix == ".png"
