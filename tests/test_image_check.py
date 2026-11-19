"""The dataset checker loads on classify, stays inside 4 GiB, and unloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from voxprint_forge.lora.image_check import CHECK_VRAM_BUDGET, ImageChecker, allocated_bytes, batch_count, measure_check
from voxprint_forge.ui.check_gate import CancelToken


def test_batch_count_never_plans_past_4gb() -> None:
    budget = CHECK_VRAM_BUDGET
    samples = (
        (0, 1024),
        (budget, 1),
        (budget // 2, budget // 2),
        (budget // 2, budget // 4),
        (1, budget),
        (0, budget + 1),
        (budget - 10, 100),
    )
    for weight, image in samples:
        count = batch_count(weight, image, budget)
        assert weight + count * image <= budget
        if image > 0 and weight + image > budget:
            assert count == 0


def test_the_checker_loads_only_when_classify_runs(tmp_path: Path) -> None:
    path = _jpeg(tmp_path / "one.jpg", 1)
    checker = ImageChecker()
    assert not checker.loaded
    before = allocated_bytes()
    ok, message = checker.classify(path, CancelToken())
    assert checker.loaded
    assert ok or message.startswith("lora.drop.")
    assert checker.peak_bytes <= CHECK_VRAM_BUDGET
    checker.unload()
    assert not checker.loaded
    assert allocated_bytes() <= before
    assert checker.peak_bytes <= CHECK_VRAM_BUDGET


def test_fifty_images_stay_inside_the_vram_budget(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    paths = [_jpeg(tmp_path / f"p{index:02d}.jpg", index) for index in range(50)]
    seconds, peak = measure_check(paths)
    print(f"CHECK_50 seconds={seconds:.3f} peak_bytes={peak}")
    assert seconds < 120
    assert peak <= CHECK_VRAM_BUDGET
    assert allocated_bytes() == 0
    captured = capsys.readouterr()
    assert "CHECK_50" in captured.out


def _jpeg(path: Path, seed: int) -> Path:
    image = np.random.default_rng(seed).integers(0, 256, (512, 512, 3), dtype=np.uint8)
    Image.fromarray(image, "RGB").save(path, "JPEG", quality=70)
    return path
