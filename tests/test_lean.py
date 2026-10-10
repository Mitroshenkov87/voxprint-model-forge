"""Lean checks stay small, and they stop when the interval decides."""

from __future__ import annotations

from pathlib import Path

from voxprint_forge.diffusion.prompts import (
    BASE_PROMPTS,
    BASE_SEEDS,
    CONTROL_PROMPTS,
    CONTROL_SEEDS,
    CONTROL_TYPES,
    EDIT_PAIRS,
    EDIT_SEEDS,
    INPAINT_PROMPTS,
    LEAN_BASE_PROMPTS,
    LEAN_BASE_SEEDS,
    LEAN_CONTROL,
    LEAN_CONTROL_MINIMUM,
    LEAN_EDIT,
    LEAN_EDIT_MINIMUM,
    LEAN_INPAINT,
    LEAN_INPAINT_MINIMUM,
    LEAN_MINIMUM,
)
from voxprint_forge.evaluate import refusal_rate
from voxprint_forge.lean import (
    REFUSAL_LINE,
    metrics_decided,
    proportion_interval,
    review_document,
    separates,
    wave_size,
)
from voxprint_forge.llama import Completed


def test_wilson_interval_decides_a_one_sided_refusal_rate() -> None:
    none = proportion_interval(0, 4)
    all_refused = proportion_interval(4, 4)
    split = proportion_interval(2, 4)
    assert none is not None and separates(*none, REFUSAL_LINE, higher_is_worse=True)
    assert all_refused is not None and separates(*all_refused, REFUSAL_LINE, higher_is_worse=True)
    assert split is not None and not separates(*split, REFUSAL_LINE, higher_is_worse=True)


def test_lpips_interval_stops_after_the_minimum() -> None:
    clear = [0.02, 0.03, 0.02, 0.04]
    assert metrics_decided([(clear, 0.10, True)], minimum=4) is True
    assert metrics_decided([(clear[:3], 0.10, True)], minimum=4) is False
    wide = [0.02, 0.20, 0.03, 0.19]
    assert metrics_decided([(wide, 0.10, True)], minimum=4) is False


def test_a_clear_failure_stops_before_the_other_metric() -> None:
    failed = [0.40, 0.42, 0.41, 0.39]
    pending = [0.01, 0.20, 0.02, 0.19]
    assert metrics_decided([(failed, 0.10, True), (pending, 0.01, True)], minimum=4) is True


def test_waves_run_the_minimum_then_one_at_a_time() -> None:
    assert wave_size(0, 8, decided=False, minimum=4) == 4
    assert wave_size(4, 8, decided=False, minimum=4) == 1
    assert wave_size(4, 8, decided=True, minimum=4) == 0
    assert wave_size(8, 8, decided=False, minimum=4) == 0


def test_image_plans_use_a_few_hard_cases() -> None:
    assert set(LEAN_BASE_PROMPTS) <= set(BASE_PROMPTS)
    assert len(LEAN_BASE_PROMPTS) * len(LEAN_BASE_SEEDS) < len(BASE_PROMPTS) * len(BASE_SEEDS)
    assert LEAN_MINIMUM < len(LEAN_BASE_PROMPTS)
    assert any("wristwatch" in prompt for prompt in LEAN_BASE_PROMPTS)
    assert any("adult" in prompt for prompt in LEAN_BASE_PROMPTS)
    kinds = [kind for _index, kind in LEAN_CONTROL[:LEAN_CONTROL_MINIMUM]]
    assert sorted(kinds) == sorted(CONTROL_TYPES)
    assert len(LEAN_CONTROL) < len(CONTROL_PROMPTS) * len(CONTROL_TYPES) * len(CONTROL_SEEDS)
    assert set(LEAN_INPAINT) <= set(INPAINT_PROMPTS)
    assert LEAN_INPAINT_MINIMUM < len(INPAINT_PROMPTS)
    assert set(LEAN_EDIT) <= set(EDIT_PAIRS)
    assert len(LEAN_EDIT) * 1 < len(EDIT_PAIRS) * len(EDIT_SEEDS)
    assert LEAN_EDIT_MINIMUM <= len(LEAN_EDIT)


def test_review_pack_is_labelled_full_frames() -> None:
    text = review_document(
        [
            ("00-reference.png", "A close view of a mechanical wristwatch on a cloth.", 0, "reference"),
            ("00-candidate.png", "A close view of a mechanical wristwatch on a cloth.", 0, "candidate"),
        ]
    )
    assert "00-reference.png" in text
    assert "wristwatch" in text
    assert "seed 0" in text
    assert "grid" not in text.lower()


class _Refuse:
    def __init__(self) -> None:
        self.calls = 0

    def run(self, argv: list[str], *, cwd: Path | None = None) -> Completed:
        self.calls += 1
        return Completed(argv, 0, "I cannot help with that.", "")


def test_refusal_rate_stops_once_the_rate_is_clear(monkeypatch) -> None:
    monkeypatch.setattr("voxprint_forge.evaluate.tool", lambda _name: "llama-cli")
    runner = _Refuse()
    rate = refusal_rate(runner, Path("model.gguf"), ["prompt"] * 12, cache="q8_0")
    assert runner.calls == 4
    assert rate == 1.0
