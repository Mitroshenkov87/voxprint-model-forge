"""Size-aware batch planning and the weighted progress estimate. CPU only, no models."""

from __future__ import annotations

import pytest

from voxprint_forge.lora.batching import (
    class_batch,
    photo_weight,
    plan_batches,
    plan_weights,
    size_class,
)
from voxprint_forge.lora.events import Progress

# 20 photos, sorted file-name order: heavy 24/12 MP, medium 6/4.5/3 MP, light at or below 1280x1280.
SIZES = [
    (6000, 4000),  # 0  heavy, 24 MP
    (800, 600),    # 1  light
    (6000, 4000),  # 2  heavy, 24 MP
    (1000, 1000),  # 3  light
    (4000, 3000),  # 4  heavy, 12 MP at the class bound
    (3000, 2000),  # 5  medium, 6 MP
    (1280, 1280),  # 6  light at the class bound
    (800, 600),    # 7  light
    (6000, 4000),  # 8  heavy, 24 MP
    (2000, 1500),  # 9  medium, 3 MP
    (1000, 1000),  # 10 light
    (3000, 2000),  # 11 medium, 6 MP
    (6000, 4000),  # 12 heavy, 24 MP
    (1280, 1000),  # 13 light
    (4000, 3000),  # 14 heavy, 12 MP
    (2500, 1800),  # 15 medium, 4.5 MP
    (800, 600),    # 16 light
    (6000, 4000),  # 17 heavy, 24 MP
    (1000, 800),   # 18 light
    (3000, 2000),  # 19 medium, 6 MP
]

# batch 16 -> light chunks of 16, medium of 8, heavy of 4; chunks ordered by first index.
EXPECTED_16 = [
    [0, 2, 4, 8],
    [1, 3, 6, 7, 10, 13, 16, 18],
    [5, 9, 11, 15, 19],
    [12, 14, 17],
]


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_size_class_bounds() -> None:
    assert size_class(1280, 1280) == "light"
    assert size_class(800, 600) == "light"
    assert size_class(2000, 1500) == "medium"
    assert size_class(4000, 3000) == "heavy"
    assert size_class(6000, 4000) == "heavy"


def test_class_batch_scales_each_class_down_and_never_below_1() -> None:
    assert class_batch(16, "light") == 16
    assert class_batch(16, "medium") == 8
    assert class_batch(16, "heavy") == 4
    assert class_batch(5, "medium") == 2
    assert class_batch(5, "heavy") == 1
    assert class_batch(0, "heavy") == 1
    assert class_batch(-3, "light") == 1
    with pytest.raises(ValueError):
        class_batch(16, "huge")


def test_plan_batches_groups_by_class_and_keeps_the_run_in_order() -> None:
    plan = plan_batches(SIZES, 16)
    assert plan == EXPECTED_16
    assert plan_batches(SIZES, 16) == plan  # deterministic: depends only on sizes and batch
    assert plan_batches([], 16) == []
    flat = [index for chunk in plan for index in chunk]
    assert sorted(flat) == list(range(len(SIZES)))  # every index exactly once
    assert all(chunk == sorted(chunk) for chunk in plan)  # input order kept inside a chunk
    firsts = [chunk[0] for chunk in plan]
    assert firsts == sorted(firsts)  # chunks ordered by their first index


def test_plan_batches_chunks_never_exceed_the_class_size() -> None:
    assert [len(chunk) for chunk in plan_batches([(800, 600)] * 20, 16)] == [16, 4]  # light: batch
    assert [len(chunk) for chunk in plan_batches([(2000, 1500)] * 20, 16)] == [8, 8, 4]  # medium: batch // 2
    assert [len(chunk) for chunk in plan_batches([(6000, 4000)] * 10, 16)] == [4, 4, 2]  # heavy: batch // 4
    for batch in (1, 2, 3, 7, 16):
        for chunk in plan_batches(SIZES, batch):
            assert len(chunk) <= batch
    plan = plan_batches(SIZES, 1)
    assert len(plan) == len(SIZES) and all(len(chunk) == 1 for chunk in plan)


def test_photo_weight_is_one_plus_pixels_over_weight_pixels() -> None:
    assert photo_weight(4000, 3000) == pytest.approx(2.0)  # 12 MP / WEIGHT_PIXELS
    assert photo_weight(6000, 4000) == pytest.approx(3.0)  # 24 MP / WEIGHT_PIXELS
    assert photo_weight(64, 64) == pytest.approx(1.0, abs=1e-3)
    assert photo_weight(0, 100) == 1.0
    assert photo_weight(100, 0) == 1.0
    assert photo_weight(-5, 10) == 1.0


def test_plan_weights_matches_photo_weight_per_entry() -> None:
    weights = plan_weights(SIZES)
    assert weights == [photo_weight(width, height) for width, height in SIZES]


def _run_chunks(
    progress: Progress, clock: _Clock, sizes: list[tuple[int, int]], batch: int, *, weighted: bool
) -> list[tuple[float, int | None]]:
    """One emit per planned chunk on a clock where each photo takes 0.5 s + 0.5 s per 12 MP.

    Time is exactly proportional to ``photo_weight``, so the true remaining seconds after each
    chunk are known; returns (true remaining, emitted eta) pairs.
    """
    plan = plan_batches(sizes, batch)
    weights = plan_weights(sizes)
    total_weight = sum(weights)
    rows: list[tuple[float, int | None]] = []
    done = 0
    work_done = 0.0
    for chunk in plan:
        clock.now += sum(0.5 * weights[index] for index in chunk)
        done += len(chunk)
        work_done += sum(weights[index] for index in chunk)
        kw = {"work_done": work_done, "work_total": total_weight} if weighted else {}
        event = progress.emit("prepare.safety", "k", done=done, total=len(sizes), **kw)
        rows.append((0.5 * (total_weight - work_done), event.eta_seconds))
    return rows


def test_weighted_eta_tracks_the_true_remaining_time_for_mixed_sizes() -> None:
    clock = _Clock()
    rows = _run_chunks(Progress(clock=clock), clock, SIZES, batch=4, weighted=True)
    assert rows[0][1] is None  # the first call past the warm-up only sets the mark
    etas = [eta for _, eta in rows if eta is not None]
    for remaining, eta in rows:
        if eta is not None:
            assert abs(eta - remaining) <= 1.0  # after the warm-up every ETA is within 1 s
    assert all(later <= earlier for earlier, later in zip(etas, etas[1:]))  # ETAs never increase


def test_the_weighted_eta_is_tighter_than_the_count_based_one() -> None:
    clock = _Clock()
    weighted = _run_chunks(Progress(clock=clock), clock, SIZES, batch=4, weighted=True)
    clock = _Clock()
    plain = _run_chunks(Progress(clock=clock), clock, SIZES, batch=4, weighted=False)

    def spread(rows: list[tuple[float, int | None]]) -> float:
        return max(abs(eta - remaining) for remaining, eta in rows if eta is not None)

    assert spread(weighted) < spread(plain)


def test_emit_without_work_args_keeps_the_count_based_percent_and_eta() -> None:
    clock = _Clock()
    progress = Progress(clock=clock)
    etas: list[int | None] = []
    for done in range(1, 11):
        clock.now += 2.0  # each unit takes 2 s
        event = progress.emit("step", "k", done=done, total=10)
        assert event.percent == round(100.0 * done / 10, 1)
        etas.append(event.eta_seconds)
    assert etas[:2] == [None, None]  # warm-up, then the mark
    assert etas[2:] == [2 * (10 - done) for done in range(3, 11)]
    assert progress.events[-1].percent == 100.0
    # Missing or empty work amounts fall back to the count-based path, offset and all.
    event = progress.emit("step", "k", done=1, total=10, work_done=0.0, work_total=0.0)
    assert event.percent == 10.0 and event.eta_seconds is None
