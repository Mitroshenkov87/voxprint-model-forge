"""Quant selector on the four stand-in cards. No weights and no GPU."""

from __future__ import annotations

import pytest

from voxprint_forge.diffusion.quant_select import (
    KNOWN_GPUS,
    Candidate,
    QuantSelectError,
    choose,
    fits,
    known_gpu,
)


def _row(mode: str, min_sm: int, generate: float, train: float, lpips: float, clip: float) -> Candidate:
    return Candidate(mode, min_sm, generate, train, lpips, clip)


def test_known_cards_are_the_four_stand_ins() -> None:
    assert [gpu.name for gpu in KNOWN_GPUS] == [
        "RTX 3060",
        "RTX 3080 Laptop",
        "RTX 4080",
        "RTX 4090",
    ]
    assert known_gpu("RTX 3060") == KNOWN_GPUS[0]
    assert (known_gpu("RTX 3060").vram_gb, known_gpu("RTX 3060").sm) == (12.0, 86)
    assert (known_gpu("RTX 3080 Laptop").vram_gb, known_gpu("RTX 3080 Laptop").sm) == (16.0, 86)
    assert (known_gpu("RTX 4080").vram_gb, known_gpu("RTX 4080").sm) == (16.0, 89)
    assert (known_gpu("RTX 4090").vram_gb, known_gpu("RTX 4090").sm) == (24.0, 89)


def test_fp8_is_skipped_below_sm89_on_3060_and_3080_laptop() -> None:
    fp8 = _row("fp8_e4m3", 89, 10.0, 10.0, 0.01, 0.99)
    int8 = _row("int8", 70, 12.5, 12.6, 0.05, 0.95)
    nf4 = _row("nf4", 70, 9.0, 9.0, 0.08, 0.90)
    for name in ("RTX 3060", "RTX 3080 Laptop"):
        gpu = known_gpu(name)
        assert fits(fp8, gpu, 13.5) is False
        # 12.6 GB fits the 16 GB laptop and the 13.5 GB budget. It does not fit a 12 GB 3060.
        assert fits(int8, gpu, 13.5) is (name != "RTX 3060")
        assert fits(nf4, gpu, 13.5) is True
        chosen = choose((fp8, int8, nf4), gpu, 13.5)
        assert chosen.mode == ("nf4" if name == "RTX 3060" else "int8")


def test_4080_can_take_fp8_and_4090_still_obeys_the_budget() -> None:
    fp8 = _row("fp8_e4m3", 89, 12.2, 13.4, 0.02, 0.99)
    int8 = _row("int8", 70, 11.0, 11.2, 0.05, 0.97)
    nf4 = _row("nf4", 70, 9.0, 9.1, 0.09, 0.91)
    assert choose((fp8, int8, nf4), known_gpu("RTX 4080"), 13.5).mode == "fp8_e4m3"
    assert choose((fp8, int8, nf4), known_gpu("RTX 4090"), 13.5).mode == "fp8_e4m3"
    over = _row("fp8_e4m3", 89, 13.6, 20.0, 0.01, 0.99)
    assert fits(over, known_gpu("RTX 4090"), 13.5) is False
    assert choose((over, int8, nf4), known_gpu("RTX 4090"), 13.5).mode == "int8"


def test_exact_budget_fits_and_a_higher_clip_breaks_an_lpips_tie() -> None:
    gpu = known_gpu("RTX 4090")
    edge = _row("int8", 70, 13.5, 13.5, 0.04, 0.90)
    assert fits(edge, gpu, 13.5) is True
    assert fits(_row("int8", 70, 13.51, 10.0, 0.01, 0.99), gpu, 13.5) is False
    low_clip = _row("int8", 70, 10.0, 10.0, 0.04, 0.80)
    high_clip = _row("nf4", 70, 10.0, 10.0, 0.04, 0.95)
    assert choose((low_clip, high_clip), gpu, 13.5).mode == "nf4"


def test_none_fit_on_a_4090_when_every_peak_is_over_the_budget() -> None:
    rows = (
        _row("fp8_e4m3", 89, 13.8, 13.9, 0.02, 0.99),
        _row("int8", 70, 14.0, 13.7, 0.04, 0.97),
        _row("nf4", 70, 13.6, 13.6, 0.08, 0.90),
    )
    with pytest.raises(QuantSelectError) as caught:
        choose(rows, known_gpu("RTX 4090"), 13.5)
    assert caught.value.code == "moss_prequant_none_fit"
