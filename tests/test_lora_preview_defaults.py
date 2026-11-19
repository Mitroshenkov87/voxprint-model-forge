"""Preview defaults (owner decisions 2026-11-16): vertical 2:3 full-figure pictures drawn with LoRA strength 1.3."""

from __future__ import annotations

import pytest

from voxprint_forge.lora.preview_format import PREVIEW_ASPECT, preview_size
from voxprint_forge.lora.sample_prompts import (
    PERSON_PREVIEW,
    PREVIEW_CLOTHED,
    PREVIEW_NUDE,
    preview_prompt_set,
)


def test_preview_size_is_vertical_two_to_three() -> None:
    assert PREVIEW_ASPECT == (2, 3)
    assert preview_size(1024) == (688, 1024)
    assert preview_size(512) == (336, 512)
    for resolution in (256, 512, 768, 1024):
        width, height = preview_size(resolution)
        assert height == resolution
        assert width % 16 == 0
        assert width < height


def test_preview_lora_scale_is_13() -> None:
    pytest.importorskip("torch")
    from voxprint_forge.lora.train import PREVIEW_LORA_SCALE

    assert PREVIEW_LORA_SCALE == 1.3


def test_person_preview_prompt_is_a_full_figure_vertical() -> None:
    assert "full figure" in PERSON_PREVIEW and "vertical" in PERSON_PREVIEW


def test_lora_strength_scales_the_delta_and_restores_it() -> None:
    torch = pytest.importorskip("torch")
    from torch import nn

    from voxprint_forge.lora.layers import LoRALinear, lora_strength

    layer = LoRALinear(nn.Linear(4, 4), rank=2, alpha=2)
    with torch.no_grad():
        layer.lora_A.weight.fill_(0.5)
        layer.lora_B.weight.fill_(0.5)
    x = torch.ones(1, 4)
    base = layer.base(x)
    usual = layer(x) - base

    layer.strength = 0.7  # a pre-set value comes back, not the default
    with lora_strength(layer, 1.3):
        assert layer.strength == 1.3
        torch.testing.assert_close(layer(x) - base, usual * 1.3)
    assert layer.strength == 0.7
    torch.testing.assert_close(layer(x) - base, usual * 0.7)

    layer.strength = 1.0
    with lora_strength(layer, 1.3):
        torch.testing.assert_close(layer(x) - base, usual * 1.3)
    assert layer.strength == 1.0
    torch.testing.assert_close(layer(x) - base, usual)


def test_lora_strength_restores_the_previous_value_on_exception() -> None:
    pytest.importorskip("torch")
    from torch import nn

    from voxprint_forge.lora.layers import LoRALinear, lora_strength

    layer = LoRALinear(nn.Linear(4, 4), rank=2, alpha=2)
    with pytest.raises(RuntimeError, match="boom"), lora_strength(layer, 1.3):
        raise RuntimeError("boom")
    assert layer.strength == 1.0


def test_preview_prompt_set_person_clothed_only_unless_adult_gets_the_pair() -> None:
    captions = [{"short": "A woman standing in a park"}]

    single = preview_prompt_set("zq7", "person", captions, adult=False)
    assert len(single) == 1
    assert single[0].endswith(f", {PREVIEW_CLOTHED}")
    assert "full figure" in single[0] and "vertical" in single[0]
    assert PREVIEW_NUDE not in single[0]

    pair = preview_prompt_set("zq7", "person", captions, adult=True)
    assert len(pair) == 2
    assert pair[0].endswith(f", {PREVIEW_CLOTHED}")
    assert pair[1].endswith(f", {PREVIEW_NUDE}")
    for prompt in pair:
        assert "full figure" in prompt and "vertical" in prompt


def test_preview_prompt_set_object_and_style_stay_a_single_plain_prompt() -> None:
    for kind in ("object", "style"):
        plain = preview_prompt_set("tcube", kind, [], adult=False)
        assert preview_prompt_set("tcube", kind, [], adult=True) == plain
        assert len(plain) == 1
        single = plain[0]
        assert PREVIEW_CLOTHED not in single and PREVIEW_NUDE not in single
        assert "full figure" not in single


def test_default_preview_prompts_prefers_the_release_adult_generation_flag(tmp_path) -> None:
    from tests.lora_fakes import make_project
    from voxprint_forge.lora.train import default_preview_prompts

    project = make_project(tmp_path / "proj", count=10)
    data = project.read()
    data["kind"] = "person"

    data["subject"] = {"name": "Ann", "trigger": "tann", "adult_content": True, "adult_generation": False}
    single = default_preview_prompts(project, data)
    assert len(single) == 1  # adult_generation False wins over adult_content True: no nude pair
    assert single[0].endswith(f", {PREVIEW_CLOTHED}")

    data["subject"] = {"name": "Ann", "trigger": "tann", "adult_content": True, "adult_generation": True}
    pair = default_preview_prompts(project, data)
    assert len(pair) == 2
    assert pair[1].endswith(f", {PREVIEW_NUDE}")

    data["subject"] = {"name": "Ann", "trigger": "tann", "adult_content": True}
    assert len(default_preview_prompts(project, data)) == 2  # the adult_content fallback stays


def test_training_previews_are_drawn_at_preview_strength(tmp_path) -> None:
    pytest.importorskip("torch")
    from tests.lora_fakes import gradient, make_project
    from tests.test_lora_review import DrawingAdapter, _config
    from voxprint_forge.lora.events import Progress
    from voxprint_forge.lora.layers import LoRALinear
    from voxprint_forge.lora.train import PREVIEW_LORA_SCALE, train

    seen: list[float] = []

    class Recording(DrawingAdapter):
        def preview(self, model, prompt, seed, config, steps=None):
            seen.extend(m.strength for m in model.modules() if isinstance(m, LoRALinear))
            return gradient(seed, 32)

    project = make_project(tmp_path, count=10)
    config = _config(steps=2, resolution=64, save_every=1, previews=True, preview_after=0, preview_every=2)
    train(project, config, Recording(), Progress())
    assert seen and all(strength == PREVIEW_LORA_SCALE for strength in seen)
