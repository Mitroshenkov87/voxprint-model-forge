"""CLIP text paths truncate long prompts to the model's 77 tokens."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from voxprint_forge.compare.score import LiveMetrics, clip_text_limit

LIMIT = 77


class FakeProcessor:
    """Tokenizes by words and fails like CLIP when a prompt is longer than 77 tokens."""

    def __call__(self, text=None, images=None, truncation=False, max_length=None, **_):
        if images is not None:
            count = len(images) if isinstance(images, list) else 1
            return {"pixel_values": torch.zeros(count, 3, 4, 4)}
        ids = list(range(len(text[0].split()) + 2))
        if truncation and max_length is not None:
            ids = ids[:max_length]
        if len(ids) > LIMIT:
            raise ValueError(f"Sequence length must be less than max_position_embeddings (got {len(ids)}, max 77)")
        tensor = torch.tensor([ids])
        return {"input_ids": tensor, "attention_mask": torch.ones_like(tensor)}


class FakeClip:
    config = SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=LIMIT))

    def get_image_features(self, pixel_values):
        return torch.ones(pixel_values.shape[0], 8)

    def get_text_features(self, input_ids, attention_mask=None):
        assert input_ids.shape[1] <= LIMIT
        return torch.ones(input_ids.shape[0], 8)


LONG = " ".join(["word"] * 214)
IMAGE = Image.new("RGB", (8, 8))


def test_limit_comes_from_the_model() -> None:
    assert clip_text_limit(FakeClip()) == 77
    assert clip_text_limit(object()) == 77


def test_compare_clip_t_accepts_a_long_prompt() -> None:
    scorer = LiveMetrics.__new__(LiveMetrics)
    scorer.device = torch.device("cpu")
    scorer.clip = FakeClip()
    scorer.clip_processor = FakeProcessor()
    assert abs(scorer._clip_text(IMAGE, LONG) - 1.0) < 1e-6


def test_diffusion_clip_scorer_accepts_a_long_prompt() -> None:
    pytest.importorskip("lpips")
    pytest.importorskip("pillow_avif")
    from voxprint_forge.diffusion.generate import ClipScorer

    scorer = ClipScorer.__new__(ClipScorer)
    scorer.device = torch.device("cpu")
    scorer.model = FakeClip()
    scorer.processor = FakeProcessor()
    assert abs(scorer.text_image(IMAGE, LONG) - 1.0) < 1e-6
