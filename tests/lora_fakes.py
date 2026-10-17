"""Tiny CPU stand-ins for LoRA tests: a random 2-layer model and procedural images."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from voxprint_forge.lora.bases import BaseAdapter, BaseInfo, Sample
from voxprint_forge.lora.project import Item, Project


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(16, 16)
        self.to_v = nn.Linear(16, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_v(torch.tanh(self.to_q(x)))


class TinyAdapter(BaseAdapter):
    info = BaseInfo("tiny-test", "example/tiny", "0" * 40, "Apache-2.0", "Tiny")
    target_modules = ("to_q", "to_v")
    requires_gpu = False
    nan_after: int | None = None

    def __init__(self) -> None:
        self.calls = 0
        self.released = False

    def load(self, weights_dir, config, device):
        torch.manual_seed(1234)
        return TinyNet()

    def encode(self, images, captions, config):
        out = []
        for image, caption in zip(images, captions, strict=True):
            small = image.convert("L").resize((4, 4))
            latents = torch.from_numpy(np.asarray(small, dtype=np.float32).reshape(-1)) / 255.0
            encoded = {k: _embed(v) for k, v in caption.items() if v}
            out.append(Sample(latents, encoded))
        return out

    def release_encoders(self):
        self.released = True

    def loss(self, model, batch, caption_keys, rng):
        self.calls += 1
        if self.nan_after is not None and self.calls > self.nan_after:
            return torch.tensor(float("nan"), requires_grad=True)
        x = torch.stack([s.latents for s in batch])
        cond = torch.stack(
            [s.captions.get(k, torch.zeros(16)) if k else torch.zeros(16) for s, k in zip(batch, caption_keys)]
        )
        noise = torch.randn_like(x)
        return torch.nn.functional.mse_loss(model(x + 0.1 * noise + cond), x)

    def seconds_per_step(self, config):
        return 0.5

    def vram_need_gb(self, config):
        return 0.0


def _embed(text: str) -> torch.Tensor:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(16, generator=generator) * 0.1


def gradient(seed: int, size: int = 64) -> Image.Image:
    rnd = random.Random(seed)
    a, b = [tuple(rnd.randrange(256) for _ in range(3)) for _ in range(2)]
    image = Image.new("RGB", (size, size))
    image.putdata(
        [
            tuple(int(a[c] + (b[c] - a[c]) * ((x + y) / (2 * size))) for c in range(3))
            for y in range(size)
            for x in range(size)
        ]
    )
    return image


def make_project(
    root: Path, count: int = 10, *, gate_passed: bool = True, mode: str = "simple", rounds=None
) -> Project:
    project = Project(root)
    project.prepared.mkdir(parents=True, exist_ok=True)
    items = []
    for index in range(count):
        name = f"prepared/p{index:02d}.png"
        gradient(index).save(root / name)
        rnd = 0 if rounds is None else rounds[index]
        items.append(Item(f"p{index:02d}", name, f"a long description {index}", f"short {index}", rnd))
    project.write_items(items)
    project.write(
        {
            "kind": "object",
            "mode": mode,
            "gate_passed": gate_passed,
            "subject": {"name": "Test cube", "trigger": "tcube"},
        }
    )
    return project
