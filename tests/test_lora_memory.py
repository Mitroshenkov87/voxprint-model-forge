"""Memory plan and fail-fast checks for LoRA training (CALIB-R2/R3: the transformer OOMed on a pinned text encoder)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tests.lora_fakes import TinyAdapter, make_project
from voxprint_forge.lora import memory
from voxprint_forge.lora.bases import ZImageAdapter
from voxprint_forge.lora.errors import LoraStageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.presets import build_config
from voxprint_forge.lora.train import estimate, train


@pytest.fixture(autouse=True)
def _no_installed_weights(monkeypatch):
    monkeypatch.setattr("voxprint_forge.lora.bases._installed", lambda: None)


def test_24_gb_plan_fits_22_gib_under_the_23_5_cap() -> None:
    plan = ZImageAdapter().memory_plan(build_config("balanced", "balanced", 24), samples=7)
    assert 11.4 < plan.transformer_gb < 11.5  # BF16 Z-Image transformer, 12.31 GB
    assert plan.peak_gb <= 22.0
    assert plan.train_gb > plan.encode_gb
    memory.check_plan(plan, 23.5, 93.1)


def test_16_gb_plan_fits_a_16_gb_card() -> None:
    plan = ZImageAdapter().memory_plan(build_config("balanced", "balanced", 16), samples=7)
    assert plan.transformer_gb < 6.0  # FP8 storage
    memory.check_plan(plan, None, 15.9)
    assert plan.peak_gb <= 14.5


def test_plan_reproduces_the_uncheckpointed_first_calibration() -> None:
    # synth4 LORA-CALIB: batch 2 at 1024 px without checkpointing reached 79.3 GB of activations.
    config = build_config("balanced", "balanced", 24, batch=2, grad_accum=1, grad_checkpointing=False)
    plan = ZImageAdapter().memory_plan(config)
    assert 79.3 * 0.9 < plan.activations_gb * memory.GIB / 1e9 < 79.3 * 1.1
    with pytest.raises(LoraStageError) as caught:
        memory.check_plan(plan, 23.5, 93.1)
    assert caught.value.key == "lora.err.vram"


def test_vram_need_follows_the_plan() -> None:
    adapter = ZImageAdapter()
    assert adapter.vram_need_gb(build_config("balanced", "balanced", 16)) < 16
    assert adapter.vram_need_gb(build_config("balanced", "balanced", 24)) <= 23


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4)  # requires_grad, as a loaded encoder has before load() freezes it


class _Pipe:
    def __init__(self) -> None:
        self.text_encoder, self.vae, self.transformer = _Encoder(), _Encoder(), _Encoder()

    def encode_prompt(self, prompt, device, do_classifier_free_guidance):
        return [self.text_encoder.proj(torch.ones(3, 4))], None


def test_preview_embeddings_carry_no_graph() -> None:
    # The R2/R3 leak: prepare_previews ran outside no_grad, so each cached embedding kept the
    # encoder's card weights and every layer's activations alive through its grad_fn.
    adapter = ZImageAdapter()
    adapter.pipe, adapter.device = _Pipe(), "cpu"
    adapter.prepare_previews(["a red cube"])
    for value in adapter._prompts.values():
        assert value.grad_fn is None and not value.requires_grad


def _fake_cuda(monkeypatch, allocated_gb: float) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: int(allocated_gb * memory.GIB))
    monkeypatch.setattr(memory, "card_gb", lambda: 93.1)
    monkeypatch.setenv("FORGE_VRAM_CAP_GB", "23.5")


def _released(monkeypatch, allocated_gb: float) -> list[str]:
    moved: list[str] = []
    adapter = ZImageAdapter()
    adapter.pipe = _Pipe()
    adapter.pipe.transformer.to = lambda device: moved.append(device)  # type: ignore[method-assign]
    adapter._plan = adapter.memory_plan(build_config("balanced", "balanced", 24), 7)
    adapter._samples_bytes = 40 * 2**20
    _fake_cuda(monkeypatch, allocated_gb)
    adapter.release_encoders()
    return moved


def test_release_stops_on_a_pinned_encoder_before_the_transformer_moves(monkeypatch) -> None:
    moved: list[str] = []
    with pytest.raises(LoraStageError) as caught:
        moved = _released(monkeypatch, 16.0)  # what R3 left allocated after the encoders "left"
    assert caught.value.key == "lora.err.encoders_stuck" and moved == []


def test_release_moves_the_transformer_when_only_the_caches_remain(monkeypatch) -> None:
    assert _released(monkeypatch, 0.05) == ["cuda"]


def test_train_fails_fast_when_the_plan_exceeds_the_cap(monkeypatch, tmp_path) -> None:
    loaded = []

    class Big(TinyAdapter):
        requires_gpu = True

        def load(self, weights_dir, config, device):
            loaded.append(device)
            return super().load(weights_dir, config, device)

        def memory_plan(self, config, samples=0, weights_dir=None):
            return memory.MemoryPlan(8.0, 1.0, 11.5, 0.6, 9.0, 0.1)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: type("P", (), {"total_memory": 93 * 2**30})())
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda *a: None)
    monkeypatch.setenv("FORGE_VRAM_CAP_GB", "23.5")
    config = build_config("quick", "balanced", 24, steps=2, resolution=64, save_every=2, previews=False)
    with pytest.raises(LoraStageError) as caught:
        train(make_project(tmp_path), config, Big(), Progress())
    assert caught.value.key == "lora.err.vram" and loaded == []


def test_dry_run_reports_the_plan(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("FORGE_VRAM_CAP_GB", "23.5")
    result = estimate(make_project(tmp_path), build_config("balanced", "balanced", 24), ZImageAdapter())
    assert result["memory_plan"]["peak_gb"] <= 22.0 and result["plan_fits_cap"] is True
