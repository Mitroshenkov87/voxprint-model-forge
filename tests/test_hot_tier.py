"""16 GB VRAM guards and the session RAM park. CUDA stats are mocked."""

from __future__ import annotations

import gc
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from voxprint_common.model_park import RamTier, for_process

import voxprint_forge.lora.vram_stage as vram_stage
import voxprint_forge.model_park as model_park
from voxprint_forge.diffusion import moss
from voxprint_forge.lora import backends, bases, captioner, refset
from voxprint_forge.lora.bases import ZImageAdapter
from voxprint_forge.lora.errors import LoraStageError
from voxprint_forge.lora.layers import inject
from voxprint_forge.lora.presets import build_config
from voxprint_forge.lora.train import VramPhases
from voxprint_forge.lora.vram_stage import (
    assert_stage_clear,
    fail_peak,
    hold_heavy,
    release_heavy,
)
from voxprint_forge.model_park import events, park


@pytest.fixture(autouse=True)
def _clear_park() -> None:
    model_park.reset()
    vram_stage.reset()
    yield
    model_park.reset()
    vram_stage.reset()


def _cuda(monkeypatch, *, allocated: int = 0, peak: int | None = None) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args: allocated)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args: allocated if peak is None else peak)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *args: allocated if peak is None else peak)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)


def test_park_reads_each_key_once() -> None:
    calls: list[int] = []

    def loader() -> dict[str, int]:
        calls.append(1)
        return {"weights": 1}

    assert park("base", loader, stage_name="load") == {"weights": 1}
    assert park("base", loader, stage_name="encode")["weights"] == 1
    assert calls == [1]
    stats = for_process().stats()
    assert stats.disk_reads["base"] == 1
    assert stats.reloads == 1
    rows = events()
    assert [row["action"] for row in rows] == ["park", "hit"]
    assert rows[0]["stage"] == "load" and rows[1]["stage"] == "encode"
    assert rows[0]["seconds"] >= 0
    assert "ram_gb" in rows[0] and "vram_gb" in rows[0]


def test_ram32_keeps_a_model_after_the_parked_handle_is_collected() -> None:
    for_process(RamTier.RAM32, app="model-forge")
    calls: list[int] = []

    def loader() -> dict[str, int]:
        calls.append(1)
        return {"weights": 1}

    held = park("kept", loader, stage_name="load")
    del held
    gc.collect()
    again = park("kept", loader, stage_name="encode")
    stats = for_process().stats()
    assert for_process().tier is RamTier.RAM32
    assert calls == [1]
    assert again["weights"] == 1
    assert stats.disk_reads["kept"] == 1


def test_peak_fails_only_above_14_on_a_16_gb_tier() -> None:
    fail_peak("step", 13.9, tier=16)
    fail_peak("step", 20.0, tier=24)
    with pytest.raises(LoraStageError) as caught:
        fail_peak("step", 14.1, tier=16)
    assert caught.value.key == "lora.err.vram_peak"
    assert caught.value.params["limit"] == 14


def test_near_zero_guard_subtracts_cached_samples(monkeypatch) -> None:
    gib = 2**30
    _cuda(monkeypatch, allocated=int(0.30 * gib))
    with pytest.raises(LoraStageError) as caught:
        assert_stage_clear("encode", tier=16, samples_gb=0.0)
    assert caught.value.key == "lora.err.vram_held"
    _cuda(monkeypatch, allocated=int(0.04 * gib))
    assert_stage_clear("encode", tier=16, samples_gb=0.0)
    _cuda(monkeypatch, allocated=int(0.30 * gib))
    assert_stage_clear("encode", tier=16, samples_gb=0.2)
    _cuda(monkeypatch, allocated=int(8 * gib))
    assert_stage_clear("encode", tier=24, samples_gb=0.0)


def test_one_heavy_model_on_a_16_gb_tier() -> None:
    hold_heavy("text_encoder", tier=16)
    hold_heavy("text_encoder", tier=16)
    with pytest.raises(LoraStageError) as caught:
        hold_heavy("transformer", tier=16)
    assert caught.value.key == "lora.err.vram_one"
    assert caught.value.params["stage"] == "text_encoder"
    hold_heavy("transformer", tier=24)
    release_heavy("text_encoder")
    hold_heavy("transformer", tier=16)
    assert vram_stage.heavy() == "transformer"


def test_phase_mark_fails_when_the_mocked_peak_passes_14(monkeypatch) -> None:
    _cuda(monkeypatch, peak=int(14.2 * 2**30))
    phases = VramPhases(tier=16)
    with pytest.raises(LoraStageError) as caught:
        phases.mark("preview", "step")
    assert caught.value.key == "lora.err.vram_peak"
    assert phases.peaks["preview"]["alloc_gb"] > 14
    VramPhases(tier=24).mark("preview", "step")


def test_zimage_16_without_fp8_does_not_read_weights(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "diffusers.ZImagePipeline.from_pretrained", lambda *args, **kwargs: pytest.fail("read the base")
    )
    weights = tmp_path / "z"
    weights.mkdir()
    config = build_config("quick", "balanced", 16, base_fp8=False)
    with pytest.raises(LoraStageError) as caught:
        ZImageAdapter().load(weights, config, "cpu")
    assert caught.value.key == "lora.err.vram_quant"


def test_zimage_second_load_reuses_the_park_and_can_inject(monkeypatch, tmp_path: Path) -> None:
    calls: list[int] = []

    class Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = nn.Module()
            self.attention.to_q = nn.Linear(4, 4)

        def enable_layerwise_casting(self, **kwargs: object) -> None:
            self.cast = kwargs

        def enable_gradient_checkpointing(self) -> None:
            self.checkpointed = True

    class Pipe:
        def __init__(self) -> None:
            self.transformer = Transformer()
            self.text_encoder = nn.Linear(4, 4)
            self.vae = nn.Linear(4, 4)

    def from_pretrained(*args: object, **kwargs: object) -> Pipe:
        calls.append(1)
        return Pipe()

    monkeypatch.setattr("diffusers.ZImagePipeline.from_pretrained", from_pretrained)
    weights = tmp_path / "z"
    weights.mkdir()
    config = build_config("quick", "balanced", 16, steps=1, resolution=64, rank=2, alpha=2)
    adapter = ZImageAdapter()
    first = adapter.load(weights, config, "cpu")
    inject(first, adapter.target_modules, config.rank, config.alpha)
    second = adapter.load(weights, config, "cpu")
    layers = inject(second, adapter.target_modules, config.rank, config.alpha)
    assert calls == [1]
    assert layers
    assert adapter.pipe.transformer.cast["storage_dtype"] == torch.float8_e4m3fn
    assert [row["action"] for row in events()].count("park") == 1


def test_preview_on_16_gb_does_not_bring_the_text_encoder_back() -> None:
    adapter = ZImageAdapter()
    adapter._vram = 16
    moved: list[str] = []

    class Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(1, 1)

        def to(self, device, *args, **kwargs):  # type: ignore[no-untyped-def]
            moved.append(str(device))
            return super().to(device, *args, **kwargs)

    adapter.pipe = SimpleNamespace(text_encoder=Encoder(), vae=nn.Linear(1, 1))
    adapter._prompts = {}
    with pytest.raises(LoraStageError) as caught:
        adapter.preview(nn.Linear(1, 1), "unseen prompt", 0, SimpleNamespace())
    assert caught.value.key == "lora.err.vram_one"
    assert moved == []


def test_moss_16_errors_before_the_dit_is_read(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(moss, "load_pipeline", lambda *args, **kwargs: pytest.fail("read the DiT"))
    adapter = bases.get_adapter("moss-image-base")
    with pytest.raises(LoraStageError) as caught:
        adapter.load(tmp_path, build_config("quick", "balanced", 16), "cpu")
    assert caught.value.key == "lora.err.vram_quant"


def test_qwen_stays_resident_after_unload(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(backends, "detect_vram_gb", lambda: 24)
    calls: list[str] = []

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(1))

    def load(folder: Path, device: str, mode: str) -> dict[str, object]:
        calls.append(mode)
        return {"processor": object(), "model": Model(), "quant": mode}

    monkeypatch.setattr(backends, "_load_qwen", load)
    folder = tmp_path / "qwen"
    folder.mkdir()
    first = backends.VlmJudge(folder, "cpu", quant="none")
    model = first.model
    first.unload()
    assert first.model is model
    second = backends.VlmJudge(folder, "cpu", quant="none")
    captioner.QwenCaptioner(folder, "cpu")
    assert calls == ["none"]
    assert second.model is model
    assert "hit" in [row["action"] for row in events()]


def test_qwen_16_refuses_bf16_and_cpu(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(backends, "detect_vram_gb", lambda: 16)
    monkeypatch.setattr(backends, "_load_qwen", lambda *args, **kwargs: pytest.fail("read qwen"))
    folder = tmp_path / "qwen"
    folder.mkdir()
    with pytest.raises(LoraStageError) as quant:
        backends.VlmJudge(folder, "cuda", quant="none")
    assert quant.value.key == "lora.err.vram_quant"
    with pytest.raises(LoraStageError) as cpu:
        backends.VlmJudge(folder, "cpu")
    assert cpu.value.key == "lora.err.cpu_fallback"


def test_load_gate_reraises_stage_errors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(backends, "installed_dir", lambda mdl_id: tmp_path)

    def boom(folder: Path, device: str = "cuda", quant: str | None = None) -> None:
        raise LoraStageError("lora.err.vram_quant", model="qwen3-vl-4b-instruct")

    monkeypatch.setattr(backends, "VlmJudge", boom)
    with pytest.raises(LoraStageError) as caught:
        backends.load_gate("cpu")
    assert caught.value.key == "lora.err.vram_quant"


def test_captioner_on_16_gb_does_not_substitute_florence(monkeypatch) -> None:
    monkeypatch.setattr(captioner, "detect_vram_gb", lambda: 16)

    def folder(name: str) -> Path | None:
        if name == captioner.FALLBACK:
            return Path("/florence")
        return None

    monkeypatch.setattr(captioner, "installed_dir", folder)
    monkeypatch.setitem(captioner._LOADERS, captioner.FALLBACK, lambda path: pytest.fail("florence"))
    assert captioner.load_captioner(captioner.DEFAULT) is None

    def broken(path: Path) -> None:
        raise LoraStageError("lora.err.te_quant_missing", mode="int8")

    monkeypatch.setattr(captioner, "installed_dir", lambda name: Path("/qwen"))
    monkeypatch.setitem(captioner._LOADERS, captioner.DEFAULT, broken)
    with pytest.raises(LoraStageError) as caught:
        captioner.load_captioner(captioner.DEFAULT)
    assert caught.value.key == "lora.err.te_quant_missing"


class _Image:
    size = (32, 32)

    def convert(self, mode: str) -> _Image:
        return self

    def resize(self, size: tuple[int, int]) -> _Image:
        return self


def test_turbo_16_parks_once_casts_fp8_and_fails_a_high_peak(monkeypatch, tmp_path: Path) -> None:
    disks: list[int] = []

    class _Place:
        """A stand-in module. ``.to`` records the device and does not touch CUDA."""

        def __init__(self) -> None:
            self.devices: list[object] = []

        def to(self, device: object) -> _Place:
            self.devices.append(device)
            return self

    class Transformer(_Place):
        def enable_layerwise_casting(self, **kwargs: object) -> None:
            self.cast = kwargs

    class Pipe:
        def __init__(self) -> None:
            self.transformer = Transformer()
            self.vae = _Place()
            self.text_encoder = _Place()

        def set_progress_bar_config(self, **kwargs: object) -> None:
            return None

        def enable_model_cpu_offload(self) -> None:
            raise AssertionError("16 GB must not offload the preview pipeline to the CPU")

        def __call__(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(images=["picture"])

    def from_pretrained(*args: object, **kwargs: object) -> Pipe:
        disks.append(1)
        return Pipe()

    monkeypatch.setattr(refset, "detect_vram_gb", lambda: 16)
    monkeypatch.setattr("diffusers.ZImageImg2ImgPipeline.from_pretrained", from_pretrained)
    weights = tmp_path / "turbo"
    weights.mkdir()
    _cuda(monkeypatch, allocated=0, peak=int(14.2 * 2**30))
    teacher = refset.TurboImg2Img(weights, "cuda")
    again = refset.TurboImg2Img(weights, "cuda")
    assert disks == [1]
    assert again.pipe is teacher.pipe
    assert teacher.pipe.transformer.cast["storage_dtype"] == torch.float8_e4m3fn
    with pytest.raises(LoraStageError) as caught:
        teacher.generate(_Image(), "a face", seed=1, strength=0.5, steps=4)
    assert caught.value.key == "lora.err.vram_peak"
    teacher.unload()
    assert teacher.pipe is again.pipe
    actions = [row["action"] for row in events()]
    assert "park" in actions and "unpark" in actions


def test_turbo_without_fp8_is_an_error(monkeypatch, tmp_path: Path) -> None:
    disks: list[int] = []

    class Pipe:
        def __init__(self) -> None:
            self.transformer = nn.Linear(1, 1)

    def from_pretrained(*args: object, **kwargs: object) -> Pipe:
        disks.append(1)
        return Pipe()

    monkeypatch.setattr(refset, "detect_vram_gb", lambda: 16)
    monkeypatch.setattr("diffusers.ZImageImg2ImgPipeline.from_pretrained", from_pretrained)
    weights = tmp_path / "turbo"
    weights.mkdir()
    with pytest.raises(LoraStageError) as caught:
        refset.TurboImg2Img(weights, "cuda")
    assert caught.value.key == "lora.err.vram_quant"
    with pytest.raises(LoraStageError):
        refset.TurboImg2Img(weights, "cuda")
    assert disks == [1, 1]


def test_turbo_24_still_offloads(monkeypatch, tmp_path: Path) -> None:
    offload: list[str] = []

    class Pipe:
        def set_progress_bar_config(self, **kwargs: object) -> None:
            return None

        def enable_model_cpu_offload(self) -> None:
            offload.append("offload")

        def to(self, device: object) -> Pipe:
            offload.append(str(device))
            return self

    monkeypatch.setattr(refset, "detect_vram_gb", lambda: 24)
    monkeypatch.setattr("diffusers.ZImageImg2ImgPipeline.from_pretrained", lambda *args, **kwargs: Pipe())
    weights = tmp_path / "turbo"
    weights.mkdir()
    teacher = refset.TurboImg2Img(weights, "cuda")
    refset.TurboImg2Img(weights, "cuda")
    assert offload == ["offload"]
    teacher.unload()
    assert teacher.pipe is not None
