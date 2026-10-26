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
from voxprint_forge.diffusion import moss, moss_fp8
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


def test_moss_16_parks_the_fp8_dit_and_attaches_the_encoder_per_load(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(moss, "load_pipeline", lambda *args, **kwargs: pytest.fail("read the bf16 DiT"))
    built: list[nn.Module] = []

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList()
            self.anchor = nn.Linear(2, 2)

    def fake_transformer(root: Path) -> nn.Module:
        module = Tiny()
        built.append(module)
        return module

    assemblies: list[bool] = []

    def fake_assemble(
        root: Path,
        transformer: nn.Module,
        *,
        with_text_encoder: bool,
        te_quant: str,
        te_device: str,
    ) -> SimpleNamespace:
        del root, te_quant, te_device
        assemblies.append(with_text_encoder)
        encoder = nn.Linear(1, 1) if with_text_encoder else None
        return SimpleNamespace(transformer=transformer, vae=nn.Linear(1, 1), text_encoder=encoder)

    monkeypatch.setattr("voxprint_forge.diffusion.moss_fp8.load_fp8_transformer", fake_transformer)
    monkeypatch.setattr("voxprint_forge.diffusion.moss_fp8.assemble_pipe", fake_assemble)
    adapter = bases.get_adapter("moss-image-base")
    monkeypatch.setattr(adapter, "_check", lambda _weights, _config: SimpleNamespace(quant="int8", device="cuda"))
    config = build_config("quick", "balanced", 16)
    first = adapter.load(tmp_path, config, "cpu")
    assert adapter.pipe.text_encoder is not None
    adapter._staged = True
    second = adapter.load(tmp_path, config, "cpu")
    assert first is second
    assert adapter.pipe.text_encoder is None
    third = adapter.load(tmp_path, config, "cpu")
    assert third is first
    assert adapter.pipe.text_encoder is not None
    assert len(built) == 1
    assert assemblies == [True, False, True]
    assert "hit" in [row["action"] for row in events()]


def test_unstaged_fp8_load_reuses_the_encoder_after_the_dit_leaves_the_card(monkeypatch, tmp_path: Path) -> None:
    order: list[tuple[str, object]] = []

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList()
            self.anchor = nn.Linear(2, 2)

    module = Tiny()

    def fake_transformer(root: Path) -> nn.Module:
        del root
        return module

    real_bind = moss_fp8.bind_device_move

    def tracking_bind(transformer: nn.Module) -> nn.Module:
        real_bind(transformer)
        original = transformer.to

        def wrapped(*args: object, **kwargs: object) -> nn.Module:
            order.append(("to", args[0] if args else kwargs.get("device")))
            return original(*args, **kwargs)

        transformer.to = wrapped  # type: ignore[method-assign]
        return transformer

    def fake_assemble(
        root: Path,
        transformer: nn.Module,
        *,
        with_text_encoder: bool,
        te_quant: str,
        te_device: str,
    ) -> SimpleNamespace:
        del root, transformer, te_quant, te_device
        order.append(("assemble", with_text_encoder))
        encoder = nn.Linear(1, 1) if with_text_encoder else None
        return SimpleNamespace(transformer=module, vae=nn.Linear(1, 1), text_encoder=encoder)

    monkeypatch.setattr("voxprint_forge.diffusion.moss_fp8.load_fp8_transformer", fake_transformer)
    monkeypatch.setattr("voxprint_forge.diffusion.moss_fp8.bind_device_move", tracking_bind)
    monkeypatch.setattr("voxprint_forge.diffusion.moss_fp8.assemble_pipe", fake_assemble)
    adapter = bases.get_adapter("moss-image-base")
    monkeypatch.setattr(adapter, "_check", lambda _weights, _config: SimpleNamespace(quant="int8", device="cuda"))
    resident = nn.Linear(1, 1)
    adapter._resident_te = resident
    returned = adapter.load(tmp_path, build_config("quick", "balanced", 16), "cpu")
    assert returned is module
    assert ("to", "cpu") in order
    assert order.index(("to", "cpu")) < order.index(("assemble", False))
    assert ("assemble", True) not in order
    assert adapter.pipe.text_encoder is resident


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


class _CudaBlob:
    """One graphics allocation. ``to`` returns a new blob and leaves this one in place."""

    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes
        self.cuda = True

    @property
    def is_cuda(self) -> bool:
        return self.cuda

    @property
    def data(self) -> _CudaBlob:
        return self

    @data.setter
    def data(self, value: object) -> None:
        cuda = getattr(value, "is_cuda", None)
        if isinstance(cuda, bool):
            self.cuda = cuda

    def to(self, device: object, *args: object, **kwargs: object) -> _CudaBlob:
        kind = device.split(":", 1)[0] if isinstance(device, str) else getattr(device, "type", "")
        moved = _CudaBlob(self.nbytes)
        moved.cuda = str(kind).startswith("cuda")
        return moved


class _Card:
    """Allocated bytes are the blobs still marked CUDA. ``empty_cache`` does not clear them."""

    def __init__(self) -> None:
        self.blobs: list[_CudaBlob] = []

    def tensor(self, nbytes: int) -> _CudaBlob:
        blob = _CudaBlob(nbytes)
        self.blobs.append(blob)
        return blob

    def allocated(self) -> int:
        return sum(blob.nbytes for blob in self.blobs if blob.cuda)


def test_caption_unload_drops_the_int8_buffer_and_keeps_the_baseline_guard(
    monkeypatch, tmp_path: Path
) -> None:
    """8-bit Qwen leaves the weights on ``state.CB`` after ``module.to``.

    ``nn.Module.to`` moves parameters and does not visit that buffer, so unload
    has to drop it, collect, and empty the cache. The guard then measures
    ``memory_allocated`` against the pre-load baseline and still rejects
    anything more than 0.25 GB above it, including a blob this unload cannot
    see. Reserved memory is not the measurement.
    """
    gib = 2**30
    allowance = int(vram_stage.NEAR_ZERO_GB * gib)
    card = _Card()
    # Resident before the captioner loads (CUDA context). Above 0.25 GB so an
    # absolute check would fail even after the weights leave.
    context = card.tensor(int(0.40 * gib))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args: card.allocated())
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *args: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(backends, "detect_vram_gb", lambda: 16)
    monkeypatch.setattr(backends, "check_available", lambda mode: None)
    calls: list[str] = []

    class Root(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = nn.Linear(1, 1)
            # After the first forward, Linear8bitLt keeps the int8 weights here.
            # nn.Module.to moves parameters and never visits this attribute.
            self.block.state = SimpleNamespace(CB=card.tensor(int(3.77 * gib)))

        def to(self, *args: object, **kwargs: object) -> Root:
            return self

    class Processor:
        def to(self, device: object) -> Processor:
            return self

    def load(folder: Path, device: str, mode: str) -> dict[str, object]:
        calls.append(mode)
        return {"processor": Processor(), "model": Root(), "quant": mode}

    monkeypatch.setattr(backends, "_load_qwen", load)
    folder = tmp_path / "qwen"
    folder.mkdir()
    judge = backends.VlmJudge(folder, "cuda", quant="int8")
    model = judge.model
    assert card.allocated() > context.nbytes
    judge.unload()
    assert card.allocated() == context.nbytes
    assert judge.model is model
    again = backends.VlmJudge(folder, "cuda", quant="int8")
    assert calls == ["int8"]
    assert again.model is model
    assert card.allocated() == context.nbytes + int(3.77 * gib)
    again.unload()
    assert card.allocated() == context.nbytes

    lingering = card.tensor(allowance + 1)
    with pytest.raises(LoraStageError) as caught:
        assert_stage_clear("caption", tier=16, baseline_gb=context.nbytes / gib)
    assert caught.value.key == "lora.err.vram_held"
    assert vram_stage.NEAR_ZERO_GB == 0.25
    lingering.cuda = False
    assert_stage_clear("caption", tier=16, baseline_gb=context.nbytes / gib)


def test_int8_restage_keeps_one_copy_of_the_weight(monkeypatch) -> None:
    """``module.to`` copies the parameter and leaves ``state.CB``. Retie drops the extra."""
    gib = 2**30
    card = _Card()

    class _DType:
        def is_floating_point(self) -> bool:
            return False

    class _Weight:
        has_fp16_weights = False

        def __init__(self, blob: _CudaBlob) -> None:
            self._data = blob
            self.SCB = None

        @property
        def data(self) -> _CudaBlob:
            return self._data

        @data.setter
        def data(self, value: object) -> None:
            if self._data is not value and isinstance(self._data, _CudaBlob):
                self._data.cuda = False
            if isinstance(value, _CudaBlob):
                self._data = value

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            weight = card.tensor(int(3.77 * gib))
            weight.dtype = _DType()
            self.weight = _Weight(weight)
            self.state = SimpleNamespace(CB=card.tensor(int(3.77 * gib)), has_fp16_weights=False)

        def to(self, device: object) -> Block:
            # Parameter copy, the way ``nn.Module._apply`` moves a weight. CB stays put.
            kind = device.split(":", 1)[0] if isinstance(device, str) else getattr(device, "type", "")
            self.weight.data.cuda = str(kind).startswith("cuda")
            return self

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    block = Block()
    assert card.allocated() == 2 * int(3.77 * gib)
    model_park.drop_vram(block, key="int8-copy", stage_name="caption")
    assert card.allocated() == 0
    model_park.stage(block, "cuda", key="int8-copy", stage_name="caption")
    assert card.allocated() == int(3.77 * gib)
    assert block.weight.data is block.state.CB

    class Refused(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.state = SimpleNamespace(CB=card.tensor(int(3.77 * gib)))

        def to(self, device: object) -> Refused:
            raise ValueError("`.to` is not supported for 4-bit or 8-bit bitsandbytes models")

    refused = Refused()
    assert card.allocated() == 2 * int(3.77 * gib)
    model_park.drop_vram(refused, key="int8-refused", stage_name="caption")
    assert card.allocated() == int(3.77 * gib)


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
