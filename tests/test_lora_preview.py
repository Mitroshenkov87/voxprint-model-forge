"""Preview device regression (CALIB-R4): offloaded text encoder vs. a generator on the card.

diffusers infers the execution device from the first pipeline module in sorted order, which is
the text encoder, parked on the CPU during training. R4 then drew latents on the CPU with a
CUDA generator and crashed. The fake pipe below reproduces that inference rule on a CPU-only
box: the parked encoder sits on ``meta`` while the transformer and VAE sit on the training
device (``cpu`` here), so any regression shows up as a device mismatch without a GPU.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from voxprint_forge.lora.bases import ZImageAdapter, _check_execution_device
from voxprint_forge.lora.errors import LoraStageError
from voxprint_forge.lora.presets import build_config


class _Module(nn.Module):
    def __init__(self, device: str) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, device=device), requires_grad=False)

    def to(self, *args, **kwargs):  # the adapter parks/unparks modules; keep the configured device
        return self


class _Image:
    def save(self, path) -> None:  # pragma: no cover - not reached in these tests
        pass


class _Result:
    def __init__(self) -> None:
        self.images = [_Image()]


class _DiffusersLikePipe:
    """Device inference and latent sampling as in diffusers 0.37 (sorted modules; randn_tensor)."""

    def __init__(self, encoder_device: str, card: str) -> None:
        self.text_encoder = _Module(encoder_device)
        self.transformer = _Module(card)
        self.vae = _Module(card)
        self.calls: list[dict] = []

    @property
    def _execution_device(self) -> torch.device:
        for name in sorted(("text_encoder", "tokenizer", "transformer", "vae", "scheduler")):
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                return next(module.parameters()).device
        return torch.device("cpu")

    def __call__(
        self, *, prompt_embeds, negative_prompt_embeds, height, width, generator, num_inference_steps=None,
    ):
        device = self._execution_device
        shape = (1, 16, 2 * (height // 16), 2 * (width // 16))
        # randn_tensor: a CPU generator draws on the CPU and moves; any other generator must
        # match the target device, or torch raises (R4's "Cannot generate a cpu tensor ...").
        if generator.device.type == "cpu":
            latents = torch.randn(shape, generator=generator).to(device)
        elif generator.device.type != device.type:
            raise RuntimeError(
                f"Cannot generate a {device.type} tensor from a generator of type {generator.device.type}."
            )
        else:
            latents = torch.randn(shape, generator=generator, device=device)
        card = next(self.transformer.parameters()).device
        if latents.device != card or prompt_embeds[0].device != card:
            raise RuntimeError(f"Expected all tensors to be on the same device ({latents.device} vs {card})")
        self.calls.append({
            "device": device,
            "generator": generator.device.type,
            "latents": latents,
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
        })
        return _Result()


def _adapter(encoder_device: str = "meta", card: str = "cpu") -> ZImageAdapter:
    adapter = ZImageAdapter()
    adapter.pipe, adapter.device = _DiffusersLikePipe(encoder_device, card), card
    adapter._prompts = {"a red cube": torch.ones(3, 4), "": torch.zeros(3, 4)}
    return adapter


@pytest.mark.parametrize("vram", [24, 16])
def test_preview_runs_with_the_text_encoder_parked_elsewhere(vram) -> None:
    adapter = _adapter()
    config = build_config("balanced", "balanced", vram)
    image = adapter.preview(nn.Identity(), "a red cube", 7, config)
    assert image is not None
    (call,) = adapter.pipe.calls
    assert call["device"].type == "cpu" and call["generator"] == "cpu"
    # The encoder is visible to the pipeline again afterwards.
    assert isinstance(adapter.pipe.text_encoder, nn.Module)


def test_preview_noise_is_seeded_and_device_independent() -> None:
    config = build_config("balanced", "balanced", 24)
    first, second = _adapter(), _adapter()
    first.preview(nn.Identity(), "a red cube", 11, config)
    second.preview(nn.Identity(), "a red cube", 11, config)
    assert torch.equal(first.pipe.calls[0]["latents"], second.pipe.calls[0]["latents"])


def test_encoder_is_restored_when_the_preview_raises() -> None:
    adapter = _adapter()

    def boom(**kwargs):
        raise RuntimeError("decode failed")

    adapter.pipe.__class__ = type("Boom", (_DiffusersLikePipe,), {"__call__": lambda self, **kw: boom(**kw)})
    with pytest.raises(RuntimeError):
        adapter.preview(nn.Identity(), "a red cube", 1, build_config("balanced", "balanced", 24))
    assert isinstance(adapter.pipe.text_encoder, nn.Module)


def test_review_samples_pass_an_explicit_step_count_and_the_training_resolution() -> None:
    adapter = _adapter()
    config = build_config("balanced", "balanced", 24)
    adapter.preview(nn.Identity(), "a red cube", 7, config, steps=28)
    call = adapter.pipe.calls[-1]
    assert call["num_inference_steps"] == 28
    assert call["height"] == config.resolution == call["width"]
    adapter.preview(nn.Identity(), "a red cube", 8, config)
    assert adapter.pipe.calls[-1]["num_inference_steps"] is None


def test_execution_device_mismatch_is_a_named_error() -> None:
    pipe = _DiffusersLikePipe("meta", "meta")
    with pytest.raises(LoraStageError) as caught:
        _check_execution_device(pipe, "cpu")
    assert caught.value.key == "lora.err.preview_device"
