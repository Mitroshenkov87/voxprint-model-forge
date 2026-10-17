"""Base-model adapters. The engine talks only to :class:`BaseAdapter`; the base is picked by MDL id.

Built in: Z-Image (``z-image``, the Canvas hand-off base, default) and
MOSS-Image Base (``moss-image-base``, internal). Both are Apache-2.0. A
restrictive base may only be an optional MDL link to its official source.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from voxprint_forge.lora.errors import LoraStageError, LoraUsageError
from voxprint_forge.lora.presets import TrainConfig

PERMISSIVE = frozenset({"apache-2.0", "mit", "bsd-3-clause", "bsd-2-clause"})


@dataclass(frozen=True)
class BaseInfo:
    """Registry row: MDL id, official source, licence."""

    mdl_id: str
    repo: str
    revision: str
    licence: str
    display_name: str

    @property
    def permissive(self) -> bool:
        """True when trained LoRAs may be shared (``lora.err.base_restrictive`` otherwise)."""
        return self.licence.lower() in PERMISSIVE


@dataclass
class Sample:
    """One encoded training item: latents and the encoded caption choices."""

    latents: torch.Tensor
    captions: dict[str, Any]
    round: int = 0


class BaseAdapter(ABC):
    """What the trainer needs from a base model. Adapters hold no hardcoded weight paths."""

    info: BaseInfo
    target_modules: tuple[str, ...] = ()
    requires_gpu: bool = True
    trainable: bool = True  # False while the adapter has no training loader yet

    @abstractmethod
    def load(self, weights_dir: Path, config: TrainConfig, device: str) -> nn.Module:
        """Load the base and return the trainable transformer (encoders stay inside the adapter)."""

    @abstractmethod
    def encode(self, images: Sequence[Any], captions: Sequence[dict[str, str]], config: TrainConfig) -> list[Sample]:
        """Encode prepared images and captions once, before training."""

    def release_encoders(self) -> None:
        """Free encoder memory after :meth:`encode`. Optional."""

    @abstractmethod
    def loss(
        self, model: nn.Module, batch: Sequence[Sample], caption_keys: Sequence[str], rng: random.Random
    ) -> torch.Tensor:
        """Training loss for one micro-batch."""

    def prepare_previews(self, prompts: Sequence[str]) -> None:
        """Encode the test prompts while the encoders are still loaded. Optional."""

    def preview(self, model: nn.Module, prompt: str, seed: int, config: TrainConfig) -> Any | None:
        """A test picture (PIL image) or ``None`` when the adapter cannot draw one."""
        return None

    @abstractmethod
    def seconds_per_step(self, config: TrainConfig) -> float:
        """Estimated wall time per optimizer step on the chosen memory tier."""

    @abstractmethod
    def vram_need_gb(self, config: TrainConfig) -> float:
        """Estimated graphics memory for ``config``."""


_REGISTRY: dict[str, type[BaseAdapter]] = {}
DEFAULT_BASE = "z-image"


def register(adapter: type[BaseAdapter]) -> type[BaseAdapter]:
    """Add an adapter class to the registry under its MDL id."""
    _REGISTRY[adapter.info.mdl_id] = adapter
    return adapter


def unregister(mdl_id: str) -> None:
    """Remove an adapter (tests)."""
    _REGISTRY.pop(mdl_id, None)


def available() -> list[BaseInfo]:
    """Registered bases."""
    return [cls.info for cls in _REGISTRY.values()]


def trainable() -> list[BaseInfo]:
    """Registered bases this build can actually train on (what the app offers)."""
    return [cls.info for cls in _REGISTRY.values() if cls.trainable]


def get_adapter(mdl_id: str) -> BaseAdapter:
    """A fresh adapter for ``mdl_id``. Unknown ids raise ``lora.err.base_missing``."""
    cls = _REGISTRY.get(mdl_id)
    if cls is None:
        raise LoraUsageError("lora.err.base_missing")
    return cls()


def weights_dir(mdl_id: str) -> Path:
    """Where MDL installs a base for Model Forge: ``<models_dir>/model-forge/<mdl_id>``."""
    from voxprint_common.paths import models_dir

    return models_dir() / "model-forge" / mdl_id


def _dtype(config: TrainConfig) -> torch.dtype:
    return torch.bfloat16 if config.precision == "bf16" else torch.float16


class ZImageAdapter(BaseAdapter):
    """Z-Image (Tongyi-MAI/Z-Image, Apache-2.0), flow matching, via diffusers ``ZImagePipeline``.

    LoRAs trained here are for Z-Image. Loading them on Z-Image-Turbo is
    untested; revisit after a Canvas test. The real path needs a CUDA GPU; CI
    covers the engine with a tiny stand-in.
    """

    info = BaseInfo("z-image", "Tongyi-MAI/Z-Image", "main", "Apache-2.0", "Z-Image")
    target_modules = (
        "attention.to_q",
        "attention.to_k",
        "attention.to_v",
        "attention.to_out.0",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
    )

    def __init__(self) -> None:
        self.pipe: Any = None
        self.device = "cuda"
        self._empty: Any = None
        self._prompts: dict[str, Any] = {}

    def load(self, weights_dir: Path, config: TrainConfig, device: str) -> nn.Module:
        """Load the pipeline; return its transformer, optionally FP8 storage and gradient checkpointing.

        The transformer stays on the CPU until :meth:`release_encoders`, so it is
        never on the card together with the text encoder (about 8 GB) and the VAE.
        """
        if config.train_text_encoder:
            raise LoraUsageError("lora.err.text_encoder")
        if not weights_dir.is_dir():
            raise LoraStageError("lora.err.base_missing")
        from diffusers import ZImagePipeline

        self.device = device
        self.pipe = ZImagePipeline.from_pretrained(str(weights_dir), torch_dtype=_dtype(config))
        transformer = self.pipe.transformer
        if config.base_fp8 and hasattr(transformer, "enable_layerwise_casting"):
            transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=_dtype(config))
        if config.grad_checkpointing and hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
        return transformer

    @torch.no_grad()
    def encode(self, images: Sequence[Any], captions: Sequence[dict[str, str]], config: TrainConfig) -> list[Sample]:
        """VAE latents and text embeddings for each item, computed once."""
        import numpy as np

        pipe = self.pipe
        pipe.vae.to(self.device)
        pipe.text_encoder.to(self.device)
        vae_cfg = pipe.vae.config
        samples: list[Sample] = []
        self._empty = self._embed("")
        for image, caption in zip(images, captions, strict=True):
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
            pixels = torch.from_numpy(array).permute(2, 0, 1)[None].to(self.device, pipe.vae.dtype)
            latents = pipe.vae.encode(pixels).latent_dist.sample()[0]
            latents = (latents - vae_cfg.shift_factor) * vae_cfg.scaling_factor
            encoded = {key: self._embed(value) for key, value in caption.items() if value}
            samples.append(Sample(latents.to(_dtype(config)), encoded))
        return samples

    def _embed(self, prompt: str) -> Any:
        embeds, _ = self.pipe.encode_prompt(prompt, device=self.device, do_classifier_free_guidance=False)
        return embeds[0]

    def prepare_previews(self, prompts: Sequence[str]) -> None:
        """Encode each test prompt and the empty negative once, so previews never load the text encoder again."""
        for prompt in prompts:
            if prompt not in self._prompts:
                self._prompts[prompt] = self._embed(prompt).to("cpu")
        if prompts and "" not in self._prompts:
            self._prompts[""] = self._embed("").to("cpu")

    def release_encoders(self) -> None:
        """Move the VAE and text encoder to the CPU, free their cache, then the transformer to the card.

        Embeddings were cached on the card during :meth:`encode`; they stay (a few MB). Every
        cached preview prompt is on the CPU already, so the text encoder never returns.
        """
        import gc

        self.pipe.vae.to("cpu")
        self.pipe.text_encoder.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.pipe.transformer.to(self.device)

    def loss(
        self, model: nn.Module, batch: Sequence[Sample], caption_keys: Sequence[str], rng: random.Random
    ) -> torch.Tensor:
        """Flow matching: x_s = (1 - s) x0 + s noise; the model predicts x0 - noise at timestep 1 - s."""
        x0 = torch.stack([item.latents for item in batch])
        noise = torch.randn_like(x0)
        sigma = torch.sigmoid(torch.randn(x0.shape[0], device=x0.device)).to(x0.dtype)
        noisy = (1 - sigma.view(-1, 1, 1, 1)) * x0 + sigma.view(-1, 1, 1, 1) * noise
        embeds = [item.captions.get(key, self._empty) if key else self._empty for item, key in zip(batch, caption_keys)]
        out = model(list(noisy.unsqueeze(2).unbind(0)), 1 - sigma, embeds, return_dict=False)[0]
        pred = torch.stack([item.float() for item in out]).squeeze(2)
        return torch.nn.functional.mse_loss(pred, (x0 - noise).float())

    @torch.no_grad()
    def preview(self, model: nn.Module, prompt: str, seed: int, config: TrainConfig) -> Any | None:
        """Draw one test picture with the current LoRA. Only the VAE joins the transformer on the card."""
        if prompt not in self._prompts:
            self.pipe.text_encoder.to(self.device)
            try:
                self.prepare_previews([prompt])
            finally:
                self.pipe.text_encoder.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.pipe.vae.to(self.device)
        # Tiled decode: a full 1024 px decode is the largest single buffer of a preview.
        if hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()
        try:
            generator = torch.Generator(self.device).manual_seed(seed)
            return self.pipe(
                prompt_embeds=[self._prompts[prompt].to(self.device)],
                negative_prompt_embeds=[self._prompts[""].to(self.device)],
                height=config.resolution,
                width=config.resolution,
                generator=generator,
            ).images[0]
        finally:
            self.pipe.vae.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def seconds_per_step(self, config: TrainConfig) -> float:
        """[U] until the 16 GB calibration run: about 1.6 s (24 GB) or 3.0 s (16 GB) per step at 1024."""
        per = 3.0 if config.vram == 16 else 1.6
        return per * (config.resolution / 1024) ** 2 * config.batch * config.grad_accum

    def vram_need_gb(self, config: TrainConfig) -> float:
        """[U] FP8 storage plus checkpointing fits 16 GB; BF16 needs 24 GB."""
        return 15.0 if config.base_fp8 else 22.0


class MossImageAdapter(BaseAdapter):
    """MOSS-Image Base (OpenMOSS-Team/MOSS-Image-Base, Apache-2.0), the internal base.

    Registered so it can be chosen by MDL id. Its training loader lands with
    the GPU calibration run; until then loading raises ``lora.err.base_missing``.
    """

    info = BaseInfo("moss-image-base", "OpenMOSS-Team/MOSS-Image-Base", "main", "Apache-2.0", "MOSS-Image Base")
    trainable = False

    def load(self, weights_dir: Path, config: TrainConfig, device: str) -> nn.Module:
        """Not available in this release."""
        raise LoraStageError("lora.err.base_missing")

    def encode(self, images: Sequence[Any], captions: Sequence[dict[str, str]], config: TrainConfig) -> list[Sample]:
        """Not available in this release."""
        raise LoraStageError("lora.err.base_missing")

    def loss(
        self, model: nn.Module, batch: Sequence[Sample], caption_keys: Sequence[str], rng: random.Random
    ) -> torch.Tensor:
        """Not available in this release."""
        raise LoraStageError("lora.err.base_missing")

    def seconds_per_step(self, config: TrainConfig) -> float:
        """[U] Same order as Z-Image."""
        return ZImageAdapter().seconds_per_step(config)

    def vram_need_gb(self, config: TrainConfig) -> float:
        """[U] Same order as Z-Image."""
        return ZImageAdapter().vram_need_gb(config)


register(ZImageAdapter)
register(MossImageAdapter)
