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

from voxprint_forge.lora import memory
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

    def memory_plan(
        self, config: TrainConfig, samples: int = 0, weights_dir: Path | None = None
    ) -> memory.MemoryPlan | None:
        """Projected memory per phase, checked before any GPU work. ``None``: the adapter has no plan."""
        return None


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


# Z-Image at the pinned revision (Tongyi-MAI/Z-Image@04cc4abb): read from the weights when present.
_Z_DIM, _Z_LAYERS, _Z_REFINERS = 3840, 30, 2
_Z_TRANSFORMER_BYTES = 12_309_817_472  # transformer/diffusion_pytorch_model.safetensors.index.json, BF16
_Z_TEXT_ENCODER_BYTES = 8_044_936_192  # text_encoder/model.safetensors.index.json (Qwen3-4B), BF16
_Z_VAE_BYTES = 167_666_902  # vae/diffusion_pytorch_model.safetensors (FLUX VAE), BF16
_Z_CAPTION_TOKENS = 512  # encode_prompt max_sequence_length
# Saved activations of one transformer block, in units of tokens x dim x 2 bytes. Measured on one
# real-width block with LoRA (35 units); +7 for the gradients alive while a checkpointed block
# recomputes. Without checkpointing every block keeps its 35 units: 34 blocks x 35 x 2 samples of
# 4608 tokens = 79 GiB, the 79.3 GB the first synth4 calibration measured at batch 2.
_Z_BLOCK_SAVED, _Z_BLOCK_BACKWARD = 35, 42
# VAE encode and text-encoder forward at 1024 px, no grad: R3 measured 0.7 GiB over the weights.
_Z_ENCODE_ACT_GB = 1.0


def _z_shapes(weights_dir: Path | None) -> tuple[int, int, int, int, int]:
    """dim, blocks, transformer bytes, text encoder bytes, VAE bytes; from the weights when installed."""
    import json

    dim, layers, refiners = _Z_DIM, _Z_LAYERS, _Z_REFINERS
    sizes = [_Z_TRANSFORMER_BYTES, _Z_TEXT_ENCODER_BYTES, _Z_VAE_BYTES]
    if weights_dir is not None and weights_dir.is_dir():
        try:
            cfg = json.loads((weights_dir / "transformer" / "config.json").read_text(encoding="utf-8"))
            dim, layers, refiners = int(cfg["dim"]), int(cfg["n_layers"]), int(cfg["n_refiner_layers"])
        except (OSError, ValueError, KeyError):
            pass
        for index, name in enumerate(
            (
                "transformer/diffusion_pytorch_model.safetensors.index.json",
                "text_encoder/model.safetensors.index.json",
            )
        ):
            try:
                index_file = json.loads((weights_dir / name).read_text(encoding="utf-8"))
                sizes[index] = int(index_file["metadata"]["total_size"])
            except (OSError, ValueError, KeyError):
                pass
        vae = weights_dir / "vae" / "diffusion_pytorch_model.safetensors"
        if vae.is_file():
            sizes[2] = vae.stat().st_size
    # Main layers plus the noise and context refiners; every one is checkpointed and carries LoRA.
    return dim, layers + 2 * refiners, sizes[0], sizes[1], sizes[2]


class ZImageAdapter(BaseAdapter):
    """Z-Image (Tongyi-MAI/Z-Image, Apache-2.0), flow matching, via diffusers ``ZImagePipeline``.

    LoRAs trained here are for Z-Image. Loading them on Z-Image-Turbo is
    untested; revisit after a Canvas test. The real path needs a CUDA GPU; CI
    covers the engine with a tiny stand-in.
    """

    info = BaseInfo(
        "z-image", "Tongyi-MAI/Z-Image", "04cc4abb7c5069926f75c9bfde9ef43d49423021", "Apache-2.0", "Z-Image"
    )
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
        self._plan: memory.MemoryPlan | None = None
        self._samples_bytes = 0

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
        # The encoders only ever run forward. Frozen, no output can carry an autograd graph that
        # pins their card copies (CALIB-R2/R3: 7.5 GiB of weights plus 8.8 GiB of activations).
        for part in (self.pipe.text_encoder, self.pipe.vae):
            part.requires_grad_(False)
            part.eval()
        self._plan = self.memory_plan(config, 0, weights_dir)
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
        self._samples_bytes = sum(_nbytes(s.latents) + sum(_nbytes(v) for v in s.captions.values()) for s in samples)
        self._samples_bytes += _nbytes(self._empty)
        return samples

    @torch.no_grad()
    def _embed(self, prompt: str) -> Any:
        """One prompt's embedding, detached: never part of a graph (previews call this outside ``encode``)."""
        embeds, _ = self.pipe.encode_prompt(prompt, device=self.device, do_classifier_free_guidance=False)
        embed = embeds[0]
        return embed.detach() if isinstance(embed, torch.Tensor) else embed

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
        cached preview prompt is on the CPU already, so the text encoder never returns. Before the
        transformer moves, whatever is still allocated must be those caches: anything more means an
        encoder is pinned, and the run stops here (``lora.err.encoders_stuck``) instead of OOMing.
        """
        import gc

        self.pipe.vae.to("cpu")
        self.pipe.text_encoder.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            residual = torch.cuda.memory_allocated() / memory.GIB
            budget = memory.budget_gb(memory.vram_cap_gb(), memory.card_gb())
            memory.check_residual(residual, self._samples_bytes / memory.GIB, self._plan, budget)
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
        """What the card must have: the projected peak (:meth:`memory_plan`) plus the budget headroom."""
        plan = self.memory_plan(config)
        assert plan is not None
        return round(plan.peak_gb + memory.HEADROOM_GB, 1)

    def memory_plan(
        self, config: TrainConfig, samples: int = 0, weights_dir: Path | None = None
    ) -> memory.MemoryPlan:
        """Projected GiB per phase from the base's shapes and stored sizes (no GPU, no weights loaded).

        Encode: text encoder and VAE (BF16) plus the measured forward buffers. Train: the
        transformer (BF16, or about 1 byte per weight with FP8 storage), LoRA weights, gradients
        and AdamW moments, one micro-batch of activations (block inputs plus one recomputed block
        with checkpointing, every block's activations without), and the cached samples.
        """
        dim, blocks, t_bytes, te_bytes, vae_bytes = _z_shapes(weights_dir if weights_dir is not None else _installed())
        ffn = int(dim / 3 * 8)
        attn = ((dim, dim),) * 4
        mlp = ((dim, ffn), (dim, ffn), (ffn, dim))
        adapter = memory.lora_params(blocks, attn + mlp, config.rank)
        side = config.resolution // 16  # VAE /8, patch 2
        tokens = config.batch * (side * side + _Z_CAPTION_TOKENS)
        unit = tokens * dim * 2 / memory.GIB
        if config.grad_checkpointing:
            act = blocks * unit + _Z_BLOCK_BACKWARD * unit
        else:
            act = blocks * _Z_BLOCK_SAVED * unit
        weights = t_bytes / 2 + 0.1 * memory.GIB if config.base_fp8 else float(t_bytes)  # BF16 and FP16: 2 bytes
        scale = (config.resolution / 1024) ** 2
        # Latents (16 x side*2 x side*2, BF16) and up to two caption embeddings plus the empty one.
        sample = 16 * (2 * side) ** 2 * 2 + 3 * _Z_CAPTION_TOKENS * 2560 * 2
        return memory.MemoryPlan(
            encoders_gb=(te_bytes + vae_bytes) / memory.GIB,
            encode_activations_gb=_Z_ENCODE_ACT_GB * max(scale, 0.25),
            transformer_gb=weights / memory.GIB,
            # The adapter trains in BF16 next to BF16/FP8 storage and in FP32 next to FP16 (layers.py).
            lora_gb=memory.lora_bytes(adapter, 2 if config.precision == "bf16" else 4) / memory.GIB,
            activations_gb=act,
            samples_gb=max(samples, 30) * sample / memory.GIB,
        )


def _nbytes(value: Any) -> int:
    return int(value.numel() * value.element_size()) if isinstance(value, torch.Tensor) else 0


def _installed() -> Path | None:
    try:
        folder = weights_dir(ZImageAdapter.info.mdl_id)
    except Exception:  # noqa: BLE001 - no models dir configured: use the pinned shapes
        return None
    return folder if folder.is_dir() else None


class MossImageAdapter(BaseAdapter):
    """MOSS-Image Base (OpenMOSS-Team/MOSS-Image-Base, Apache-2.0), the internal base.

    Registered so it can be chosen by MDL id. Its training loader lands with
    the GPU calibration run; until then loading raises ``lora.err.base_missing``.
    """

    info = BaseInfo(
        "moss-image-base",
        "OpenMOSS-Team/MOSS-Image-Base",
        "1ceede80badddd6a7812069619edaa5633587192",
        "Apache-2.0",
        "MOSS-Image Base",
    )
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
