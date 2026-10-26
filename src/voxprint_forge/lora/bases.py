"""Base-model adapters. The engine talks only to :class:`BaseAdapter`; the base is picked by MDL id.

Built in: Z-Image (``z-image``, the Canvas hand-off base, default) and
MOSS-Image Base (``moss-image-base``). Both are Apache-2.0 and both can train.
A restrictive base may only be an optional MDL link to its official source.
"""

from __future__ import annotations

import gc
import json
import os
import random
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.checkpoint as checkpoint
from torch import nn

from voxprint_forge.diffusion import moss as moss_runtime
from voxprint_forge.lora import (
    DEFAULT_BASE,  # noqa: F401 - re-export for callers that read bases.DEFAULT_BASE
    memory,
    te_quant,
)
from voxprint_forge.lora.errors import LoraStageError, LoraUsageError
from voxprint_forge.lora.layers import LoRALinear
from voxprint_forge.lora.presets import TrainConfig, measured
from voxprint_forge.lora.vram_stage import assert_stage_clear, hold_heavy, release_heavy, require_quantized
from voxprint_forge.model_park import drop_vram, park, stage
from voxprint_forge.model_views import ensure_view

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
    supports_negatives: bool = False  # True when disliked review samples can enter the loss
    lora_dtype: torch.dtype | None = None  # None: the adapter dtype follows the base weight

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

    def preview(
        self,
        model: nn.Module,
        prompt: str,
        seed: int,
        config: TrainConfig,
        steps: int | None = None,
    ) -> Any | None:
        """A test picture (PIL image) or ``None`` when the adapter cannot draw one.

        ``steps`` is the denoising count for a review picture. ``None`` leaves the pipeline default,
        which is what test pictures during a stage use.
        """
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


def register(adapter: type[BaseAdapter]) -> type[BaseAdapter]:
    """Add an adapter class to the registry under its MDL id."""
    _REGISTRY[adapter.info.mdl_id] = adapter
    return adapter


def _inference_steps(steps: int | None) -> dict[str, int]:
    """Pipeline kwargs for an explicit denoising count. Empty when the pipeline default applies."""
    if steps is None:
        return {}
    return {"num_inference_steps": int(steps)}


def unregister(mdl_id: str) -> None:
    """Remove an adapter (tests)."""
    _REGISTRY.pop(mdl_id, None)


def available() -> list[BaseInfo]:
    """Registered bases."""
    return [cls.info for cls in _REGISTRY.values()]


# MOSS-Image Base as a LoRA base. VOXPRINT_FORGE_MOSS_LORA=0 hides it (app list and ``lora train``).
MOSS_LORA_DEFAULT = True


def hidden() -> frozenset[str]:
    """MDL ids switched off as LoRA bases in this build or by the environment."""
    raw = os.environ.get("VOXPRINT_FORGE_MOSS_LORA", "").strip().lower()
    shown = MOSS_LORA_DEFAULT if raw == "" else raw not in ("0", "false", "no", "off")
    return frozenset() if shown else frozenset({"moss-image-base"})


def trainable() -> list[BaseInfo]:
    """Registered bases this build can actually train on (what the app offers)."""
    off = hidden()
    return [cls.info for cls in _REGISTRY.values() if cls.trainable and cls.info.mdl_id not in off]


def get_adapter(mdl_id: str, *, training: bool = False) -> BaseAdapter:
    """A fresh adapter for ``mdl_id``. Unknown ids, and hidden ones for ``training``, raise ``lora.err.base_missing``.

    Export and other reads of an existing run still get a hidden base.
    """
    cls = _REGISTRY.get(mdl_id)
    if cls is None or (training and mdl_id in hidden()):
        raise LoraUsageError("lora.err.base_missing")
    return cls()


def weights_dir(mdl_id: str) -> Path:
    """Where MDL installs a base for Model Forge: ``<models_dir>/model-forge/<mdl_id>``."""
    from voxprint_common.paths import models_dir

    ensure_view(mdl_id)
    return models_dir() / "model-forge" / mdl_id


def _dtype(config: TrainConfig) -> torch.dtype:
    return torch.bfloat16 if config.precision == "bf16" else torch.float16


def _unwrap_lora(module: nn.Module) -> None:
    """Restore base linears on a parked module so a later ``train()`` can inject again.

    The parked object is the RAM copy. A resume in this process reuses it instead of
    reading the files again, and the previous run's wrappers would hide the linears.
    """
    wrapped = [name for name, child in module.named_modules() if isinstance(child, LoRALinear)]
    for name in wrapped:
        parent_name, _, attr = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attr, getattr(parent, attr).base)


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
        self._vram = 24

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
        self._vram = config.vram
        if self.pipe is not None and getattr(self.pipe, "transformer", None) is not None:
            drop_vram(self.pipe.transformer, key=f"{self.info.mdl_id}:transformer", stage_name="train")
        release_heavy("transformer")
        release_heavy("text_encoder")
        require_quantized(config.vram, config.base_fp8, self.info.mdl_id)
        key = f"{self.info.mdl_id}:{weights_dir}:{id(self)}"

        def read() -> object:
            pipe = ZImagePipeline.from_pretrained(str(weights_dir), torch_dtype=_dtype(config))
            if config.base_fp8:
                transformer = pipe.transformer
                cast = getattr(transformer, "enable_layerwise_casting", None)
                if not callable(cast):
                    require_quantized(config.vram, False, self.info.mdl_id)
                    return pipe
                cast(storage_dtype=torch.float8_e4m3fn, compute_dtype=_dtype(config))
                transformer._voxprint_fp8 = True
            return pipe

        self.pipe = park(key, read, stage_name="load", replay_rng=True)
        # The encoders only ever run forward. Frozen, no output can carry an autograd graph that
        # pins their card copies (CALIB-R2/R3: 7.5 GiB of weights plus 8.8 GiB of activations).
        for part in (self.pipe.text_encoder, self.pipe.vae):
            part.requires_grad_(False)
            part.eval()
        self._plan = self.memory_plan(config, 0, weights_dir)
        transformer = self.pipe.transformer
        _unwrap_lora(transformer)
        if (
            config.grad_checkpointing
            and hasattr(transformer, "enable_gradient_checkpointing")
            and not getattr(transformer, "_voxprint_checkpointed", False)
        ):
            transformer.enable_gradient_checkpointing()
            transformer._voxprint_checkpointed = True
        return transformer

    @torch.no_grad()
    def encode(self, images: Sequence[Any], captions: Sequence[dict[str, str]], config: TrainConfig) -> list[Sample]:
        """VAE latents and text embeddings for each item, computed once."""
        import numpy as np

        pipe = self.pipe
        hold_heavy("text_encoder", tier=self._vram)
        stage(pipe.text_encoder, self.device, key=f"{self.info.mdl_id}:text_encoder", stage_name="encode")
        stage(pipe.vae, self.device, key=f"{self.info.mdl_id}:vae", stage_name="encode")
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
        drop_vram(self.pipe.vae, key=f"{self.info.mdl_id}:vae", stage_name="encode")
        drop_vram(self.pipe.text_encoder, key=f"{self.info.mdl_id}:text_encoder", stage_name="encode")
        release_heavy("text_encoder")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            residual = torch.cuda.memory_allocated() / memory.GIB
            budget = memory.budget_gb(memory.vram_cap_gb(), memory.card_gb())
            memory.check_residual(residual, self._samples_bytes / memory.GIB, self._plan, budget)
            assert_stage_clear("encode", tier=self._vram, samples_gb=self._samples_bytes / memory.GIB)
        hold_heavy("transformer", tier=self._vram)
        stage(self.pipe.transformer, self.device, key=f"{self.info.mdl_id}:transformer", stage_name="train")

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
    def preview(
        self,
        model: nn.Module,
        prompt: str,
        seed: int,
        config: TrainConfig,
        steps: int | None = None,
    ) -> Any | None:
        """Draw one test picture with the current LoRA. Only the VAE joins the transformer on the card."""
        if prompt not in self._prompts:
            if self._vram == 16:
                raise LoraStageError("lora.err.vram_one", stage="transformer", model="text_encoder")
            self.pipe.text_encoder.to(self.device)
            try:
                self.prepare_previews([prompt])
            finally:
                self.pipe.text_encoder.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        stage(self.pipe.vae, self.device, key=f"{self.info.mdl_id}:vae", stage_name="preview")
        # Tiled decode: a full 1024 px decode is the largest single buffer of a preview.
        if hasattr(self.pipe.vae, "enable_tiling"):
            self.pipe.vae.enable_tiling()
        try:
            # CALIB-R4: the pipeline infers its execution device from the first module in sorted
            # order, which is the offloaded text encoder (CPU), so latents were drawn on the CPU
            # with a CUDA generator. Hide the parked encoder for the call and seed a CPU generator:
            # ``randn_tensor`` draws on the CPU and moves to any device, so the noise is the same on
            # every card and can never mismatch the generator.
            generator = torch.Generator("cpu").manual_seed(seed)
            with _encoder_parked(self.pipe):
                _check_execution_device(self.pipe, self.device)
                return self.pipe(
                    prompt_embeds=[self._prompts[prompt].to(self.device)],
                    negative_prompt_embeds=[self._prompts[""].to(self.device)],
                    height=config.resolution,
                    width=config.resolution,
                    generator=generator,
                    **_inference_steps(steps),
                ).images[0]
        finally:
            drop_vram(self.pipe.vae, key=f"{self.info.mdl_id}:vae", stage_name="preview")

    def seconds_per_step(self, config: TrainConfig) -> float:
        """Seconds for one optimizer step, scaled from 1024 px.

        The 16 GB tier uses the measured 9.7 s at 1024 px (:data:`presets.MEASURED`).
        The 24 GB figure, 1.6 s at 1024 px, has not been measured.
        """
        run = measured(self.info.mdl_id, config.vram)
        per = run.s_per_step / run.micro_batches if run is not None else 1.6
        return per * (config.resolution / 1024) ** 2 * config.batch * config.grad_accum

    def vram_need_gb(self, config: TrainConfig) -> float:
        """What the card must have: the projected peak (:meth:`memory_plan`) plus the budget headroom."""
        plan = self.memory_plan(config)
        assert plan is not None
        return round(plan.peak_gb + memory.HEADROOM_GB, 1)

    def memory_plan(self, config: TrainConfig, samples: int = 0, weights_dir: Path | None = None) -> memory.MemoryPlan:
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


@contextmanager
def _encoder_parked(pipe: Any) -> Iterator[None]:
    """Hide the text encoder from device inference while previews run on precomputed embeddings."""
    encoder = getattr(pipe, "text_encoder", None)
    # object.__setattr__: DiffusionPipeline.__setattr__ would also rewrite the pipeline config.
    object.__setattr__(pipe, "text_encoder", None)
    try:
        yield
    finally:
        object.__setattr__(pipe, "text_encoder", encoder)


def _check_execution_device(pipe: Any, device: str | torch.device) -> None:
    """Stop with a named error instead of a deep generator/device mismatch inside the pipeline."""
    found = getattr(pipe, "_execution_device", None)
    if found is None:
        return
    if torch.device(found).type != torch.device(device).type:
        raise LoraStageError(
            "lora.err.preview_device", devices=f"{torch.device(found).type}->{torch.device(device).type}"
        )


def _nbytes(value: Any) -> int:
    return int(value.numel() * value.element_size()) if isinstance(value, torch.Tensor) else 0


def _installed() -> Path | None:
    try:
        folder = weights_dir(ZImageAdapter.info.mdl_id)
    except Exception:  # noqa: BLE001 - no models dir configured: use the pinned shapes
        return None
    return folder if folder.is_dir() else None


# MOSS-Image Base at the pinned revision. Shapes match the vendored transformer; byte sizes are the
# resident bf16 DiT (10.17 GB) and the fp32 FLUX.2 VAE (0.25 GB). The Qwen3.5-9B text encoder stays
# on the CPU: it does not fit beside the DiT on a 16 GB card.
_M_DIM, _M_FFN, _M_LAYERS = 3072, 8192, 42
_M_LATENT_CHANNELS = 128
_M_TEXT_DIM = 4096
_M_CAPTION_TOKENS = 128
_M_TRANSFORMER_BYTES = int(10.17 * 10**9)
_M_VAE_BYTES = int(0.25 * 10**9)
# Checkpointed activations, in units of one bf16 [tokens, dim] buffer. 1.3 of those, at rank 32,
# 1024 px and batch 1, is the amount that puts the plan on the measured 11.8 GiB peak (RTX 4090,
# bf16 DiT, fp32 LoRA, manual per-block checkpointing).
_M_CHECKPOINT_UNITS = 1.3
_M_BLOCK_UNITS = 20  # activations kept per block when checkpointing is off
# Seconds per micro-batch (one image) at 1024 px, rank 32, when a tier has no measurement:
# the first RTX 4090 run with the text encoder on the CPU. Measured tiers use presets.MEASURED.
_M_SECONDS = 0.7


def _moss_shapes(weights_dir: Path | None) -> tuple[int, int, int, int, int]:
    """dim, ffn, layers, transformer bytes, VAE bytes; from the snapshot when it is installed."""
    dim, ffn, layers = _M_DIM, _M_FFN, _M_LAYERS
    sizes = [_M_TRANSFORMER_BYTES, _M_VAE_BYTES]
    if weights_dir is not None and weights_dir.is_dir():
        try:
            cfg = json.loads((weights_dir / "transformer" / "config.json").read_text(encoding="utf-8"))
            dim = int(cfg.get("dim", dim))
            ffn = int(cfg.get("ffn_dim", ffn))
            layers = int(cfg.get("num_layers", layers))
        except (OSError, ValueError, KeyError, TypeError):
            pass
        try:
            index_file = json.loads(
                (weights_dir / "transformer" / "diffusion_pytorch_model.safetensors.index.json").read_text(
                    encoding="utf-8"
                )
            )
            sizes[0] = int(index_file["metadata"]["total_size"])
        except (OSError, ValueError, KeyError, TypeError):
            single = weights_dir / "transformer" / "diffusion_pytorch_model.safetensors"
            if single.is_file():
                sizes[0] = single.stat().st_size
        vae = weights_dir / "vae" / "diffusion_pytorch_model.safetensors"
        if vae.is_file():
            sizes[1] = vae.stat().st_size
    return dim, ffn, layers, sizes[0], sizes[1]


def _moss_weights_dir() -> Path | None:
    try:
        folder = weights_dir("moss-image-base")
    except Exception:  # noqa: BLE001 - no models dir configured: use the pinned shapes
        return None
    return folder if folder.is_dir() else None


def _moss_runtime_problems() -> list[str]:
    """Installed torch, diffusers and transformers below what the vendored MOSS file needs."""
    from importlib.metadata import PackageNotFoundError, version

    got = []
    for name in ("torch", "diffusers", "transformers"):
        try:
            got.append(version(name))
        except PackageNotFoundError:
            got.append("0")
    return moss_runtime.check_versions(*got)


def _checkpoint_blocks(blocks: Sequence[nn.Module]) -> None:
    """Recompute each block on the backward. The MOSS transformer has no checkpointing switch."""
    for block in blocks:
        forward = block.forward

        def wrapped(*args: object, _forward: Any = forward, **kwargs: object) -> object:
            return checkpoint.checkpoint(_forward, *args, use_reentrant=False, **kwargs)

        block.forward = wrapped  # type: ignore[method-assign]


def _pad_text(
    pairs: Sequence[tuple[torch.Tensor, torch.Tensor]], device: str, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack prompt rows that may differ in length. Padding is masked off."""
    length = max(int(emb.shape[0]) for emb, _mask in pairs)
    width = int(pairs[0][0].shape[-1])
    embeds = torch.zeros(len(pairs), length, width, device=device, dtype=dtype)
    mask = torch.zeros(len(pairs), length, device=device, dtype=torch.bool)
    for index, (emb, bits) in enumerate(pairs):
        count = int(emb.shape[0])
        embeds[index, :count] = emb.to(device=device, dtype=dtype)
        mask[index, : int(bits.shape[0])] = bits.to(device=device).bool()
    return embeds, mask


def _pixels(image: Any) -> torch.Tensor:
    """RGB picture to a ``[1, 3, H, W]`` tensor in ``[-1, 1]``, the VAE's range."""
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(np.ascontiguousarray(array)).permute(2, 0, 1)[None]


class MossImageAdapter(BaseAdapter):
    """MOSS-Image Base (OpenMOSS-Team/MOSS-Image-Base, Apache-2.0).

    The DiT trains in bf16 with the LoRA in fp32. 4-bit and FP8 storage break the
    MOSS forward (it reads ``weight.dtype`` and casts activations to it). A 24 GB
    card keeps the bf16 weights. A 16 GB card has no quantized DiT, so :meth:`load`
    raises ``lora.err.vram_quant`` before the DiT is read. The text encoder encodes
    on the card (bf16 on 24 GB, int8 on 16 GB). On 24 GB, CPU encode is a fallback;
    on 16 GB it raises ``lora.err.cpu_fallback``. The transformer stays on the CPU
    until :meth:`release_encoders`. Flow matching follows the reference: predict
    ``noise - x0`` at timestep ``sigma * 1000``.
    """

    info = BaseInfo(
        "moss-image-base",
        "OpenMOSS-Team/MOSS-Image-Base",
        "1ceede80badddd6a7812069619edaa5633587192",
        "Apache-2.0",
        "MOSS-Image Base",
    )
    target_modules = (
        "proj.q",
        "proj.k",
        "proj.v",
        "proj.o",
        "proj.ffn_up_gate",
        "proj.ffn_down",
    )
    trainable = True
    lora_dtype = torch.float32

    def __init__(self) -> None:
        self.pipe: Any = None
        self.device = "cuda"
        self._dtype = torch.bfloat16
        self._empty: tuple[torch.Tensor, torch.Tensor] | None = None
        self._prompts: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._plan: memory.MemoryPlan | None = None
        self._samples_bytes = 0
        self._vram = 24

    encodes_before_load = True

    def _check(self, weights_dir: Path, config: TrainConfig) -> te_quant.TePlan:
        if config.train_text_encoder:
            raise LoraUsageError("lora.err.text_encoder")
        if weights_dir is None or not Path(weights_dir).is_dir():
            raise LoraStageError("lora.err.base_missing")
        problems = _moss_runtime_problems()
        if problems:
            raise LoraStageError("lora.err.base_runtime", need="; ".join(problems))
        if not str(self.device).startswith("cuda"):
            if config.vram == 16:
                raise LoraStageError("lora.err.cpu_fallback", tier=16, reason="no CUDA device")
            return te_quant.TePlan("none", "cpu", "no CUDA device")
        return te_quant.plan(getattr(config, "te_quant", None), getattr(config, "te_device", None), config.vram)

    def load_encoders(self, weights_dir: Path, config: TrainConfig, device: str) -> None:
        """Read the tokenizer, text encoder and VAE only. The DiT is read by :meth:`load` after
        :meth:`free_text_encoder`, so the bf16 encoder and the DiT are never in RAM together.

        The encoder runs on the card (bf16 on the 24 GB tier, int8 on 16 GB; see
        :func:`voxprint_forge.lora.te_quant.plan`). A 16 GB tier does not fall back to the CPU.
        """
        self.device = device
        self._vram = config.vram
        if self.pipe is not None and getattr(self.pipe, "transformer", None) is not None:
            drop_vram(self.pipe.transformer, key=f"{self.info.mdl_id}:transformer", stage_name="train")
        release_heavy("transformer")
        choice = self._check(weights_dir, config)
        self._dtype = torch.bfloat16
        self._te = choice
        key = f"{self.info.mdl_id}:encoders:{weights_dir}:{id(self)}"
        self.pipe = park(
            key,
            lambda: moss_runtime.load_encoder_parts(Path(weights_dir), te_quant=choice.quant, te_device=choice.device),
            stage_name="load",
            replay_rng=True,
        )
        # free_text_encoder drops the working pointer so the DiT load does not keep a second
        # reference. The module itself stays on the adapter, and a later train() puts it back
        # without reading the files again.
        resident = getattr(self, "_resident_te", None)
        if getattr(self.pipe, "text_encoder", None) is None and resident is not None:
            self.pipe.text_encoder = resident
        self._staged = True

    def free_text_encoder(self) -> None:
        """Drop the text encoder (CPU bf16 or quantized on the card). Cached embeddings stay."""
        if self.pipe is not None and getattr(self.pipe, "text_encoder", None) is not None:
            drop_vram(self.pipe.text_encoder, key=f"{self.info.mdl_id}:text_encoder", stage_name="encode")
            self._resident_te = self.pipe.text_encoder
            self.pipe.text_encoder = None
        release_heavy("text_encoder")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            assert_stage_clear("encode", tier=getattr(self, "_vram", 24))

    def load(self, weights_dir: Path, config: TrainConfig, device: str) -> nn.Module:
        """Load the bf16 transformer and return it, parked on the CPU. Encoders stay frozen.

        After :meth:`load_encoders` and :meth:`free_text_encoder` the pipeline is built without a
        text encoder. Called on its own it reads the whole pipeline, as before.
        """
        self.device = device
        self._vram = config.vram
        require_quantized(config.vram, False, self.info.mdl_id)
        choice = self._check(weights_dir, config)
        self._te = choice
        self._dtype = torch.bfloat16
        staged = getattr(self, "_staged", False)
        self._staged = False
        prior = self.pipe
        key = f"{self.info.mdl_id}:dit:{weights_dir}:{id(self)}:{int(staged)}"
        self.pipe = park(
            key,
            lambda: moss_runtime.load_pipeline(
                Path(weights_dir),
                dtype=torch.bfloat16,
                with_text_encoder=not staged,
                te_quant=choice.quant,
                te_device=choice.device,
            ),
            stage_name="load",
            replay_rng=True,
        )
        if staged and prior.vae is not None:
            self.pipe.vae = prior.vae
        transformer = self.pipe.transformer
        _unwrap_lora(transformer)
        transformer.to(dtype=torch.bfloat16)
        transformer.to("cpu")
        for part in (self.pipe.text_encoder, self.pipe.vae):
            if part is None:
                continue
            part.requires_grad_(False)
            part.eval()
        self._plan = self.memory_plan(config, 0, Path(weights_dir))
        if config.grad_checkpointing and not getattr(transformer, "_voxprint_checkpointed", False):
            _checkpoint_blocks(transformer.layers)
            transformer._voxprint_checkpointed = True
        return transformer

    @torch.no_grad()
    def encode(self, images: Sequence[Any], captions: Sequence[dict[str, str]], config: TrainConfig) -> list[Sample]:
        """Cached packed VAE latents plus one ``encode_prompt`` embedding and mask per caption."""
        hold_heavy("text_encoder", tier=self._vram)
        pipe = self.pipe
        stage(pipe.vae, self.device, key=f"{self.info.mdl_id}:vae", stage_name="encode")
        self._empty = self._on_device(self._encode_prompt(""))
        samples: list[Sample] = []
        for image, caption in zip(images, captions, strict=True):
            latents = moss_runtime.encode_image(pipe.vae, _pixels(image))[0].detach()
            encoded = {key: self._on_device(self._encode_prompt(value)) for key, value in caption.items() if value}
            samples.append(Sample(latents, encoded))
        self._samples_bytes = sum(
            _nbytes(item.latents) + sum(_nbytes(part) for pair in item.captions.values() for part in pair)
            for item in samples
        )
        if self._empty is not None:
            self._samples_bytes += _nbytes(self._empty[0]) + _nbytes(self._empty[1])
        return samples

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """One prompt's embedding and mask on the CPU. The encoder runs where it was loaded."""
        pipe = self.pipe
        if getattr(pipe, "tokenizer", None) is not None and getattr(pipe, "text_encoder", None) is not None:
            embeds, mask = moss_runtime.encode_on_cpu(pipe, prompt)
        else:
            embeds, mask = pipe.encode_prompt(prompt)
        return embeds[0].detach().to("cpu"), mask[0].detach().to("cpu").bool()

    def _on_device(self, pair: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        emb, mask = pair
        return emb.to(self.device), mask.to(self.device)

    def prepare_previews(self, prompts: Sequence[str]) -> None:
        """Encode each test prompt and the empty negative once, on the CPU text encoder."""
        for prompt in prompts:
            if prompt not in self._prompts:
                self._prompts[prompt] = self._encode_prompt(prompt)
        if prompts and "" not in self._prompts:
            self._prompts[""] = self._encode_prompt("")

    def release_encoders(self) -> None:
        """Move the VAE and text encoder off the card, then the transformer onto it.

        The text encoder was never on the card. The residual check is the same one Z-Image uses:
        anything still allocated beyond the cached samples means an encoder is pinned.
        """
        drop_vram(self.pipe.vae, key=f"{self.info.mdl_id}:vae", stage_name="encode")
        if self.pipe.text_encoder is not None:
            drop_vram(self.pipe.text_encoder, key=f"{self.info.mdl_id}:text_encoder", stage_name="encode")
            self.pipe.text_encoder = None
        release_heavy("text_encoder")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            residual = torch.cuda.memory_allocated() / memory.GIB
            budget = memory.budget_gb(memory.vram_cap_gb(), memory.card_gb())
            memory.check_residual(residual, self._samples_bytes / memory.GIB, self._plan, budget)
            assert_stage_clear("encode", tier=self._vram, samples_gb=self._samples_bytes / memory.GIB)
        hold_heavy("transformer", tier=self._vram)
        stage(self.pipe.transformer, self.device, key=f"{self.info.mdl_id}:transformer", stage_name="train")

    def loss(
        self, model: nn.Module, batch: Sequence[Sample], caption_keys: Sequence[str], rng: random.Random
    ) -> torch.Tensor:
        """Flow matching: ``x_t = (1 - σ) x0 + σ noise``; the model predicts ``noise - x0``."""
        if self._empty is None:
            raise LoraStageError("lora.err.base_missing")
        x0 = torch.stack([item.latents for item in batch]).to(device=self.device, dtype=torch.float32)
        pairs = []
        for item, key in zip(batch, caption_keys, strict=True):
            packed = item.captions.get(key) if key else None
            pairs.append(self._empty if packed is None else packed)
        embeds, mask = _pad_text(pairs, self.device, self._dtype)
        noise = torch.randn_like(x0)
        sigma = torch.sigmoid(torch.randn(x0.shape[0], device=x0.device)).view(-1, 1, 1, 1)
        xt = (1 - sigma) * x0 + sigma * noise
        # requires_grad so per-block checkpointing has an input to track; the LoRA is what trains.
        hidden = xt.to(dtype=self._dtype).requires_grad_(True)
        pred = model(hidden, sigma.view(-1) * 1000.0, embeds, mask).sample.float()
        return torch.nn.functional.mse_loss(pred, (noise - x0).float())

    @torch.no_grad()
    def preview(
        self,
        model: nn.Module,
        prompt: str,
        seed: int,
        config: TrainConfig,
        steps: int | None = None,
    ) -> Any | None:
        """Draw one test picture. The text encoder stays parked; noise comes from a CPU generator."""
        if prompt not in self._prompts or "" not in self._prompts:
            if self._vram == 16 and getattr(self.pipe, "text_encoder", None) is not None:
                raise LoraStageError("lora.err.vram_one", stage="transformer", model="text_encoder")
            self.prepare_previews([prompt])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        stage(self.pipe.vae, self.device, key=f"{self.info.mdl_id}:vae", stage_name="preview")
        try:
            positive = self._prompts[prompt]
            negative = self._prompts[""]
            generator = torch.Generator("cpu").manual_seed(seed)
            with _encoder_parked(self.pipe):
                _check_execution_device(self.pipe, self.device)
                return self.pipe(
                    prompt_embeds=positive[0].unsqueeze(0).to(self.device),
                    prompt_attention_mask=positive[1].unsqueeze(0).to(self.device),
                    negative_prompt_embeds=negative[0].unsqueeze(0).to(self.device),
                    negative_prompt_attention_mask=negative[1].unsqueeze(0).to(self.device),
                    height=config.resolution,
                    width=config.resolution,
                    generator=generator,
                    **_inference_steps(steps),
                ).images[0]
        finally:
            drop_vram(self.pipe.vae, key=f"{self.info.mdl_id}:vae", stage_name="preview")

    def seconds_per_step(self, config: TrainConfig) -> float:
        """Measured time per micro-batch on the chosen tier (:data:`presets.MEASURED`), scaled to ``config``.

        moss-memprobe v2: 4.89 s (16 GB) and 4.90 s (24 GB) per optimizer step at 1024 px, batch 1 x
        accum 2, so about 2.45 s per image. An unmeasured tier falls back to :data:`_M_SECONDS`.
        """
        run = measured(self.info.mdl_id, config.vram)
        per = run.s_per_step / run.micro_batches if run is not None else _M_SECONDS
        return per * (config.resolution / 1024) ** 2 * config.batch * config.grad_accum

    def vram_need_gb(self, config: TrainConfig) -> float:
        """What the card must have: the projected peak plus the headroom, never below the measured peak."""
        plan = self.memory_plan(config)
        need = round(plan.peak_gb + memory.HEADROOM_GB, 1)
        run = measured(self.info.mdl_id, config.vram)
        return max(need, run.peak_reserved_gb) if run is not None else need

    def memory_plan(self, config: TrainConfig, samples: int = 0, weights_dir: Path | None = None) -> memory.MemoryPlan:
        """Projected GiB. Both tiers keep the bf16 DiT: FP8 and 4-bit break the MOSS forward.

        Encode is the VAE only (the text encoder stays on the CPU). Train is the DiT, the fp32
        LoRA with its gradients and AdamW moments, one micro-batch of activations and the cache.
        At rank 32, 1024 px and batch 1 the peak is the measured 11.8 GiB.
        """
        dim, ffn, layers, t_bytes, vae_bytes = _moss_shapes(
            weights_dir if weights_dir is not None else _moss_weights_dir()
        )
        shapes = ((dim, dim),) * 4 + ((dim, 2 * ffn), (ffn, dim))
        adapter = memory.lora_params(layers, shapes, config.rank)
        side = config.resolution // 16
        tokens = config.batch * (side * side + _M_CAPTION_TOKENS)
        unit = tokens * dim * 2 / memory.GIB
        if config.grad_checkpointing:
            act = _M_CHECKPOINT_UNITS * unit * (layers / _M_LAYERS)
        else:
            act = layers * _M_BLOCK_UNITS * unit
        sample = _M_LATENT_CHANNELS * side * side * 2 + 3 * (_M_CAPTION_TOKENS * _M_TEXT_DIM * 2 + _M_CAPTION_TOKENS)
        scale = (config.resolution / 1024) ** 2
        return memory.MemoryPlan(
            encoders_gb=vae_bytes / memory.GIB,
            encode_activations_gb=max(scale, 0.25),
            transformer_gb=t_bytes / memory.GIB,
            lora_gb=memory.lora_bytes(adapter, 4) / memory.GIB,
            activations_gb=act,
            samples_gb=max(samples, 30) * sample / memory.GIB,
        )


register(ZImageAdapter)
register(MossImageAdapter)
