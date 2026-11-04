"""Peak VRAM and RSS for a compact MOSS build (``forge moss-peak``).

The command reads ``voxprint-moss-image-sft-compact`` (the ``moss-prequant``
``--out`` directory, or that weights directory itself). It checks sha256
sidecars, loads the weights read-only, and attaches a seeded dummy LoRA in
memory. ``gen`` denoises for ``--steps`` and drops the latent. ``train`` runs
``--train-steps`` of AdamW on a fresh dummy LoRA over a tiny synthetic
flow-matching batch. Nothing is saved: no image, no checkpoint, no LoRA file,
no optimizer state. The only write is the JSON report at ``--out``, and that
path is refused when it resolves inside the build or any model directory.

``*_gb`` values are bytes / 1024**3, the same unit as the prequant probe.
Linux ``ru_maxrss`` is kilobytes and is stored as bytes. Tests pass
:class:`Services` stand-ins. :func:`live_services` is the CUDA path and is
not what the tests call. This module does not download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from voxprint_forge.diffusion.moss_prequant import MODEL_ID, WEIGHTS_DIR, _assert_sidecars
from voxprint_forge.errors import StageError, UsageError

DEFAULT_RANK = 16
DEFAULT_STEPS = 4
DEFAULT_TRAIN_STEPS = 5
DEFAULT_SEED = 0
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_PROMPT = "A ceramic teapot and two cups on a wooden table, window light."
MODES = ("gen", "train", "both")
ATTENTION_TARGETS = ("proj.q", "proj.k", "proj.v", "proj.o")
TRAIN_EDGE = 4
TRAIN_LR = 1e-4
CAPTION_TOKENS = 8
_BYTES_PER_GB = 1024**3


class ModelSession(Protocol):
    """In-memory load, generation, and training. Implementations must not write files."""

    def load(self, weights: Path, *, quant_mode: str) -> None:
        """Read ``weights``. Do not save and do not download."""

    def generate(
        self,
        *,
        steps: int,
        seed: int,
        rank: int,
        prompt: str,
        width: int,
        height: int,
    ) -> None:
        """Attach a seeded dummy LoRA, run ``steps``, and drop the image."""

    def train(self, *, steps: int, seed: int, rank: int, prompt: str) -> None:
        """Attach a fresh seeded dummy LoRA and run ``steps`` AdamW updates. Save nothing."""


class MemoryProbe(Protocol):
    """CUDA peak counters and the process RSS."""

    def reset(self) -> None:
        """Clear the CUDA peak so the next sample is this phase only."""

    def sample(self) -> MemorySample:
        """Peaks since :meth:`reset`."""

    def gpu_name(self) -> str:
        """GPU name recorded on the report."""


class Clock(Protocol):
    """Report timestamp."""

    def timestamp(self) -> str:
        """An ISO-8601 UTC time."""


@dataclass(frozen=True)
class MemorySample:
    """One phase's peaks, in bytes. ``run`` adds the GB fields."""

    allocated_bytes: int
    reserved_bytes: int
    rss_bytes: int


@dataclass(frozen=True)
class BuildLocation:
    """Resolved compact build. ``root`` holds ``voxprint-model.json``. ``weights`` is loaded."""

    root: Path
    weights: Path
    manifest: Path
    build_id: str
    quant_mode: str


@dataclass
class Services:
    """Injected backends. Tests pass stand-ins. The CLI uses :func:`live_services`."""

    session: ModelSession
    probe: MemoryProbe
    clock: Clock


def rss_to_bytes(raw: int, platform: str | None = None) -> int:
    """Convert ``ru_maxrss`` to bytes. Linux counts kilobytes; macOS counts bytes."""
    system = sys.platform if platform is None else platform
    if system == "darwin":
        return int(raw)
    return int(raw) * 1024


def rss_bytes() -> int:
    """Peak process RSS in bytes. Windows uses psutil ``peak_wset`` when installed, else 0."""
    try:
        import resource
    except ImportError:
        try:
            import psutil
        except ImportError:
            return 0
        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", 0) or getattr(info, "rss", 0))
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss_to_bytes(raw)


def locate_build(path: Path) -> BuildLocation:
    """Find the compact build. Accepts the prequant ``--out`` directory or the weights directory."""
    folder = Path(path).expanduser().resolve()
    if not folder.is_dir():
        raise StageError(f"build directory missing: {folder}", code="moss_peak_missing")
    nested = folder / WEIGHTS_DIR
    manifest_here = folder / "voxprint-model.json"
    if nested.is_dir() and manifest_here.is_file():
        if not _is_weights_dir(nested):
            raise StageError(f"{nested} is not a compact weights directory", code="moss_peak_missing")
        weights = nested
        root = folder
        manifest = manifest_here
    elif _is_weights_dir(folder):
        weights = folder
        root, manifest = _manifest_near(folder)
    else:
        raise StageError(
            f"{folder} is not a {MODEL_ID} build",
            code="moss_peak_missing",
        )
    document = _read_json_object(manifest, "moss_peak_manifest")
    build_id = document.get("id")
    if not isinstance(build_id, str) or not build_id:
        raise StageError(f"{manifest.name} has no id", code="moss_peak_manifest")
    if build_id != MODEL_ID:
        raise StageError(f"build id {build_id} is not {MODEL_ID}", code="moss_peak_manifest")
    return BuildLocation(
        root=root.resolve(),
        weights=weights.resolve(),
        manifest=manifest.resolve(),
        build_id=build_id,
        quant_mode=_quant_mode(document),
    )


def run(
    build: Path,
    out: Path,
    *,
    mode: str,
    rank: int = DEFAULT_RANK,
    steps: int = DEFAULT_STEPS,
    train_steps: int = DEFAULT_TRAIN_STEPS,
    seed: int = DEFAULT_SEED,
    prompt: str = DEFAULT_PROMPT,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    services: Services | None = None,
) -> dict[str, object]:
    """Measure and write one JSON file. ``services`` is required for tests.

    The report carries ``mode``, ``steps``, ``train_steps``, ``rank``, ``seed``,
    ``resolution``, ``build_id``, ``quant_mode``, ``gpu``, and ``timestamp``.
    ``gen`` and ``train`` sections are present for the phases that ran. Each
    section has allocated and reserved bytes and GB, and peak RSS in bytes.
    """
    _check_config(mode, rank, steps, train_steps, seed, width, height)
    location = locate_build(build)
    destination = Path(out).expanduser()
    _refuse_output(destination, location)
    _assert_sidecars(location.root)
    backends = services if services is not None else live_services()
    gpu = backends.probe.gpu_name()
    backends.session.load(location.weights, quant_mode=location.quant_mode)
    phases: dict[str, dict[str, object]] = {}
    if mode in ("gen", "both"):
        backends.probe.reset()
        backends.session.generate(
            steps=steps,
            seed=seed,
            rank=rank,
            prompt=prompt,
            width=width,
            height=height,
        )
        phases["gen"] = _phase(steps, backends.probe.sample())
    if mode in ("train", "both"):
        backends.probe.reset()
        backends.session.train(steps=train_steps, seed=seed, rank=rank, prompt=prompt)
        phases["train"] = _phase(train_steps, backends.probe.sample())
    report: dict[str, object] = {
        "mode": mode,
        "build_id": location.build_id,
        "quant_mode": location.quant_mode,
        "gpu": gpu,
        "timestamp": backends.clock.timestamp(),
        "rank": rank,
        "seed": seed,
        "steps": steps,
        "train_steps": train_steps,
        "resolution": {"width": width, "height": height},
        "prompt": prompt,
    }
    report.update(phases)
    _write_json(destination, report)
    return report


def execute(args: argparse.Namespace) -> dict[str, object]:
    """Run from the ``forge moss-peak`` arguments. Uses the live backends."""
    return run(
        Path(args.build),
        Path(args.out),
        mode=str(args.mode),
        rank=int(args.lora_rank),
        steps=int(args.steps),
        train_steps=int(args.train_steps),
        seed=int(args.seed),
        prompt=str(args.prompt),
        width=int(args.width),
        height=int(args.height),
    )


def live_services() -> Services:
    """CUDA backends. Tests pass their own :class:`Services` and do not call this."""
    return Services(LiveSession(), LiveProbe(), LiveClock())


def add_command(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``forge moss-peak``."""
    command = sub.add_parser(
        "moss-peak",
        help="Measure peak VRAM and RSS for the compact MOSS build. Never downloads.",
    )
    command.add_argument(
        "--mode",
        required=True,
        choices=MODES,
        help="gen, train, or both. both runs generation, then training.",
    )
    command.add_argument(
        "--build",
        type=Path,
        required=True,
        help="moss-prequant --out directory, or the voxprint-moss-image-sft-compact weights directory.",
    )
    command.add_argument(
        "--out",
        type=Path,
        required=True,
        help="JSON report path. Refused when it resolves inside the build or any model directory.",
    )
    command.add_argument(
        "--lora-rank",
        type=int,
        default=DEFAULT_RANK,
        help=f"Dummy LoRA rank. Default: {DEFAULT_RANK}. The adapter is not saved.",
    )
    command.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Generation steps. Default: {DEFAULT_STEPS}.",
    )
    command.add_argument(
        "--train-steps",
        type=int,
        default=DEFAULT_TRAIN_STEPS,
        help=f"AdamW steps on the dummy LoRA. Default: {DEFAULT_TRAIN_STEPS}. Nothing is saved.",
    )
    command.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed for the generator, the LoRA, and the synthetic batch. Default: {DEFAULT_SEED}.",
    )
    command.add_argument("--prompt", default=DEFAULT_PROMPT, help="Caption. Default: a neutral still life.")
    command.add_argument(
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Generation width. Default: {DEFAULT_WIDTH}.",
    )
    command.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Generation height. Default: {DEFAULT_HEIGHT}.",
    )


class LiveClock:
    """UTC time from the system clock."""

    def timestamp(self) -> str:
        """ISO-8601 timestamp with a timezone."""
        return datetime.now(timezone.utc).isoformat()


class LiveProbe:
    """``reset_peak_memory_stats`` before a phase and the CUDA peaks after it."""

    def reset(self) -> None:
        """Clear the CUDA peak counters for the phase that is about to run."""
        _torch_cuda().cuda.reset_peak_memory_stats()

    def sample(self) -> MemorySample:
        """``max_memory_allocated``, ``max_memory_reserved``, and peak RSS."""
        torch = _torch_cuda()
        return MemorySample(
            allocated_bytes=int(torch.cuda.max_memory_allocated()),
            reserved_bytes=int(torch.cuda.max_memory_reserved()),
            rss_bytes=rss_bytes(),
        )

    def gpu_name(self) -> str:
        """Name of CUDA device 0."""
        torch = _torch_cuda()
        return str(torch.cuda.get_device_properties(0).name)


class LiveSession:
    """Load the compact transformer and text encoder and keep them in memory."""

    def __init__(self) -> None:
        self._transformer: Any = None
        self._encoder: Any = None

    def load(self, weights: Path, *, quant_mode: str) -> None:
        """Read the compact transformer and text encoder. Does not download or save."""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        mode = _read_mode(weights)
        if mode != quant_mode:
            raise StageError(
                f"weights mode {mode} does not match voxprint-model.json quant mode {quant_mode}",
                code="moss_peak_quant",
            )
        import torch

        from voxprint_forge.diffusion.moss import import_vendored
        from voxprint_forge.diffusion.moss_prequant import materialize_quantization_config

        configs = materialize_quantization_config(mode)
        mod = import_vendored()
        transformer = mod.MOSSImageTransformer.from_pretrained(
            weights / "sft",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            quantization_config=configs["dit"],
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(
            weights / "encoder",
            dtype=torch.bfloat16,
            use_safetensors=True,
            trust_remote_code=False,
            quantization_config=configs["encoder"],
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self._transformer = transformer
        self._encoder = encoder

    def generate(
        self,
        *,
        steps: int,
        seed: int,
        rank: int,
        prompt: str,
        width: int,
        height: int,
    ) -> None:
        """Seeded dummy LoRA, ``steps`` of denoising, latent dropped. No image file."""
        transformer, encoder = self._ready()
        device = _cuda_device()
        transformer.to(device)
        encoder.to(device)
        # A is seeded kaiming and B is zero. Both stay in memory.
        layers = _attach_lora(transformer, rank, seed)
        transformer.eval()
        try:
            embeds, mask = _encode_caption(encoder, prompt, device)
            _denoise(
                transformer,
                embeds,
                mask,
                steps=steps,
                seed=seed,
                width=width,
                height=height,
                device=device,
            )
        finally:
            _detach_lora(transformer, layers)

    def train(self, *, steps: int, seed: int, rank: int, prompt: str) -> None:
        """Fresh seeded dummy LoRA and ``steps`` AdamW updates. No checkpoint and no LoRA file."""
        transformer, encoder = self._ready()
        device = _cuda_device()
        transformer.to(device)
        encoder.to(device)
        layers = _attach_lora(transformer, rank, seed)
        transformer.train()
        try:
            embeds, mask = _encode_caption(encoder, prompt, device)
            _train_steps(transformer, embeds, mask, layers, steps=steps, seed=seed, device=device)
        finally:
            _detach_lora(transformer, layers)

    def _ready(self) -> tuple[Any, Any]:
        if self._transformer is None or self._encoder is None:
            raise StageError("compact build is not loaded", code="moss_peak_load")
        return self._transformer, self._encoder


def _check_config(
    mode: str,
    rank: int,
    steps: int,
    train_steps: int,
    seed: int,
    width: int,
    height: int,
) -> None:
    if mode not in MODES:
        raise UsageError("mode must be gen, train, or both", code="moss_peak_mode")
    if rank < 1:
        raise UsageError("lora rank must be at least 1", code="moss_peak_args")
    if steps < 1:
        raise UsageError("steps must be at least 1", code="moss_peak_args")
    if train_steps < 1:
        raise UsageError("train steps must be at least 1", code="moss_peak_args")
    if seed < 0:
        raise UsageError("seed must be zero or positive", code="moss_peak_args")
    if width < 1 or height < 1:
        raise UsageError("width and height must be at least 1", code="moss_peak_args")


def _is_weights_dir(path: Path) -> bool:
    if not (path / "mode.json").is_file():
        return False
    return (path / "sft").is_dir() or (path / "encoder").is_dir()


def _manifest_near(weights: Path) -> tuple[Path, Path]:
    local = weights / "voxprint-model.json"
    if local.is_file():
        return weights, local
    parent = weights.parent / "voxprint-model.json"
    if parent.is_file():
        return weights.parent, parent
    raise StageError(
        f"voxprint-model.json not found in {weights} or {weights.parent}",
        code="moss_peak_manifest",
    )


def _quant_mode(document: Mapping[str, Any]) -> str:
    prequant = document.get("prequant")
    if isinstance(prequant, dict):
        chosen = prequant.get("chosen")
        if isinstance(chosen, str) and chosen:
            return chosen
    flags = document.get("flags")
    if isinstance(flags, list):
        for flag in flags:
            if isinstance(flag, str) and flag.startswith("chosen="):
                value = flag.split("=", 1)[1].strip()
                if value:
                    return value
    raise StageError("voxprint-model.json has no quant mode", code="moss_peak_quant")


def _read_json_object(path: Path, code: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageError(f"cannot read {path.name}", code=code) from exc
    if not isinstance(payload, dict):
        raise StageError(f"{path.name} is not a JSON object", code=code)
    return payload


def _read_mode(weights: Path) -> str:
    payload = _read_json_object(weights / "mode.json", "moss_peak_mode")
    mode = payload.get("mode")
    if not isinstance(mode, str) or not mode:
        raise StageError("mode.json has no mode", code="moss_peak_mode")
    return mode


def _is_inside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _looks_like_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "voxprint-model.json").is_file():
        return True
    return _is_weights_dir(path)


def _refuse_output(out: Path, location: BuildLocation) -> None:
    """Refuse a report path that resolves inside the build or any model directory."""
    target = out.resolve()
    protected = [location.root, location.weights]
    for root in protected:
        if _is_inside(root, target):
            raise StageError(
                f"refusing to write {out} inside model directory {root}",
                code="moss_peak_out",
            )
    cursor = target if target.is_dir() else target.parent
    for parent in [cursor, *cursor.parents]:
        resolved = parent.resolve()
        if _looks_like_model_dir(resolved):
            raise StageError(
                f"refusing to write {out} inside model directory {resolved}",
                code="moss_peak_out",
            )
        if resolved == resolved.parent:
            break


def _phase(steps: int, sample: MemorySample) -> dict[str, object]:
    allocated = int(sample.allocated_bytes)
    reserved = int(sample.reserved_bytes)
    return {
        "steps": int(steps),
        "allocated_bytes": allocated,
        "allocated_gb": allocated / _BYTES_PER_GB,
        "reserved_bytes": reserved,
        "reserved_gb": reserved / _BYTES_PER_GB,
        "rss_bytes": int(sample.rss_bytes),
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    """The only write this module performs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def _torch_cuda() -> Any:
    import torch

    if not torch.cuda.is_available():
        raise StageError("moss-peak needs a CUDA GPU to measure peaks", code="moss_peak_gpu")
    return torch


def _cuda_device() -> Any:
    return _torch_cuda().device("cuda")


def _attach_lora(model: Any, rank: int, seed: int) -> dict[str, Any]:
    """Seeded in-memory LoRA on the attention projections. Nothing is written."""
    import torch

    from voxprint_forge.lora.layers import inject

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        return inject(model, ATTENTION_TARGETS, int(rank), int(rank))
    except ValueError as exc:
        raise StageError(str(exc), code="moss_peak_lora") from exc


def _detach_lora(model: Any, layers: Mapping[str, Any]) -> None:
    for name, layer in layers.items():
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, layer.base)


def _latent_channels(transformer: Any) -> int:
    value = getattr(getattr(transformer, "config", None), "latent_channels", None)
    if not isinstance(value, int) or value < 1:
        raise StageError("transformer config has no latent_channels", code="moss_peak_config")
    return value


def _vocab_size(encoder: Any) -> int:
    config = getattr(encoder, "config", None)
    for owner in (config, getattr(config, "text_config", None)):
        value = getattr(owner, "vocab_size", None)
        if isinstance(value, int) and value >= 2:
            return value
    raise StageError("text encoder config has no vocab_size", code="moss_peak_encoder")


def _caption_token_ids(prompt: str, length: int, vocab: int) -> list[int]:
    """Deterministic token ids for ``prompt``. Not a tokenizer and not a download."""
    pool = hashlib.sha256(prompt.encode("utf-8")).digest()
    ids: list[int] = []
    while len(ids) < length:
        pool = hashlib.sha256(pool).digest()
        for byte in pool:
            ids.append(1 + (byte % (vocab - 1)))
            if len(ids) == length:
                break
    return ids


def _encode_caption(encoder: Any, prompt: str, device: Any) -> tuple[Any, Any]:
    """Fixed caption through the build's text encoder. The ids are derived, not downloaded."""
    import torch

    vocab = _vocab_size(encoder)
    ids = _caption_token_ids(prompt, CAPTION_TOKENS, vocab)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(input_ids)
    encoder.eval()
    with torch.no_grad():
        output = encoder(input_ids=input_ids, attention_mask=mask)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        hidden = output[0]
    return hidden, mask


def _denoise(
    transformer: Any,
    embeds: Any,
    mask: Any,
    *,
    steps: int,
    seed: int,
    width: int,
    height: int,
    device: Any,
) -> None:
    """Euler steps from noise. The compact build has no VAE; the latent is dropped, not written."""
    import torch

    channels = _latent_channels(transformer)
    latent_h = max(1, int(height) // 16)
    latent_w = max(1, int(width) // 16)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    latents = torch.randn(
        1,
        channels,
        latent_h,
        latent_w,
        generator=generator,
        dtype=torch.bfloat16,
        device="cpu",
    ).to(device)
    embeds = embeds.to(device)
    mask = mask.to(device)
    step = 1.0 / float(steps)
    image = latents
    with torch.no_grad():
        for index in range(int(steps)):
            sigma = 1.0 - index * step
            timestep = torch.tensor([sigma * 1000.0], device=device)
            pred = transformer(image, timestep, embeds, mask).sample
            if not isinstance(pred, torch.Tensor):
                raise StageError("generation got no sample", code="moss_peak_gen")
            image = image - pred.to(dtype=image.dtype) * step
    del image


def _train_steps(
    transformer: Any,
    embeds: Any,
    mask: Any,
    layers: Mapping[str, Any],
    *,
    steps: int,
    seed: int,
    device: Any,
) -> None:
    """AdamW on the LoRA only. Synthetic latents, flow-matching MSE (``noise - x0``). Nothing saved."""
    import torch

    from voxprint_forge.lora.layers import lora_parameters

    channels = _latent_channels(transformer)
    dtype = next(transformer.parameters()).dtype
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    embeds = embeds.to(device)
    mask = mask.to(device)
    opt = torch.optim.AdamW(lora_parameters(layers), lr=TRAIN_LR)
    try:
        for _ in range(int(steps)):
            x0 = torch.randn(
                1, channels, TRAIN_EDGE, TRAIN_EDGE, generator=generator, dtype=torch.float32
            ).to(device)
            noise = torch.randn(
                1, channels, TRAIN_EDGE, TRAIN_EDGE, generator=generator, dtype=torch.float32
            ).to(device)
            sigma = torch.sigmoid(torch.randn((), generator=generator)).to(device=device, dtype=torch.float32)
            xt = (1.0 - sigma) * x0 + sigma * noise
            pred = transformer(xt.to(dtype=dtype), sigma.reshape(1) * 1000.0, embeds, mask).sample
            if not isinstance(pred, torch.Tensor):
                raise StageError("train step got no sample", code="moss_peak_train")
            loss = torch.nn.functional.mse_loss(pred.float(), (noise - x0).float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    finally:
        opt.zero_grad(set_to_none=True)
        del opt
