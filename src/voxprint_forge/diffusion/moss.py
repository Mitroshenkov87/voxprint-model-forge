"""MOSS-Image loader. It uses the vendored pipeline file and never ``trust_remote_code``.

The pinned author snapshots are in ``REPOS``. ``load_pipeline`` builds each component from a
local snapshot directory with ``local_files_only=True`` and ``use_safetensors=True`` under
``HF_HUB_OFFLINE=1``, so nothing is fetched and no repo code runs. diffusers/transformers are
imported lazily, so importing this module needs neither.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import math
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from torch import nn

from voxprint_forge.errors import ChecksumError, StageError

VENDORED = Path(__file__).resolve().parent.parent / "vendored" / "moss_image" / "pipeline_moss_image.py"
VENDORED_SHA256 = "1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31"
LICENSE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
MIN_DIFFUSERS = (0, 38)
MIN_TRANSFORMERS = (5, 3)
MIN_TORCH = (2, 9)


@dataclass(frozen=True)
class MossRepo:
    """One pinned author repo and our byte-identical backup mirror."""

    key: str
    repo: str
    revision: str
    mirror: str


REPOS = {
    "base": MossRepo(
        "base",
        "OpenMOSS-Team/MOSS-Image-Base",
        "1ceede80badddd6a7812069619edaa5633587192",
        "Mitroshenkov87/voxprint-mirror-moss-image-base",
    ),
    "sft": MossRepo(
        "sft",
        "OpenMOSS-Team/MOSS-Image-SFT",
        "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        "Mitroshenkov87/voxprint-mirror-moss-image-sft",
    ),
}

EXPECTED_INDEX = {
    "_class_name": "MOSSImagePipeline",
    "scheduler": ["pipeline_moss_image", "MOSSImageScheduler"],
    "text_encoder": ["transformers", "Qwen3_5Model"],
    "tokenizer": ["transformers", "Qwen2Tokenizer"],
    "transformer": ["pipeline_moss_image", "MOSSImageTransformer"],
    "vae": ["diffusers", "AutoencoderKLFlux2"],
}
UNSAFE_SUFFIXES = (".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt")
# Bare builtins only: ``module.eval()`` is torch's eval mode, not ``eval``.
FORBIDDEN_BUILTINS = frozenset({"exec", "eval", "compile", "__import__"})
# ``receiver.attr`` pairs: deserialisers and shells.
FORBIDDEN_ATTRS = frozenset(
    {("torch", "load"), ("pickle", "load"), ("pickle", "loads"), ("marshal", "loads"), ("dill", "load"),
     ("yaml", "load"), ("os", "system"), ("os", "popen")}
)
FORBIDDEN_MODULES = frozenset(
    {"pickle", "dill", "marshal", "subprocess", "socket", "requests", "urllib", "http", "huggingface_hub", "importlib"}
)


def sha256_bytes(data: bytes) -> str:
    """Hex sha256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def verify_vendored(path: Path = VENDORED) -> str:
    """Raise ChecksumError unless the vendored file matches the pinned sha256."""
    got = sha256_bytes(path.read_bytes())
    if got != VENDORED_SHA256:
        raise ChecksumError(f"vendored MOSS pipeline sha256 {got} != pinned {VENDORED_SHA256}", code="checksum")
    return got


def audit_source(source: str) -> list[str]:
    """Static review findings: forbidden imports and calls. Empty means clean."""
    findings: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                    findings.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in FORBIDDEN_MODULES:
                findings.append(f"line {node.lineno}: from {node.module} import")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_BUILTINS:
                findings.append(f"line {node.lineno}: call {func.id}()")
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if (func.value.id, func.attr) in FORBIDDEN_ATTRS:
                    findings.append(f"line {node.lineno}: call {func.value.id}.{func.attr}()")
            for kw in node.keywords:
                if kw.arg == "trust_remote_code" and not (
                    isinstance(kw.value, ast.Constant) and kw.value.value is False
                ):
                    findings.append(f"line {node.lineno}: trust_remote_code not False")
    return findings


def check_snapshot(root: Path) -> dict[str, object]:
    """Check a local snapshot: model_index classes as pinned, safetensors weights only."""
    index_path = root / "model_index.json"
    if not index_path.is_file():
        raise StageError(f"{root}: model_index.json missing", code="moss_snapshot")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    for key, want in EXPECTED_INDEX.items():
        if index.get(key) != want:
            raise StageError(f"model_index.json {key}={index.get(key)!r}, expected {want!r}", code="moss_snapshot")
    unsafe = sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.suffix.lower() in UNSAFE_SUFFIXES)
    if unsafe:
        raise StageError(f"non-safetensors weights in snapshot: {unsafe}", code="moss_snapshot")
    for sub in ("transformer", "text_encoder", "tokenizer", "vae", "scheduler"):
        if not (root / sub).is_dir():
            raise StageError(f"{root}: {sub}/ missing", code="moss_snapshot")
    return index


def _version(text: str) -> tuple[int, ...]:
    parts = []
    for piece in text.split("+")[0].split(".")[:2]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts)


def check_versions(torch_v: str, diffusers_v: str, transformers_v: str) -> list[str]:
    """Return problems with the runtime versions the vendored file needs."""
    problems = []
    for name, got, need in (
        ("torch", torch_v, MIN_TORCH),
        ("diffusers", diffusers_v, MIN_DIFFUSERS),
        ("transformers", transformers_v, MIN_TRANSFORMERS),
    ):
        if _version(got) < need:
            problems.append(f"{name} {got} < {'.'.join(map(str, need))}")
    return problems


def import_vendored() -> object:
    """Verify the pinned sha, then import the vendored module."""
    verify_vendored()
    from voxprint_forge.vendored.moss_image import pipeline_moss_image

    return pipeline_moss_image


def load_text_encoder(
    root: Path, *, dtype: object | None = None, te_quant: str = "none", device: str = "cpu"
) -> tuple[object, object]:
    """Tokenizer and text encoder from a local snapshot. Offline, safetensors only.

    ``device`` ``cuda`` loads the weights straight onto the card (bf16, or quantized with
    ``te_quant`` ``int8``/``nf4``, see :mod:`voxprint_forge.lora.te_quant`); ``cpu`` keeps bf16 in RAM.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    from voxprint_forge.lora import te_quant as quant

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    dtype = dtype or torch.bfloat16
    common = {"local_files_only": True}
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", trust_remote_code=False, **common)
    extra: dict[str, object] = {"low_cpu_mem_usage": True}
    qc = quant.quant_config(te_quant) if device == "cuda" else None
    if qc is not None:
        extra["quantization_config"] = qc
    if device == "cuda":
        extra["device_map"] = {"": "cuda"}
    text_encoder = AutoModel.from_pretrained(
        root / "text_encoder", dtype=dtype, use_safetensors=True, trust_remote_code=False, **common, **extra
    ).eval()
    return tokenizer, text_encoder


def load_pipeline(
    root: Path,
    *,
    dtype: object | None = None,
    vae_dtype: object | None = None,
    importer: Callable[[], object] = import_vendored,
    lora_path: Path | None = None,
    lora_scale: float = 1.0,
    with_text_encoder: bool = True,
    te_quant: str = "none",
    te_device: str = "cpu",
) -> object:
    """Build MOSSImagePipeline from a local snapshot. Offline, safetensors only, no remote code.

    ``lora_path`` is a ``lora export`` safetensors file. ``lora_scale`` multiplies the
    trained adapter (alpha / rank, or 1 when the file has no rank). 1 keeps that scale.
    ``with_text_encoder=False`` builds the pipeline with the tokenizer only, for callers that
    encode first and free the encoder before the DiT is read (see :func:`encode_staged`).
    """
    import torch
    from diffusers import AutoencoderKLFlux2
    from transformers import AutoTokenizer

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    check_snapshot(root)
    mod = importer()
    dtype = dtype or torch.bfloat16
    vae_dtype = vae_dtype or torch.float32
    common = {"local_files_only": True}
    if with_text_encoder:
        tokenizer, text_encoder = load_text_encoder(root, dtype=dtype, te_quant=te_quant, device=te_device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", trust_remote_code=False, **common)
        text_encoder = None
    transformer = mod.MOSSImageTransformer.from_pretrained(
        root / "transformer", torch_dtype=dtype, use_safetensors=True, low_cpu_mem_usage=True, **common
    ).eval()
    vae = AutoencoderKLFlux2.from_pretrained(root / "vae", torch_dtype=vae_dtype, use_safetensors=True, **common).eval()
    scheduler = mod.MOSSImageScheduler.from_pretrained(root / "scheduler", **common)
    pipe = mod.MOSSImagePipeline(transformer, scheduler, text_encoder, tokenizer, vae)
    if lora_path is not None:
        apply_lora(pipe.transformer, lora_path, lora_scale)
    return pipe


def load_encoder_parts(root: Path, *, te_quant: str = "none", te_device: str = "cpu") -> object:
    """Tokenizer, text encoder and VAE, frozen, without the DiT. ``transformer`` is ``None``."""
    import torch
    from diffusers import AutoencoderKLFlux2

    check_snapshot(root)
    tokenizer, text_encoder = load_text_encoder(root, te_quant=te_quant, device=te_device)
    vae = AutoencoderKLFlux2.from_pretrained(
        root / "vae", torch_dtype=torch.float32, use_safetensors=True, local_files_only=True
    ).eval()
    for part in (text_encoder, vae):
        part.requires_grad_(False)  # type: ignore[attr-defined]
    return SimpleNamespace(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=None)


def free_module(module: object | None) -> None:
    """Drop the last reference the caller handed over, then collect and empty the CUDA cache."""
    import gc

    import torch

    del module
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def encode_staged(
    root: Path,
    prompts: list[str],
    *,
    te_quant: str = "none",
    te_device: str = "cpu",
    loader: Callable[..., tuple[object, object]] | None = None,
) -> dict[str, tuple[object, object]]:
    """Encode ``prompts`` (and the empty negative) with only the text encoder resident, then free it.

    Returns CPU ``(embeds, mask)`` per prompt, batch dim kept. The DiT is not read until this returns,
    so the bf16 encoder (~18 GB) and the DiT (~10 GB) are never in RAM together.
    """
    tokenizer, text_encoder = (loader or load_text_encoder)(root, te_quant=te_quant, device=te_device)
    holder = SimpleNamespace(tokenizer=tokenizer, text_encoder=text_encoder)
    out: dict[str, tuple[object, object]] = {}
    try:
        for prompt in dict.fromkeys([*prompts, ""]):
            emb, mask = encode_on_cpu(holder, prompt)
            out[prompt] = (emb.detach().to("cpu"), mask.detach().to("cpu"))  # type: ignore[attr-defined]
    finally:
        holder.text_encoder = None
        del text_encoder
        free_module(None)
    return out


def pack_latents(vae: object, latents: object) -> object:
    """VAE sample ``[B, 32, H/8, W/8]`` to the normalized DiT latent ``[B, 128, H/16, W/16]``.

    The inverse of the unpacking in :func:`decode_latents`: patchify 2x2, then the VAE batch-norm.
    """
    zc = latents
    b, c, height2, width2 = zc.shape  # type: ignore[attr-defined]
    height, width = height2 // 2, width2 // 2
    z = zc.reshape(b, c, height, 2, width, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, height, width)
    bn = vae.bn  # type: ignore[attr-defined]
    eps = vae.config.batch_norm_eps  # type: ignore[attr-defined]
    mean = bn.running_mean.to(z).view(1, -1, 1, 1)
    scale = (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1)
    return ((z - mean) / scale).float()


def encode_image(vae: object, image_bchw_m1p1: object) -> object:
    """``image`` in ``[-1, 1]``, ``[B, 3, H, W]``, to the packed DiT latent."""
    device = next(vae.parameters()).device  # type: ignore[attr-defined]
    dtype = getattr(vae, "dtype", None)
    if not isinstance(dtype, torch.dtype):
        dtype = next(vae.parameters()).dtype  # type: ignore[attr-defined]
    posterior = vae.encode(image_bchw_m1p1.to(device=device, dtype=dtype))  # type: ignore[attr-defined]
    return pack_latents(vae, posterior.latent_dist.sample())


_LORA_PREFIXES = ("base_model.model.", "transformer.")


def _lora_module(key: str) -> tuple[str, str] | None:
    """``(module name, "A"|"B")`` for a Forge or PEFT LoRA key, else ``None``."""
    part = ""
    name = ""
    if key.endswith(".lora_A.weight"):
        name, part = key[: -len(".lora_A.weight")], "A"
    elif key.endswith(".lora_B.weight"):
        name, part = key[: -len(".lora_B.weight")], "B"
    else:
        return None
    changed = True
    while changed:
        changed = False
        for prefix in _LORA_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name, part


def _trained_scale(metadata: dict[str, str] | None, scale: float) -> float:
    if not math.isfinite(scale):
        raise StageError("LoRA scale must be a finite number", code="moss_lora")
    meta = metadata or {}
    rank, alpha = meta.get("voxprint.lora.rank"), meta.get("voxprint.lora.alpha")
    trained = 1.0
    if rank not in (None, "") and alpha not in (None, ""):
        trained = float(alpha) / float(rank)
    return float(scale) * trained


def apply_lora(module: object, path: Path, scale: float = 1.0) -> int:
    """Merge a ``lora export`` file into ``module`` (a MOSS transformer). Returns how many linears changed.

    The merge is ``W + scale * (alpha / rank) * B @ A``, which is what training adds on the forward.
    The modules stay ``nn.Linear``, so the vendored forward can still read ``weight.dtype``.
    """
    groups: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(str(path), framework="pt") as handle:
        factor = _trained_scale(handle.metadata(), scale)
        for key in handle.keys():
            parsed = _lora_module(key)
            if parsed is None:
                continue
            name, part = parsed
            groups.setdefault(name, {})[part] = handle.get_tensor(key)
    if not groups:
        raise StageError(f"{path.name} has no LoRA tensors", code="moss_lora")
    updated = 0
    for name, parts in groups.items():
        if "A" not in parts or "B" not in parts:
            raise StageError(f"{path.name} is missing a LoRA half for {name}", code="moss_lora")
        try:
            linear = module.get_submodule(name)  # type: ignore[attr-defined]
        except AttributeError as exc:
            raise StageError(f"{path.name} does not match this MOSS transformer ({name})", code="moss_lora") from exc
        if not isinstance(linear, nn.Linear):
            raise StageError(f"{path.name} does not match this MOSS transformer ({name})", code="moss_lora")
        delta = parts["B"].float() @ parts["A"].float()
        if tuple(delta.shape) != tuple(linear.weight.shape):
            raise StageError(f"{path.name} does not match this MOSS transformer ({name})", code="moss_lora")
        with torch.no_grad():
            linear.weight.add_(delta.to(dtype=linear.weight.dtype, device=linear.weight.device) * factor)
        updated += 1
    return updated


def generate(
    snapshot: Path,
    prompt: str,
    *,
    seed: int = 0,
    steps: int = 50,
    height: int = 1024,
    width: int = 1024,
    guidance: float = 3.0,
    lora_path: Path | None = None,
    lora_scale: float = 1.0,
    out: Path | None = None,
    te_quant: str | None = None,
    te_device: str | None = None,
) -> object:
    """One MOSS image from a local snapshot. ``lora_path`` is optional; ``lora_scale`` multiplies it.

    The prompt is encoded on the card first and the text encoder freed before the DiT is read (see
    :func:`encode_staged`). ``te_quant`` (``auto``, ``none``, ``int8``, ``nf4``) and ``te_device``
    (``gpu``, ``cpu``) follow :func:`voxprint_forge.lora.te_quant.plan` for the detected tier.
    With no CUDA card, ``auto`` keeps the 24 GB CPU path: there is no 16 GB card to refuse.
    """
    from voxprint_forge.lora import te_quant as quant
    from voxprint_forge.lora.presets import detect_vram_gb

    tier = detect_vram_gb() if quant.resolve(te_quant) == "auto" and torch.cuda.is_available() else 24
    choice = quant.plan(te_quant, te_device, tier)
    embeds = encode_staged(snapshot, [prompt], te_quant=choice.quant, te_device=choice.device)
    pipe = load_pipeline(snapshot, lora_path=lora_path, lora_scale=lora_scale, with_text_encoder=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe.transformer.to(device)
    if pipe.vae is not None:
        pipe.vae.to(device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    positive, negative = embeds[prompt], embeds[""]
    image = pipe(
        prompt_embeds=positive[0].to(device),
        prompt_attention_mask=positive[1].to(device),
        negative_prompt_embeds=negative[0].to(device),
        negative_prompt_attention_mask=negative[1].to(device),
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
    ).images[0]
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        image.save(out)
    return image


def generation_parser() -> argparse.ArgumentParser:
    """CLI for one MOSS image. ``--lora`` and ``--lora-scale`` are optional."""
    parser = argparse.ArgumentParser(prog="python -m voxprint_forge.diffusion.moss")
    parser.add_argument("--snapshot", type=Path, required=True, help="Local MOSS snapshot directory.")
    parser.add_argument("--prompt", required=True, help="Caption already written out. No extra tags are added.")
    parser.add_argument("--out", type=Path, required=True, help="Where to write the picture.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--lora", type=Path, default=None, help="LoRA .safetensors written by lora export.")
    parser.add_argument("--lora-scale", type=float, default=1.0, help="Multiplier on the trained LoRA. 1 keeps it.")
    parser.add_argument(
        "--te-quant", choices=("auto", "none", "int8", "nf4"), default=None,
        help="Text encoder format. Default: VOXPRINT_FORGE_TE_QUANT or auto (by graphics memory).",
    )
    parser.add_argument(
        "--te-device", choices=("gpu", "cpu"), default=None,
        help="Where the text encoder runs. Default: VOXPRINT_FORGE_TE_DEVICE or gpu; cpu is slower.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate one image. Same flags as :func:`generation_parser`."""
    args = generation_parser().parse_args(argv)
    generate(
        args.snapshot,
        args.prompt,
        seed=args.seed,
        steps=args.steps,
        height=args.height,
        width=args.width,
        guidance=args.guidance,
        lora_path=args.lora,
        lora_scale=args.lora_scale,
        out=args.out,
        te_quant=args.te_quant,
        te_device=args.te_device,
    )
    return 0


# Module-level kernels in transformers' Qwen3.5 linear-attention layers. With causal-conv1d or
# flash-linear-attention installed, transformers binds these names to the package kernel, which
# is CUDA-only (``Expected x.is_cuda()``) whatever device the module sits on.
CUDA_ONLY_KERNELS = (
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
)


@contextmanager
def torch_reference_kernels(model: object) -> Iterator[list[str]]:
    """Inside the block, the modeling modules of ``model`` call the pure-PyTorch kernels.

    Every name in ``CUDA_ONLY_KERNELS`` found in a module that defines one of ``model``'s
    submodule classes is replaced by ``inspect.unwrap`` of it (the decorated torch function)
    and restored on exit. Yields the ``module.name`` entries it swapped.
    """
    mods = {type(m).__module__ for m in model.modules()}  # type: ignore[attr-defined]
    saved: list[tuple[object, str, object]] = []
    try:
        for name in sorted(mods):
            mod = sys.modules.get(name)
            if mod is None:
                continue
            for fn_name in CUDA_ONLY_KERNELS:
                fn = getattr(mod, fn_name, None)
                if fn is None:
                    continue
                ref = inspect.unwrap(fn)
                if ref is not fn:
                    saved.append((mod, fn_name, fn))
                    setattr(mod, fn_name, ref)
        yield [f"{m.__name__}.{n}" for m, n, _ in saved]  # type: ignore[attr-defined]
    finally:
        for mod, fn_name, fn in reversed(saved):
            setattr(mod, fn_name, fn)


def encode_on_cpu(pipe: object, prompt: str, max_sequence_length: int = 4096) -> tuple[object, object]:
    """Same as the vendored ``encode_prompt`` for one prompt, with the text encoder on CPU.

    The vendored method moves the inputs to ``_execution_device``. That is the GPU once the
    transformer sits there, which breaks a CPU text encoder. The inputs follow the encoder's
    own device, and on CPU the CUDA-only conv/delta-rule kernels are swapped for the torch
    reference (see ``torch_reference_kernels``). Embeds and mask come back on that device; the
    caller moves both to the transformer's device.
    """
    import torch

    device = next(pipe.text_encoder.parameters()).device
    formatted = pipe.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    inputs = pipe.tokenizer(formatted, return_tensors="pt", truncation=True, max_length=max_sequence_length).to(device)
    with torch.no_grad():
        if device.type == "cpu":
            with torch_reference_kernels(pipe.text_encoder):
                row = pipe.text_encoder(**inputs, use_cache=False, return_dict=True).last_hidden_state
        else:
            row = pipe.text_encoder(**inputs, use_cache=False, return_dict=True).last_hidden_state
    mask = torch.ones(row.shape[:2], dtype=torch.bool, device=row.device)
    return row, mask



def decode_latents(pipe: object, latents: object) -> object:
    """The vendored ``__call__`` post-processing for ``output_type="latent"`` results, then PIL images.

    Same math as the pipeline after its denoise loop: batch-norm de-normalisation, 2x2 unpatchify,
    ``vae.decode`` (untiled), clamp, postprocess. Called after the denoise working set is released,
    so the decode does not stack on cached denoise blocks. Tiled decode is not offered: on the pinned
    FLUX.2 VAE, 512 px tiles give 23.3 dB PSNR against the untiled decode at 1024x1024 (global colour
    shift from the mid-block attention and group norms seeing one tile).
    """
    import torch

    vae = pipe.vae  # type: ignore[attr-defined]
    with torch.no_grad():
        z = latents.to(dtype=vae.dtype)  # type: ignore[attr-defined]
        bn = vae.bn
        eps = vae.config.batch_norm_eps
        z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(1, -1, 1, 1)
        b, c, h, w = z.shape
        z = z.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
        images = vae.decode(z, return_dict=False)[0].clamp(-1, 1)
    return pipe.image_processor.postprocess(images, output_type="pil")  # type: ignore[attr-defined]


if __name__ == "__main__":
    raise SystemExit(main())
