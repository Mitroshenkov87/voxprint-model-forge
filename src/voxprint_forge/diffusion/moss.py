"""MOSS-Image loader. It uses the vendored pipeline file and never ``trust_remote_code``.

The pinned author snapshots are in ``REPOS``. ``load_pipeline`` builds each component from a
local snapshot directory with ``local_files_only=True`` and ``use_safetensors=True`` under
``HF_HUB_OFFLINE=1``, so nothing is fetched and no repo code runs. diffusers/transformers are
imported lazily: CI pins diffusers <0.38, which has no ``AutoencoderKLFlux2``.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

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


def load_pipeline(
    root: Path,
    *,
    dtype: object | None = None,
    vae_dtype: object | None = None,
    importer: Callable[[], object] = import_vendored,
) -> object:
    """Build MOSSImagePipeline from a local snapshot. Offline, safetensors only, no remote code."""
    import torch
    from diffusers import AutoencoderKLFlux2
    from transformers import AutoModel, AutoTokenizer

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    check_snapshot(root)
    mod = importer()
    dtype = dtype or torch.bfloat16
    vae_dtype = vae_dtype or torch.float32
    common = {"local_files_only": True}
    transformer = mod.MOSSImageTransformer.from_pretrained(
        root / "transformer", torch_dtype=dtype, use_safetensors=True, **common
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", trust_remote_code=False, **common)
    text_encoder = AutoModel.from_pretrained(
        root / "text_encoder", dtype=dtype, use_safetensors=True, trust_remote_code=False, **common
    ).eval()
    vae = AutoencoderKLFlux2.from_pretrained(root / "vae", torch_dtype=vae_dtype, use_safetensors=True, **common).eval()
    scheduler = mod.MOSSImageScheduler.from_pretrained(root / "scheduler", **common)
    return mod.MOSSImagePipeline(transformer, scheduler, text_encoder, tokenizer, vae)


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
