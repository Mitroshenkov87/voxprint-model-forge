"""FP8 e4m3 weight-only MOSS DiT, saved once and loaded from disk.

torchao ``Float8Tensor.dtype`` stays bf16, so the vendored MOSS block still
casts activations to bf16 and the matmul runs in bf16. The bytes on disk are
float8_e4m3fn with a per-row scale (torchao 0.18.0, ``Float8WeightOnlyConfig``
version 2). The first materialize writes ``dit-fp8.safetensors`` and a
``voxprint-model/1`` manifest. Later loads read that file.

The manifest's primary source is our mirror. The revision and weight sha256
stay placeholders until the picture-quality check has passed and the upload
is pinned. Until then, and whenever a download fails, the bf16 snapshot is
quantized locally. ``models verify`` uses the same rule and does not hand
these ids to the upstream downloader.

torch and torchao are imported inside the quantize and load functions.
Importing them here would pull both into ``--version`` and into a verify
that only hashes files.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from voxprint_forge import __version__
from voxprint_forge.errors import ChecksumError, StageError
from voxprint_forge.mdl import (
    CUDA_CC,
    MANIFEST_NAME,
    build_document,
    infer_app_ids,
    write_manifest,
)

log = logging.getLogger("voxprint.moss_fp8")

WEIGHT_NAME = "dit-fp8.safetensors"
FORMAT = "voxprint-moss-fp8-v1"
TORCHAO_VERSION = "0.18.0"
QUANT_ID = "float8_e4m3fn,per_row,v2"
UNPUBLISHED = "unpublished"
ZERO_SHA = "0" * 64
CARD_DATE = "2026-10-25"
APACHE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"
RUNTIME_REQ = ">=0.38"
_CONFIG_KEYS = (
    "dim",
    "ffn_dim",
    "num_heads",
    "num_layers",
    "text_dim",
    "freq_dim",
    "latent_channels",
    "rope_base",
)

# Shared by Base and SFT at the pinned revisions (hashed from the snapshots).
_CONFIG = ("transformer/config.json", 240, "3291c3a1c7adc22b8182a0de25ca8ec97474d72715ce84adbab09cc0be24f666")
_INDEX = (
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    40298,
    "b609327fc75dbe5080e7e095d8b727d74d728b4f98e8b047133b0082cfa2fb03",
)
_BASE_SHARDS = (
    (
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        4991978016,
        "ae085fba123b6df67daf90aa9698fadee60c73e45bcaf3f3043d15f34e7a1804",
    ),
    (
        "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
        4983137016,
        "806e876d80c7fba560663a3a95580e6c7033da3f9ea3c52765245ef0e1818f75",
    ),
    (
        "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
        189549184,
        "5f319b72c40324294007c270d78112425627b537fcd8769b3c00f9f070e506a9",
    ),
)
_SFT_SHARDS = (
    (
        "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
        4991978016,
        "e40d3444940a0b0fd7cd45ba7959a5bd3d5efd7ddf87aae5ce4d87ce8aa03d01",
    ),
    (
        "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
        4983137016,
        "d106da790ec2924ffd8af04b4ffa8f8de281a974b58e55649c3ec596dc8ef033",
    ),
    (
        "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
        189549184,
        "11f637ab17c7093344a6110a1e87ffda8e56558dc55183d8e790222cfcdcf52d",
    ),
)


@dataclass(frozen=True)
class SourceFile:
    """One bf16 file the FP8 DiT is quantized from."""

    path: str
    size: int
    sha256: str


def _sources(shards: Sequence[tuple[str, int, str]]) -> tuple[SourceFile, ...]:
    rows = (_CONFIG, _INDEX, *shards)
    return tuple(SourceFile(path, size, digest) for path, size, digest in rows)


@dataclass(frozen=True)
class Fp8Pin:
    """One FP8 DiT: the bf16 snapshot it comes from, and our mirror once published."""

    key: str
    mdl_id: str
    source_mdl: str
    source_repo: str
    source_revision: str
    source_mirror: str
    source_mirror_revision: str
    mirror_repo: str
    mirror_revision: str
    weight_sha256: str
    weight_size: int
    display_name: str
    sources: tuple[SourceFile, ...]

    @property
    def published(self) -> bool:
        """True when the mirror revision and the weight sha256 are real pins, not placeholders."""
        revision = self.mirror_revision
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            return False
        digest = self.weight_sha256
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            return False
        if digest == ZERO_SHA or self.weight_size <= 0:
            return False
        return True


def _pin(
    key: str,
    *,
    source_repo: str,
    source_revision: str,
    source_mirror: str,
    source_mirror_revision: str,
    shards: Sequence[tuple[str, int, str]],
    display_name: str,
) -> Fp8Pin:
    return Fp8Pin(
        key,
        f"moss-image-{key}-fp8",
        f"moss-image-{key}",
        source_repo,
        source_revision,
        source_mirror,
        source_mirror_revision,
        f"Mitroshenkov87/voxprint-mirror-moss-image-{key}-fp8",
        UNPUBLISHED,
        ZERO_SHA,
        0,
        display_name,
        _sources(shards),
    )


PINS: tuple[Fp8Pin, ...] = (
    _pin(
        "base",
        source_repo="OpenMOSS-Team/MOSS-Image-Base",
        source_revision="1ceede80badddd6a7812069619edaa5633587192",
        source_mirror="Mitroshenkov87/voxprint-mirror-moss-image-base",
        source_mirror_revision="85cc5ca1dc0b6b74555715dec6164bb3e81f3c55",
        shards=_BASE_SHARDS,
        display_name="MOSS-Image Base DiT FP8",
    ),
    _pin(
        "sft",
        source_repo="OpenMOSS-Team/MOSS-Image-SFT",
        source_revision="fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
        source_mirror="Mitroshenkov87/voxprint-mirror-moss-image-sft",
        source_mirror_revision="2d0da610484827b96bdb81c44d702b5cf79a0689",
        shards=_SFT_SHARDS,
        display_name="MOSS-Image SFT DiT FP8",
    ),
)


def model_ids() -> frozenset[str]:
    """FP8 model ids. These are checked here and left out of the upstream downloader."""
    return frozenset(pin.mdl_id for pin in PINS)


def pin_for_source(folder: Path) -> Fp8Pin:
    """The FP8 pin for a snapshot directory named ``moss-image-base`` or ``moss-image-sft``."""
    name = folder.name
    for pin in PINS:
        if name == pin.source_mdl or name == pin.key:
            return pin
    raise StageError(f"no MOSS FP8 pin for {name}", code="moss_fp8")


def fp8_dir(source: Path, pin: Fp8Pin) -> Path:
    """``<models>/model-forge/moss-image-*-fp8``, next to the bf16 snapshot."""
    return source.parent / pin.mdl_id


def file_sha256(path: Path) -> str:
    """Hex sha256 of ``path``, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance_flags(pin: Fp8Pin) -> list[str]:
    """Flags a saved manifest must contain. A missing one means the file is stale."""
    flags = [
        "derivative",
        "modified",
        "fallback=quantize-bf16",
        f"source-repo={pin.source_repo}",
        f"source-revision={pin.source_revision}",
        f"source-mirror={pin.source_mirror}",
        f"torchao={TORCHAO_VERSION}",
        f"quant={QUANT_ID}",
    ]
    flags.extend(f"source-sha={item.path}:{item.sha256}" for item in pin.sources)
    return flags


def repo_root() -> Path:
    """Checkout root (the directory that contains ``LICENSE``)."""
    return Path(__file__).resolve().parents[3]


def card_text(pin: Fp8Pin, weight_sha: str, weight_size: int) -> str:
    """Model card. The weight sha256 is the placeholder until the file exists."""
    if weight_sha == ZERO_SHA or weight_size <= 0:
        weight_line = (
            f"`{WEIGHT_NAME}` sha256: pending (mirror revision `{pin.mirror_revision}`; "
            f"placeholder `{ZERO_SHA}` until the file is built and the mirror is pinned)."
        )
    else:
        weight_line = f"`{WEIGHT_NAME}` size {weight_size} sha256 `{weight_sha}`."
    source_rows = "\n".join(
        f"| `{item.path}` | {item.size} | `{item.sha256}` |" for item in pin.sources
    )
    return (
        "---\n"
        "license: apache-2.0\n"
        f"base_model: {pin.source_repo}\n"
        "tags:\n"
        "- voxprint\n"
        "- quantized\n"
        "- fp8\n"
        "---\n"
        "\n"
        f"# {pin.display_name}\n"
        "\n"
        "This is a **modified** Apache-2.0 derivative of "
        f"`{pin.source_repo}` at revision `{pin.source_revision}`.\n"
        f"The bf16 backup mirror is `{pin.source_mirror}` at `{pin.source_mirror_revision}`.\n"
        "LICENSE and NOTICE are included. The NOTICE states the modification.\n"
        "\n"
        "The DiT weights are stored as FP8 e4m3 (`float8_e4m3fn`) with per-row scales.\n"
        f"torchao `{TORCHAO_VERSION}`, `Float8WeightOnlyConfig(weight_dtype=float8_e4m3fn, "
        "granularity=PerRow(), version=2, set_inductor_config=False)`.\n"
        f"Quant id: `{QUANT_ID}`. Format: `{FORMAT}`.\n"
        "Activations and the matmul stay in bf16. `Float8Tensor.dtype` reports bf16, which is "
        "what the MOSS block reads before it casts activations.\n"
        "\n"
        "## Source files\n"
        "\n"
        "| Path | Size | sha256 |\n"
        "| --- | --- | --- |\n"
        f"{source_rows}\n"
        "\n"
        "## Quantized file\n"
        "\n"
        f"{weight_line}\n"
        "\n"
        f"Primary download: `{pin.mirror_repo}` revision `{pin.mirror_revision}`.\n"
        "Until that revision is a published commit, and if a download of the pinned file fails, "
        "the loader quantizes the bf16 snapshot above and saves the result here.\n"
        "\n"
        f"Card date: {CARD_DATE}.\n"
    )


def notice_text(pin: Fp8Pin) -> str:
    """NOTICE for the derivative. Apache-2.0 asks a modified work to say so."""
    return (
        f"{pin.display_name}\n"
        "Copyright 2026 Aleksandr Mitroshenkov\n"
        "\n"
        "This is a modified work. Voxprint quantized the MOSS-Image DiT to FP8 e4m3\n"
        f"weight-only with torchao {TORCHAO_VERSION} ({QUANT_ID}). Compute stays in bf16.\n"
        "The source revision, the source file sha256 values, the torchao version and the\n"
        "quant config are recorded in voxprint-model.json and README.md.\n"
        "\n"
        "Licensed under the Apache License, Version 2.0. The license text is the LICENSE\n"
        "file in this directory.\n"
    )


def upload_steps(built_root: Path | None = None) -> str:
    """Hugging Face commands for after the picture-quality check. Nothing here uploads."""
    lines = [
        "Dry run. Nothing was uploaded.",
        "Upload only after the picture-quality check passes.",
        "Upload the built model folder (it contains dit-fp8.safetensors), not this git template.",
        "The template has the card and a placeholder sha256; the weight is produced by the first quantize.",
    ]
    for pin in PINS:
        folder = str(built_root / pin.mdl_id) if built_root is not None else f"<models>/model-forge/{pin.mdl_id}"
        lines.append(f"huggingface-cli repo create {pin.mirror_repo} --type model")
        lines.append(f"huggingface-cli upload {pin.mirror_repo} {folder} . --repo-type model")
        lines.append(
            f"Pin {pin.mdl_id}: set mirror_revision to that 40-hex commit and "
            f"weight_sha256/weight_size from {WEIGHT_NAME}, then run this script again."
        )
    lines.append(
        "Until mirror_revision is 40 hex and the weight sha256 is not the zero placeholder, "
        "loading and check/repair fall back to quantizing the pinned bf16 snapshot."
    )
    return "\n".join(lines) + "\n"


def write_templates(output: Path) -> list[Path]:
    """Write the card, LICENSE, NOTICE and placeholder manifest for each pin. Does not upload."""
    written: list[Path] = []
    for pin in PINS:
        written.append(write_bundle(output / pin.mdl_id, pin, weight_sha=ZERO_SHA, weight_size=0))
    return written


def write_bundle(
    folder: Path,
    pin: Fp8Pin,
    *,
    weight_sha: str,
    weight_size: int,
    extra_flags: Sequence[str] = (),
) -> Path:
    """Write the card files and ``voxprint-model.json``. Returns the manifest path."""
    folder.mkdir(parents=True, exist_ok=True)
    _copy_license(folder)
    # newline="\n": Path.write_text otherwise turns \n into \r\n on Windows, and the
    # card sha256 in the manifest would not match the LF files this repo commits.
    (folder / "NOTICE").write_text(notice_text(pin), encoding="utf-8", newline="\n")
    (folder / "README.md").write_text(card_text(pin, weight_sha, weight_size), encoding="utf-8", newline="\n")
    files: list[dict[str, object]] = [
        _file_row(folder / "LICENSE"),
        _file_row(folder / "NOTICE"),
        _file_row(folder / "README.md"),
        {"path": WEIGHT_NAME, "size": int(weight_size), "sha256": weight_sha},
    ]
    flags = provenance_flags(pin)
    for flag in extra_flags:
        if flag not in flags:
            flags.append(flag)
    document = build_document(
        model_id=pin.mdl_id,
        display_name=pin.display_name,
        app_ids=infer_app_ids("text-to-image"),
        task="text-to-image",
        license="Apache-2.0",
        files=files,
        mirrors=[
            {"kind": "hf", "ref": pin.mirror_repo, "revision": pin.mirror_revision, "ours": True},
        ],
        min_vram_gb=16,
        recommended_vram_gb=16,
        gpu_min_cc=CUDA_CC,
        engine="diffusers",
        version_req=RUNTIME_REQ,
        base_model=pin.source_repo,
        card_url=f"https://huggingface.co/{pin.mirror_repo}",
        flags=flags,
        on_demand=True,
        forge_version=__version__,
        tag="",
        date=CARD_DATE,
    )
    return write_manifest(folder, document)


def status(folder: Path, pin: Fp8Pin) -> str:
    """``ok``, ``missing``, ``stale`` or ``corrupt``.

    ``stale`` is a provenance mismatch (source revision, source sha256, torchao
    version or quant config), or a published pin whose file is not the pinned
    mirror object and was not marked as a local fallback. ``corrupt`` is a
    readable manifest whose weight bytes do not match that manifest.
    """
    weight = folder / WEIGHT_NAME
    manifest = folder / MANIFEST_NAME
    if not weight.is_file() or not manifest.is_file():
        return "missing"
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return "corrupt"
    if not isinstance(document, dict) or not _provenance_ok(document, pin):
        return "stale"
    row = _weight_row(document)
    if row is None:
        return "corrupt"
    try:
        size = weight.stat().st_size
        digest = file_sha256(weight)
    except OSError:
        return "corrupt"
    if size != row.get("size") or digest != row.get("sha256"):
        return "corrupt"
    flags = document.get("flags")
    held = set(flags) if isinstance(flags, list) else set()
    if pin.published and digest != pin.weight_sha256 and "fallback-used=1" not in held:
        return "stale"
    return "ok"


def source_ready(snapshot: Path, pin: Fp8Pin) -> str:
    """``ok`` when every pinned source file matches, else ``missing`` or ``mismatch``."""
    if not snapshot.is_dir():
        return "missing"
    for item in pin.sources:
        path = snapshot / item.path
        if not path.is_file():
            return "missing"
        if path.stat().st_size != item.size or file_sha256(path) != item.sha256:
            return "mismatch"
    return "ok"


def materialize(
    folder: Path,
    pin: Fp8Pin,
    *,
    factory: Callable[[], object],
    produce: Callable[[], object],
    download: Callable[[], bool] | None = None,
) -> tuple[object, str]:
    """Return ``(module, action)`` with action ``cache``, ``download`` or ``quantize``.

    A cache hit does not call ``produce`` or ``download``. An unpublished mirror
    does not call ``download``. A download that raises or leaves a bad file falls
    through to ``produce``, which quantizes the bf16 snapshot.
    """
    if status(folder, pin) == "ok":
        module = factory()
        load_module(module, folder / WEIGHT_NAME)
        log.info("moss fp8 %s action=cache", pin.mdl_id)
        return module, "cache"
    got = False
    if pin.published:
        fetch = download if download is not None else (lambda: fetch_mirror(pin, folder))
        try:
            got = bool(fetch()) and status(folder, pin) == "ok"
        except Exception:
            log.info("moss fp8 %s mirror download failed; quantizing the bf16 snapshot", pin.mdl_id)
            got = False
    if not got:
        module = produce()
        save_module(module, folder, pin)
        log.info("moss fp8 %s action=quantize", pin.mdl_id)
        return module, "quantize"
    module = factory()
    load_module(module, folder / WEIGHT_NAME)
    log.info("moss fp8 %s action=download", pin.mdl_id)
    return module, "download"


def fetch_mirror(pin: Fp8Pin, folder: Path) -> bool:
    """Download the pinned mirror into ``folder``. Unpublished pins return False without a request."""
    if not pin.published:
        return False
    # Imported here so a verify that never downloads does not need the Hub call path.
    from huggingface_hub import snapshot_download

    folder.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=pin.mirror_repo, revision=pin.mirror_revision, local_dir=str(folder))
    weight = folder / WEIGHT_NAME
    if not weight.is_file():
        raise StageError(f"{pin.mirror_repo} has no {WEIGHT_NAME}", code="moss_fp8")
    digest = file_sha256(weight)
    if digest != pin.weight_sha256 or weight.stat().st_size != pin.weight_size:
        raise ChecksumError(
            f"{pin.mdl_id} mirror sha256 {digest} size {weight.stat().st_size} "
            f"!= pinned {pin.weight_sha256} size {pin.weight_size}",
            code="checksum",
        )
    return status(folder, pin) == "ok"


def quantize_module(module: object) -> object:
    """Replace every ``nn.Linear`` weight with torchao FP8 weight-only storage, in place."""
    _require_torchao()
    # torchao stays out of process startup; see the module docstring.
    import torch
    from torchao.quantization import Float8WeightOnlyConfig, quantize_
    from torchao.quantization.granularity import PerRow

    quantize_(
        module,
        Float8WeightOnlyConfig(
            weight_dtype=torch.float8_e4m3fn,
            granularity=PerRow(),
            version=2,
            set_inductor_config=False,
        ),
    )
    return module


def quantize_snapshot(source: Path, pin: Fp8Pin, *, verify_source: bool = True) -> object:
    """Load the bf16 DiT and quantize it. Refuses a snapshot whose shards do not match the pin."""
    if verify_source:
        ready = source_ready(source, pin)
        if ready != "ok":
            raise StageError(f"bf16 source {ready} for {pin.mdl_id}", code="moss_fp8_source")
    module = _load_bf16_transformer(source)
    return quantize_module(module)


def save_module(module: object, folder: Path, pin: Fp8Pin) -> Path:
    """Write ``dit-fp8.safetensors``, the card and a manifest that records the file sha256."""
    folder.mkdir(parents=True, exist_ok=True)
    weight = folder / WEIGHT_NAME
    _write_safetensors(module, weight)
    digest = file_sha256(weight)
    size = weight.stat().st_size
    extra: tuple[str, ...] = ()
    if pin.published and digest != pin.weight_sha256:
        extra = ("fallback-used=1",)
    return write_bundle(folder, pin, weight_sha=digest, weight_size=size, extra_flags=extra)


def load_module(module: object, path: Path) -> object:
    """Load a ``voxprint-moss-fp8-v1`` file onto ``module`` without allocating a bf16 copy of the weights."""
    # torch stays out of process startup; see the module docstring.
    import torch
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        meta = handle.metadata() or {}
        if meta.get("format") != FORMAT:
            raise StageError(f"{path.name} is not {FORMAT}", code="moss_fp8")
        storage = meta.get("qdata_storage", "float8_e4m3fn")
        blocks = json.loads(meta.get("blocks") or "{}")
        prefs = json.loads(meta.get("kernel_preference") or "{}")
        blobs = {key: handle.get_tensor(key) for key in handle.keys()}
    if not isinstance(blocks, dict) or not isinstance(prefs, dict):
        raise StageError(f"{path.name} is missing FP8 metadata", code="moss_fp8")
    seen: set[str] = set()
    for name, _param in list(module.named_parameters()):  # type: ignore[attr-defined]
        qkey = f"{name}.qdata"
        if qkey in blobs:
            qdata = blobs[qkey]
            if storage == "uint8":
                qdata = qdata.view(torch.float8_e4m3fn)
            block = blocks.get(name)
            block_size = [int(item) for item in block] if isinstance(block, list) and block else None
            tensor = _float8_tensor(
                qdata,
                blobs[f"{name}.scale"],
                block_size=block_size,
                kernel_name=str(prefs.get(name, "AUTO")),
            )
            _assign(module, name, tensor, requires_grad=False)
            seen.add(name)
        elif name in blobs:
            _assign(module, name, blobs[name], requires_grad=False)
            seen.add(name)
    for name, _buf in list(module.named_buffers()):  # type: ignore[attr-defined]
        if name in blobs and name not in seen:
            _assign_buffer(module, name, blobs[name])
    return module


def bind_device_move(module: object) -> object:
    """Make ``module.to`` move FP8 storage onto a device without casting ``qdata`` to bf16.

    ``stage`` and ``drop_vram`` call ``module.to``. A dtype cast would turn the
    float8 bytes back into bf16. Device-only moves rebuild ``Float8Tensor`` so
    ``qdata`` stays ``float8_e4m3fn``.
    """
    if getattr(module, "_voxprint_fp8_move", False):
        return module

    def moved(*args: object, **kwargs: object) -> object:
        # torch stays out of process startup; see the module docstring.
        import torch

        device, _dtype, non_blocking, _fmt = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            _move_module(module, device, bool(non_blocking))
        return module

    module.to = moved  # type: ignore[method-assign]
    module._voxprint_fp8_move = True  # type: ignore[attr-defined]
    return module


def empty_transformer(source: Path) -> object:
    """A MOSS DiT shell on the meta device, so a reload does not allocate bf16 weights."""
    # torch stays out of process startup; see the module docstring.
    import torch

    from voxprint_forge.diffusion.moss import import_vendored

    cfg = json.loads((source / "transformer" / "config.json").read_text(encoding="utf-8"))
    for key in ("_class_name", "_diffusers_version", "_name_or_path"):
        cfg.pop(key, None)
    kwargs = {key: cfg[key] for key in _CONFIG_KEYS if key in cfg}
    mod = import_vendored()
    with torch.device("meta"):
        return mod.MOSSImageTransformer(**kwargs)


def load_fp8_transformer(root: Path) -> object:
    """The quantized DiT. ``ModelPark.get`` keeps this module, not the pipeline.

    A later load attaches or omits the text encoder without reading the DiT again.
    The park boxes the module, so its tensor copy does not rewrite FP8 ``qdata``.
    """
    pin = pin_for_source(root)
    folder = fp8_dir(root, pin)

    def factory() -> object:
        return empty_transformer(root)

    def produce() -> object:
        return quantize_snapshot(root, pin)

    transformer, _action = materialize(folder, pin, factory=factory, produce=produce)
    bind_device_move(transformer)
    transformer.eval()  # type: ignore[attr-defined]
    transformer.requires_grad_(False)  # type: ignore[attr-defined]
    return transformer


def assemble_pipe(
    root: Path,
    transformer: object,
    *,
    with_text_encoder: bool = True,
    te_quant: str = "none",
    te_device: str = "cpu",
) -> object:
    """Pipeline around an already quantized DiT. Does not read the DiT weights."""
    # diffusers and transformers match moss.load_pipeline: imported on use, not at startup.
    import torch
    from diffusers import AutoencoderKLFlux2
    from transformers import AutoTokenizer

    from voxprint_forge.diffusion import moss as runtime

    with _hub_offline():
        runtime.check_snapshot(root)
        common = {"local_files_only": True}
        if with_text_encoder:
            tokenizer, text_encoder = runtime.load_text_encoder(
                root, dtype=torch.bfloat16, te_quant=te_quant, device=te_device
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", trust_remote_code=False, **common)
            text_encoder = None
        vae = AutoencoderKLFlux2.from_pretrained(
            root / "vae", torch_dtype=torch.float32, use_safetensors=True, **common
        ).eval()
        mod = runtime.import_vendored()
        scheduler = mod.MOSSImageScheduler.from_pretrained(root / "scheduler", **common)
        return mod.MOSSImagePipeline(transformer, scheduler, text_encoder, tokenizer, vae)


def load_training_pipe(
    root: Path,
    *,
    with_text_encoder: bool = True,
    te_quant: str = "none",
    te_device: str = "cpu",
) -> object:
    """MOSS pipeline whose DiT is the FP8 file. Quantizes the bf16 snapshot when that file is absent.

    Training parks :func:`load_fp8_transformer` and calls :func:`assemble_pipe` on each
    load, so the text encoder follows that load instead of the first one.
    """
    transformer = load_fp8_transformer(root)
    return assemble_pipe(
        root,
        transformer,
        with_text_encoder=with_text_encoder,
        te_quant=te_quant,
        te_device=te_device,
    )


def cover(root: Path, *, repair: bool) -> list[dict[str, object]]:
    """Verify an FP8 DiT that is already on disk. With ``repair``, rebuild a bad file.

    A bf16 snapshot alone is not a missing derivative: the 24 GB path never writes
    this file, and a plain verify must not fail until something creates the directory.
    Repair does not invent weights when the pinned source shards are absent or do not match.
    """
    app = root / "model-forge"
    if not app.is_dir():
        return []
    rows: list[dict[str, object]] = []
    for pin in PINS:
        folder = app / pin.mdl_id
        source = app / pin.source_mdl
        if not folder.exists():
            continue
        state = status(folder, pin)
        if repair and state != "ok":
            ready = source_ready(source, pin)
            if ready != "ok":
                rows.append(
                    {
                        "id": pin.mdl_id,
                        "status": "missing" if ready == "missing" else "corrupt",
                        "detail": f"source {ready}",
                    }
                )
                continue
            try:
                materialize(
                    folder,
                    pin,
                    factory=lambda source=source: empty_transformer(source),
                    produce=lambda source=source, pin=pin: quantize_snapshot(source, pin, verify_source=False),
                )
            except (OSError, StageError, ChecksumError, RuntimeError, ValueError) as exc:
                rows.append({"id": pin.mdl_id, "status": "corrupt", "detail": str(exc)})
                continue
            state = "repaired" if status(folder, pin) == "ok" else status(folder, pin)
        rows.append({"id": pin.mdl_id, "status": state})
    return rows


def _require_torchao() -> None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        found = version("torchao")
    except PackageNotFoundError as exc:
        raise StageError("torchao 0.18.0 is required to quantize the MOSS DiT", code="moss_fp8") from exc
    if found != TORCHAO_VERSION:
        raise StageError(f"torchao {found} != pinned {TORCHAO_VERSION}", code="moss_fp8")


def _load_bf16_transformer(source: Path) -> object:
    import torch

    from voxprint_forge.diffusion.moss import import_vendored

    mod = import_vendored()
    with _hub_offline():
        return mod.MOSSImageTransformer.from_pretrained(
            source / "transformer",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()


@contextmanager
def _hub_offline() -> Iterator[None]:
    """Read the local snapshot, then restore the process Hub flags.

    ``from_pretrained`` follows these variables. Leaving them set stops a later
    ``models verify --repair`` from downloading any other model.
    """
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = "1"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _copy_license(folder: Path) -> Path:
    src = repo_root() / "LICENSE"
    dest = folder / "LICENSE"
    shutil.copyfile(src, dest)
    digest = file_sha256(dest)
    if digest != APACHE_SHA256:
        raise StageError(f"LICENSE sha256 {digest} != Apache-2.0 {APACHE_SHA256}", code="moss_fp8")
    return dest


def _file_row(path: Path) -> dict[str, object]:
    return {"path": path.name, "size": path.stat().st_size, "sha256": file_sha256(path)}


def _provenance_ok(document: Mapping[str, object], pin: Fp8Pin) -> bool:
    flags = document.get("flags")
    if not isinstance(flags, list):
        return False
    held = {item for item in flags if isinstance(item, str)}
    return all(flag in held for flag in provenance_flags(pin))


def _weight_row(document: Mapping[str, object]) -> Mapping[str, object] | None:
    files = document.get("files")
    if not isinstance(files, list):
        return None
    for item in files:
        if isinstance(item, Mapping) and item.get("path") == WEIGHT_NAME:
            return item
    return None


def _as_fp8(tensor: object) -> object | None:
    if hasattr(tensor, "qdata") and hasattr(tensor, "scale"):
        return tensor
    data = getattr(tensor, "data", None)
    if data is not None and data is not tensor and hasattr(data, "qdata") and hasattr(data, "scale"):
        return data
    return None


def _float8_tensor(qdata: object, scale: object, *, block_size: list[int] | None, kernel_name: str) -> object:
    from torchao.quantization.quantize_.common import KernelPreference
    from torchao.quantization.quantize_.workflows import Float8Tensor

    try:
        preference = KernelPreference[kernel_name]
    except KeyError:
        preference = KernelPreference.AUTO
    import torch

    return Float8Tensor(
        qdata,
        scale,
        block_size=block_size,
        mm_config=None,
        act_quant_kwargs=None,
        kernel_preference=preference,
        dtype=torch.bfloat16,
    )


def _write_safetensors(module: object, path: Path) -> None:
    import torch
    from safetensors.torch import save_file

    tensors: dict[str, object] = {}
    blocks: dict[str, list[int]] = {}
    prefs: dict[str, str] = {}
    for name, param in module.named_parameters():  # type: ignore[attr-defined]
        fp8 = _as_fp8(param)
        if fp8 is None:
            tensors[name] = param.detach().cpu().contiguous()
            continue
        qdata = fp8.qdata.detach().cpu().contiguous()
        tensors[f"{name}.qdata"] = qdata
        tensors[f"{name}.scale"] = fp8.scale.detach().cpu().float().contiguous()
        block = getattr(fp8, "block_size", None)
        blocks[name] = [int(item) for item in block] if block is not None else []
        preference = getattr(fp8, "kernel_preference", None)
        prefs[name] = str(getattr(preference, "name", "AUTO"))
    for name, buf in module.named_buffers():  # type: ignore[attr-defined]
        if name not in tensors:
            tensors[name] = buf.detach().cpu().contiguous()
    metadata = {
        "format": FORMAT,
        "torchao": TORCHAO_VERSION,
        "quant": QUANT_ID,
        "qdata_storage": "float8_e4m3fn",
        "blocks": json.dumps(blocks, sort_keys=True),
        "kernel_preference": json.dumps(prefs, sort_keys=True),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        save_file(tensors, str(tmp), metadata=metadata)
    except (RuntimeError, ValueError, TypeError):
        converted = {}
        for key, value in tensors.items():
            if key.endswith(".qdata") and getattr(value, "dtype", None) == torch.float8_e4m3fn:
                converted[key] = value.view(torch.uint8)
            else:
                converted[key] = value
        metadata["qdata_storage"] = "uint8"
        save_file(converted, str(tmp), metadata=metadata)
    tmp.replace(path)


def _assign(module: object, name: str, tensor: object, *, requires_grad: bool) -> None:
    import torch
    from torch import nn

    parent, child = _split(module, name)
    current = getattr(parent, child, None)
    fp8 = _as_fp8(tensor)
    if (
        fp8 is None
        and isinstance(current, nn.Parameter)
        and not getattr(current, "is_meta", False)
        and isinstance(tensor, torch.Tensor)
    ):
        try:
            current.data.copy_(tensor.to(dtype=current.dtype))
            current.requires_grad_(requires_grad)
            return
        except (RuntimeError, TypeError, ValueError):
            pass
    if not isinstance(tensor, nn.Parameter):
        tensor = nn.Parameter(tensor, requires_grad=requires_grad)  # type: ignore[arg-type]
    setattr(parent, child, tensor)


def _assign_buffer(module: object, name: str, tensor: object) -> None:
    parent, child = _split(module, name)
    current = getattr(parent, child, None)
    try:
        if current is not None and not getattr(current, "is_meta", False):
            current.data.copy_(tensor)  # type: ignore[attr-defined]
            return
    except (RuntimeError, TypeError, ValueError, AttributeError):
        pass
    setattr(parent, child, tensor)


def _split(module: object, name: str) -> tuple[object, str]:
    parent_name, _, child = name.rpartition(".")
    parent = module.get_submodule(parent_name) if parent_name else module  # type: ignore[attr-defined]
    return parent, child


def _move_module(module: object, device: object, non_blocking: bool) -> None:
    import torch

    target = torch.device(device)  # type: ignore[arg-type]
    for name, param in list(module.named_parameters()):  # type: ignore[attr-defined]
        # Float8Tensor does not support ``or``: it tries to become a bool.
        fp8 = _as_fp8(param)
        if _same_device(fp8 if fp8 is not None else param, target):
            continue
        moved = _relocate(param, target, non_blocking)
        if _as_fp8(param) is not None or _as_fp8(moved) is not None:
            _assign(module, name, moved, requires_grad=bool(param.requires_grad))
        else:
            param.data = moved
    for name, buf in list(module.named_buffers()):  # type: ignore[attr-defined]
        if _same_device(buf, target):
            continue
        moved = _relocate(buf, target, non_blocking)
        try:
            buf.data = moved
        except (RuntimeError, TypeError, ValueError):
            _assign_buffer(module, name, moved)


def _relocate(tensor: object, device: object, non_blocking: bool) -> object:
    fp8 = _as_fp8(tensor)
    if fp8 is None:
        return tensor.to(device=device, non_blocking=non_blocking)  # type: ignore[attr-defined]
    qdata = fp8.qdata.to(device=device, non_blocking=non_blocking)
    scale = fp8.scale.to(device=device, non_blocking=non_blocking)
    block = getattr(fp8, "block_size", None)
    preference = getattr(fp8, "kernel_preference", None)
    return _float8_tensor(
        qdata,
        scale,
        block_size=[int(item) for item in block] if block is not None else None,
        kernel_name=str(getattr(preference, "name", "AUTO")),
    )


def _same_device(tensor: object, device: object) -> bool:
    import torch

    found = getattr(tensor, "device", None)
    if found is None:
        qdata = getattr(tensor, "qdata", None)
        found = getattr(qdata, "device", None)
    if found is None:
        return False
    return torch.device(found) == torch.device(device)  # type: ignore[arg-type]
