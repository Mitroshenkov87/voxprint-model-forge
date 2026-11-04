"""One-time MOSS pre-quant build (``forge moss-prequant``).

The command checks a local MOSS-Image SFT snapshot and the resharded
Qwen3.5-9B encoder against pinned per-file sha256. It never downloads.
MOSS-Image Base is not a default target. It caps the process at the VRAM
budget (default 13.5 GB, allowed 13–14), measures fp8_e4m3, int8, and nf4,
and keeps the best quality that fits. fp8_e4m3 is skipped below compute
capability 8.9 (sm 89). The manifest id and the weights directory are
``voxprint-moss-image-sft-compact``.

Heavy pieces (loader, quantizer, renderer, scorer, GPU probe) are injected.
Tests pass stand-ins. The live objects are the calibration-VM path and are
not what the tests call.

Upload is out of scope. Load-time quantization stays a fallback for a card
this file does not fit; this command is the build that publishes the file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Protocol

from voxprint_forge.diffusion.prompts import BASE_PROMPTS
from voxprint_forge.diffusion.quant_select import (
    MODES,
    Candidate,
    Gpu,
    QuantSelectError,
    choose,
    mode_by_name,
)
from voxprint_forge.errors import ChecksumError, StageError, UsageError
from voxprint_forge.lora.presets import MOSS_VRAM_BUDGET_GB

DEFAULT_BUDGET_GB = MOSS_VRAM_BUDGET_GB
BUDGET_MIN_GB = 13.0
BUDGET_MAX_GB = 14.0
VRAM_CAP_ENV = "FORGE_VRAM_CAP_GB"
MODEL_ID = "voxprint-moss-image-sft-compact"
DISPLAY_NAME = "MOSS-Image SFT compact 16GB"
MIRROR_REPO = "Mitroshenkov87/voxprint-moss-image-sft-compact"
MIRROR_REVISION = "unpublished"
SFT_REPO = "OpenMOSS-Team/MOSS-Image-SFT"
SFT_REVISION = "fc5de0d56fc910cbca6f6625a70f5f582b26deb1"
ENCODER_REPO = "Qwen/Qwen3.5-9B"
VAE_REPO = "black-forest-labs/FLUX.2-small-decoder"
GENERATE_STEPS = 50
TRAIN_PROBE_EDGE = 16
WEIGHTS_DIR = "voxprint-moss-image-sft-compact"

# MOSS-Image Base file list. Only the text_encoder/ rows feed the encoder pins.
# The app catalog id moss-image-base lists the SFT snapshot. These Base
# transformer hashes are not a default prequant target.
_BASE_ROWS: tuple[tuple[str, int, str], ...] = (
    ("LICENSE", 11358, "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
    ("model_index.json", 412, "ca80b1f562a506e0eb24afb1acb464fb583163e3c240c04e4bc69fea090db173"),
    ("pipeline_moss_image.py", 25060, "1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31"),
    ("scheduler/pipeline_moss_image.py", 25060, "1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31"),
    ("scheduler/scheduler_config.json", 494, "707322832cc507dfb8c51c0e0cd04d44ee2d30a9b036b6f3c47d6c17aa71adf0"),
    ("text_encoder/config.json", 2813, "b31ee18a242bcfcccbe740f825f6b1d3ae231b0655eeecb5de3e98d504b7d68c"),
    (
        "text_encoder/model-00001-of-00004.safetensors",
        4973910744,
        "abfaa3d3a8612508c85ee8f6da9c734b862f10298b903765c4ad384177c32b0b",
    ),
    (
        "text_encoder/model-00002-of-00004.safetensors",
        4988364288,
        "9112637fa6bf4d07bafd6ae268852b18d3d0c1242f4ee901c5579006e5bd4894",
    ),
    (
        "text_encoder/model-00003-of-00004.safetensors",
        4954202112,
        "956f1ef51d0cfe610b0d22987c0b7821c36e3ce228abceccef523518bcf079dc",
    ),
    (
        "text_encoder/model-00004-of-00004.safetensors",
        1869001848,
        "9e21d20e8adde88c3e0b65d4a0d251f143d98ee9d52db36b4348361091e4e208",
    ),
    (
        "text_encoder/model.safetensors.index.json",
        64641,
        "0af6fa3fa36ba8dcbb6b7ad0325a36c5b1b69862a9a1f1486b1b80385bd8f5ee",
    ),
    ("tokenizer/chat_template.jinja", 7756, "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"),
    ("tokenizer/tokenizer.json", 19989325, "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
    ("tokenizer/tokenizer_config.json", 1123, "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87"),
    ("transformer/config.json", 240, "3291c3a1c7adc22b8182a0de25ca8ec97474d72715ce84adbab09cc0be24f666"),
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
    (
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        40298,
        "b609327fc75dbe5080e7e095d8b727d74d728b4f98e8b047133b0082cfa2fb03",
    ),
    ("transformer/pipeline_moss_image.py", 25060, "1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31"),
    ("vae/config.json", 838, "cc77f3f0ce0357039c71018dc19edaa4fcceb0c2f7488fdc2b9c0580d6baafa1"),
    (
        "vae/diffusion_pytorch_model.safetensors",
        249521372,
        "c3bdf94dc67ec69b30ea9fe1fa8340085679cae64d019f7c7e4a9921f1a7f17a",
    ),
)
# SFT transformer shards differ from Base. config and index match. text_encoder,
# tokenizer, and vae are byte-identical to Base (plans/moss-synth4.toml).
_SFT_ROWS: tuple[tuple[str, int, str], ...] = (
    ("transformer/config.json", 240, "3291c3a1c7adc22b8182a0de25ca8ec97474d72715ce84adbab09cc0be24f666"),
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
    (
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        40298,
        "b609327fc75dbe5080e7e095d8b727d74d728b4f98e8b047133b0082cfa2fb03",
    ),
    ("text_encoder/config.json", 2813, "b31ee18a242bcfcccbe740f825f6b1d3ae231b0655eeecb5de3e98d504b7d68c"),
    (
        "text_encoder/model-00001-of-00004.safetensors",
        4973910744,
        "abfaa3d3a8612508c85ee8f6da9c734b862f10298b903765c4ad384177c32b0b",
    ),
    (
        "text_encoder/model-00002-of-00004.safetensors",
        4988364288,
        "9112637fa6bf4d07bafd6ae268852b18d3d0c1242f4ee901c5579006e5bd4894",
    ),
    (
        "text_encoder/model-00003-of-00004.safetensors",
        4954202112,
        "956f1ef51d0cfe610b0d22987c0b7821c36e3ce228abceccef523518bcf079dc",
    ),
    (
        "text_encoder/model-00004-of-00004.safetensors",
        1869001848,
        "9e21d20e8adde88c3e0b65d4a0d251f143d98ee9d52db36b4348361091e4e208",
    ),
    (
        "text_encoder/model.safetensors.index.json",
        64641,
        "0af6fa3fa36ba8dcbb6b7ad0325a36c5b1b69862a9a1f1486b1b80385bd8f5ee",
    ),
    ("tokenizer/chat_template.jinja", 7756, "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"),
    ("tokenizer/tokenizer.json", 19989325, "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"),
    ("tokenizer/tokenizer_config.json", 1123, "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87"),
    ("vae/config.json", 838, "cc77f3f0ce0357039c71018dc19edaa4fcceb0c2f7488fdc2b9c0580d6baafa1"),
    (
        "vae/diffusion_pytorch_model.safetensors",
        249521372,
        "c3bdf94dc67ec69b30ea9fe1fa8340085679cae64d019f7c7e4a9921f1a7f17a",
    ),
)
# The encoder directory is the resharded Qwen3.5-9B text encoder (MOSS
# ``text_encoder/``), not the upstream Qwen repo. Those weights are not byte-identical.
_ENCODER_ROWS: tuple[tuple[str, int, str], ...] = tuple(
    (path.removeprefix("text_encoder/"), size, digest)
    for path, size, digest in _BASE_ROWS
    if path.startswith("text_encoder/")
)


class NoQuantFits(StageError):
    """No fp8_e4m3, int8, or nf4 candidate fits the VRAM budget."""

    def __init__(self, message: str, *, report: dict[str, object] | None = None) -> None:
        super().__init__(message, code="moss_prequant_none_fit", report=report)


@dataclass(frozen=True)
class PinnedFile:
    """One file that must already be on disk. ``path`` is relative to the snapshot root."""

    path: str
    sha256: str
    size: int


@dataclass(frozen=True)
class SnapshotSet:
    """Local originals. Nothing here is fetched.

    Default targets are the SFT snapshot and the Qwen3.5-9B encoder.
    """

    sft: Path
    encoder: Path

    def roots(self) -> dict[str, Path]:
        """Role name to directory."""
        return {"sft": self.sft, "encoder": self.encoder}


@dataclass(frozen=True)
class PromptCase:
    """One reference-set item. Every mode sees the same prompt and seed."""

    prompt: str
    seed: int


@dataclass(frozen=True)
class Quality:
    """One pair. Lower ``lpips`` is better. Higher ``clip`` is better."""

    lpips: float
    clip: float


@dataclass(frozen=True)
class PrequantResult:
    """What ``forge moss-prequant`` prints."""

    chosen: str
    budget_gb: float
    out: Path
    manifest: Path
    peaks: dict[str, dict[str, float]]
    quality: dict[str, dict[str, float]]
    skipped: tuple[str, ...]
    min_sm: dict[str, int]
    gpu: Gpu

    def to_dict(self) -> dict[str, object]:
        """JSON-ready fields, including the manifest's quant record."""
        return {
            "chosen": self.chosen,
            "budget_gb": self.budget_gb,
            "out": str(self.out),
            "manifest": str(self.manifest),
            "quant_modes": [mode.name for mode in MODES],
            "measured_peak_gb": self.peaks,
            "min_sm": self.min_sm,
            "skipped": list(self.skipped),
            "quality": self.quality,
            "gpu": {"name": self.gpu.name, "vram_gb": self.gpu.vram_gb, "sm": self.gpu.sm},
        }


class Loader(Protocol):
    """Opens the local originals. Implementations must not download."""

    def open(self, snapshots: SnapshotSet) -> None:
        """Prepare ``snapshots`` for a local read."""


class Quantizer(Protocol):
    """Builds, reloads, and probes one quant mode."""

    def config(self, mode: str) -> Mapping[str, object]:
        """Quantization spec for ``mode``."""

    def build(self, mode: str, snapshots: SnapshotSet, dest: Path) -> None:
        """Write ``mode`` into ``dest`` from the local snapshots."""

    def reload(self, dest: Path) -> None:
        """Load ``dest`` back. Leave the files unchanged when the load matches."""

    def train_probe(self, dest: Path) -> None:
        """One short LoRA training step on ``dest``."""


class Renderer(Protocol):
    """Draws the reference and one candidate. Same prompt and seed for every mode."""

    def render_reference(self, snapshots: SnapshotSet, prompt: str, seed: int) -> bytes:
        """Render the unquantized reference."""

    def render(self, weights: Path, prompt: str, seed: int, *, lora: bool) -> bytes:
        """Render ``weights``. ``lora`` is true for the generation-with-LoRA measurement."""


class Scorer(Protocol):
    """CLIP and LPIPS against the reference. The live scorer uses ``compare.score``."""

    def score(self, reference: bytes, candidate: bytes, prompt: str) -> Quality:
        """Score one pair."""


class GpuProbe(Protocol):
    """The card, the process cap, and a peak around one callback."""

    def gpu(self) -> Gpu:
        """The card this process measures on."""

    def cap(self, budget_gb: float) -> None:
        """Apply ``budget_gb`` as the process VRAM cap."""

    def measure(self, mode: str, kind: str, fn: Callable[[], None]) -> float:
        """Run ``fn`` and return the peak in GB. ``kind`` is ``generate`` or ``train``."""


@dataclass
class Services:
    """Injected backends. Tests pass stand-ins. The CLI uses :func:`live_services`."""

    loader: Loader
    quantizer: Quantizer
    renderer: Renderer
    scorer: Scorer
    probe: GpuProbe


def _pins(rows: Sequence[tuple[str, int, str]]) -> tuple[PinnedFile, ...]:
    return tuple(PinnedFile(path, digest, size) for path, size, digest in rows)


def default_pins() -> dict[str, tuple[PinnedFile, ...]]:
    """Pinned files for the SFT snapshot and the Qwen3.5-9B encoder.

    MOSS-Image Base is not a default target. ``_BASE_ROWS`` stays so the
    encoder pin list (the resharded ``text_encoder/`` files) has one source.
    """
    return {"sft": _pins(_SFT_ROWS), "encoder": _pins(_ENCODER_ROWS)}


def default_cases() -> tuple[PromptCase, ...]:
    """The reference set: the first public prompts, each at seed 0."""
    return tuple(PromptCase(prompt, 0) for prompt in BASE_PROMPTS[:3])


def check_budget(budget_gb: float) -> float:
    """Return ``budget_gb`` when it is inside 13–14 GB."""
    value = float(budget_gb)
    if value < BUDGET_MIN_GB or value > BUDGET_MAX_GB:
        raise UsageError(
            f"vram budget {budget_gb} GB is outside {BUDGET_MIN_GB:g}–{BUDGET_MAX_GB:g}",
            code="moss_prequant_budget",
        )
    return value


def quantization_spec(mode: str) -> dict[str, object]:
    """Diffusers/transformers quantization_config fields for ``mode``. No heavy imports.

    The live builder turns this into a bitsandbytes ``BitsAndBytesConfig`` or a
    torchao ``TorchAoConfig`` (``Float8WeightOnlyConfig``, float8_e4m3fn).
    """
    item = mode_by_name(mode)
    if item.name == "fp8_e4m3":
        return {
            "mode": item.name,
            "backend": "torchao",
            "libraries": ["diffusers", "torchao"],
            "class": "TorchAoConfig",
            "weight_dtype": "float8_e4m3fn",
            "granularity": "per_row",
            "version": 2,
            "min_sm": item.min_sm,
        }
    if item.name == "int8":
        return {
            "mode": item.name,
            "backend": "bitsandbytes",
            "libraries": ["diffusers", "transformers"],
            "class": "BitsAndBytesConfig",
            "load_in_8bit": True,
            "min_sm": item.min_sm,
        }
    return {
        "mode": item.name,
        "backend": "bitsandbytes",
        "libraries": ["diffusers", "transformers"],
        "class": "BitsAndBytesConfig",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_use_double_quant": False,
        "min_sm": item.min_sm,
    }


def verify_snapshots(
    snapshots: SnapshotSet,
    pins: Mapping[str, Sequence[PinnedFile]] | None = None,
) -> None:
    """Hash the pinned files. Missing files and mismatches stop the build. Never downloads."""
    table = pins if pins is not None else default_pins()
    for role, root in snapshots.roots().items():
        if role not in table:
            raise StageError(f"no pins for {role}", code="moss_prequant_pin")
        if not root.is_dir():
            raise StageError(f"{role} snapshot missing: {root}", code="moss_prequant_missing")
        for item in table[role]:
            path = _safe_file(root, item.path)
            if not path.is_file():
                raise StageError(f"{role} missing {item.path}", code="moss_prequant_missing")
            digest = _sha256_file(path)
            if digest != item.sha256 or path.stat().st_size != item.size:
                raise ChecksumError(
                    f"{role} {item.path} sha256 {digest} size {path.stat().st_size} "
                    f"!= pinned {item.sha256} size {item.size}",
                    code="moss_prequant_sha",
                )


def run(
    snapshots: SnapshotSet,
    out: Path,
    *,
    budget_gb: float = DEFAULT_BUDGET_GB,
    sm: int | None = None,
    pins: Mapping[str, Sequence[PinnedFile]] | None = None,
    cases: Sequence[PromptCase] | None = None,
    services: Services | None = None,
) -> PrequantResult:
    """Verify, measure, choose, save, reload, and write the card and manifest.

    ``services`` is required for tests. The CLI passes :func:`live_services`.
    """
    budget = check_budget(budget_gb)
    if not cases:
        cases = default_cases()
    verify_snapshots(snapshots, pins)
    backends = services if services is not None else live_services(snapshots)
    previous = os.environ.get(VRAM_CAP_ENV)
    os.environ[VRAM_CAP_ENV] = f"{budget:g}"
    try:
        backends.probe.cap(budget)
        backends.loader.open(snapshots)
        gpu = backends.probe.gpu()
        if sm is not None:
            gpu = Gpu(gpu.name, gpu.vram_gb, int(sm))
        references = [backends.renderer.render_reference(snapshots, case.prompt, case.seed) for case in cases]
        skipped: list[str] = []
        candidates: list[Candidate] = []
        built: dict[str, tuple[Path, list[bytes]]] = {}
        quality: dict[str, dict[str, float]] = {}
        with tempfile.TemporaryDirectory(prefix="moss-prequant-") as tmp:
            root = Path(tmp)
            for mode in MODES:
                if gpu.sm < mode.min_sm:
                    skipped.append(mode.name)
                    continue
                spec = backends.quantizer.config(mode.name)
                if spec.get("mode") != mode.name:
                    raise StageError(
                        f"{mode.name} quantization_config mode is {spec.get('mode')!r}",
                        code="moss_prequant_config",
                    )
                dest = root / mode.name
                dest.mkdir(parents=True)
                backends.quantizer.build(mode.name, snapshots, dest)
                images: list[bytes] = []

                def _generate(dest: Path = dest, images: list[bytes] = images) -> None:
                    for case in cases:
                        images.append(backends.renderer.render(dest, case.prompt, case.seed, lora=True))

                generate_gb = float(backends.probe.measure(mode.name, "generate", _generate))
                train_gb = float(
                    backends.probe.measure(mode.name, "train", lambda dest=dest: backends.quantizer.train_probe(dest))
                )
                scores = [
                    backends.scorer.score(reference, image, case.prompt)
                    for case, reference, image in zip(cases, references, images, strict=True)
                ]
                lpips = _mean([item.lpips for item in scores])
                clip = _mean([item.clip for item in scores])
                quality[mode.name] = {"lpips": lpips, "clip": clip}
                candidates.append(
                    Candidate(mode.name, mode.min_sm, generate_gb, train_gb, lpips, clip)
                )
                built[mode.name] = (dest, images)
            try:
                chosen = choose(candidates, gpu, budget)
            except QuantSelectError as exc:
                report = {
                    "budget_gb": budget,
                    "gpu": {"name": gpu.name, "vram_gb": gpu.vram_gb, "sm": gpu.sm},
                    "skipped": list(skipped),
                    "measured_peak_gb": {
                        item.mode: {"generate": item.generate_gb, "train": item.train_gb} for item in candidates
                    },
                }
                raise NoQuantFits(str(exc), report=report) from exc
            weights = out / WEIGHTS_DIR
            if weights.exists():
                raise StageError(f"{weights} already exists", code="moss_prequant_out")
            out.mkdir(parents=True, exist_ok=True)
            shutil.copytree(built[chosen.mode][0], weights)
            first_images = built[chosen.mode][1]
        _write_sidecars(weights)
        backends.quantizer.reload(weights)
        _assert_sidecars(weights)
        again = [backends.renderer.render(weights, case.prompt, case.seed, lora=True) for case in cases]
        if again != first_images:
            raise StageError("reload render does not match the candidate", code="moss_prequant_reload")
        peaks = {item.mode: {"generate": item.generate_gb, "train": item.train_gb} for item in candidates}
        min_sm = {mode.name: mode.min_sm for mode in MODES}
        card = render_card(
            chosen=chosen.mode,
            budget_gb=budget,
            peaks=peaks,
            quality=quality,
            skipped=tuple(skipped),
            min_sm=min_sm,
            gpu=gpu,
        )
        manifest = _publish(out, card, budget_gb=budget, peaks=peaks, min_sm=min_sm, chosen=chosen.mode)
        return PrequantResult(
            chosen.mode,
            budget,
            out,
            manifest,
            peaks,
            quality,
            tuple(skipped),
            min_sm,
            gpu,
        )
    finally:
        if previous is None:
            os.environ.pop(VRAM_CAP_ENV, None)
        else:
            os.environ[VRAM_CAP_ENV] = previous


def render_card(
    *,
    chosen: str,
    budget_gb: float,
    peaks: Mapping[str, Mapping[str, float]],
    quality: Mapping[str, Mapping[str, float]],
    skipped: Sequence[str],
    min_sm: Mapping[str, int],
    gpu: Gpu,
) -> str:
    """Model card. Apache-2.0 credit to OpenMOSS, Qwen, and the FLUX.2 VAE."""
    rows = []
    for name in [mode.name for mode in MODES]:
        if name in skipped:
            rows.append(f"| `{name}` | skipped (needs sm>={min_sm[name]}) | | | | | |")
            continue
        peak = peaks[name]
        score = quality[name]
        mark = "yes" if name == chosen else ""
        rows.append(
            f"| `{name}` | {peak['generate']:.2f} | {peak['train']:.2f} | "
            f"{max(peak['generate'], peak['train']):.2f} | {score['lpips']:.4f} | {score['clip']:.4f} | {mark} |"
        )
    table = "\n".join(rows)
    skipped_line = ", ".join(f"`{name}`" for name in skipped) if skipped else "none"
    return (
        "---\n"
        "license: apache-2.0\n"
        "base_model:\n"
        f"- {SFT_REPO}\n"
        f"- {ENCODER_REPO}\n"
        "tags:\n"
        "- voxprint\n"
        "- quantized\n"
        "- moss-image\n"
        "---\n"
        "\n"
        f"# {DISPLAY_NAME}\n"
        "\n"
        "This is a modified Apache-2.0 build of MOSS-Image. Voxprint quantized it once "
        f"and publishes this file. The process cap for the measurement was {budget_gb:g} GB "
        f"(`{VRAM_CAP_ENV}`), which is the default MOSS budget of {MOSS_VRAM_BUDGET_GB:g} GB "
        "when the flag is left at its default. Load-time quantization is not the default path.\n"
        "\n"
        "## Credits\n"
        "\n"
        f"- OpenMOSS Team, `{SFT_REPO}` at `{SFT_REVISION}`, Apache-2.0.\n"
        f"- Qwen team, `{ENCODER_REPO}` text encoder (the resharded MOSS text encoder), Apache-2.0.\n"
        f"- Black Forest Labs, `{VAE_REPO}` (FLUX.2 small VAE), Apache-2.0.\n"
        "\n"
        "LICENSE and NOTICE are included.\n"
        "\n"
        "## Choice\n"
        "\n"
        f"Chosen mode: `{chosen}`. Budget: {budget_gb:g} GB. "
        f"Measured on {gpu.name} (sm {gpu.sm}, {gpu.vram_gb:g} GB).\n"
        f"Skipped: {skipped_line}.\n"
        "\n"
        "fp8_e4m3 needs sm >= 89. int8 and nf4 are bitsandbytes weight-only configs. "
        "Quality is mean LPIPS (lower is better) and mean CLIP (higher is better) "
        "against the bf16 reference on the same prompts and seeds.\n"
        "\n"
        "| Mode | Generate GB | Train GB | Peak GB | LPIPS | CLIP | Chosen |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
        f"{table}\n"
        "\n"
        "## Memory\n"
        "\n"
        f"The peak covers generation with a LoRA loaded and a short training probe. "
        f"min_sm: fp8_e4m3={min_sm['fp8_e4m3']}, int8={min_sm['int8']}, nf4={min_sm['nf4']}.\n"
        "\n"
        f"Card date: {date.today().isoformat()}.\n"
    )


def execute(args: argparse.Namespace) -> dict[str, object]:
    """Run from the ``forge moss-prequant`` arguments. Uses the live backends."""
    snapshots = SnapshotSet(Path(args.sft), Path(args.encoder))
    sm = int(args.sm) if getattr(args, "sm", None) is not None else None
    result = run(snapshots, Path(args.out), budget_gb=float(args.vram_budget_gb), sm=sm)
    return result.to_dict()


def live_services(snapshots: SnapshotSet) -> Services:
    """Calibration-VM backends. Tests do not call this."""
    return Services(LiveLoader(), LiveQuantizer(), LiveRenderer(snapshots), LiveScorer(), LiveProbe())


class LiveLoader:
    """Marks the process offline and checks that the snapshot directories exist."""

    def open(self, snapshots: SnapshotSet) -> None:
        """Set the Hub offline flags. Does not fetch."""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        for role, root in snapshots.roots().items():
            if not root.is_dir():
                raise StageError(f"{role} snapshot missing: {root}", code="moss_prequant_missing")


class LiveQuantizer:
    """Quantize with diffusers/transformers ``quantization_config``. Local files only."""

    def config(self, mode: str) -> Mapping[str, object]:
        """The plain spec. Objects are built in :meth:`build`."""
        return quantization_spec(mode)

    def build(self, mode: str, snapshots: SnapshotSet, dest: Path) -> None:
        """Load each component with ``local_files_only`` and save it under ``dest``."""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        configs = materialize_quantization_config(mode)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "mode.json").write_text(
            json.dumps({"mode": mode, "spec": quantization_spec(mode)}, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        import torch

        from voxprint_forge.diffusion.moss import import_vendored

        mod = import_vendored()
        common = {"local_files_only": True, "low_cpu_mem_usage": True}
        sft = mod.MOSSImageTransformer.from_pretrained(
            snapshots.sft / "transformer",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            quantization_config=configs["dit"],
            **common,
        )
        sft.save_pretrained(dest / "sft", safe_serialization=True)
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(
            snapshots.encoder,
            dtype=torch.bfloat16,
            use_safetensors=True,
            trust_remote_code=False,
            quantization_config=configs["encoder"],
            **common,
        )
        encoder.save_pretrained(dest / "encoder", safe_serialization=True)

    def reload(self, dest: Path) -> None:
        """Load the saved components with ``local_files_only`` to prove the files read back."""
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        mode = json.loads((dest / "mode.json").read_text(encoding="utf-8"))["mode"]
        configs = materialize_quantization_config(str(mode))
        import torch

        from voxprint_forge.diffusion.moss import import_vendored

        mod = import_vendored()
        common = {"local_files_only": True, "low_cpu_mem_usage": True}
        mod.MOSSImageTransformer.from_pretrained(
            dest / "sft",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            quantization_config=configs["dit"],
            **common,
        )
        from transformers import AutoModel

        AutoModel.from_pretrained(
            dest / "encoder",
            dtype=torch.bfloat16,
            use_safetensors=True,
            trust_remote_code=False,
            quantization_config=configs["encoder"],
            local_files_only=True,
        )

    def train_probe(self, dest: Path) -> None:
        """One AdamW step on a rank-4 LoRA over ``video_embedder``, at a small latent."""
        import torch

        from voxprint_forge.diffusion.moss import import_vendored
        from voxprint_forge.lora.layers import inject, lora_parameters

        os.environ["HF_HUB_OFFLINE"] = "1"
        mode = json.loads((dest / "mode.json").read_text(encoding="utf-8"))["mode"]
        configs = materialize_quantization_config(str(mode))
        mod = import_vendored()
        model = mod.MOSSImageTransformer.from_pretrained(
            dest / "sft",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            quantization_config=configs["dit"],
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.train()
        layers = inject(model, ("video_embedder",), 4, 4)
        opt = torch.optim.AdamW(lora_parameters(layers), lr=1e-4)
        config = json.loads((dest / "sft" / "config.json").read_text(encoding="utf-8"))
        channels = int(config["latent_channels"])
        text_dim = int(config["text_dim"])
        edge = TRAIN_PROBE_EDGE
        hidden = torch.randn(1, channels, edge, edge, device=device, dtype=torch.bfloat16)
        text = torch.randn(1, 4, text_dim, device=device, dtype=torch.bfloat16)
        mask = torch.ones(1, 4, device=device)
        step = torch.tensor([500.0], device=device)
        pred = model(hidden, step, text, mask).sample
        if not isinstance(pred, torch.Tensor):
            raise StageError("train probe got no sample", code="moss_prequant_train")
        loss = pred.float().pow(2).mean()
        loss.backward()
        opt.step()
        del model, layers


class LiveRenderer:
    """Renders through the vendored MOSS pipeline. Reference is the SFT snapshot."""

    def __init__(self, snapshots: SnapshotSet, steps: int = GENERATE_STEPS) -> None:
        self.snapshots = snapshots
        self.steps = steps

    def render_reference(self, snapshots: SnapshotSet, prompt: str, seed: int) -> bytes:
        """One bf16 image from the local SFT snapshot."""
        from io import BytesIO

        from voxprint_forge.diffusion.moss import generate

        image = generate(snapshots.sft, prompt, seed=seed, steps=self.steps)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def render(self, weights: Path, prompt: str, seed: int, *, lora: bool) -> bytes:
        """Render the saved quant. ``lora`` wraps one linear so the peak includes an adapter."""
        from io import BytesIO

        import torch

        from voxprint_forge.diffusion.moss import import_vendored
        from voxprint_forge.lora.layers import inject

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        mode = json.loads((weights / "mode.json").read_text(encoding="utf-8"))["mode"]
        configs = materialize_quantization_config(str(mode))
        mod = import_vendored()
        common = {"local_files_only": True}
        transformer = mod.MOSSImageTransformer.from_pretrained(
            weights / "sft",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            quantization_config=configs["dit"],
            low_cpu_mem_usage=True,
            **common,
        )
        if lora:
            inject(transformer, ("video_embedder",), 4, 4)
        from diffusers import AutoencoderKLFlux2
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.snapshots.sft / "tokenizer", trust_remote_code=False, **common)
        text_encoder = AutoModel.from_pretrained(
            weights / "encoder",
            dtype=torch.bfloat16,
            use_safetensors=True,
            trust_remote_code=False,
            quantization_config=configs["encoder"],
            low_cpu_mem_usage=True,
            **common,
        )
        vae = AutoencoderKLFlux2.from_pretrained(
            self.snapshots.sft / "vae", torch_dtype=torch.float32, use_safetensors=True, **common
        )
        scheduler = mod.MOSSImageScheduler.from_pretrained(self.snapshots.sft / "scheduler", **common)
        pipe = mod.MOSSImagePipeline(transformer, scheduler, text_encoder, tokenizer, vae)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe.transformer.to(device)
        if pipe.vae is not None:
            pipe.vae.to(device)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        image = pipe(prompt=prompt, generator=generator, num_inference_steps=self.steps).images[0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


class LiveScorer:
    """``compare.score.make_scorer`` (LPIPS and CLIP). Imported only when scoring."""

    def __init__(self) -> None:
        self._score: Callable[..., object] | None = None

    def score(self, reference: bytes, candidate: bytes, prompt: str) -> Quality:
        """Score PNG bytes. The prompt is the CLIP text."""
        from io import BytesIO

        from PIL import Image

        from voxprint_forge.compare.score import make_scorer

        if self._score is None:
            self._score = make_scorer()
        left = Image.open(BytesIO(reference))
        right = Image.open(BytesIO(candidate))
        pair = self._score(left, right, _Prompt(prompt))
        return Quality(lpips=float(pair["lpips"]), clip=float(pair["clip_i"]))


class LiveProbe:
    """Sets the CUDA process fraction and reads ``max_memory_reserved``."""

    def gpu(self) -> Gpu:
        """The first CUDA card. Refuses a machine with no card."""
        import torch

        if not torch.cuda.is_available():
            raise StageError("moss-prequant needs a CUDA GPU to measure peaks", code="moss_prequant_gpu")
        props = torch.cuda.get_device_properties(0)
        sm = int(props.major) * 10 + int(props.minor)
        vram_gb = float(props.total_memory) / (1024**3)
        return Gpu(str(props.name), vram_gb, sm)

    def cap(self, budget_gb: float) -> None:
        """``torch.cuda.set_per_process_memory_fraction`` from ``budget_gb`` GiB."""
        import torch

        if not torch.cuda.is_available():
            return
        total = float(torch.cuda.get_device_properties(0).total_memory)
        want = float(budget_gb) * (1024**3)
        torch.cuda.set_per_process_memory_fraction(min(1.0, want / total), 0)

    def measure(self, mode: str, kind: str, fn: Callable[[], None]) -> float:
        """Peak reserved GiB while ``fn`` runs. ``mode`` and ``kind`` label the sample."""
        del mode, kind
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        fn()
        if not torch.cuda.is_available():
            raise StageError("no CUDA peak to measure", code="moss_prequant_gpu")
        return float(torch.cuda.max_memory_reserved()) / (1024**3)


def materialize_quantization_config(mode: str) -> dict[str, object]:
    """The diffusers and transformers config objects for ``mode``.

    fp8_e4m3 is a torchao ``Float8WeightOnlyConfig`` inside ``TorchAoConfig``.
    int8 and nf4 are bitsandbytes configs (diffusers for the DiT, transformers
    for the encoder).
    """
    spec = quantization_spec(mode)
    if spec["backend"] == "torchao":
        import torch
        from diffusers import TorchAoConfig
        from torchao.quantization import Float8WeightOnlyConfig
        from torchao.quantization.granularity import PerRow

        config = TorchAoConfig(
            Float8WeightOnlyConfig(
                weight_dtype=torch.float8_e4m3fn,
                granularity=PerRow(),
                version=2,
                set_inductor_config=False,
            )
        )
        return {"dit": config, "encoder": config}
    import torch
    from diffusers import BitsAndBytesConfig as DiffusersBits
    from transformers import BitsAndBytesConfig as TransformersBits

    if mode == "int8":
        return {"dit": DiffusersBits(load_in_8bit=True), "encoder": TransformersBits(load_in_8bit=True)}
    kwargs = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": torch.bfloat16,
        "bnb_4bit_use_double_quant": False,
    }
    return {"dit": DiffusersBits(**kwargs), "encoder": TransformersBits(**kwargs)}


class _Prompt:
    """The ``prompt`` attribute :func:`voxprint_forge.compare.score.make_scorer` reads."""

    def __init__(self, prompt: str) -> None:
        self.prompt = prompt


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise StageError("reference set is empty", code="moss_prequant_reference")
    return sum(values) / len(values)


def _safe_file(root: Path, relative: str) -> Path:
    parts = Path(relative).parts
    if not relative or relative.startswith(("/", "\\")) or "\\" in relative or ".." in parts:
        raise StageError(f"pinned path {relative!r} is not a relative file", code="moss_prequant_pin")
    return root / relative


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and not path.name.endswith(".sha256") and path.name != "voxprint-model.json"
    )


def _write_sidecars(folder: Path) -> None:
    from voxprint_forge.hashing import write_sha256

    for path in _files(folder):
        write_sha256(path, _sha256_file(path))


def _assert_sidecars(folder: Path) -> None:
    for path in _files(folder):
        sidecar = Path(str(path) + ".sha256")
        if not sidecar.is_file():
            raise StageError(f"missing sha256 sidecar for {path.name}", code="moss_prequant_reload")
        token = sidecar.read_text(encoding="utf-8").split()
        if not token or token[0] != _sha256_file(path):
            raise StageError(
                f"reloaded {path.name} does not match its sha256 sidecar",
                code="moss_prequant_reload",
            )


def _publish(
    out: Path,
    card: str,
    *,
    budget_gb: float,
    peaks: Mapping[str, Mapping[str, float]],
    min_sm: Mapping[str, int],
    chosen: str,
) -> Path:
    notice = (
        f"{DISPLAY_NAME}\n"
        "Copyright 2026 Aleksandr Mitroshenkov\n"
        "\n"
        "This is a modified work. Voxprint quantized MOSS-Image once "
        f"({chosen}) under a {budget_gb:g} GB peak budget.\n"
        f"Credits: OpenMOSS ({SFT_REPO}), Qwen ({ENCODER_REPO}), "
        f"Black Forest Labs ({VAE_REPO}). Apache-2.0.\n"
    )
    (out / "NOTICE").write_text(notice, encoding="utf-8", newline="\n")
    (out / "README.md").write_text(card, encoding="utf-8", newline="\n")
    _copy_license(out)
    _write_sidecars(out)
    from voxprint_forge import __version__
    from voxprint_forge.mdl import build_document, hash_tree, infer_app_ids, write_manifest

    chosen_sm = min_sm[chosen] / 10.0
    document = build_document(
        model_id=MODEL_ID,
        display_name=DISPLAY_NAME,
        app_ids=infer_app_ids("text-to-image"),
        task="text-to-image",
        license="Apache-2.0",
        files=hash_tree(out),
        mirrors=[{"kind": "hf", "ref": MIRROR_REPO, "revision": MIRROR_REVISION, "ours": True}],
        min_vram_gb=16,
        recommended_vram_gb=16,
        gpu_min_cc=chosen_sm,
        engine="diffusers",
        version_req=">=0.38",
        base_model=SFT_REPO,
        card_url=f"https://huggingface.co/{MIRROR_REPO}",
        flags=[
            "derivative",
            "modified",
            "prequant",
            f"chosen={chosen}",
            f"budget_gb={budget_gb:g}",
            f"source-sft={SFT_REVISION}",
        ],
        on_demand=True,
        forge_version=__version__,
        tag="",
        date=date.today().isoformat(),
    )
    document["prequant"] = {
        "quant_modes": [mode.name for mode in MODES],
        "measured_peak_gb": {
            name: {"generate": float(row["generate"]), "train": float(row["train"])} for name, row in peaks.items()
        },
        "min_sm": {name: int(value) for name, value in min_sm.items()},
        "budget_gb": float(budget_gb),
        "chosen": chosen,
    }
    return write_manifest(out, document)


def _copy_license(folder: Path) -> None:
    src = Path(__file__).resolve().parents[3] / "LICENSE"
    if not src.is_file():
        raise StageError(f"missing {src}", code="moss_prequant_license")
    shutil.copyfile(src, folder / "LICENSE")


def add_command(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``forge moss-prequant``."""
    command = sub.add_parser(
        "moss-prequant",
        help="Build the one-time MOSS pre-quant from local snapshots. Never downloads.",
    )
    command.add_argument("--sft", type=Path, required=True, help="Local MOSS-Image-SFT snapshot.")
    command.add_argument(
        "--encoder",
        type=Path,
        required=True,
        help="Local resharded Qwen3.5-9B encoder (the MOSS text_encoder directory).",
    )
    command.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory next to the originals for the chosen build.",
    )
    command.add_argument(
        "--vram-budget-gb",
        "--budget",
        dest="vram_budget_gb",
        type=float,
        default=DEFAULT_BUDGET_GB,
        help=f"Peak VRAM budget in GB ({BUDGET_MIN_GB:g}–{BUDGET_MAX_GB:g}). "
        f"Default: {DEFAULT_BUDGET_GB:g}. Sets {VRAM_CAP_ENV}.",
    )
    command.add_argument(
        "--sm",
        type=int,
        default=None,
        help="Override the GPU compute capability times 10 (89 means sm_89). Default: the card.",
    )
