"""Z-Image-Turbo de-distillation training adapter: frozen, train-only, never saved.

Ostris's adapter is merged into Turbo at scale 1.0 while the character LoRA
trains, then disabled for previews, export and inference so the published
LoRA runs on plain Turbo. The default pin is Apache-2.0; fetch it from the
official Hugging Face repo on the VM, never from a mirror on this box.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from torch import nn

from voxprint_forge.lora.errors import LoraUsageError
from voxprint_forge.lora.layers import LoRALinear

DEFAULT_REPO = "ostris/zimage_turbo_training_adapter"
DEFAULT_REVISION = "654cd1bf8b3589d9442cb1c5e35221879e1af4f3"
DEFAULT_FILE = "zimage_turbo_training_adapter_v2.safetensors"
DEFAULT_SHA256 = "20a541d3e016ab8de0da076321b48b6cd9b3ffd072d9df830a068220ab2265f6"
TURBO_BASE = "z-image-turbo"
# Official Turbo pin, short form z-image-turbo@f332072a.
TURBO_REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
_PREFIXES = ("diffusion_model.", "transformer.", "lora_unet_", "lora_transformer_")


@dataclass(frozen=True)
class TrainingAdapterSpec:
    """Where the training adapter is, and how to check the default file."""

    source: str  # local path or org/repo
    revision: str = DEFAULT_REVISION
    filename: str = DEFAULT_FILE
    sha256: str = ""
    path: Path | None = None  # set once the file is on disk

    @property
    def pin(self) -> str:
        """``repo@revision`` or a local path, for job logs."""
        if "/" in self.source and not Path(self.source).exists():
            return f"{self.source}@{self.revision[:12] if self.revision else 'main'}"
        return self.source


def default_spec() -> TrainingAdapterSpec:
    """The pinned Ostris v2 adapter used when training on Turbo."""
    return TrainingAdapterSpec(
        source=DEFAULT_REPO,
        revision=DEFAULT_REVISION,
        filename=DEFAULT_FILE,
        sha256=DEFAULT_SHA256,
    )


def resolve_spec(
    source: str | None,
    revision: str | None = None,
    filename: str | None = None,
    *,
    sha256: str | None = None,
) -> TrainingAdapterSpec:
    """Build a spec from CLI flags. A local file or folder wins over a repo id."""
    text = (source or DEFAULT_REPO).strip()
    revision = (revision or DEFAULT_REVISION).strip()
    filename = (filename or DEFAULT_FILE).strip()
    digest = (sha256 or "").strip()
    path = Path(text).expanduser()
    if path.is_file():
        if not digest and path.name == DEFAULT_FILE:
            digest = DEFAULT_SHA256
        return TrainingAdapterSpec(source=text, revision=revision, filename=path.name, sha256=digest, path=path)
    if path.is_dir():
        found = path / filename
        if found.is_file():
            if not digest and filename == DEFAULT_FILE:
                digest = DEFAULT_SHA256
            return TrainingAdapterSpec(
                source=text, revision=revision, filename=filename, sha256=digest, path=found
            )
        raise LoraUsageError("lora.err.training_adapter_missing", path=str(found))
    if not digest and filename == DEFAULT_FILE and (not source or text == DEFAULT_REPO):
        digest = DEFAULT_SHA256
    return TrainingAdapterSpec(source=text, revision=revision, filename=filename, sha256=digest, path=None)


def bind(adapter: Any, source: str | None, revision: str | None = None, filename: str | None = None) -> None:
    """Attach a training-adapter spec to a Turbo adapter, or refuse it on any other base."""
    mdl_id = getattr(getattr(adapter, "info", None), "mdl_id", "")
    given = any(value is not None and str(value).strip() for value in (source, revision, filename))
    if mdl_id != TURBO_BASE:
        if given:
            raise LoraUsageError("lora.err.training_adapter_base")
        return
    adapter.training_adapter = resolve_spec(source, revision, filename)


def sha256_file(path: Path) -> str:
    """Hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_local(spec: TrainingAdapterSpec) -> Path:
    """Return the adapter file on disk. Downloads from the official repo only when needed.

    Tests pass a local path and never reach the Hugging Face client. The default
    v2 file is checked against :data:`DEFAULT_SHA256` when a digest is set.
    """
    path = spec.path
    if path is None or not path.is_file():
        path = _download(spec)
    if spec.sha256:
        found = sha256_file(path)
        if found.lower() != spec.sha256.lower():
            raise LoraUsageError("lora.err.training_adapter_hash", path=str(path))
    return path


def _download(spec: TrainingAdapterSpec) -> Path:
    """Fetch ``spec.filename`` from Hugging Face. Not used in the offline suite."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise LoraUsageError("lora.err.training_adapter_missing", path=spec.source) from exc
    try:
        downloaded = hf_hub_download(
            repo_id=spec.source,
            filename=spec.filename,
            revision=spec.revision or None,
        )
    except Exception as exc:  # noqa: BLE001 - surface a stable LoRA error
        raise LoraUsageError("lora.err.training_adapter_missing", path=f"{spec.source}/{spec.filename}") from exc
    return Path(downloaded)


def remap_key(name: str) -> str:
    """Strip ComfyUI / PEFT prefixes so keys line up with diffusers module names."""
    text = name.strip()
    for prefix in _PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    text = text.replace(".lora_down.", ".lora_A.").replace(".lora_up.", ".lora_B.")
    text = text.replace(".lora.down.", ".lora_A.").replace(".lora.up.", ".lora_B.")
    return text


def load_pairs(path: Path) -> dict[str, tuple[torch.Tensor, torch.Tensor, float]]:
    """Map module name -> ``(A, B, lora_scale)`` from a safetensors LoRA file."""
    raw = {remap_key(key): value for key, value in load_file(str(path)).items()}
    alpha = _alpha(raw)
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    a_keys = [key for key in raw if key.endswith(".lora_A.weight") or key.endswith(".lora_down.weight")]
    for a_key in a_keys:
        stem = a_key.removesuffix(".lora_A.weight").removesuffix(".lora_down.weight")
        b_key = next(
            (
                candidate
                for candidate in (f"{stem}.lora_B.weight", f"{stem}.lora_up.weight")
                if candidate in raw
            ),
            None,
        )
        if b_key is None:
            continue
        a_weight, b_weight = raw[a_key], raw[b_key]
        rank = int(a_weight.shape[0]) if a_weight.ndim == 2 else 1
        scale = (alpha / rank) if alpha is not None and rank else 1.0
        pairs[stem] = (a_weight, b_weight, scale)
    if not pairs:
        raise LoraUsageError("lora.err.training_adapter_keys", path=str(path))
    return pairs


def _alpha(raw: dict[str, torch.Tensor]) -> float | None:
    """LoRA alpha from a scalar tensor, if the file stored one."""
    for key in ("alpha", "lora_alpha"):
        value = raw.get(key)
        if value is not None and hasattr(value, "reshape"):
            return float(value.reshape(-1)[0].item())
    return None


def attach_frozen(layers: dict[str, LoRALinear], path: Path, *, scale: float = 1.0) -> int:
    """Load the adapter onto the character LoRA layers at ``scale``, frozen.

    Returns how many layers received tensors. Unknown adapter keys are ignored
    when at least one layer matches; none matching is an error.
    """
    pairs = load_pairs(path)
    used = 0
    for name, layer in layers.items():
        pair = pairs.get(name)
        if pair is None:
            # Allow the adapter to use a shorter suffix (Comfy names vs our tree).
            pair = next((item for key, item in pairs.items() if name.endswith(key) or key.endswith(name)), None)
        if pair is None:
            continue
        a_weight, b_weight, lora_scale = pair
        layer.set_training_adapter(a_weight, b_weight, scale=scale, lora_scale=lora_scale)
        used += 1
    if not used:
        raise LoraUsageError("lora.err.training_adapter_keys", path=str(path))
    return used


def set_enabled(model: nn.Module, enabled: bool) -> None:
    """Turn the frozen training adapter on (train) or off (preview / export)."""
    scale = 1.0 if enabled else 0.0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.training_adapter_scale = scale if module.has_training_adapter else 0.0
