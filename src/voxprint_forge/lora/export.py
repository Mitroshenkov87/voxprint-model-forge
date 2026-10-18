"""``lora export``: one ``.safetensors`` with typed metadata plus the ``voxprint-model/1`` sidecar (Canvas contract).

Header keys are ``voxprint.subject.*`` and ``voxprint.lora.*`` only, all typed by
the user or set by the trainer: no paths, usernames, machine ids or EXIF. No
key may look like a gate or threshold setting; Canvas refuses such files.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from voxprint_forge import __version__
from voxprint_forge.lora.bases import get_adapter
from voxprint_forge.lora.consent import default_trigger
from voxprint_forge.lora.errors import LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.project import Project

SCHEMA = "voxprint-model/1"
MIN_SCHEMA = 1
DEFAULT_STRENGTH = 0.8
STRENGTH_RANGE = (0.0, 1.5)
SIDECAR = "voxprint-model.json"
PRODUCER = f"voxprint-model-forge {__version__}"
RESERVED_TOKENS = frozenset({"safety", "minor", "minors", "age", "gate", "gates", "threshold", "thresholds", "nsfw"})
_LICENCES = Path(__file__).resolve().parent / "licences"


def gate_like(key: str) -> bool:
    """True for a key that reads like a gate or threshold setting (never written)."""
    return any(token in RESERVED_TOKENS for token in re.split(r"[._\-]", key.lower()))


def header_metadata(data: dict[str, object]) -> dict[str, str]:
    """The ``voxprint.subject.*`` and ``voxprint.lora.*`` header for a trained project."""
    subject = dict(data.get("subject") or {})  # type: ignore[call-overload]
    train = dict(data.get("train") or {})  # type: ignore[call-overload]
    config = dict(train.get("config") or {})
    kind = str(subject.get("kind") or data.get("kind") or "")
    meta = {
        "voxprint.subject.name": str(subject.get("name", "")),
        "voxprint.subject.stage_name": str(subject.get("stage_name", "")),
        "voxprint.subject.trigger": str(subject.get("trigger", "")),
        "voxprint.subject.consent_obtained": _flag(subject.get("consent_obtained")),
        "voxprint.subject.adult_confirmed": _flag(subject.get("adult_confirmed")),
        "voxprint.subject.provenance_note": str(subject.get("provenance_note", "")),
        "voxprint.subject.release_attached": _flag(subject.get("release_attached")),
        "voxprint.subject.release_sha256": str(subject.get("release_sha256", "")),
        "voxprint.lora.base": str(train.get("base", "")),
        "voxprint.lora.base_revision": str(train.get("base_revision", "")),
        "voxprint.lora.rank": str(config.get("rank", "")),
        "voxprint.lora.alpha": str(config.get("alpha", "")),
        "voxprint.lora.steps": str(train.get("step", "")),
        "voxprint.lora.kind": kind,
        "voxprint.lora.trained_on": str(train.get("trained_on", "")),
        "voxprint.lora.adult_content": _flag(subject.get("adult_content")),
    }
    if kind == "person":
        meta["voxprint.subject.consent_method"] = consent_method(subject)
    if kind in ("object", "style"):
        meta["voxprint.subject.rights_confirmed"] = _flag(subject.get("rights_confirmed"))
    for key in meta:
        if gate_like(key):
            raise AssertionError(f"gate-like key {key}")
    return meta


def consent_method(subject: dict[str, object]) -> str:
    """``verbal`` or ``file`` for a person. Projects from before the field read ``file`` only with a release."""
    method = str(subject.get("consent_method") or "")
    if method in ("verbal", "file"):
        return method
    return "file" if subject.get("release_attached") is True else "verbal"


def user_models_dir() -> Path:
    """``<models dir>/user-models``, always from ``voxprint_common.paths.models_dir()``.

    LoRAs go to ``<that>/lora/<id>/``. There is no Forge-only override: Canvas and
    Manage models read the same folder, so a different root would hide the LoRA.
    """
    from voxprint_common.paths import models_dir

    return models_dir() / "user-models"


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Raw little-endian bytes of a tensor in its own dtype.

    NumPy has no bfloat16, so ``tensor.numpy()`` fails on BF16 LoRAs (the
    trainer's default). A ``uint8`` view gives the same bytes for every dtype,
    and the same bytes as ``.numpy().tobytes()`` for FP32 and FP16.
    """
    flat = tensor.detach().cpu().contiguous().reshape(-1)
    return flat.view(torch.uint8).numpy().tobytes()


def tensors_digest(tensors: dict[str, torch.Tensor]) -> str:
    """sha256 over sorted names and raw tensor bytes; the id hex comes from it."""
    digest = hashlib.sha256()
    for key in sorted(tensors):
        digest.update(key.encode())
        digest.update(tensor_bytes(tensors[key]))
    return digest.hexdigest()


def model_id(data: dict[str, object], tensors_digest: str) -> str:
    """``<trigger slug>-<8 hex>``; the hex is from the LoRA tensors, so the same LoRA keeps its id."""
    subject = dict(data.get("subject") or {})  # type: ignore[call-overload]
    slug = default_trigger(str(subject.get("trigger") or subject.get("name") or ""), "")
    slug = re.sub(r"[^a-z0-9_]", "", slug.encode("ascii", "ignore").decode()) or "lora"
    return f"{slug[:40]}-{tensors_digest[:8]}"


def export(
    project: Project,
    progress: Progress,
    *,
    checkpoint: int | None = None,
    out: Path | None = None,
    install_to_mdl: bool = False,
    tags: tuple[str, ...] = (),
) -> dict[str, object]:
    """Write ``<id>.safetensors`` and ``voxprint-model.json``; optionally install them for Canvas."""
    data = project.read()
    train = data.get("train")
    if data.get("gate_passed") is not True or not isinstance(train, dict) or not train.get("checkpoint"):
        raise LoraUsageError("lora.err.no_checkpoint")
    name = f"step-{checkpoint:06d}.safetensors" if checkpoint is not None else str(train["checkpoint"])
    source = project.checkpoints / name
    if not source.is_file():
        raise LoraUsageError("lora.err.no_checkpoint")
    info = get_adapter(str(train["base"])).info
    progress.emit("export.write", "lora.progress.export", reason_key="lora.reason.export")
    if not info.permissive:
        progress.emit("export.licence", "lora.err.base_restrictive")
    tensors = {key: value.detach().cpu().contiguous() for key, value in load_file(str(source)).items()}
    mid = model_id(data, tensors_digest(tensors))
    meta = header_metadata({**data, "train": {**train, "step": _step(name)}})
    target = Path(out) if out is not None else project.root / "export" / f"{mid}.safetensors"
    if target.suffix != ".safetensors":
        raise LoraUsageError("lora.err.no_checkpoint")
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(target), metadata=meta)
    preview = _preview(project, _step(name), target.parent)
    sidecar = sidecar_document(mid, data, target, info, preview, tags)
    _write_json(target.parent / SIDECAR, sidecar)
    if info.licence.lower() == "apache-2.0":
        shutil.copyfile(_LICENCES / "Apache-2.0.txt", target.parent / "LICENSE")
        (target.parent / "NOTICE").write_text(
            f"This LoRA was trained on {info.display_name} ({info.repo}), licensed under the Apache License 2.0.\n"
            f"Produced by {PRODUCER}.\n",
            encoding="utf-8",
        )
    result: dict[str, object] = {"state": "done", "id": mid, "file": str(target), "sha256": sidecar["sha256"]}
    if install_to_mdl:
        progress.emit("export.install", "lora.progress.install", reason_key="lora.reason.install")
        folder = user_models_dir() / "lora" / mid
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target, folder / f"{mid}.safetensors")
        _write_json(folder / SIDECAR, {**sidecar, "file": f"{mid}.safetensors"})
        for item in ("LICENSE", "NOTICE", "preview.webp"):
            if (target.parent / item).is_file():
                shutil.copyfile(target.parent / item, folder / item)
        result["installed"] = str(folder)
    progress.emit("export.done", "lora.progress.done", state="done")
    return result


def sidecar_document(mid, data, target: Path, info, preview: Path | None, tags) -> dict[str, object]:
    """The ``voxprint-model/1`` descriptor agreed with Canvas."""
    raw = target.read_bytes()
    subject = dict(data.get("subject") or {})  # type: ignore[call-overload]
    kind = str(subject.get("kind") or data.get("kind") or "")
    doc: dict[str, object] = {
        "schema": SCHEMA,
        "min_schema": MIN_SCHEMA,
        "id": mid,
        "kind": kind,
        "type": "lora",
        "producer": PRODUCER,
        "file": target.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
        "base": {"mdl_id": info.mdl_id, "revision": str(dict(data.get("train") or {}).get("base_revision", ""))},
        "licence": info.licence,
        "default_strength": DEFAULT_STRENGTH,
        "strength_range": list(STRENGTH_RANGE),
        "tags": [str(tag) for tag in tags if str(tag).strip()],
        "voxprint.lora.adult_content": subject.get("adult_content") is True,
    }
    if kind == "person":
        doc["voxprint.subject.consent_method"] = consent_method(subject)
    if preview is not None:
        doc["preview"] = preview.name
        doc["preview_sha256"] = hashlib.sha256(preview.read_bytes()).hexdigest()
    return doc


def _preview(project: Project, step: int, folder: Path) -> Path | None:
    found = sorted(project.previews.glob(f"step-{step:06d}-*.png")) if project.previews.is_dir() else []
    if not found:
        return None
    target = folder / "preview.webp"
    with Image.open(found[0]) as image:
        image.convert("RGB").save(target, "WEBP", quality=90)
    return target


def _step(name: str) -> int:
    match = re.search(r"step-(\d+)", name)
    return int(match.group(1)) if match else 0


def _flag(value: object) -> str:
    return "true" if value is True else "false"


def _write_json(path: Path, doc: dict[str, object]) -> None:
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
