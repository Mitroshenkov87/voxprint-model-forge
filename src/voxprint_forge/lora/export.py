"""``lora export``: one ``.safetensors`` with typed metadata plus the ``voxprint-model/1`` sidecar (Canvas contract).

Header keys are ``voxprint.subject.*`` and ``voxprint.lora.*`` only, all typed by
the user or set by the trainer: no paths, usernames, machine ids or EXIF. No
key may look like a gate or threshold setting; Canvas refuses such files.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from voxprint_forge import __version__
from voxprint_forge.lora.bases import get_adapter
from voxprint_forge.lora.consent import default_trigger
from voxprint_forge.lora.errors import LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.project import Project, package_stem
from voxprint_forge.lora.review import load_state, passes_output_check

SCHEMA = "voxprint-model/1"
MIN_SCHEMA = 1
DEFAULT_STRENGTH = 0.8
STRENGTH_RANGE = (0.0, 1.5)
SCALE_DEFAULT = 0.8
SCALE_MIN = 0.0
SCALE_MAX = 1.2
PREVIEW_LIMIT = 1024
PREVIEW_EXTRAS = 6
PACKAGE_KINDS = {"person": "subject", "thing": "object", "object": "object", "style": "style"}
SIDECAR = "voxprint-model.json"
PRODUCER = f"voxprint-model-forge {__version__}"
RESERVED_TOKENS = frozenset({"safety", "minor", "minors", "age", "gate", "gates", "threshold", "thresholds", "nsfw"})
_LICENCES = Path(__file__).resolve().parent / "licences"


def gate_like(key: str) -> bool:
    """True for a key that reads like a gate or threshold setting (never written)."""
    return any(token in RESERVED_TOKENS for token in re.split(r"[._\-]", key.lower()))


def header_metadata(
    data: dict[str, object], tensors: dict[str, torch.Tensor] | None = None
) -> dict[str, str]:
    """The ``voxprint.subject.*`` and ``voxprint.lora.*`` header for a trained project."""
    # Circular: attr.encode calls tensors_digest in this module.
    from voxprint_forge.lora import attr as subject_attr

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
    planned = train.get("steps_planned")
    if isinstance(planned, int) and not isinstance(planned, bool):
        meta["voxprint.lora.steps_planned"] = str(planned)
    if kind == "person":
        meta["voxprint.subject.consent_method"] = consent_method(subject)
    if kind in ("object", "style"):
        meta["voxprint.subject.rights_confirmed"] = _flag(subject.get("rights_confirmed"))
    if tensors is not None:
        encoded = subject_attr.encode(subject.get("attr"), tensors)
        if encoded is not None:
            meta[subject_attr.KEY] = encoded
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
    shown = _export_train(train, name, checkpoint is None)
    meta = header_metadata({**data, "train": shown}, tensors)
    target = Path(out) if out is not None else project.root / "export" / f"{mid}.safetensors"
    if target.suffix != ".safetensors":
        raise LoraUsageError("lora.err.no_checkpoint")
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(target), metadata=meta)
    preview = _preview(project, _step(name), target.parent)
    sidecar = sidecar_document(mid, {**data, "train": shown}, target, info, preview, tags)
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
    trained = dict(data.get("train") or {})
    done_steps = trained.get("step")
    planned = trained.get("steps_planned")
    if isinstance(done_steps, int) and not isinstance(done_steps, bool):
        doc["steps"] = done_steps
    if isinstance(planned, int) and not isinstance(planned, bool):
        doc["steps_planned"] = planned
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


def _export_train(train: dict[str, object], name: str, final: bool) -> dict[str, object]:
    """Steps recorded on an export.

    The final file reports the steps the LoRA got. An earlier ``--checkpoint`` reports that file's step.
    ``steps_planned`` stays the preset total when the run stored one.
    """
    file_step = _step(name)
    recorded = train.get("step")
    step = recorded if final and isinstance(recorded, int) and not isinstance(recorded, bool) else file_step
    shown = {**train, "step": step}
    return shown


def _step(name: str) -> int:
    match = re.search(r"step-(\d+)", name)
    return int(match.group(1)) if match else 0


def _flag(value: object) -> str:
    return "true" if value is True else "false"


def _write_json(path: Path, doc: dict[str, object]) -> None:
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def package_kind(kind: str) -> str:
    """Canvas kind for a package: a person is ``subject``, a thing or object is ``object``."""
    return PACKAGE_KINDS.get(kind, "object")


def export_package(
    project: Project,
    progress: Progress,
    *,
    checkpoint: int | None = None,
    out: Path | None = None,
) -> dict[str, object]:
    """Write ``<stage name or name>.voxlora``: the small LoRA package, and nothing merged.

    The zip is stored uncompressed. It holds ``model.safetensors`` (today's
    ``voxprint.subject.*`` and ``voxprint.lora.*`` header), ``voxprint-model.json``,
    one ``preview.webp`` and up to six ``previews/NN.webp``. Pictures come only
    from generated samples that already passed :func:`passes_output_check`.
    Source photos are not included.
    """
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
    tensors = _lora_tensors(source)
    shown = _export_train(train, name, checkpoint is None)
    meta = header_metadata({**data, "train": shown}, tensors)
    target = _package_target(project, data, out)
    target.parent.mkdir(parents=True, exist_ok=True)
    weights = _safetensors_bytes(tensors, meta)
    pictures = _package_pictures(project)
    doc = _package_document(data, shown, info.mdl_id, hashlib.sha256(weights).hexdigest(), pictures)
    _write_package(target, weights, doc, pictures)
    progress.emit("export.done", "lora.progress.done", state="done")
    return {"state": "done", "file": str(target), "sha256": doc["weights_sha256"], "name": target.name}


def _lora_tensors(source: Path) -> dict[str, torch.Tensor]:
    """``<module>.lora_A.weight`` and ``<module>.lora_B.weight`` only."""
    loaded = load_file(str(source))
    tensors = {
        key: value.detach().cpu().contiguous()
        for key, value in loaded.items()
        if key.endswith(".lora_A.weight") or key.endswith(".lora_B.weight")
    }
    if not tensors:
        raise LoraUsageError("lora.err.no_checkpoint")
    return tensors


def _safetensors_bytes(tensors: dict[str, torch.Tensor], meta: dict[str, str]) -> bytes:
    """One safetensors blob, header included, without touching the checkpoint file."""
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "model.safetensors"
        save_file(tensors, str(path), metadata=meta)
        return path.read_bytes()


def _package_target(project: Project, data: dict[str, object], out: Path | None) -> Path:
    """Default ``<project>/export/<stem>.voxlora``, or ``out`` with that suffix."""
    stem = package_stem(data)
    if out is None:
        return project.root / "export" / f"{stem}.voxlora"
    path = Path(out)
    if path.exists() and path.is_dir():
        return path / f"{stem}.voxlora"
    if path.suffix.lower() != ".voxlora":
        path = path.with_suffix(".voxlora") if path.suffix else Path(f"{path}.voxlora")
    return path


def _package_document(
    data: dict[str, object],
    train: dict[str, object],
    base: str,
    digest: str,
    pictures: list[tuple[str, bytes]],
) -> dict[str, object]:
    """The ``voxprint-model/1`` package descriptor. ``attest`` is two booleans."""
    subject = dict(data.get("subject") or {})  # type: ignore[call-overload]
    kind = package_kind(str(subject.get("kind") or data.get("kind") or ""))
    done = train.get("step")
    planned = train.get("steps_planned")
    steps = done if isinstance(done, int) and not isinstance(done, bool) else 0
    steps_planned = planned if isinstance(planned, int) and not isinstance(planned, bool) else steps
    return {
        "schema": SCHEMA,
        "name": package_stem(data),
        "base": base,
        "trigger": str(subject.get("trigger") or ""),
        "scale_default": SCALE_DEFAULT,
        "kind": kind,
        "forge_version": __version__,
        "created": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "scale_min": SCALE_MIN,
        "scale_max": SCALE_MAX,
        "weights_sha256": digest,
        "steps": steps,
        "steps_planned": steps_planned,
        "previews": [name for name, _blob in pictures],
        "attest": {
            "consent_obtained": subject.get("consent_obtained") is True,
            "subject_adult": subject.get("adult_confirmed") is True,
        },
    }


def _package_pictures(project: Project) -> list[tuple[str, bytes]]:
    """``preview.webp`` plus up to six ``previews/NN.webp``, metadata stripped."""
    blobs: list[bytes] = []
    for path in _generated_samples(project):
        blob = _clean_webp(path)
        if blob is not None:
            blobs.append(blob)
        if len(blobs) >= 1 + PREVIEW_EXTRAS:
            break
    named: list[tuple[str, bytes]] = []
    if blobs:
        named.append(("preview.webp", blobs[0]))
    for index, blob in enumerate(blobs[1:1 + PREVIEW_EXTRAS], start=1):
        named.append((f"previews/{index:02d}.webp", blob))
    return named


def _generated_samples(project: Project) -> list[Path]:
    """Review samples and test pictures inside the project. Source photos are skipped."""
    found: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        if not path.is_file():
            return
        try:
            relative = path.resolve().relative_to(project.root.resolve())
        except ValueError:
            return
        if not relative.parts or relative.parts[0] not in {"previews", "review"}:
            return
        key = path.resolve()
        if key in seen:
            return
        seen.add(key)
        found.append(path)

    if project.previews.is_dir():
        for pattern in ("*.png", "*.webp"):
            for path in sorted(project.previews.glob(pattern)):
                add(path)
    state = load_state(project)
    samples = state.get("samples")
    if isinstance(samples, list):
        for sample in samples:
            if isinstance(sample, dict) and sample.get("file"):
                add(project.root / str(sample["file"]))
    return found


def _clean_webp(path: Path) -> bytes | None:
    """One RGB WebP at most 1024 px, or ``None`` when the picture fails the output check."""
    try:
        with Image.open(path) as image:
            image.load()
            if not passes_output_check(image):
                return None
            rgb = image.convert("RGB")
    except (OSError, ValueError):
        return None
    clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
    if max(clean.size) > PREVIEW_LIMIT:
        clean.thumbnail((PREVIEW_LIMIT, PREVIEW_LIMIT), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    clean.save(buffer, format="WEBP", quality=90)
    return buffer.getvalue()


def _write_package(target: Path, weights: bytes, doc: dict[str, object], pictures: list[tuple[str, bytes]]) -> None:
    """Write the zip with :data:`zipfile.ZIP_STORED` so every member is uncompressed."""
    payload = json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    members = [("model.safetensors", weights), (SIDECAR, payload), *pictures]
    with zipfile.ZipFile(target, "w") as archive:
        for name, blob in members:
            info = zipfile.ZipInfo(filename=name)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            archive.writestr(info, blob)
