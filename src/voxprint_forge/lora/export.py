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
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from voxprint_forge import __version__
from voxprint_forge.lora import legacy_base, resolve_base
from voxprint_forge.lora.bases import get_adapter
from voxprint_forge.lora.errors import LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.project import Item, Project, package_stem
from voxprint_forge.lora.review import load_state, passes_output_check
from voxprint_forge.lora.trigger import token_slug

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
# Short base tag in export file names. Unknown ids fall back to a compact slug.
# The stored tag stays moss5b for the MOSS registry id. The SFT/Base split is base_variant.
_BASE_SHORT = {"moss-image-base": "moss5b", "z-image": "zimg6b", "z-image-turbo": "zimgturbo"}
_VARIANTS = frozenset({"sft", "base", "z-image"})
# Canvas contract: trained_base is "<mdl_id>@<first 8 of the revision>".
_ZIMAGE_COMPATIBLE = ("z-image", "z-image-turbo")
_MOSS_COMPATIBLE = ("moss-image-base",)


def trained_base_tag(mdl_id: str, revision: str = "") -> str:
    """``z-image@04cc4abb``, ``z-image-turbo@f332072a`` or ``moss-image-base@fc5de0d5``."""
    ident = (mdl_id or "").strip() or "z-image"
    short = (revision or "").strip()[:8]
    if not short:
        try:
            short = str(get_adapter(ident).info.revision).strip()[:8]
        except (LoraUsageError, KeyError, AttributeError):
            short = ""
    return f"{ident}@{short}" if short else ident


def compatible_bases_list(mdl_id: str) -> list[str]:
    """Sidecar ``compatible_bases``.

    A full Z-Image LoRA also loads on Z-Image-Turbo. A LoRA trained on Turbo
    with the de-distillation adapter is Turbo-only and is not claimed for
    full Z-Image.
    """
    ident = (mdl_id or "").strip() or "z-image"
    if ident == "z-image":
        return list(_ZIMAGE_COMPATIBLE)
    if ident == "z-image-turbo":
        return ["z-image-turbo"]
    if ident == "moss-image-base":
        return list(_MOSS_COMPATIBLE)
    return [ident]


def compatible_bases_json(mdl_id: str) -> str:
    """Safetensors ``voxprint.lora.compatible_bases``: compact JSON, no spaces."""
    return json.dumps(compatible_bases_list(mdl_id), separators=(",", ":"))



def gate_like(key: str) -> bool:
    """True for a key that reads like a gate or threshold setting (never written)."""
    return any(token in RESERVED_TOKENS for token in re.split(r"[._\-]", key.lower()))


def header_metadata(
    data: dict[str, object], tensors: dict[str, torch.Tensor] | None = None, *, dataset_id: str = ""
) -> dict[str, str]:
    """The ``voxprint.subject.*`` and ``voxprint.lora.*`` header for a trained project."""
    # Circular: attr.encode calls tensors_digest in this module.
    from voxprint_forge.lora import attr as subject_attr

    subject = dict(data.get("subject") or {})  # type: ignore[call-overload]
    train = dict(data.get("train") or {})  # type: ignore[call-overload]
    config = dict(train.get("config") or {})
    kind = str(subject.get("kind") or data.get("kind") or "")
    base = str(train.get("base", ""))
    meta = {
        "voxprint.subject.name": str(subject.get("name", "")),
        "voxprint.subject.stage_name": str(subject.get("stage_name", "")),
        "voxprint.subject.trigger": str(subject.get("trigger", "")),
        "voxprint.subject.consent_obtained": _flag(subject.get("consent_obtained")),
        "voxprint.subject.adult_confirmed": _flag(subject.get("adult_confirmed")),
        "voxprint.subject.provenance_note": str(subject.get("provenance_note", "")),
        "voxprint.subject.release_attached": _flag(subject.get("release_attached")),
        "voxprint.subject.release_sha256": str(subject.get("release_sha256", "")),
        "voxprint.lora.base": base,
        "voxprint.lora.base_family": base,
        "voxprint.lora.base_variant": base_variant(base, train.get("base_variant")),
        "voxprint.lora.base_revision": str(train.get("base_revision", "")),
        "voxprint.lora.base_sha256": str(train.get("base_sha256", "")),
        "voxprint.lora.trained_base": trained_base_tag(resolve_base(base), str(train.get("base_revision", ""))),
        "voxprint.lora.compatible_bases": compatible_bases_json(resolve_base(base)),
        "voxprint.lora.dataset_id": dataset_id,
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
    slug = token_slug(str(subject.get("trigger") or subject.get("name") or ""))
    slug = re.sub(r"[^a-z0-9_]", "", slug.encode("ascii", "ignore").decode()) or "lora"
    return f"{slug[:40]}-{tensors_digest[:8]}"


def base_variant(mdl_id: str, recorded: object = None) -> str:
    """``sft``, ``base`` or ``z-image`` for LoRA metadata.

    A stored ``voxprint.lora.base_variant`` wins when it is one of those three.
    The MOSS registry id ``moss-image-base`` defaults to ``sft`` (the weights in
    ``diffusion.moss.REPOS["sft"]``). A blank id is a LoRA from before any base
    was recorded, which is Z-Image. Any other id, including a test stand-in,
    records ``base``.
    """
    if isinstance(recorded, str) and recorded.strip() in _VARIANTS:
        return recorded.strip()
    text = mdl_id.strip()
    if text in ("", "z-image"):
        return "z-image"
    if text == "moss-image-base":
        return "sft"
    return "base"


def lora_variant_of(metadata: Mapping[str, object]) -> str:
    """Variant recorded on a LoRA file. Missing metadata follows :func:`base_variant`."""
    return base_variant(lora_base_of(metadata), metadata.get("voxprint.lora.base_variant"))


def base_short(mdl_id: str) -> str:
    """Short base tag for export file names: ``moss-image-base`` -> ``moss5b``, ``z-image`` -> ``zimg6b``.

    An unregistered id becomes a compact lowercase slug, so stand-in adapters in tests
    still get a file-safe tag.
    """
    known = _BASE_SHORT.get(mdl_id)
    if known is not None:
        return known
    return re.sub(r"[^a-z0-9]+", "", mdl_id.lower()) or "base"


def lora_base_of(metadata: Mapping[str, object]) -> str:
    """The base MDL id recorded in safetensors header metadata.

    New exports carry ``voxprint.lora.base_family``; older ones only
    ``voxprint.lora.base``. A file with neither predates base recording, and
    such a LoRA is a Z-Image one (the only base then). The file name is not
    consulted: an old name without a base short tag still reads.
    """
    for key in ("voxprint.lora.base_family", "voxprint.lora.base"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return legacy_base()


def _base_sha(info: object) -> str:
    """Pinned base digest on the adapter, or empty when the adapter has none yet."""
    return str(getattr(info, "sha256", "") or "")


def dataset_digest(items: Sequence[Item]) -> str:
    """sha256 of the sorted item ids and their caption text.

    One value per prepared dataset: it does not change with the item order and
    is the same no matter which base the dataset was encoded for.
    """
    digest = hashlib.sha256()
    for item in sorted(items, key=lambda entry: entry.id):
        for part in (item.id, item.long, item.short):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
    return digest.hexdigest()


def export(
    project: Project,
    progress: Progress,
    *,
    checkpoint: int | None = None,
    out: Path | None = None,
    install_to_mdl: bool = False,
    tags: tuple[str, ...] = (),
) -> dict[str, object]:
    """Write ``<id>.<base>.safetensors`` and ``voxprint-model.json``; optionally install them for Canvas."""
    data = project.read()
    train = data.get("train")
    if data.get("gate_passed") is not True or not isinstance(train, dict) or not train.get("checkpoint"):
        raise LoraUsageError("lora.err.no_checkpoint")
    name = f"step-{checkpoint:06d}.safetensors" if checkpoint is not None else str(train["checkpoint"])
    source = project.checkpoints / name
    if not source.is_file():
        raise LoraUsageError("lora.err.no_checkpoint")
    # Projects saved before the base was recorded trained on Z-Image.
    base_id = resolve_base(train.get("base"))
    info = get_adapter(base_id).info
    progress.emit("export.write", "lora.progress.export", reason_key="lora.reason.export")
    if not info.permissive:
        progress.emit("export.licence", "lora.err.base_restrictive")
    tensors = {key: value.detach().cpu().contiguous() for key, value in load_file(str(source)).items()}
    mid = model_id(data, tensors_digest(tensors))
    shown = {
        **_export_train(train, name, checkpoint is None),
        "base": base_id,
        "base_variant": base_variant(base_id, train.get("base_variant") or getattr(info, "variant", "")),
        "base_sha256": _base_sha(info) or str(train.get("base_sha256", "")),
    }
    meta = header_metadata({**data, "train": shown}, tensors, dataset_id=dataset_digest(project.items()))
    file_name = f"{mid}.{base_short(info.mdl_id)}.safetensors"
    target = Path(out) if out is not None else project.export / file_name
    if target.suffix != ".safetensors":
        raise LoraUsageError("lora.err.no_checkpoint")
    target.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(target), metadata=meta)
    preview = _preview(project, _step(name), target.parent)
    sidecar = sidecar_document(mid, {**data, "train": shown}, target, info, preview, tags)
    _add_likeness(sidecar, data, _step(name))
    _write_json(target.parent / SIDECAR, sidecar)
    if info.licence.lower() == "apache-2.0":
        shutil.copyfile(_LICENCES / "Apache-2.0.txt", target.parent / "LICENSE")
        (target.parent / "NOTICE").write_text(
            f"This LoRA was trained on {info.display_name} ({info.repo}), licensed under the Apache License 2.0.\n"
            f"Produced by {PRODUCER}.\n",
            encoding="utf-8",
            newline="\n",
        )
    result: dict[str, object] = {"state": "done", "id": mid, "file": str(target), "sha256": sidecar["sha256"]}
    if install_to_mdl:
        progress.emit("export.install", "lora.progress.install", reason_key="lora.reason.install")
        folder = user_models_dir() / "lora" / mid
        folder.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(target, folder / file_name)
        _write_json(folder / SIDECAR, {**sidecar, "file": file_name})
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
        "base": {
            "mdl_id": info.mdl_id,
            "revision": str(dict(data.get("train") or {}).get("base_revision", "")),
            "sha256": str(getattr(info, "sha256", "")),
            "mirror": str(getattr(info, "mirror", "")),
        },
        "trained_base": trained_base_tag(
            info.mdl_id, str(dict(data.get("train") or {}).get("base_revision", ""))
        ),
        "compatible_bases": compatible_bases_list(info.mdl_id),
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


def _add_likeness(doc: dict[str, object], data: dict[str, object], step: int) -> None:
    """Copy the advisory likeness score onto the model card when it was measured on ``step``.

    The score is informational. Whatever its status, export goes on unchanged.
    """
    from voxprint_forge.lora.likeness import card_fields

    fields = card_fields(data, step)
    if fields is not None:
        doc["likeness"] = fields


def _preview(project: Project, step: int, folder: Path) -> Path | None:
    found = sorted(project.previews.glob(f"step-{step:06d}-*.png")) if project.previews.is_dir() else []
    if not found:
        return None
    from voxprint_forge.imageio_save import save_image

    with Image.open(found[0]) as image:
        rgb = image.convert("RGB")
    # A regular picture: lossy WebP at quality 90, with no camera or text metadata.
    return save_image(rgb, folder / "preview", "regular")


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
    path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


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
    """Write ``<stage name or name>.<base>.voxlora``: the small LoRA package, and nothing merged.

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
    # Projects saved before the base was recorded trained on Z-Image.
    base_id = resolve_base(train.get("base"))
    info = get_adapter(base_id).info
    progress.emit("export.write", "lora.progress.export", reason_key="lora.reason.export")
    tensors = _lora_tensors(source)
    shown = {
        **_export_train(train, name, checkpoint is None),
        "base": base_id,
        "base_variant": base_variant(base_id, train.get("base_variant") or getattr(info, "variant", "")),
        "base_sha256": _base_sha(info) or str(train.get("base_sha256", "")),
    }
    meta = header_metadata({**data, "train": shown}, tensors, dataset_id=dataset_digest(project.items()))
    target = _package_target(project, data, out, base_short(info.mdl_id))
    target.parent.mkdir(parents=True, exist_ok=True)
    weights = _safetensors_bytes(tensors, meta)
    pictures = _package_pictures(project)
    doc = _package_document(data, shown, info.mdl_id, hashlib.sha256(weights).hexdigest(), pictures)
    _add_likeness(doc, data, _step(name))
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


def _package_target(project: Project, data: dict[str, object], out: Path | None, short: str) -> Path:
    """Default ``<project>/export/<stem>.<base>.voxlora``, or ``out`` with that suffix."""
    stem = f"{package_stem(data)}.{short}"
    if out is None:
        return project.export / f"{stem}.voxlora"
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
        if not relative.parts or relative.parts[0] not in {project.previews.name, project.review.name}:
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
                add(project.resolve(str(sample["file"])))
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
    from voxprint_forge.imageio_save import image_bytes

    return image_bytes(clean, "regular")


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
