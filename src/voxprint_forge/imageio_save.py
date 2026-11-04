"""Save pictures. This is the Forge-local writer.

It is planned to move to voxprint-common with the shared ``save_image``.
Import it from here until that move, so the call has one place to change.

The default format is WebP. ``master``, ``series`` and ``handoff`` are
lossless WebP or PNG. ``regular`` is lossy at quality 90. ``fmt`` may be
``webp``, ``png``, ``jpeg`` or ``avif``. JPEG has no lossless mode, so it
stays at quality 90. AVIF is written only when this Pillow build can encode
it. When that encoder honors ``lossless=True`` it is used; otherwise a
master, series or hand-off is quality 100 with 4:4:4 sampling.

Metadata (prompt, seed, steps, base, LoRA name and strength, sha256) is
embedded only when asked: pass ``embed_metadata=True``, or turn on
:func:`embed_metadata_scope` from the ``--embed-metadata`` flag. The app
setting defaults to off, and a mapping passed while that switch is off is
not written.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from xml.sax.saxutils import escape

__all__ = [
    "LOSSY_QUALITY",
    "ImageSaveError",
    "embed_metadata_enabled",
    "embed_metadata_scope",
    "format_of",
    "image_bytes",
    "read_metadata",
    "save_image",
]

#: Lossy quality for a regular picture, and for JPEG (JPEG has no lossless mode).
LOSSY_QUALITY = 90
#: AVIF quality when the encoder has no real lossless switch.
_AVIF_BEST_QUALITY = 100
_AVIF_LOSSLESS: bool | None = None
_LOSSLESS_KINDS = frozenset({"master", "series", "handoff"})
_KINDS = _LOSSLESS_KINDS | {"regular"}
_EXTENSIONS = {"webp": ".webp", "png": ".png", "avif": ".avif", "jpeg": ".jpg"}
_PIL_FORMATS = {"webp": "WEBP", "png": "PNG", "avif": "AVIF", "jpeg": "JPEG"}
_XMP_NS = "https://voxprint.ai/ns/image/1.0/"
_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_PNG_XMP_KEY = "XML:com.adobe.xmp"
_TEXT_ESCAPES = {'"': "&quot;", "'": "&apos;"}
_EXIF_FORMATS = frozenset({"webp", "avif", "jpeg"})
_EMPTY_LORA = {"name": None, "strength": None, "sha256": None, "base": None}
_EMBED: ContextVar[bool] = ContextVar("voxprint_forge_embed_metadata", default=False)


class ImageSaveError(ValueError):
    """The picture could not be saved in the requested format."""


@contextmanager
def embed_metadata_scope(enabled: bool) -> Iterator[None]:
    """Opt in to metadata for the pictures saved inside this block.

    The ``--embed-metadata`` flag uses this. A save made outside the block,
    or inside a block opened with ``False``, stays free of the document.
    """
    token = _EMBED.set(bool(enabled))
    try:
        yield
    finally:
        _EMBED.reset(token)


def embed_metadata_enabled() -> bool:
    """Whether the current block asked for picture metadata. Default off."""
    return bool(_EMBED.get())


def format_of(path: Path | str) -> str | None:
    """The format named by ``path``'s suffix, or None when it names none.

    ``.jpg`` and ``.jpeg`` are both JPEG. Callers that already chose a file
    name (a review PNG, ``--out picture.png``) pass this back as ``fmt`` so
    the written path stays the one the caller named.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".jpeg":
        return "jpeg"
    for name, extension in _EXTENSIONS.items():
        if suffix == extension:
            return name
    return None


def save_image(
    image: Any,
    path_stem: Path | str,
    kind: str,
    fmt: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    embed_metadata: bool = False,
) -> Path:
    """Write ``image`` and return the path, including the format's extension.

    ``path_stem`` may already end in an extension; it is replaced by the
    extension of the format actually written. JPEG stays ``.jpeg`` when the
    stem already used that suffix, and is ``.jpg`` otherwise.

    Args:
        image: A Pillow image. Camera and text metadata on it are dropped.
        path_stem: Destination without relying on its suffix.
        kind: ``master``, ``series`` and ``handoff`` are lossless. ``regular``
            is lossy WebP at :data:`LOSSY_QUALITY` when ``fmt`` is omitted.
        fmt: ``webp`` (the default), ``png``, ``jpeg`` or ``avif``.
        metadata: Prompt, seed, steps, base, LoRA strength and sha256.
        embed_metadata: Write ``metadata`` into the file. False still writes
            it when :func:`embed_metadata_scope` is on. Otherwise the mapping
            is ignored.

    Raises:
        ImageSaveError: The kind or format is unknown, AVIF is unavailable,
            or ``image`` is not a Pillow image.
    """
    target = _target(Path(path_stem), _format(fmt))
    _publish(target, image_bytes(image, kind, fmt, metadata, embed_metadata))
    return target


def image_bytes(
    image: Any,
    kind: str,
    fmt: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    embed_metadata: bool = False,
) -> bytes:
    """Encode ``image`` the way :func:`save_image` would write it."""
    chosen = _kind(kind)
    format_name = _format(fmt)
    _require_avif(format_name)
    clean = _rgb(image)
    kwargs = _save_kwargs(chosen, format_name)
    if metadata and (bool(embed_metadata) or _EMBED.get()):
        document = _document(metadata)
        if _has_metadata(document):
            packet = _xmp(document)
            if format_name == "png":
                kwargs["pnginfo"] = _png_info(packet)
            else:
                kwargs["xmp"] = packet
                if format_name in _EXIF_FORMATS:
                    kwargs["exif"] = _exif(document)
    buffer = io.BytesIO()
    clean.save(buffer, format=_PIL_FORMATS[format_name], **kwargs)
    return buffer.getvalue()


def read_metadata(path: Path | str) -> dict[str, Any]:
    """Read the Voxprint metadata written by :func:`save_image`.

    Missing entries are None, and ``loras`` is a list (empty when absent).
    ``base_model`` repeats ``base``. A file with no Voxprint packet returns
    that empty document.
    """
    packet = _open_xmp(path)
    if not packet:
        return _blank()
    return _parse_xmp(packet)


def _target(path: Path, fmt: str) -> Path:
    if fmt == "jpeg" and path.suffix.lower() == ".jpeg":
        return path
    return path.with_suffix(_EXTENSIONS[fmt])


def _kind(kind: str) -> str:
    name = str(kind).strip().lower().replace("-", "")
    if name not in _KINDS:
        raise ImageSaveError(f"kind must be master, series, handoff or regular, got {kind!r}")
    return name


def _format(fmt: str | None) -> str:
    if fmt is None or fmt == "":
        return "webp"
    name = str(fmt).strip().lower()
    if name == "jpg":
        name = "jpeg"
    if name not in _EXTENSIONS:
        raise ImageSaveError(f"format must be webp, png, avif or jpeg, got {fmt!r}")
    return name


def _require_avif(fmt: str) -> None:
    if fmt != "avif":
        return
    features = _pillow_features()
    if not features.check("avif"):
        raise ImageSaveError(
            "AVIF needs a Pillow build with AVIF support. This Pillow cannot encode AVIF."
        )


def _save_kwargs(kind: str, fmt: str) -> dict[str, Any]:
    lossless = kind in _LOSSLESS_KINDS
    if fmt == "png":
        return {}
    if fmt == "webp":
        return {"lossless": True} if lossless else {"quality": LOSSY_QUALITY}
    if fmt == "avif":
        return _avif_kwargs(lossless)
    return {"quality": LOSSY_QUALITY}


def _avif_kwargs(lossless: bool) -> dict[str, Any]:
    """Encoder options for AVIF. True lossless when this Pillow honors it."""
    if lossless and _avif_lossless():
        return {"lossless": True}
    if lossless:
        return {"quality": _AVIF_BEST_QUALITY, "subsampling": "4:4:4"}
    return {"quality": LOSSY_QUALITY}


def _avif_lossless() -> bool:
    """Whether ``lossless=True`` round-trips pixels on this Pillow build."""
    global _AVIF_LOSSLESS
    if _AVIF_LOSSLESS is not None:
        return _AVIF_LOSSLESS
    Image = _pillow()
    source = Image.frombytes("RGB", (8, 8), bytes(range(8 * 8 * 3)))
    buffer = io.BytesIO()
    try:
        source.save(buffer, format="AVIF", lossless=True)
        opened = Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
        _AVIF_LOSSLESS = opened.tobytes() == source.tobytes()
    except (OSError, ValueError):
        _AVIF_LOSSLESS = False
    return _AVIF_LOSSLESS


def _rgb(image: Any) -> Any:
    """A fresh RGB image. The copy drops EXIF, XMP and text from the source."""
    Image = _pillow()
    if not isinstance(image, Image.Image):
        raise ImageSaveError("save_image needs a Pillow image")
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    return Image.frombytes("RGB", rgb.size, rgb.tobytes())


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(name)
        raise


def _blank() -> dict[str, Any]:
    return {
        "prompt": None,
        "seed": None,
        "steps": None,
        "base": None,
        "base_model": None,
        "base_variant": None,
        "lora_strength": None,
        "sha256": None,
        "loras": [],
    }


def _document(metadata: Mapping[str, Any]) -> dict[str, Any]:
    document = _blank()
    prompt = metadata.get("prompt")
    if prompt is not None:
        document["prompt"] = str(prompt)
    seed = _as_int(metadata.get("seed"))
    if seed is not None:
        document["seed"] = seed
    steps = _as_int(metadata.get("steps"))
    if steps is not None:
        document["steps"] = steps
    base = metadata.get("base", metadata.get("base_model"))
    if base is not None:
        document["base"] = str(base)
        document["base_model"] = str(base)
    variant = metadata.get("base_variant")
    if isinstance(variant, str) and variant.strip() in {"sft", "base", "z-image"}:
        document["base_variant"] = variant.strip()
    strength = _as_float(metadata.get("lora_strength", metadata.get("strength")))
    if strength is not None:
        document["lora_strength"] = strength
    sha = metadata.get("sha256")
    if sha is not None:
        document["sha256"] = str(sha)
    loras = [_lora(entry) for entry in metadata.get("loras") or ()]
    name = metadata.get("lora", metadata.get("lora_name"))
    if not loras and (name is not None or strength is not None or sha is not None):
        loras.append({
            "name": None if name is None else str(name),
            "strength": strength,
            "sha256": None if sha is None else str(sha),
            "base": None if base is None else str(base),
        })
    document["loras"] = loras
    if document["sha256"] is None and len(loras) == 1 and loras[0]["sha256"]:
        document["sha256"] = loras[0]["sha256"]
    if document["lora_strength"] is None and len(loras) == 1 and loras[0]["strength"] is not None:
        document["lora_strength"] = loras[0]["strength"]
    return document


def _has_metadata(document: Mapping[str, Any]) -> bool:
    if document["loras"]:
        return True
    return any(document[key] is not None for key in ("prompt", "seed", "steps", "base", "lora_strength", "sha256"))


def _lora(entry: Any) -> dict[str, Any]:
    if isinstance(entry, str):
        return {**_EMPTY_LORA, "name": entry}
    if not isinstance(entry, Mapping):
        return dict(_EMPTY_LORA)
    base = entry.get("base", entry.get("base_model"))
    return {
        "name": None if entry.get("name") is None else str(entry.get("name")),
        "strength": _as_float(entry.get("strength")),
        "sha256": None if entry.get("sha256") is None else str(entry.get("sha256")),
        "base": None if base is None else str(base),
    }


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    with contextlib.suppress(TypeError, ValueError):
        return int(value)
    return None


def _as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    with contextlib.suppress(TypeError, ValueError):
        return float(value)
    return None


def _xml(value: str) -> str:
    text = escape(str(value), _TEXT_ESCAPES)
    return text.replace("\r\n", "&#13;\n").replace("\r", "&#13;")


def _xmp(document: Mapping[str, Any]) -> bytes:
    attributes = ""
    if document["seed"] is not None:
        attributes += f' vp:seed="{document["seed"]}"'
    if document["steps"] is not None:
        attributes += f' vp:steps="{document["steps"]}"'
    if document["base"] is not None:
        attributes += f' vp:base="{_xml(str(document["base"]))}"'
    if document["base_variant"] is not None:
        attributes += f' vp:base_variant="{_xml(str(document["base_variant"]))}"'
    if document["lora_strength"] is not None:
        attributes += f' vp:lora_strength="{document["lora_strength"]!r}"'
    if document["sha256"] is not None:
        attributes += f' vp:sha256="{_xml(str(document["sha256"]))}"'
    elements = ""
    if document["prompt"] is not None:
        elements += f"<vp:prompt>{_xml(str(document['prompt']))}</vp:prompt>"
    if document["loras"]:
        items = "".join(_lora_xml(lora) for lora in document["loras"])
        elements += f"<vp:loras><rdf:Bag>{items}</rdf:Bag></vp:loras>"
    packet = (
        '<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        f'  <rdf:RDF xmlns:rdf="{_RDF_NS}">\n'
        f'    <rdf:Description rdf:about="" xmlns:vp="{_XMP_NS}"{attributes}>'
        f"{elements}</rdf:Description>\n"
        "  </rdf:RDF>\n"
        "</x:xmpmeta>\n"
        '<?xpacket end="w"?>'
    )
    return packet.encode("utf-8")


def _lora_xml(lora: Mapping[str, Any]) -> str:
    attributes = ""
    if lora["name"] is not None:
        attributes += f' vp:name="{_xml(str(lora["name"]))}"'
    if lora["strength"] is not None:
        attributes += f' vp:strength="{lora["strength"]!r}"'
    if lora["sha256"] is not None:
        attributes += f' vp:sha256="{_xml(str(lora["sha256"]))}"'
    if lora["base"] is not None:
        attributes += f' vp:base="{_xml(str(lora["base"]))}"'
    return f"<rdf:li><rdf:Description{attributes}/></rdf:li>"


def _parse_xmp(packet: bytes | str) -> dict[str, Any]:
    text = packet.decode("utf-8", "replace") if isinstance(packet, bytes) else packet
    found = _blank()
    root = ElementTree.fromstring(text)
    description = root.find(f".//{{{_RDF_NS}}}Description")
    if description is None:
        return found
    prompt = description.find(f"{{{_XMP_NS}}}prompt")
    if prompt is not None and prompt.text is not None:
        found["prompt"] = prompt.text
    found["seed"] = _as_int(description.get(f"{{{_XMP_NS}}}seed"))
    found["steps"] = _as_int(description.get(f"{{{_XMP_NS}}}steps"))
    base = description.get(f"{{{_XMP_NS}}}base")
    if base is not None:
        found["base"] = base
        found["base_model"] = base
    variant = description.get(f"{{{_XMP_NS}}}base_variant")
    if variant is not None:
        found["base_variant"] = variant
    found["lora_strength"] = _as_float(description.get(f"{{{_XMP_NS}}}lora_strength"))
    sha = description.get(f"{{{_XMP_NS}}}sha256")
    if sha is not None:
        found["sha256"] = sha
    bag = description.find(f"{{{_XMP_NS}}}loras/{{{_RDF_NS}}}Bag")
    if bag is not None:
        found["loras"] = [_parse_lora(item) for item in bag.findall(f"{{{_RDF_NS}}}li")]
    return found


def _parse_lora(item: ElementTree.Element) -> dict[str, Any]:
    nested = item.find(f"{{{_RDF_NS}}}Description")
    node = nested if nested is not None else item
    return {
        "name": node.get(f"{{{_XMP_NS}}}name"),
        "strength": _as_float(node.get(f"{{{_XMP_NS}}}strength")),
        "sha256": node.get(f"{{{_XMP_NS}}}sha256"),
        "base": node.get(f"{{{_XMP_NS}}}base"),
    }


def _exif(document: Mapping[str, Any]) -> Any:
    """A small EXIF copy of the same document, for tools that skip XMP."""
    Image = _pillow()
    payload = json.dumps(_exif_document(document), ensure_ascii=True, sort_keys=True)
    exif = Image.Exif()
    exif[0x010E] = payload  # ImageDescription
    exif[0x9286] = b"ASCII\x00\x00\x00" + payload.encode("ascii")  # UserComment
    return exif


def _exif_document(document: Mapping[str, Any]) -> dict[str, Any]:
    def drop(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: drop(item) for key, item in value.items() if item is not None}
        if isinstance(value, list):
            return [drop(item) for item in value]
        return value

    return drop(dict(document))


def _png_info(packet: bytes) -> Any:
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_itxt(_PNG_XMP_KEY, packet.decode("utf-8"))
    return info


def _open_xmp(path: Path | str) -> bytes | None:
    Image = _pillow()
    with Image.open(Path(path)) as image:
        packet = image.info.get("xmp")
        if packet is None:
            packet = image.info.get(_PNG_XMP_KEY)
    if isinstance(packet, str):
        return packet.encode("utf-8")
    return packet


def _pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImageSaveError("saving a picture needs Pillow") from exc
    return Image


def _pillow_features() -> Any:
    try:
        from PIL import features
    except ImportError as exc:
        raise ImageSaveError("saving a picture needs Pillow") from exc
    return features
