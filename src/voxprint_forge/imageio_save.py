"""Save pictures. This is the Forge-local writer.

It is planned to move to voxprint-common with the shared ``save_image``.
Import it from here until that move, so the call has one place to change.

The default format is WebP. ``master``, ``series`` and ``handoff`` are
lossless WebP; when this Pillow build cannot encode WebP the same call falls
back to JPEG, never PNG. ``regular`` is lossy at quality 90. ``fmt`` may be
``webp``, ``jpeg`` or ``avif``. An old ``png`` request — a leftover ``.png``
path, an accidental ``fmt="png"`` — is coerced to WebP instead of writing
PNG. JPEG has no lossless mode, so it stays at quality 90. AVIF is written
only when this Pillow build can encode it. When that encoder honors
``lossless=True`` it is used; otherwise a master, series or hand-off is
quality 100 with 4:4:4 sampling. Reading is unchanged: PNG, JPEG, WebP and
AVIF inputs all open.

:func:`save_lossless_webp` is the pixel-exact lossless WebP writer for prepared
training crops and other intermediates: the decoded pixels are the pixels written.

Metadata (prompt, seed, steps, base, LoRA name and strength, sha256) is
embedded only when asked: pass ``embed_metadata=True``, or turn on
:func:`embed_metadata_scope` from the ``--embed-metadata`` flag. The app
setting defaults to off, and a mapping passed while that switch is off is
not written. A training-preview stamp (``voxprint.*`` keys) is different:
it is written whenever ``stamp`` is passed, so a preview always records how
it was drawn.
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
from typing import Any, BinaryIO
from xml.etree import ElementTree
from xml.sax.saxutils import escape

__all__ = [
    "LOSSLESS_WEBP_METHOD",
    "LOSSLESS_WEBP_QUALITY",
    "LOSSY_QUALITY",
    "WEBP_MAX_SIDE",
    "ImageSaveError",
    "embed_metadata_enabled",
    "embed_metadata_scope",
    "format_of",
    "image_bytes",
    "read_metadata",
    "read_stamp",
    "save_image",
    "save_lossless_webp",
]

#: Lossy quality for a regular picture, and for JPEG (JPEG has no lossless mode).
LOSSY_QUALITY = 90
#: Encoder effort for :func:`save_lossless_webp`: in lossless mode quality is the encoder
#: effort, and method 0 / quality 25 measured 3-5x faster than PNG and 19-28% smaller
#: on 1024 px photo crops, with identical pixels.
LOSSLESS_WEBP_METHOD = 0
LOSSLESS_WEBP_QUALITY = 25
#: The WebP format limit on a picture's longer side.
WEBP_MAX_SIDE = 16383
#: AVIF quality when the encoder has no real lossless switch.
_AVIF_BEST_QUALITY = 100
_AVIF_LOSSLESS: bool | None = None
_LOSSLESS_KINDS = frozenset({"master", "series", "handoff"})
_KINDS = _LOSSLESS_KINDS | {"regular"}
#: Formats this writer emits. PNG is not among them; a PNG request becomes WebP.
_EXTENSIONS = {"webp": ".webp", "avif": ".avif", "jpeg": ".jpg"}
_PIL_FORMATS = {"webp": "WEBP", "avif": "AVIF", "jpeg": "JPEG"}
_XMP_NS = "https://voxprint.ai/ns/image/1.0/"
_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_PNG_XMP_KEY = "XML:com.adobe.xmp"
_TEXT_ESCAPES = {'"': "&quot;", "'": "&apos;"}
_EXIF_FORMATS = frozenset({"webp", "avif", "jpeg"})
#: The ``voxprint.*`` preview-stamp keys, without their prefix, as XMP attributes.
_STAMP_KEYS = ("seed", "lora_scale", "base_model", "lora_sha256", "step")
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
    """The writable format named by ``path``'s suffix, or None when it names none.

    ``.jpg`` and ``.jpeg`` are both JPEG. A ``.png`` suffix names no writable
    format anymore — pictures are never written as PNG — so it returns None
    and the caller's save falls back to the WebP default. Reading PNG inputs
    is unaffected.
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
    stamp: Mapping[str, str] | None = None,
) -> Path:
    """Write ``image`` and return the path, including the format's extension.

    ``path_stem`` may already end in an extension; it is replaced by the
    extension of the format actually written (WebP, or JPEG when the WebP
    encode fails — never PNG). JPEG stays ``.jpeg`` when the stem already
    used that suffix, and is ``.jpg`` otherwise.

    Args:
        image: A Pillow image. Camera and text metadata on it are dropped.
        path_stem: Destination without relying on its suffix.
        kind: ``master``, ``series`` and ``handoff`` are lossless. ``regular``
            is lossy WebP at :data:`LOSSY_QUALITY` when ``fmt`` is omitted.
        fmt: ``webp`` (the default), ``jpeg`` or ``avif``. ``png`` is coerced
            to ``webp``.
        metadata: Prompt, seed, steps, base, LoRA strength and sha256.
        embed_metadata: Write ``metadata`` into the file. False still writes
            it when :func:`embed_metadata_scope` is on. Otherwise the mapping
            is ignored.
        stamp: ``voxprint.*`` text values (a training preview's seed, scale,
            base, LoRA sha256 and step) written into the XMP packet whenever
            given, independent of ``embed_metadata``.

    Raises:
        ImageSaveError: The kind or format is unknown, AVIF is unavailable,
            or ``image`` is not a Pillow image.
    """
    payload, written = _encode(image, kind, fmt, metadata, embed_metadata, stamp)
    target = _target(Path(path_stem), written)
    _publish(target, payload)
    return target


def image_bytes(
    image: Any,
    kind: str,
    fmt: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    embed_metadata: bool = False,
    stamp: Mapping[str, str] | None = None,
) -> bytes:
    """Encode ``image`` the way :func:`save_image` would write it."""
    payload, _written = _encode(image, kind, fmt, metadata, embed_metadata, stamp)
    return payload


def save_lossless_webp(image: Any, target: Path | str | BinaryIO) -> None:
    """Write ``image`` to ``target`` as lossless WebP, with no metadata.

    ``target`` is a path or a binary file object. The decoded pixels are exactly
    the pixels written.

    Raises:
        ImageSaveError: A side of ``image`` is over :data:`WEBP_MAX_SIDE`, the WebP limit.
    """
    width, height = image.size
    if max(width, height) > WEBP_MAX_SIDE:
        raise ImageSaveError(f"WebP sides are limited to {WEBP_MAX_SIDE}px, got {width}x{height}")
    image.save(
        target,
        format="WEBP",
        lossless=True,
        method=LOSSLESS_WEBP_METHOD,
        quality=LOSSLESS_WEBP_QUALITY,
        exact=True,
    )


def _encode(
    image: Any,
    kind: str,
    fmt: str | None,
    metadata: Mapping[str, Any] | None,
    embed_metadata: bool,
    stamp: Mapping[str, str] | None,
) -> tuple[bytes, str]:
    """The encoded picture and the format actually written (JPEG after a failed WebP encode)."""
    chosen = _kind(kind)
    format_name = _format(fmt)
    _require_avif(format_name)
    clean = _rgb(image)
    document = _document(metadata) if (metadata and (bool(embed_metadata) or _EMBED.get())) else None
    packet: bytes | None = None
    if stamp or (document is not None and _has_metadata(document)):
        packet = _xmp(document or _blank(), stamp)
    buffer = io.BytesIO()
    try:
        clean.save(buffer, format=_PIL_FORMATS[format_name], **_save_kwargs(chosen, format_name, packet, document))
    except (OSError, ValueError, KeyError):
        if format_name != "webp":
            raise
        # A failed WebP encode falls back to JPEG, never PNG.
        format_name = "jpeg"
        buffer = io.BytesIO()
        clean.save(buffer, format="JPEG", **_save_kwargs(chosen, format_name, packet, document))
    return buffer.getvalue(), format_name


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


def read_stamp(path: Path | str) -> dict[str, str]:
    """The ``voxprint.*`` preview stamp written with the ``stamp`` argument.

    The five stamp keys (``seed``, ``lora_scale``, ``base_model``,
    ``lora_sha256``, ``step``) are read back from the XMP packet with their
    ``voxprint.`` prefix restored. A picture without the packet — or an old
    PNG preview whose stamp lives in text chunks — returns an empty mapping;
    readers of old previews fall back to the PNG text chunks first.
    """
    packet = _open_xmp(path)
    if not packet:
        return {}
    text = packet.decode("utf-8", "replace") if isinstance(packet, bytes) else packet
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return {}
    description = root.find(f".//{{{_RDF_NS}}}Description")
    if description is None:
        return {}
    found: dict[str, str] = {}
    for name in _STAMP_KEYS:
        value = description.get(f"{{{_XMP_NS}}}{name}")
        if value is not None:
            found[f"voxprint.{name}"] = str(value)
    return found


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
    if name == "png":
        # Pictures are never written as PNG. An old PNG request becomes WebP
        # (or JPEG when this Pillow cannot encode WebP).
        return "webp"
    if name not in _EXTENSIONS:
        raise ImageSaveError(f"format must be webp, avif or jpeg, got {fmt!r}")
    return name


def _require_avif(fmt: str) -> None:
    if fmt != "avif":
        return
    features = _pillow_features()
    if not features.check("avif"):
        raise ImageSaveError(
            "AVIF needs a Pillow build with AVIF support. This Pillow cannot encode AVIF."
        )


def _save_kwargs(
    kind: str, fmt: str, packet: bytes | None, document: Mapping[str, Any] | None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    lossless = kind in _LOSSLESS_KINDS
    if fmt == "webp":
        kwargs = {"lossless": True} if lossless else {"quality": LOSSY_QUALITY}
    elif fmt == "avif":
        kwargs = _avif_kwargs(lossless)
    elif fmt == "jpeg":
        kwargs = {"quality": LOSSY_QUALITY}
    if packet is not None:
        kwargs["xmp"] = packet
        if document is not None and _has_metadata(document) and fmt in _EXIF_FORMATS:
            kwargs["exif"] = _exif(document)
    return kwargs


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


def _xmp(document: Mapping[str, Any], stamp: Mapping[str, str] | None = None) -> bytes:
    attributes = ""
    written: set[str] = set()

    def add(name: str, value: str) -> None:
        nonlocal attributes
        if name in written:
            return
        written.add(name)
        attributes += f' vp:{name}="{value}"'

    if document["seed"] is not None:
        add("seed", str(document["seed"]))
    if document["steps"] is not None:
        add("steps", str(document["steps"]))
    if document["base"] is not None:
        add("base", _xml(str(document["base"])))
    if document["base_variant"] is not None:
        add("base_variant", _xml(str(document["base_variant"])))
    if document["lora_strength"] is not None:
        add("lora_strength", repr(document["lora_strength"]))
    if document["sha256"] is not None:
        add("sha256", _xml(str(document["sha256"])))
    if stamp:
        # Preview stamp: ``voxprint.seed`` becomes a ``vp:seed`` attribute and so on.
        # Written whenever given, unlike the document above, so a preview always
        # records how it was drawn. A key the document already wrote carries the
        # same value, so the document wins and the attribute stays unique.
        for key, value in stamp.items():
            add(str(key).removeprefix("voxprint."), _xml(str(value)))
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
