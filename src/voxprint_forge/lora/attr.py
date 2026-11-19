"""An optional typed subject attribute stored with integrity binding."""

from __future__ import annotations

import base64
import hashlib
import hmac
from collections.abc import Mapping
from pathlib import Path

from safetensors import safe_open

KEY = "voxprint.subject.attr_v"
LOW = 18
HIGH = 99
_SPAN = 2
_TAG = hashlib.sha256().digest_size
_KEY = bytes.fromhex(
    "6ac13e9457b208d6419fe32c7015a84b"
    "c0d7e91a3f58264b8e0c5d7a19b4f263"
)


def accepted(value: object) -> int | None:
    """``value`` when it is a whole number from 18 to 99."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if LOW <= value <= HIGH:
        return value
    return None


def encode(value: object, tensors: Mapping[str, object]) -> str | None:
    """Encoded form of an optional typed subject attribute stored with integrity binding."""
    number = accepted(value)
    if number is None:
        return None
    hidden = _hide(number)
    tag = _tag(_digest(tensors), hidden)
    return base64.b64encode(hidden + tag).decode("ascii")


def read_declared(
    path_or_metadata: str | Path | Mapping[str, object],
    tensors: Mapping[str, object] | None = None,
) -> int | None:
    """Return the optional typed subject attribute when its integrity binding verifies.

    ``path_or_metadata`` is a ``.safetensors`` path (metadata and tensors are loaded),
    or a metadata mapping used with ``tensors``. A missing field, a failed binding,
    or a value outside 18-99 yields ``None``.
    """
    try:
        return _read(path_or_metadata, tensors)
    except Exception:
        return None


def _read(
    path_or_metadata: str | Path | Mapping[str, object],
    tensors: Mapping[str, object] | None,
) -> int | None:
    if isinstance(path_or_metadata, (str, Path)):
        loaded_meta, loaded_tensors = _load(Path(path_or_metadata))
        meta: Mapping[str, object] = loaded_meta
        if tensors is None:
            tensors = loaded_tensors
    elif isinstance(path_or_metadata, Mapping):
        meta = path_or_metadata
    else:
        return None
    if tensors is None:
        return None
    raw = meta.get(KEY)
    if not isinstance(raw, str) or not raw:
        return None
    blob = base64.b64decode(raw, validate=True)
    if len(blob) != _SPAN + _TAG:
        return None
    hidden, tag = blob[:_SPAN], blob[_SPAN:]
    if not hmac.compare_digest(tag, _tag(_digest(tensors), hidden)):
        return None
    return accepted(_show(hidden))


def _load(path: Path) -> tuple[dict[str, str], dict[str, object]]:
    with safe_open(str(path), framework="pt") as handle:
        meta = {str(key): str(value) for key, value in (handle.metadata() or {}).items()}
        tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    return meta, tensors


def _hide(value: int) -> bytes:
    raw = value.to_bytes(_SPAN, "big")
    mask = hashlib.sha256(_KEY).digest()
    return bytes(byte ^ mask[index] for index, byte in enumerate(raw))


def _show(hidden: bytes) -> int:
    mask = hashlib.sha256(_KEY).digest()
    raw = bytes(byte ^ mask[index] for index, byte in enumerate(hidden))
    return int.from_bytes(raw, "big")


def _tag(digest: str, hidden: bytes) -> bytes:
    return hmac.new(_KEY, digest.encode("ascii") + hidden, hashlib.sha256).digest()


def _digest(tensors: Mapping[str, object]) -> str:
    # Circular: export.header_metadata imports this module, and tensors_digest is defined there.
    from voxprint_forge.lora.export import tensors_digest

    return tensors_digest(dict(tensors))
