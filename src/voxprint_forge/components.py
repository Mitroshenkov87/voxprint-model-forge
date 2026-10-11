"""Model component files live under the shared models root.

A new component is written under ``VOXPRINT_MODEL_CACHE`` when that variable is
an absolute path, and otherwise under :func:`voxprint_common.paths.models_dir`.
An absolute path is accepted only when it is already inside one of those roots.
The shared manifest records the same rule.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from voxprint_common.hashing import sha256_file
from voxprint_common.paths import models_dir
from voxprint_common.runtime_manifest import add_ref

_WEIGHT_SUFFIXES = frozenset({".gguf", ".safetensors", ".bin", ".pt", ".ckpt", ".onnx"})
_ID_BREAK = re.compile(r"[^a-z0-9._-]+")


def model_component_root() -> Path:
    """Return the directory a new model component is written to.

    ``VOXPRINT_MODEL_CACHE`` wins when it is absolute. Otherwise the directory
    is :func:`voxprint_common.paths.models_dir`.
    """
    cache = os.environ.get("VOXPRINT_MODEL_CACHE", "").strip()
    if not cache:
        return models_dir()
    folder = Path(os.path.expandvars(os.path.expanduser(cache)))
    if not folder.is_absolute():
        raise ValueError("VOXPRINT_MODEL_CACHE must be an absolute path")
    return folder


def component_roots() -> tuple[Path, ...]:
    """Return the absolute directories a model component path may occupy."""
    roots = [models_dir(), model_component_root()]
    found: list[Path] = []
    for root in roots:
        if all(root != item for item in found):
            found.append(root)
    return tuple(found)


def resolve_model_component(name: str) -> Path:
    """Return a component path that stays under the shared models root.

    A relative name is joined onto :func:`model_component_root`. An absolute
    path must already sit under :func:`component_roots`.
    """
    text = name.strip().replace("\\", "/")
    if not text or ".." in Path(text).parts:
        raise ValueError("model component path must stay under the shared models root")
    candidate = Path(text)
    if candidate.is_absolute() or text.startswith("/"):
        if not _inside(candidate, component_roots()):
            raise ValueError("model component path must stay under the shared models root")
        return candidate
    dest = model_component_root() / candidate
    if not _inside(dest, component_roots()):
        raise ValueError("model component path must stay under the shared models root")
    return dest


def component_name(path: Path) -> str:
    """Relative component name for a finished weight file."""
    parts = path.parts
    if "work" in parts:
        index = parts.index("work")
        tail = "/".join(parts[index + 1 :])
        return f"model-forge/{tail}"
    return f"model-forge/{path.name}"


def place_model_component(source: Path, name: str, *, version: str, sha256: str) -> Path:
    """Link ``source`` into the shared models root and record it as a model component.

    The link is a hard link when the volumes allow it. A same-file digest is
    reused. The manifest path is the absolute component path.
    """
    dest = resolve_model_component(name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not dest.is_file():
        raise ValueError(f"{dest} is not a file")
    if not dest.is_file() or sha256_file(dest) != sha256:
        if dest.is_file() or dest.is_symlink():
            dest.unlink()
        target = source.resolve()
        try:
            os.link(target, dest)
        except OSError:
            shutil.copy2(target, dest)
    add_ref(
        _component_id(name),
        "model-forge",
        kind="model",
        version=version,
        path=str(dest.resolve()),
        sha256=sha256,
    )
    return dest


def place_weight(path: Path, *, digest: str | None = None) -> Path | None:
    """Place a weight file under the shared models root. Other files are skipped."""
    if path.suffix.lower() not in _WEIGHT_SUFFIXES or not path.is_file():
        return None
    value = digest if digest is not None else sha256_file(path)
    return place_model_component(path, component_name(path), version=value[:12], sha256=value)


def _component_id(name: str) -> str:
    text = name.strip().lower().replace("\\", "/").replace("/", ".")
    text = _ID_BREAK.sub("-", text).strip(".-")
    if not text or not text[0].isalnum():
        text = f"file-{text}" if text else "file"
    if text.startswith("model-forge"):
        return text
    return f"model-forge.{text}"


def _inside(candidate: Path, roots: tuple[Path, ...]) -> bool:
    cand = os.path.normcase(str(candidate.resolve()))
    for root in roots:
        base = os.path.normcase(str(root.resolve()))
        if cand == base or cand.startswith(base + os.sep) or cand.startswith(base + "/"):
            return True
    return False
