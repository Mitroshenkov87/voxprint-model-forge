"""Read both synthetic-set manifest layouts.

A row requires ``prompt`` and an integer ``seed``. ``negative_prompt`` (or
``negative``) is optional. ``width`` and ``height``, or ``size``, are optional:
a missing size is stored as 0 so the job width and height are used. ``model``,
``model_id``, ``revision``, ``steps``, ``guidance``, ``guidance_scale``, and
``cfg`` are ignored, because the job names the model under test and supplies
its steps and CFG. ``keep: false`` drops the row.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from voxprint_forge.compare.codes import CompareSetup


@dataclass(frozen=True)
class ManifestItem:
    """One image row. Prompt text stays in memory and is not written out."""

    image_key: str
    prompt: str
    negative: str
    seed: int
    width: int
    height: int


def find_manifest(folder: Path) -> Path | None:
    """``manifest.json`` at the set root, or ``dataset/manifest.json``."""
    top = folder / "manifest.json"
    nested = folder / "dataset" / "manifest.json"
    if top.is_file():
        return top
    if nested.is_file():
        return nested
    return None


def load_manifest(path: Path) -> list[ManifestItem]:
    """Parse one manifest file.

    ``_run`` and other non-image keys are skipped. ``keep: false`` drops the
    row before required fields are checked. A missing prompt or seed names
    that image key and is a setup error.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CompareSetup(f"manifest {path.name} is not JSON", code="compare_manifest") from exc
    if not isinstance(payload, dict):
        raise CompareSetup(f"manifest {path.name} is not an object", code="compare_manifest")
    rows: list[ManifestItem] = []
    for key, value in payload.items():
        if not isinstance(key, str) or key.startswith("_") or not isinstance(value, dict):
            continue
        if value.get("keep") is False:
            continue
        rows.append(_item(path.name, key, value))
    if not rows:
        raise CompareSetup(f"manifest {path.name} has no images", code="compare_manifest")
    return rows


def _item(filename: str, key: str, value: dict[str, object]) -> ManifestItem:
    prompt = value.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise CompareSetup(f"manifest {filename} entry {key} requires a prompt", code="compare_manifest")
    negative = value.get("negative_prompt", value.get("negative", ""))
    if negative is None:
        negative = ""
    if not isinstance(negative, str):
        raise CompareSetup(f"manifest {filename} entry {key} has a bad negative", code="compare_manifest")
    width, height = _size(filename, key, value)
    return ManifestItem(
        image_key=key,
        prompt=prompt,
        negative=negative,
        seed=_seed(filename, key, value.get("seed")),
        width=width,
        height=height,
    )


def _seed(filename: str, key: str, value: object) -> int:
    if isinstance(value, bool):
        raise CompareSetup(f"manifest {filename} entry {key} has no integer seed", code="compare_manifest")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise CompareSetup(f"manifest {filename} entry {key} has no integer seed", code="compare_manifest")


def _size(filename: str, key: str, value: dict[str, object]) -> tuple[int, int]:
    width = value.get("width")
    height = value.get("height")
    if width is not None or height is not None:
        return _dim(filename, key, width), _dim(filename, key, height)
    size = value.get("size")
    if isinstance(size, int) and not isinstance(size, bool):
        return size, size
    if isinstance(size, list) and len(size) == 2:
        return _dim(filename, key, size[0]), _dim(filename, key, size[1])
    if isinstance(size, str) and "x" in size:
        left, _sep, right = size.partition("x")
        return _dim(filename, key, left), _dim(filename, key, right)
    return 0, 0


def _dim(filename: str, key: str, value: object) -> int:
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompareSetup(f"manifest {filename} entry {key} has a bad size", code="compare_manifest")
    if value < 1:
        raise CompareSetup(f"manifest {filename} entry {key} has a bad size", code="compare_manifest")
    return value
