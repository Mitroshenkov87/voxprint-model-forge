"""Photo import, limits, filters (size, blur, near-duplicates) and cropping."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from voxprint_forge.lora.errors import LoraUsageError

EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
SIMPLE_RANGE = (10, 30)
EXPERT_FLOOR = 5
EXPERT_CAP = 60
MIN_SIDE = 512
BLUR_THRESHOLD = 60.0
DUP_THRESHOLD = 6
PREPARED_SIDE = 1024


def collect(photos_dir: Path | None, photos: Sequence[Path], exclude: Sequence[Path]) -> list[Path]:
    """Image files from ``--photos DIR`` and ``--photo FILE``, minus ``--exclude``, sorted and unique."""
    found: list[Path] = []
    if photos_dir is not None:
        found += [p for p in sorted(Path(photos_dir).iterdir()) if p.is_file() and p.suffix.lower() in EXTENSIONS]
    found += [Path(p) for p in photos]
    skip = {Path(p).resolve() for p in exclude}
    unique: dict[Path, Path] = {}
    for path in found:
        key = path.resolve()
        if key not in skip:
            unique.setdefault(key, path)
    return list(unique.values())


def check_count(count: int, mode: str) -> str | None:
    """Raise on a count the mode refuses; return ``lora.warn.count`` when Expert is outside 10-30."""
    low, high = SIMPLE_RANGE
    if count == 0:
        raise LoraUsageError("lora.err.no_photos")
    if mode == "simple":
        if count < low:
            raise LoraUsageError("lora.err.too_few", need=low - count)
        if count > high:
            raise LoraUsageError("lora.err.too_many", extra=count - high)
        return None
    if count < EXPERT_FLOOR:
        raise LoraUsageError("lora.err.below_floor")
    if count > EXPERT_CAP:
        raise LoraUsageError("lora.err.above_cap", cap=EXPERT_CAP)
    return "lora.warn.count" if not low <= count <= high else None


def photo_id(path: Path) -> str:
    """``sha256[:12]`` of the file bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def open_photo(path: Path) -> Image.Image | None:
    """Decoded RGB image with EXIF rotation applied, or ``None`` when it is not a picture."""
    try:
        with Image.open(path) as image:
            return ImageOps.exif_transpose(image).convert("RGB")
    except Exception:  # noqa: BLE001 - any decode failure is an unreadable photo
        return None


def sharpness(image: Image.Image) -> float:
    """Variance of the Laplacian on a 512 px grey copy. Lower is blurrier."""
    grey = image.convert("L")
    grey.thumbnail((512, 512))
    a = np.asarray(grey, dtype=np.float32)
    lap = -4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    return float(lap.var())


def dhash(image: Image.Image) -> int:
    """64-bit difference hash for near-duplicate detection."""
    small = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def filter_reason(
    image: Image.Image | None,
    seen: list[int],
    *,
    min_side: int = MIN_SIDE,
    blur_threshold: float = BLUR_THRESHOLD,
    dup_threshold: int = DUP_THRESHOLD,
) -> str | None:
    """``lora.drop.*`` key for a photo to set aside, else ``None`` (and its hash is added to ``seen``)."""
    if image is None:
        return "lora.drop.unreadable"
    if min(image.size) < min_side:
        return "lora.drop.small"
    if sharpness(image) < blur_threshold:
        return "lora.drop.blurry"
    value = dhash(image)
    if any(bin(value ^ other).count("1") <= dup_threshold for other in seen):
        return "lora.drop.duplicate"
    seen.append(value)
    return None


def crop(image: Image.Image, mode: str, side: int = PREPARED_SIDE) -> Image.Image:
    """``subject``: centred square crop. ``whole``: the full photo padded to a square. Then ``side`` px."""
    if mode == "whole":
        width, height = image.size
        edge = max(width, height)
        fill = tuple(int(c) for c in np.asarray(image).reshape(-1, 3).mean(axis=0))
        canvas = Image.new("RGB", (edge, edge), fill)
        canvas.paste(image, ((edge - width) // 2, (edge - height) // 2))
        image = canvas
    else:
        image = ImageOps.fit(image, (min(image.size),) * 2, Image.Resampling.LANCZOS)
    return image.resize((side, side), Image.Resampling.LANCZOS)


@dataclass(frozen=True)
class Drop:
    """A photo set aside. ``reason`` is a ``lora.drop.*`` key; safety drops never say what was found."""

    id: str
    name: str
    reason: str
