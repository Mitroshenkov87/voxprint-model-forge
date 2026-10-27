"""Photo import, limits, filters (size, blur, near-duplicates) and cropping."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_avif  # registers AVIF when Pillow itself has none
except ImportError:
    pillow_avif = None

from voxprint_forge.lora.errors import LoraUsageError
from voxprint_forge.lora.limits import (  # noqa: F401 - re-export
    EXPERT_CAP,
    MIN_SIDE,
    RECOMMENDED_RANGE,
    need_more,
)

EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".avif")
_COLLAGE_WORD = re.compile(r"\b(?:grids?|collages?)\b", re.IGNORECASE)
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
    """Raise on a count the mode refuses; return ``lora.warn.count`` when Expert is outside the recommended range."""
    low, high = RECOMMENDED_RANGE
    if count == 0:
        raise LoraUsageError("lora.err.no_photos")
    need = need_more(count, mode)
    if need:
        raise LoraUsageError("lora.err.too_few", need=need)
    if mode == "simple":
        if count > high:
            raise LoraUsageError("lora.err.too_many", extra=count - high, high=high)
        return None
    if count > EXPERT_CAP:
        raise LoraUsageError("lora.err.above_cap", cap=EXPERT_CAP)
    return "lora.warn.count" if not low <= count <= high else None


def photo_id(path: Path) -> str:
    """``sha256[:12]`` of the file bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def avif_ready() -> bool:
    """True when Pillow itself, or pillow-avif-plugin, can decode AVIF."""
    if pillow_avif is not None:
        return True
    return ".avif" in {key.lower() for key in Image.registered_extensions()}


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
    from PIL import Image

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
    if looks_like_collage(image):
        return "lora.drop.collage"
    if sharpness(image) < blur_threshold:
        return "lora.drop.blurry"
    value = dhash(image)
    if any(bin(value ^ other).count("1") <= dup_threshold for other in seen):
        return "lora.drop.duplicate"
    seen.append(value)
    return None


def looks_like_collage(image: Image.Image) -> bool:
    """True when straight interior seams divide the picture into a grid of panels.

    A seam is a narrow spike in the row or column gradient, well above the
    typical edge. Plain noise and a smooth gradient have no such spike. A
    caption that says grid or collage is a separate check.
    """
    small = image.convert("L").resize((90, 90), Image.Resampling.BOX)
    grey = np.asarray(small, dtype=np.float32)
    vertical = _seam_count(np.abs(np.diff(grey, axis=1)).mean(axis=0))
    horizontal = _seam_count(np.abs(np.diff(grey, axis=0)).mean(axis=1))
    return 1 <= vertical <= 4 and 1 <= horizontal <= 4


def caption_marks_collage(*parts: str) -> bool:
    """True when a caption calls the picture a grid or a collage."""
    return any(_COLLAGE_WORD.search(part) for part in parts)


def _seam_count(score: np.ndarray) -> int:
    length = int(score.shape[0])
    margin = max(2, length // 10)
    if length <= margin * 2 + 2:
        return 0
    baseline = float(np.median(score))
    limit = max(baseline + 18.0, baseline * 2.2, 22.0)
    hits = 0
    index = margin
    while index < length - margin:
        value = float(score[index])
        if value >= limit:
            left = float(score[max(0, index - 2)])
            right = float(score[min(length - 1, index + 2)])
            if value >= left + 10.0 and value >= right + 10.0:
                hits += 1
                index += 5
                continue
        index += 1
    return hits


def crop(image: Image.Image, mode: str, side: int = PREPARED_SIDE) -> Image.Image:
    """``subject``: centred square crop. ``whole``: the full photo padded to a square. Then ``side`` px."""
    from PIL import Image, ImageOps

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
