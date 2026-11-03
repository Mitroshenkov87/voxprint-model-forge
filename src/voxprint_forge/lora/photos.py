"""Photo import, limits, filters (size, blur, near-duplicates) and cropping."""

from __future__ import annotations

import hashlib
import io
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
# Collage check: the picture is scaled so its long side is this many pixels, aspect kept.
_COLLAGE_SIDE = 256
COLLAGE_SCORE = 0.75
_SEAM_TOLERANCE = 0.03  # a seam may sit this share of the side off the exact split
_SEAM_MIN_STEP = 10.0
_SEAM_RATIO = 2.5
_SEAM_OFFSET = 3.0
_TEXTURE_NEAR = 3
_TEXTURE_FAR = 8
# Drop reasons that ``lora prepare --keep FILE`` may override. Nothing else.
KEEPABLE = frozenset({"lora.drop.collage", "lora.drop.duplicate"})


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
    return bytes_id(Path(path).read_bytes())


def read_photo(path: Path) -> tuple[str, Image.Image | None]:
    """``(photo_id, open_photo)`` from one read of the file. Raises ``OSError`` like :func:`photo_id`."""
    data = Path(path).read_bytes()
    return bytes_id(data), decode_photo(io.BytesIO(data))


def bytes_id(data: bytes) -> str:
    """:func:`photo_id` of file bytes already read."""
    return hashlib.sha256(data).hexdigest()[:12]


def avif_ready() -> bool:
    """True when Pillow itself, or pillow-avif-plugin, can decode AVIF."""
    if pillow_avif is not None:
        return True
    return ".avif" in {key.lower() for key in Image.registered_extensions()}


def open_photo(path: Path) -> Image.Image | None:
    """Decoded RGB image with EXIF rotation applied, or ``None`` when it is not a picture."""
    try:
        with Path(path).open("rb") as handle:
            return decode_photo(handle)
    except Exception:  # noqa: BLE001 - any decode failure is an unreadable photo
        return None


def decode_photo(source: io.IOBase) -> Image.Image | None:
    """RGB pixels of ``source`` after its EXIF rotation, with no extra full-size copy.

    The same pixels as ``ImageOps.exif_transpose(image).convert("RGB")``: the decoded
    image is rotated in place and converted only when it is not RGB already. Like
    those calls, it returns a plain in-memory image that holds no file.
    """
    try:
        image = Image.open(source)
        image.load()
        ImageOps.exif_transpose(image, in_place=True)
        return image._new(image.im) if image.mode == "RGB" else image.convert("RGB")
    except Exception:  # noqa: BLE001 - any decode failure is an unreadable photo
        return None


def sharpness(image: Image.Image) -> float:
    """Variance of the Laplacian on a 512 px grey copy. Lower is blurrier."""
    return _sharpness_of(image.convert("L"))


def _sharpness_of(grey: Image.Image) -> float:
    """:func:`sharpness` of an ``L`` image, which is left as it is."""
    grey = grey.copy()
    grey.thumbnail((512, 512))
    a = np.asarray(grey, dtype=np.float32)
    lap = -4 * a[1:-1, 1:-1] + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
    return float(lap.var())


def dhash(image: Image.Image) -> int:
    """64-bit difference hash for near-duplicate detection."""
    return _dhash_of(image.convert("L"))


def _dhash_of(grey: Image.Image) -> int:
    """:func:`dhash` of an ``L`` image."""
    small = np.asarray(grey.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


@dataclass(frozen=True)
class Measures:
    """The numbers :func:`filter_reason` decides on, measured once per photo.

    ``collage``, ``sharp`` and ``hash`` are ``None`` for a photo under the size limit.
    They depend on the pixels only, so they can be measured on worker threads.
    """

    size: tuple[int, int]
    collage: float | None = None
    sharp: float | None = None
    hash: int | None = None


def measure(image: Image.Image | None, min_side: int = MIN_SIDE) -> Measures | None:
    """:class:`Measures` of ``image`` (``None`` for an unreadable photo). One grey copy serves blur and hash."""
    if image is None:
        return None
    if min(image.size) < min_side:
        return Measures(image.size)
    grey = image.convert("L")
    return Measures(image.size, collage_score(image), _sharpness_of(grey), _dhash_of(grey))


def filter_reason(
    image: Image.Image | None,
    seen: list[int],
    *,
    min_side: int = MIN_SIDE,
    blur_threshold: float = BLUR_THRESHOLD,
    dup_threshold: int = DUP_THRESHOLD,
    allow: frozenset[str] = frozenset(),
    measures: Measures | None = None,
) -> str | None:
    """``lora.drop.*`` key for a photo to set aside, else ``None`` (and its hash is added to ``seen``).

    ``allow`` holds reasons that are not applied to this photo; only :data:`KEEPABLE` reasons may be in it.
    ``measures`` is :func:`measure` of ``image`` with the same ``min_side`` when it was taken already.
    """
    if not allow <= KEEPABLE:
        raise ValueError(f"only {sorted(KEEPABLE)} can be allowed")
    if image is None:
        return "lora.drop.unreadable"
    if measures is None:
        measures = measure(image, min_side)
    if min(measures.size) < min_side:
        return "lora.drop.small"
    if "lora.drop.collage" not in allow and measures.collage >= COLLAGE_SCORE:
        return "lora.drop.collage"
    if measures.sharp < blur_threshold:
        return "lora.drop.blurry"
    value = measures.hash
    if "lora.drop.duplicate" not in allow and near_duplicate(value, seen, dup_threshold) is not None:
        return "lora.drop.duplicate"
    seen.append(value)
    return None


def near_duplicate(value: int, seen: Sequence[int], dup_threshold: int = DUP_THRESHOLD) -> int | None:
    """Index of the first hash in ``seen`` within ``dup_threshold`` bits of ``value``, else ``None``."""
    for index, other in enumerate(seen):
        if bin(value ^ other).count("1") <= dup_threshold:
            return index
    return None


def looks_like_collage(image: Image.Image) -> bool:
    """True when straight seams split the picture into equal panels (a grid or side-by-side shots).

    See :func:`collage_score`. A caption that says grid or collage is a separate check.
    """
    return collage_score(image) >= COLLAGE_SCORE


def collage_score(image: Image.Image) -> float:
    """How strongly the picture splits into 2, 3 or 4 equal panels along one axis, from ``0`` to ``1``.

    For each seam position of an equal split, the score is the share of rows (or
    columns) where the colour jumps sharply across that line compared with the
    texture next to it. A split scores the value of its weakest seam; the result
    is the best split on either axis. A seam between panels is sharp along almost
    its full length. Natural edges in a single photo (a person against a
    backdrop, a wall or a floor line) cover only part of the frame or do not sit
    on an equal split, so they stay well below :data:`COLLAGE_SCORE`.
    """
    width, height = image.size
    scale = _COLLAGE_SIDE / max(width, height, 1)
    size = (max(24, round(width * scale)), max(24, round(height * scale)))
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    rgb = np.asarray(rgb.resize(size, Image.Resampling.BOX), dtype=np.float32)
    return max(_split_score(rgb), _split_score(rgb.transpose(1, 0, 2)))


def caption_marks_collage(*parts: str) -> bool:
    """True when a caption calls the picture a grid or a collage."""
    return any(_COLLAGE_WORD.search(part) for part in parts)


def _split_score(rgb: np.ndarray) -> float:
    """Best equal-split score for vertical seams (columns) of an ``H x W x 3`` array."""
    width = int(rgb.shape[1])
    # Colour step across two pixels, so a seam softened by the downscale still counts.
    step = np.zeros(rgb.shape[:2], dtype=np.float32)
    step[:, 1:-1] = np.abs(rgb[:, 2:] - rgb[:, :-2]).max(axis=2)
    reach = max(2, round(_SEAM_TOLERANCE * width))
    best = 0.0
    for panels in (2, 3, 4):
        weakest = 1.0
        for seam in range(1, panels):
            centre = seam * width / panels
            lo = max(_TEXTURE_FAR + 1, int(centre - reach))
            hi = min(width - _TEXTURE_FAR - 1, int(centre + reach) + 1)
            weakest = min(weakest, max((_seam_cover(step, col) for col in range(lo, hi)), default=0.0))
        best = max(best, weakest)
    return best


def _seam_cover(step: np.ndarray, col: int) -> float:
    """Share of rows where ``col`` holds a sharp colour step well above the texture beside it."""
    near = [col + o for o in range(_TEXTURE_NEAR, _TEXTURE_FAR + 1)]
    near += [col - o for o in range(_TEXTURE_NEAR, _TEXTURE_FAR + 1)]
    texture = np.median(step[:, near], axis=1)
    here = step[:, col]
    return float(((here >= _SEAM_MIN_STEP) & (here >= _SEAM_RATIO * texture + _SEAM_OFFSET)).mean())


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
