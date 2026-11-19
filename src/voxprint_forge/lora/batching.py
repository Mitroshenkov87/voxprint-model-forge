"""How many photos ``lora prepare`` sends to the vision model in one ``generate`` call.

The safety judge and the captioner share one parked Qwen3-VL-4B copy. On the 16 GB card
target it loads int8 (about 5 GB of weights on the card). ``DEFAULT_BATCH`` (16, also
``MAX_BATCH``) is the size for both stages; with CUDA it is lowered to what fits in the free
graphics memory after the model is loaded: ``HEADROOM_GIB`` stays free for the CUDA allocator,
fragmentation and the longest answer, and each image in flight is budgeted
``PER_IMAGE_GIB[stage]``, so the size is ``min(DEFAULT_BATCH, int((free - HEADROOM_GIB) /
PER_IMAGE_GIB[stage]))``, never below 1.

Measured on an RTX 4090 Laptop (16 GB, int8 Qwen3-VL-4B): batch 16 peaked at 14.5 GB in the
check and 12.0 GB in the caption stage, about 0.59 GB per image in flight in the check
(320-token JSON answers) and 0.44 GB in the caption stage; 32 photos prepared in 152.2 s
instead of 675.7 s. With about 10.9 GiB free after the weights both stages get 16; smaller
cards scale down. Without a CUDA device, or when torch was never loaded, the size is 1.

``VOXPRINT_PREPARE_BATCH`` overrides the estimate for both stages (clamped to 1..``ENV_MAX``).

Size-aware planning: the judge downsizes each photo to about 1280 px on the long side, so a
padded batch costs as much as its largest image, while the age estimator, the decode and the temp
frame written for it scale with the full pixel count — and two decoded chunks sit in RAM at once.
``size_class`` splits photos into ``light`` (up to ``LIGHT_MAX_PIXELS``), ``medium`` and ``heavy``
(from ``HEAVY_MIN_PIXELS``); ``plan_batches`` cuts each class into ``class_batch`` chunks (the
full VRAM-tier ``batch`` for light photos, half for medium, a quarter for heavy) and orders the
chunks by first photo, so the run still walks the set from the first file to the last.
``plan_weights`` gives each photo ``photo_weight`` — a fixed cost for the judge answer and the
estimator call plus the pixel count over ``WEIGHT_PIXELS`` — for a weighted progress estimate
that stays steady when photos of very different sizes share a batch.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

ENV = "VOXPRINT_PREPARE_BATCH"
DEFAULT_BATCH = 16
MAX_BATCH = DEFAULT_BATCH
ENV_MAX = 32
HEADROOM_GIB = 1.0
PER_IMAGE_GIB = {"check": 0.59, "caption": 0.44}
LIGHT_MAX_PIXELS = 1280 * 1280
HEAVY_MIN_PIXELS = 12_000_000
WEIGHT_PIXELS = 12_000_000
_GIB = 1024**3


def prepare_batch_size(stage: str = "check") -> int:
    """The batch size for ``stage`` (``check`` or ``caption``). Call it after the model is loaded.

    ``VOXPRINT_PREPARE_BATCH`` (an integer, clamped to 1..``ENV_MAX``) wins. Otherwise torch must
    already be imported (this never imports it) and see a CUDA device; else the size is 1. With a
    device, ``min(DEFAULT_BATCH, int((free - HEADROOM_GIB) / PER_IMAGE_GIB[stage]))``, at least 1.
    An unknown ``stage`` raises ``ValueError``.
    """
    if stage not in PER_IMAGE_GIB:
        raise ValueError(f"unknown prepare stage {stage!r}")
    raw = os.environ.get(ENV, "").strip()
    if raw:
        try:
            return max(1, min(ENV_MAX, int(raw)))
        except ValueError:
            pass
    torch = sys.modules.get("torch")
    if torch is None:
        return 1
    try:
        if not torch.cuda.is_available():
            return 1
        free, _total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 - no reading means one photo at a time
        return 1
    fit = int((float(free) / _GIB - HEADROOM_GIB) / PER_IMAGE_GIB[stage])
    return max(1, min(DEFAULT_BATCH, fit))


def photo_weight(width: int, height: int) -> float:
    """The progress weight of one photo: a fixed cost for the judge answer and the estimator call
    plus the pixel count over ``WEIGHT_PIXELS`` — a 12 MP photo weighs 2, a 24 MP photo 3.

    Only the relative weights matter: the progress estimate re-measures the seconds per weight
    unit as the run goes. A non-positive size weighs 1.0.
    """
    if width <= 0 or height <= 0:
        return 1.0
    return 1.0 + width * height / WEIGHT_PIXELS


def size_class(width: int, height: int) -> str:
    """``"light"`` up to ``LIGHT_MAX_PIXELS``, ``"heavy"`` from ``HEAVY_MIN_PIXELS``, else ``"medium"``.

    The judge downsizes to about 1280 px on the long side, so a light photo costs the padded batch
    no extra image tokens; the estimator, the decode and the temp frame still scale with pixels,
    which is why heavy photos are chunked smaller.
    """
    pixels = width * height
    if pixels <= LIGHT_MAX_PIXELS:
        return "light"
    if pixels >= HEAVY_MIN_PIXELS:
        return "heavy"
    return "medium"


def class_batch(batch: int, kind: str) -> int:
    """The chunk size for a size class: the VRAM-tier ``batch`` for light photos, half for medium,
    a quarter for heavy — never below 1, and a ``batch`` below 1 counts as 1.

    A padded batch costs as much as its largest image, so more pixels mean a smaller chunk while
    the tier size stays the upper bound for every class. An unknown ``kind`` raises ``ValueError``.
    """
    batch = max(1, batch)
    if kind == "light":
        return batch
    if kind == "medium":
        return max(1, batch // 2)
    if kind == "heavy":
        return max(1, batch // 4)
    raise ValueError(f"unknown size class {kind!r}")


def plan_batches(sizes: Sequence[tuple[int, int]], batch: int) -> list[list[int]]:
    """The indices of ``sizes``, grouped by size class, cut into ``class_batch`` chunks and ordered by first index.

    Inside a class the input order is kept, so ordering the chunks by their first index keeps the
    run walking the photo set from the first file to the last. Every index appears exactly once and
    the result depends only on ``sizes`` and ``batch``; empty input gives ``[]``.
    """
    by_class = {"light": [], "medium": [], "heavy": []}
    for index, (width, height) in enumerate(sizes):
        by_class[size_class(width, height)].append(index)
    chunks: list[list[int]] = []
    for kind, indices in by_class.items():
        size = class_batch(batch, kind)
        chunks.extend(indices[start : start + size] for start in range(0, len(indices), size))
    chunks.sort(key=lambda chunk: chunk[0])
    return chunks


def plan_weights(sizes: Sequence[tuple[int, int]]) -> list[float]:
    """``photo_weight`` per entry, the work amounts matching the indices of ``plan_batches``."""
    return [photo_weight(width, height) for width, height in sizes]
