"""Pinned imatrix corpus. The bytes are fetched at runtime; they are not in git.

bartowski's calibration_datav3 has no explicit licence, so the repository does
not redistribute the text. A file already at data/imatrix.txt is used only
when its sha256 matches the pin (the hot-patched VM copy). Otherwise the
pinned gist URL is downloaded into $FORGE_OUT/calibration/imatrix.txt.
"""

from __future__ import annotations

import hashlib
import os
import urllib.error
import urllib.request
from pathlib import Path

from voxprint_forge.errors import ChecksumError, DownloadError, PlanError
from voxprint_forge.evaluate import data_dir
from voxprint_forge.hashing import sha256_file
from voxprint_forge.plan import Plan

# bartowski calibration_datav3. The gist states no licence.
CALIBRATION_URL = "https://gist.github.com/bartowski1182/eb213dccb3571f863da82e99418f81e8"
CALIBRATION_RAW_URL = (
    "https://gist.githubusercontent.com/bartowski1182/"
    "eb213dccb3571f863da82e99418f81e8/raw/"
    "41ae165614f4041c262fa8fe668bbe3efd9eb695/calibration_datav3.txt"
)
CALIBRATION_SHA256 = "200e109bcd2b599fabcceaada7f52bbd1e7c8f9ae030b8dc59c011de039a8026"
# GPT-2 content tokens of that file, with no added BOS. llama.cpp adds BOS on top.
CALIBRATION_TOKENS = 74048
LICENCE_STATUS = "licence unverified, calibration-only, not redistributed in model files"
_MAX_BYTES = 2_000_000


def tokens_required(ctx: int, chunks: int) -> int:
    """Tokens required before llama-imatrix will honour ctx and chunks.

    tools/imatrix/imatrix.cpp at 98819068 errors when the tokenized file is
    shorter than 2*n_ctx. It also caps the chunk count at len(tokens)//n_ctx,
    so a request of `chunks` is real only when the file holds at least
    ctx*chunks tokens. The preflight requires both.
    """
    return max(2 * ctx, ctx * chunks)


def _cache_path() -> Path:
    root = os.environ.get("FORGE_OUT", "").strip()
    base = Path(root) if root else Path.home() / "forge-out"
    return base / "calibration" / "imatrix.txt"


def _pinned(path: Path) -> bool:
    return path.is_file() and sha256_file(path) == CALIBRATION_SHA256


def _fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "voxprint-forge"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read(_MAX_BYTES + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise DownloadError(
            f"calibration download failed: {exc.__class__.__name__}",
            code="download",
        ) from None
    if len(payload) > _MAX_BYTES:
        raise DownloadError("calibration download is larger than 2 MB", code="download")
    return payload


def _write_verified(dest: Path, payload: bytes) -> None:
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CALIBRATION_SHA256:
        raise ChecksumError(
            f"calibration download sha256 {digest} is not the pinned calibration_datav3 {CALIBRATION_SHA256}",
            code="checksum",
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".partial")
    temporary.write_bytes(payload)
    os.replace(temporary, dest)


def ensure_calibration() -> Path:
    """Return the pinned corpus, downloading it when the tree does not already have it."""
    bundled = data_dir() / "imatrix.txt"
    if _pinned(bundled):
        return bundled
    dest = _cache_path()
    if _pinned(dest):
        return dest
    _write_verified(dest, _fetch(CALIBRATION_RAW_URL))
    return dest


def assert_calibration(plan: Plan) -> int:
    """Return the pinned token count, or raise when the file or the plan is short."""
    ensure_calibration()
    need = tokens_required(plan.imatrix_ctx, plan.imatrix_chunks)
    if CALIBRATION_TOKENS < need:
        raise PlanError(
            f"calibration file has {CALIBRATION_TOKENS} tokens; need {need} "
            f"(at least 2*ctx={2 * plan.imatrix_ctx} and ctx*chunks={plan.imatrix_ctx * plan.imatrix_chunks})",
            code="plan_shape",
        )
    return CALIBRATION_TOKENS
