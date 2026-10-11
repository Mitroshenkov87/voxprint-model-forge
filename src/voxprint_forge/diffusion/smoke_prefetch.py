"""Resumable weight prefetch for the laptop smoke.

Downloads finish before the compute clock starts. A broken connection is
retried with backoff. A failure here is a setup error, not a step failure.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from pathlib import Path

from voxprint_forge.diffusion.nunchaku_edit import (
    ARMS,
    QWEN_BASE_REPO,
    QWEN_BASE_REVISION,
    ZIMAGE_BASE_REPO,
    ZIMAGE_BASE_REVISION,
)

DOWNLOAD_ATTEMPTS = 8
SIZE_TOLERANCE = 0.02
_RETRYABLE_NAMES = frozenset(
    {
        "ChunkedEncodingError",
        "IncompleteRead",
        "ConnectionError",
        "Timeout",
        "ConnectTimeout",
        "ReadTimeout",
        "ProtocolError",
        "RemoteDisconnected",
        "SSLError",
        "ProxyError",
        "ConnectionResetError",
    }
)
_RETRYABLE_TEXT = ("incompleteread", "connection broken", "timed out", "connection reset")


class DownloadError(Exception):
    """A weight download failed. The smoke process exits 2."""


def is_retryable_download(exc: BaseException) -> bool:
    """True for a dropped connection or a retryable HTTP status."""
    if type(exc).__name__ in _RETRYABLE_NAMES:
        return True
    cause = exc.__cause__
    if cause is not None and cause is not exc and is_retryable_download(cause):
        return True
    text = str(exc).lower()
    if any(needle in text for needle in _RETRYABLE_TEXT):
        return True
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code in {408, 429, 500, 502, 503, 504}


def call_with_retry(
    action: Callable[[], object],
    what: str,
    *,
    attempts: int = DOWNLOAD_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> object:
    """Run ``action`` until it succeeds or the connection stops being retryable."""
    delay = 2.0
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except DownloadError as exc:
            last = exc
            if attempt == attempts:
                raise
            sleep(delay)
            delay = min(delay * 2, 60.0)
        except Exception as exc:
            last = exc
            if not is_retryable_download(exc) or attempt == attempts:
                raise DownloadError(f"setup: download failed: {what}: {type(exc).__name__}: {exc}") from exc
            sleep(delay)
            delay = min(delay * 2, 60.0)
    raise DownloadError(f"setup: download failed: {what}: {last}")


def verify_download(path: Path, *, sha256: str, size_bytes: int) -> None:
    """Check size, then sha256. A mismatch is a setup error and deletes the file."""
    if not path.is_file():
        raise DownloadError(f"setup: download failed: missing {path.name}")
    size = path.stat().st_size
    if size_bytes > 0 and abs(size - size_bytes) / size_bytes > SIZE_TOLERANCE:
        path.unlink(missing_ok=True)
        raise DownloadError(
            f"setup: download failed: {path.name} size {size} is outside {SIZE_TOLERANCE:.0%} of {size_bytes}"
        )
    digest = sha256_file(path)
    if digest != sha256:
        path.unlink(missing_ok=True)
        raise DownloadError(f"setup: download failed: {path.name} sha256 {digest} != {sha256}")


def sha256_file(path: Path) -> str:
    """Return the sha256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def weights_ready(weight: Path, base: Path) -> None:
    """Raise before any GPU move when the transformer file or the base is absent."""
    if not weight.is_file() or weight.stat().st_size < 1_000_000:
        raise DownloadError(f"setup: download failed: transformer weights are not on disk: {weight}")
    marker = base / "model_index.json"
    config = base / "transformer" / "config.json"
    if not marker.is_file() or not config.is_file():
        raise DownloadError(f"setup: download failed: base snapshot is incomplete: {base}")


def prefetch_weights(cache: Path, steps: tuple[str, ...], *, attempts: int = DOWNLOAD_ATTEMPTS) -> None:
    """Download every weight the selected steps need. Existing sha256 matches are kept."""
    need_qwen_int4 = "quantfunc" in steps
    need_qwen_base = "quantfunc" in steps or "calib-quant-timing" in steps
    need_transformer = "calib-quant-timing" in steps
    need_zimage = "zimage" in steps
    if need_qwen_int4:
        _prefetch_arm(cache, "qwen-r128", attempts=attempts)
    if need_qwen_base:
        _prefetch_snapshot(
            cache,
            "qwen-edit-2511",
            QWEN_BASE_REPO,
            QWEN_BASE_REVISION,
            keep_transformer=need_transformer,
            attempts=attempts,
        )
    if need_zimage:
        _prefetch_arm(cache, "zimage-r128", attempts=attempts)
        _prefetch_snapshot(
            cache,
            "z-image-turbo",
            ZIMAGE_BASE_REPO,
            ZIMAGE_BASE_REVISION,
            keep_transformer=False,
            attempts=attempts,
        )


def _prefetch_arm(cache: Path, arm_id: str, *, attempts: int) -> None:
    spec = ARMS[arm_id]
    dest = cache / "weights" / arm_id
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / spec.filename
    if path.is_file():
        try:
            verify_download(path, sha256=spec.sha256, size_bytes=spec.size_bytes)
            return
        except DownloadError:
            path.unlink(missing_ok=True)

    def once() -> object:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=spec.repo,
            filename=spec.filename,
            revision=spec.revision,
            local_dir=str(dest),
            token=False,
        )
        verify_download(path, sha256=spec.sha256, size_bytes=spec.size_bytes)
        return path

    call_with_retry(once, spec.filename, attempts=attempts)


def _prefetch_snapshot(
    cache: Path,
    name: str,
    repo: str,
    revision: str,
    *,
    keep_transformer: bool,
    attempts: int,
) -> None:
    dest = cache / "weights" / name
    dest.mkdir(parents=True, exist_ok=True)
    ignore = None if keep_transformer else ["transformer/*.safetensors", "transformer/*.bin", "transformer/*.pt"]

    def once() -> object:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repo,
            revision=revision,
            ignore_patterns=ignore,
            local_dir=str(dest),
            token=False,
            max_workers=4,
        )
        _verify_snapshot(dest, keep_transformer=keep_transformer)
        return dest

    call_with_retry(once, repo, attempts=attempts)


def _verify_snapshot(dest: Path, *, keep_transformer: bool) -> None:
    marker = dest / "model_index.json"
    config = dest / "transformer" / "config.json"
    if not marker.is_file() or not config.is_file():
        raise DownloadError(
            f"setup: download failed: {dest.name} is missing model_index.json or transformer/config.json"
        )
    if not keep_transformer:
        return
    shards = list((dest / "transformer").glob("*.safetensors"))
    if not shards or max(path.stat().st_size for path in shards) < 1_000_000:
        raise DownloadError(f"setup: download failed: {dest.name} transformer weights are missing")
