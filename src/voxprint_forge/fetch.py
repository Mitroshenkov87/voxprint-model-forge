"""Download pinned files and refuse a checksum mismatch.

A pinned file with a sha256 is fetched by ``voxprint_common.download``: Hugging
Face, then ModelScope, with the digest required and the disk check before any
byte. Sidecars that have no pin still use ``huggingface_hub``.
"""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path
from typing import Protocol

from huggingface_hub import hf_hub_download
from voxprint_common.download import DiskSpaceError, download, huggingface_url, modelscope_url
from voxprint_common.download import DownloadError as CommonDownloadError
from voxprint_common.hashing import sha256_file

from voxprint_forge.errors import ChecksumError, DownloadError
from voxprint_forge.plan import PinnedFile
from voxprint_forge.scrub import scrub


class Downloader(Protocol):
    """Something that can fetch one pinned file into ``dest``."""
    def fetch(self, repo: str, revision: str, filename: str, dest: Path, *, optional: bool = False) -> None:
        """Write `filename` from `repo`@`revision` to `dest`."""


class HfDownloader:
    """huggingface_hub download. The token stays in the process environment."""

    def __init__(self, token: str | None) -> None:
        self._token = token

    def _open(self, request: urllib.request.Request, timeout: float) -> object:
        if self._token:
            request.add_header("Authorization", f"Bearer {self._token}")
        return urllib.request.urlopen(request, timeout=timeout)

    def download_verified(self, pinned: PinnedFile, dest: Path) -> None:
        """Download one pinned file from Hugging Face, then ModelScope.

        ``voxprint_common.download`` checks free space before the first byte
        and refuses a digest mismatch. The Hub token, when set, is sent only
        as a bearer header.
        """
        revision = pinned.revision or "main"
        urls = [
            huggingface_url(pinned.repo, pinned.path, revision),
            modelscope_url(pinned.repo, pinned.path, revision),
        ]
        try:
            download(
                dest,
                urls,
                sha256=pinned.sha256,
                size=pinned.size or None,
                opener=self._open,
            )
        except DiskSpaceError as exc:
            raise DownloadError(str(exc), code="download") from exc
        except CommonDownloadError as exc:
            text = str(exc)
            if "does not match" in text:
                dest.unlink(missing_ok=True)
                raise ChecksumError(text, code="checksum") from exc
            raise DownloadError(scrub(text), code="download") from exc

    def fetch(self, repo: str, revision: str, filename: str, dest: Path, *, optional: bool = False) -> None:
        """Download one Hub file that has no sha256 pin."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            downloaded = hf_hub_download(
                repo_id=repo,
                filename=filename,
                revision=revision or None,
                token=self._token,
                local_dir=str(dest.parent),
            )
        except Exception as exc:
            message = scrub(str(exc))
            if optional:
                raise DownloadError(message, code="sidecar_missing") from exc
            raise DownloadError(message, code="download") from exc
        produced = Path(downloaded)
        if produced.resolve() != dest.resolve():
            shutil.copy2(produced, dest)


def ensure_pinned(downloader: Downloader, pinned: PinnedFile, dest: Path) -> str:
    """Return the verified sha256. Sidecars with an empty pin keep the observed digest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file():
        digest = sha256_file(dest)
        if not pinned.sha256 or digest == pinned.sha256:
            if pinned.size and dest.stat().st_size != pinned.size:
                dest.unlink()
            else:
                return digest
        else:
            dest.unlink()
    if pinned.sha256 and isinstance(downloader, HfDownloader):
        downloader.download_verified(pinned, dest)
        digest = sha256_file(dest)
        if digest != pinned.sha256.lower():
            dest.unlink(missing_ok=True)
            raise ChecksumError(
                f"{pinned.path} sha256 {digest} != pinned {pinned.sha256}",
                code="checksum",
            )
        if pinned.size and dest.stat().st_size != pinned.size:
            dest.unlink(missing_ok=True)
            raise ChecksumError(
                f"{pinned.path} size {dest.stat().st_size} != pinned {pinned.size}",
                code="checksum",
            )
        return digest
    try:
        downloader.fetch(pinned.repo, pinned.revision, pinned.path, dest, optional=pinned.role == "sidecar")
    except DownloadError:
        if pinned.role == "sidecar":
            return ""
        raise
    if not dest.is_file():
        if pinned.role == "sidecar":
            return ""
        raise DownloadError(f"missing {pinned.path} after download", code="download")
    digest = sha256_file(dest)
    if pinned.sha256 and digest != pinned.sha256:
        dest.unlink()
        raise ChecksumError(
            f"{pinned.path} sha256 {digest} != pinned {pinned.sha256}",
            code="checksum",
        )
    if pinned.size and dest.stat().st_size != pinned.size:
        dest.unlink()
        raise ChecksumError(
            f"{pinned.path} size {dest.stat().st_size if dest.exists() else 0} != pinned {pinned.size}",
            code="checksum",
        )
    return digest
