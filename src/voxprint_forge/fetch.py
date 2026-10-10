"""Download pinned files and refuse a checksum mismatch."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

from huggingface_hub import hf_hub_download

from voxprint_forge.errors import ChecksumError, DownloadError
from voxprint_forge.hashing import sha256_file
from voxprint_forge.plan import PinnedFile
from voxprint_forge.scrub import scrub


class Downloader(Protocol):
    def fetch(self, repo: str, revision: str, filename: str, dest: Path, *, optional: bool = False) -> None:
        """Write `filename` from `repo`@`revision` to `dest`."""


class HfDownloader:
    """huggingface_hub download. The token stays in the process environment."""

    def __init__(self, token: str | None) -> None:
        self._token = token

    def fetch(self, repo: str, revision: str, filename: str, dest: Path, *, optional: bool = False) -> None:
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
