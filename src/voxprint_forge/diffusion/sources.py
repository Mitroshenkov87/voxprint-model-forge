"""Pinned downloads. The licence tag is read at the pinned revision, not at main."""

from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from voxprint_forge.diffusion.planload import Source
from voxprint_forge.errors import DownloadError, LicenceError
from voxprint_forge.hashing import sha256_file
from voxprint_forge.licence import decide_licence
from voxprint_forge.scrub import scrub


def _licence_of(info: object) -> str | None:
    card = getattr(info, "card_data", None)
    if card is None:
        return None
    value = card.get("license") if isinstance(card, dict) else getattr(card, "license", None)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    return str(value)


def assert_apache(repo: str, revision: str, token: str | None) -> str:
    api = HfApi(token=token)
    try:
        info = api.model_info(repo, revision=revision)
    except Exception as exc:
        raise DownloadError(scrub(f"model info {repo}@{revision}: {exc}"), code="download") from None
    tag = decide_licence(_licence_of(info), repo=repo)
    if tag != "apache-2.0":
        raise LicenceError(
            f"{repo}@{revision} licence tag {tag!r} is not Apache-2.0",
            code="non_permissive",
        )
    return tag


def tree_hashes(root: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        found[path.relative_to(root).as_posix()] = sha256_file(path)
    return found


def same_trees(left: Path, right: Path) -> bool:
    if not left.is_dir() or not right.is_dir():
        return False
    return tree_hashes(left) == tree_hashes(right)


def download_source(source: Source, dest: Path, token: str | None) -> dict[str, str]:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        if source.files:
            for name in source.files:
                hf_hub_download(
                    repo_id=source.repo,
                    filename=name,
                    revision=source.revision,
                    local_dir=str(dest),
                    token=token,
                )
        else:
            snapshot_download(
                repo_id=source.repo,
                revision=source.revision,
                local_dir=str(dest),
                token=token,
                allow_patterns=list(source.allow) or None,
            )
    except Exception as exc:
        raise DownloadError(scrub(f"download {source.repo}@{source.revision}: {exc}"), code="download") from None
    return tree_hashes(dest)


def fetch_sources(sources: tuple[Source, ...], dest: Path, token: str | None) -> dict[str, object]:
    record: dict[str, object] = {}
    for source in sources:
        tag = assert_apache(source.repo, source.revision, token)
        hashes = download_source(source, dest / source.id, token)
        record[source.id] = {
            "repo": source.repo,
            "revision": source.revision,
            "licence": tag,
            "files": hashes,
        }
    path = dest / "upstream-sha256.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
