"""Private Hugging Face and Kaggle upload for one canvas part.

Kaggle variation handles stay on the `gguf` framework because that is the
handle `HubRemote.kaggle_upload_variation` returns. The bytes are safetensors.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from voxprint_forge.diffusion.registry import Row, render_models_md
from voxprint_forge.diffusion.weights import link_or_copy
from voxprint_forge.drive import upload_drive
from voxprint_forge.errors import ForgeError, StageError
from voxprint_forge.hashing import sha256_file, write_sha256
from voxprint_forge.upload import (
    DELAYS,
    Artifact,
    HubRemote,
    assert_private_metadata,
    hf_file_url,
    kaggle_user,
    kaggle_variation_url,
    model_metadata,
    upload_verified,
    variation_metadata,
)


@dataclass(frozen=True)
class LocalFile:
    """One local file ready to publish, with its sha256 and tier."""
    path: Path
    name: str
    sha256: str
    size: int
    tier: str
    vram_gb: int | None
    repo: str
    revision: str


def _iter_files(output: Path) -> list[Path]:
    return sorted(path for path in output.rglob("*") if path.is_file() and not path.name.endswith(".sha256"))


def write_sidecars(output: Path) -> None:
    """Write a sha256 sidecar next to each published file."""
    for path in _iter_files(output):
        write_sha256(path)


def _tier_of(
    name: str,
    tiers: dict[str, tuple[str, int | None, str, str]],
    repo: str,
    revision: str,
) -> tuple[str, int | None, str, str]:
    return tiers.get(name, ("shared", None, repo, revision))


def collect_files(
    output: Path,
    tiers: dict[str, tuple[str, int | None, str, str]],
    repo: str,
    revision: str,
) -> list[LocalFile]:
    """List published files with hashes and empty source URLs."""
    found: list[LocalFile] = []
    for path in _iter_files(output):
        name = path.relative_to(output).as_posix()
        tier, vram, file_repo, file_revision = _tier_of(name, tiers, repo, revision)
        found.append(
            LocalFile(
                path=path,
                name=name,
                sha256=sha256_file(path),
                size=path.stat().st_size,
                tier=tier,
                vram_gb=vram,
                repo=file_repo,
                revision=file_revision,
            )
        )
    return found


def write_models_md(output: Path, files: list[LocalFile]) -> None:
    """Write the MODELS snippet for these local files."""
    rows = [
        Row(
            repo=item.repo,
            revision=item.revision,
            file=item.name,
            size=item.size,
            sha256=item.sha256,
            licence="Apache-2.0",
            tier=item.tier,
            vram_gb=item.vram_gb,
        )
        for item in files
        if item.name != "MODELS.md"
    ]
    (output / "MODELS.md").write_text(render_models_md(rows), encoding="utf-8")


def _shared_names(output: Path) -> list[str]:
    names = [
        "README.md",
        "NOTICE",
        "MODELS.md",
        "checks/fp8-vs-bf16.json",
        "checks/i2i-fp8-vs-bf16.json",
        "checks/grid.avif",
        "checks/grid-inpaint.avif",
        "upstream-sha256.json",
    ]
    checks = output / "checks"
    if checks.is_dir():
        for path in sorted(checks.rglob("*")):
            if path.is_file() and path.relative_to(checks).parts[0].startswith("review"):
                names.append(str(path.relative_to(output)))
    return [name for name in names if (output / name).is_file()]


def _kaggle_name(name: str) -> str:
    return name.replace("/", "__")


def _stage_variation(output: Path, folder: Path, names: list[str]) -> None:
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for name in names:
        src = output / name
        if not src.is_file():
            continue
        flat = _kaggle_name(name)
        link_or_copy(src, folder / flat)
        sidecar = Path(str(src) + ".sha256")
        if sidecar.is_file():
            link_or_copy(sidecar, folder / f"{flat}.sha256")


def _all_match(remote: HubRemote, handle: str, files: list[LocalFile]) -> bool:
    for item in files:
        remote_sha, remote_size = remote.kaggle_stat(handle, _kaggle_name(item.name))
        if remote_sha != item.sha256 or remote_size != item.size:
            return False
    return True


def _upload_variation(
    remote: HubRemote,
    owner: str,
    slug: str,
    variation: str,
    folder: Path,
    files: list[LocalFile],
    *,
    attempts: int,
    sleep: Callable[[float], None],
) -> str:
    last = f"{variation} remote sha256/size did not match"
    handle = ""
    for attempt in range(attempts):
        handle = remote.kaggle_upload_variation(owner, slug, variation, folder)
        if _all_match(remote, handle, files):
            return handle
        if attempt + 1 < attempts:
            sleep(DELAYS[min(attempt, len(DELAYS) - 1)])
    raise StageError(last, code="upload_mismatch")


def _publish_hub(
    *,
    output: Path,
    hf_repo: str,
    kaggle_slug: str,
    title: str,
    description: str,
    tiers: dict[str, tuple[str, int | None, str, str]],
    variations: dict[str, list[str]],
    repo: str,
    revision: str,
    remote: HubRemote | None = None,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = 4,
    staging: Path | None = None,
    on_verified: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """Upload every output file. A variation is done only when sha256 and size match."""
    client = remote if remote is not None else HubRemote(
        os.environ.get("HF_TOKEN"),
        os.environ.get("KAGGLE_API_TOKEN"),
    )
    write_models_md(output, collect_files(output, tiers, repo, revision))
    write_sidecars(output)
    files = collect_files(output, tiers, repo, revision)
    client.hf_private_repo(hf_repo)
    for item in files:
        upload_verified(
            lambda item=item: client.hf_upload(hf_repo, item.path, item.name),
            lambda item=item: client.hf_stat(hf_repo, item.name),
            local_sha=item.sha256,
            local_size=item.size,
            attempts=attempts,
            sleep=sleep,
        )
    for sidecar in sorted(output.rglob("*.sha256")):
        digest = sha256_file(sidecar)
        size = sidecar.stat().st_size
        name = sidecar.relative_to(output).as_posix()
        upload_verified(
            lambda sidecar=sidecar, name=name: client.hf_upload(hf_repo, sidecar, name),
            lambda name=name: client.hf_stat(hf_repo, name),
            local_sha=digest,
            local_size=size,
            attempts=attempts,
            sleep=sleep,
        )
    if on_verified is not None:
        on_verified("hf")

    owner = kaggle_user()
    stage_root = staging or (output.parent / "kaggle-staging" / kaggle_slug)
    model_dir = stage_root / "model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True)
    meta = model_metadata(owner, kaggle_slug, title, description)
    meta["subtitle"] = "Voxprint AI Canvas private weights"
    assert_private_metadata(meta)
    (model_dir / "model-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    client.kaggle_private_model(owner, kaggle_slug, model_dir)

    by_name = {item.name: item for item in files}
    records: list[dict[str, object]] = []
    shared = _shared_names(output)
    for slug, names in variations.items():
        folder = stage_root / slug
        chosen = [name for name in [*names, *shared] if name in by_name]
        _stage_variation(output, folder, chosen)
        overview = description[:4000]
        instance = variation_metadata(owner, kaggle_slug, slug, overview)
        (folder / "model-instance-metadata.json").write_text(
            json.dumps(instance, indent=2) + "\n",
            encoding="utf-8",
        )
        handle = _upload_variation(
            client,
            owner,
            kaggle_slug,
            slug,
            folder,
            [by_name[name] for name in chosen],
            attempts=attempts,
            sleep=sleep,
        )
        page = kaggle_variation_url(owner, kaggle_slug, slug)
        for name in chosen:
            item = by_name[name]
            records.append(
                {
                    "name": item.name,
                    "sha256": item.sha256,
                    "size": item.size,
                    "tier": item.tier,
                    "vram_gb": item.vram_gb,
                    "sources": {
                        "hf": hf_file_url(hf_repo, item.name),
                        "kaggle": page,
                        "kaggle_handle": handle,
                        "torrent": "placeholder",
                    },
                }
            )
    recorded = {str(record["name"]) for record in records}
    for item in files:
        if item.name in recorded:
            continue
        records.append(
            {
                "name": item.name,
                "sha256": item.sha256,
                "size": item.size,
                "tier": item.tier,
                "vram_gb": item.vram_gb,
                "sources": {
                    "hf": hf_file_url(hf_repo, item.name),
                    "kaggle": "",
                    "torrent": "placeholder",
                },
            }
        )
    if stage_root.exists():
        shutil.rmtree(stage_root)
    if on_verified is not None:
        on_verified("kaggle")
    return records


def _drive_artifacts(output: Path) -> list[Artifact]:
    found: list[Artifact] = []
    if not output.is_dir():
        return found
    for path in sorted(output.rglob("*")):
        if not path.is_file():
            continue
        if "kaggle-staging" in path.parts:
            continue
        found.append(
            Artifact(
                path,
                path.relative_to(output).as_posix(),
                sha256_file(path),
                path.stat().st_size,
                "",
                "",
                "file",
            )
        )
    return found


def publish_output(
    *,
    output: Path,
    hf_repo: str,
    kaggle_slug: str,
    title: str,
    description: str,
    tiers: dict[str, tuple[str, int | None, str, str]],
    variations: dict[str, list[str]],
    repo: str,
    revision: str,
    remote: HubRemote | None = None,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = 4,
    staging: Path | None = None,
    on_verified: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    """Upload to Hugging Face and Kaggle. Drive is the fallback when both fail."""
    try:
        return _publish_hub(
            output=output,
            hf_repo=hf_repo,
            kaggle_slug=kaggle_slug,
            title=title,
            description=description,
            tiers=tiers,
            variations=variations,
            repo=repo,
            revision=revision,
            remote=remote,
            sleep=sleep,
            attempts=attempts,
            staging=staging,
            on_verified=on_verified,
        )
    except ForgeError as exc:
        artifacts = _drive_artifacts(output)
        if not artifacts:
            raise
        day = datetime.now(timezone.utc).date().isoformat()
        try:
            records = upload_drive(artifacts, model_id=kaggle_slug, date=day)
        except ForgeError as drive_exc:
            raise StageError(
                f"HF/Kaggle failed ({exc}); Drive failed ({drive_exc})",
                code="upload",
            ) from drive_exc
        if on_verified is not None:
            on_verified("drive")
        return records
