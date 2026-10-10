"""Delete weights that a verified upload no longer needs.

A disk check lists leftovers and removes them before it looks at free space
again. It does not sit in a waiting state while those files are still here.
Removing a Hugging Face snapshot directory only removes symlinks; the blobs
are removed separately.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from huggingface_hub import scan_cache_dir
from huggingface_hub.constants import HF_HUB_CACHE

from voxprint_forge.diffusion.guard import _pid_alive
from voxprint_forge.diffusion.planload import CanvasPlan
from voxprint_forge.diffusion.retain import consumers_for
from voxprint_forge.layout import read_marker
from voxprint_forge.scrub import log
from voxprint_forge.statusfile import read_status

_STAGING_NAMES = frozenset({"upload-staging", "kaggle-staging"})
_INTERMEDIATE_NAMES = frozenset({"f16.gguf", "bf16.gguf"})
_MODEL_DIRS = ("source", "ablated", "offload")


def hub_cache() -> Path | None:
    """The Hugging Face hub cache, when that directory exists."""
    path = Path(HF_HUB_CACHE)
    if path.is_dir():
        return path
    return None


def path_bytes(path: Path) -> int:
    """Bytes a deletion will free. Symlinks count as the link, not the target."""
    if not path.exists() and not path.is_symlink():
        return 0
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_symlink() or item.is_file():
            try:
                total += item.lstat().st_size
            except OSError:
                continue
    return total


def delete_logged(path: Path) -> int:
    """Remove ``path`` and log the freed size. Missing paths free nothing."""
    if not path.exists() and not path.is_symlink():
        return 0
    size = path_bytes(path)
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)
    log(f"reclaim {path} {size} bytes")
    return size


def reclaim_model_work(model_dir: Path, *, drop_staging: bool) -> int:
    """Delete one LLM model's sources and intermediates. Keep the quants."""
    freed = 0
    for name in (*_MODEL_DIRS, *_INTERMEDIATE_NAMES):
        freed += delete_logged(model_dir / name)
    for path in sorted(model_dir.glob("*.safetensors")):
        freed += delete_logged(path)
    if drop_staging:
        freed += delete_logged(model_dir / "upload-staging")
    return freed


def free_hf_revisions(cache_dir: Path, revisions: tuple[str, ...]) -> int:
    """Drop cache revisions through ``scan_cache_dir().delete_revisions().execute()``."""
    info = scan_cache_dir(cache_dir)
    chosen = set(revisions)
    seen: set[Path] = set()
    for repo in info.repos:
        for revision in repo.revisions:
            if revision.commit_hash not in chosen:
                continue
            for item in revision.files:
                if item.blob_path in seen:
                    continue
                seen.add(item.blob_path)
                log(f"reclaim {item.blob_path} {item.size_on_disk} bytes")
    strategy = info.delete_revisions(*revisions)
    strategy.execute()
    return strategy.expected_freed_size


def free_orphaned_blobs(cache_dir: Path) -> int:
    """Remove blob files no snapshot symlink still references."""
    if not cache_dir.is_dir():
        return 0
    referenced = _snapshot_targets(cache_dir)
    freed = 0
    for blob in _blob_files(cache_dir):
        if blob.name.endswith(".incomplete"):
            freed += delete_logged(blob)
            continue
        if blob.resolve() in referenced:
            continue
        freed += delete_logged(blob)
    return freed


def reclaim_for_disk(
    forge_out: Path,
    *,
    cache_dir: Path | None = None,
    plan: CanvasPlan | None = None,
) -> int:
    """Delete leftovers, log each one, and return the bytes removed."""
    live = _live_ids(forge_out)
    freed = 0
    for path in _without_children(_deletable(forge_out, live=live, plan=plan)):
        freed += delete_logged(path)
    if cache_dir is not None:
        freed += free_orphaned_blobs(cache_dir)
    return freed


def _snapshot_targets(cache_dir: Path) -> set[Path]:
    found: set[Path] = set()
    for path in cache_dir.glob("*/snapshots/**/*"):
        if not path.is_symlink():
            continue
        found.add(path.resolve())
        raw = os.readlink(path)
        hop = Path(raw)
        if not hop.is_absolute():
            hop = (path.parent / hop).resolve()
        found.add(hop)
    return found


def _blob_files(cache_dir: Path) -> list[Path]:
    found: list[Path] = []
    for pattern in ("*/blobs/*", "blobs/*/*"):
        for path in cache_dir.glob(pattern):
            if path.is_file() or path.is_symlink():
                found.append(path)
    return found


def _live_ids(forge_out: Path) -> set[str]:
    payload = read_status(forge_out)
    items = payload.get("items")
    live: set[str] = set()
    if not isinstance(items, dict):
        return live
    for item_id, raw in items.items():
        if not isinstance(raw, dict) or str(raw.get("state", "")) != "RUNNING":
            continue
        pid = int(raw.get("pid") or 0)
        if pid and _pid_alive(pid):
            live.add(str(item_id))
    return live


def _deletable(forge_out: Path, *, live: set[str], plan: CanvasPlan | None) -> list[Path]:
    found: list[Path] = []
    found.extend(_verified_outputs(forge_out, live))
    found.extend(_finished_staging(forge_out, live))
    found.extend(_stale_partials(forge_out, live))
    found.extend(_released_sources(forge_out, plan))
    return found


def _verified_outputs(forge_out: Path, live: set[str]) -> list[Path]:
    found: list[Path] = []
    stages = forge_out / "stages"
    if not stages.is_dir():
        return found
    for marker in stages.glob("*/*.json"):
        if not _marker_verified(marker):
            continue
        item = marker.parent.name
        if item in live:
            continue
        name = marker.name
        if name == "upload.json":
            found.extend(_model_leftovers(forge_out / "work" / item))
            continue
        if not name.endswith("-upload.json"):
            continue
        part = name[: -len("-upload.json")]
        if part in live:
            continue
        found.append(forge_out / "work" / item / "upstream" / part)
    return found


def _model_leftovers(model_dir: Path) -> list[Path]:
    found = [model_dir / name for name in (*_MODEL_DIRS, *_INTERMEDIATE_NAMES, "upload-staging")]
    found.extend(model_dir.glob("*.safetensors"))
    return found


def _finished_staging(forge_out: Path, live: set[str]) -> list[Path]:
    work = forge_out / "work"
    if not work.is_dir():
        return []
    found: list[Path] = []
    for path in work.rglob("*"):
        if path.is_dir() and path.name in _STAGING_NAMES and not (_ids_for(path, forge_out) & live):
            found.append(path)
    return found


def _stale_partials(forge_out: Path, live: set[str]) -> list[Path]:
    work = forge_out / "work"
    if not work.is_dir():
        return []
    found: list[Path] = []
    for path in work.rglob("*"):
        if not path.is_file() or (_ids_for(path, forge_out) & live):
            continue
        if path.name.endswith(".partial") or path.name.endswith(".incomplete"):
            found.append(path)
            continue
        if Path(str(path) + ".sha256").is_file():
            continue
        rel = path.relative_to(work).parts
        if path.suffix == ".gguf" and "quants" in rel:
            found.append(path)
            continue
        if path.name in _INTERMEDIATE_NAMES and _intermediate_done(forge_out, rel[0]):
            found.append(path)
    return found


def _intermediate_done(forge_out: Path, model_id: str) -> bool:
    stages = forge_out / "stages" / model_id
    if (stages / "upload.json").is_file() and _marker_verified(stages / "upload.json"):
        return True
    return (stages / "quantize.json").is_file() and (stages / "imatrix.json").is_file()


def _released_sources(forge_out: Path, plan: CanvasPlan | None) -> list[Path]:
    found: list[Path] = []
    if plan is not None:
        for part in plan.parts:
            for source in part.sources:
                names = consumers_for(plan, source.id)
                if _consumers_done(forge_out, plan.id, source.id, names):
                    found.append(forge_out / "work" / plan.id / "sources" / source.id)
    work = forge_out / "work"
    if not work.is_dir():
        return found
    for folder in work.glob("*/source-releases/*"):
        listed = folder / "_consumers"
        if not listed.is_file():
            continue
        names = tuple(line.strip() for line in listed.read_text(encoding="utf-8").splitlines() if line.strip())
        if not _every_marker(folder, names):
            continue
        job = folder.parent.parent.name
        found.append(work / job / "sources" / folder.name)
    return found


def _consumers_done(forge_out: Path, job_id: str, source_id: str, names: tuple[str, ...]) -> bool:
    folder = forge_out / "work" / job_id / "source-releases" / source_id
    return _every_marker(folder, names)


def _every_marker(folder: Path, names: tuple[str, ...]) -> bool:
    return bool(names) and all((folder / name).is_file() for name in names)


def _marker_verified(path: Path) -> bool:
    payload = read_marker(path)
    if payload is None:
        return False
    if payload.get("state") == "UPLOADED":
        return True
    uploads = payload.get("uploads")
    if isinstance(uploads, dict) and any(value == "uploaded" for value in uploads.values()):
        return True
    files = payload.get("files")
    if not isinstance(files, list):
        return False
    for item in files:
        if not isinstance(item, dict):
            continue
        if item.get("target") in {"hf", "kaggle", "drive"}:
            return True
        sources = item.get("sources")
        if not isinstance(sources, dict):
            continue
        for key in ("hf", "kaggle", "drive"):
            value = str(sources.get(key, "")).strip()
            if value and value != "placeholder":
                return True
    return False


def _ids_for(path: Path, forge_out: Path) -> set[str]:
    try:
        rel = path.relative_to(forge_out / "work")
    except ValueError:
        return set()
    found: set[str] = set()
    if rel.parts:
        found.add(rel.parts[0])
    if len(rel.parts) >= 3 and rel.parts[1] == "upstream":
        found.add(rel.parts[2])
    return found


def _without_children(paths: list[Path]) -> list[Path]:
    ordered = sorted({path for path in paths if path.exists() or path.is_symlink()}, key=lambda item: len(item.parts))
    kept: list[Path] = []
    for path in ordered:
        if any(path == parent or parent in path.parents for parent in kept):
            continue
        kept.append(path)
    return kept
