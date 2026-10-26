"""Folder views over the VOXPrint MDL blob store.

MDL keeps each file at ``<models>/blobs/<sha256>``. Model Forge reads
``<models>/model-forge/<id>/<path>``. :func:`ensure_view` materialises that
folder as hard links to the blobs, or as copies when the link cannot be made
on another volume. A marker lists every path and sha256. The view is rebuilt
when the marker or the blobs differ. Nothing under ``blobs/`` is deleted.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from voxprint_common.paths import models_dir

MARKER = ".voxprint-view.json"
APP_DIR = "model-forge"
BLOBS_DIR = "blobs"


class ViewError(OSError):
    """A view file could not be linked or copied."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def manifest_path() -> Path:
    """Packaged schema 1 file. The install copy is chosen by :mod:`voxprint_forge.mdl_app`."""
    return Path(__file__).resolve().parent / "data" / "mdl.json"


def files_for(mdl_id: str, manifest: Path | None = None) -> dict[str, tuple[str, int]]:
    """``path -> (sha256, size)`` for ``mdl_id``, or an empty map when it is absent."""
    path = manifest or manifest_path()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return {}
    if not isinstance(document, dict):
        return {}
    for model in document.get("models") or []:
        if isinstance(model, dict) and model.get("id") == mdl_id:
            found: dict[str, tuple[str, int]] = {}
            for item in model.get("files") or []:
                if isinstance(item, dict) and item.get("path") and item.get("sha256"):
                    found[str(item["path"])] = (str(item["sha256"]), int(item.get("size") or 0))
            return found
    return {}


def ensure_view(mdl_id: str, *, root: Path | None = None, manifest: Path | None = None) -> Path | None:
    """Materialise ``<models>/model-forge/<id>/`` from blobs when they are present.

    Returns the folder when every listed file is already there, or after it is
    built. Returns None when the id is unknown or the blobs and the folder are
    both absent. Existing files that already match are left in place.
    """
    spec = files_for(mdl_id, manifest)
    if not spec:
        return None
    base = root or models_dir()
    folder = base / APP_DIR / mdl_id
    blobs = {path: base / BLOBS_DIR / digest for path, (digest, _size) in spec.items()}
    if not all(path.is_file() for path in blobs.values()):
        return folder if folder.is_dir() and _files_present(folder, spec) else None
    if _matches(folder, spec, base):
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    for rel, (digest, _size) in spec.items():
        target = _safe_target(folder, rel)
        _place(blobs[rel], target, base / BLOBS_DIR)
    _write_marker(folder, spec)
    return folder


def _files_present(folder: Path, spec: dict[str, tuple[str, int]]) -> bool:
    return all((folder / rel).is_file() for rel in spec)


def _matches(folder: Path, spec: dict[str, tuple[str, int]], root: Path) -> bool:
    try:
        marker = json.loads((folder / MARKER).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return False
    if not isinstance(marker, dict) or marker.get("files") != _marker_files(spec):
        return False
    for rel, (digest, size) in spec.items():
        entry = folder / rel
        blob = root / BLOBS_DIR / digest
        if not entry.is_file() or not blob.is_file():
            return False
        try:
            same = os.path.samefile(entry, blob)
        except OSError:
            return False
        if same:
            continue
        if entry.stat().st_size != blob.stat().st_size or (size and entry.stat().st_size != size):
            return False
    return True


def _marker_files(spec: dict[str, tuple[str, int]]) -> list[dict[str, str]]:
    return [{"path": rel, "sha256": digest} for rel, (digest, _size) in sorted(spec.items())]


def _write_marker(folder: Path, spec: dict[str, tuple[str, int]]) -> None:
    payload = {"files": _marker_files(spec)}
    (folder / MARKER).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def _safe_target(folder: Path, rel: str) -> Path:
    parts = Path(rel.replace("\\", "/")).parts
    if not parts or Path(rel).is_absolute() or any(part in ("", ".", "..") for part in parts):
        raise ViewError(rel)
    target = folder.joinpath(*parts)
    if folder.resolve() not in target.resolve().parents and target.resolve() != folder.resolve():
        raise ViewError(rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _place(blob: Path, target: Path, blobs_root: Path) -> None:
    if target.exists() or target.is_symlink():
        try:
            if target.is_file() and os.path.samefile(target, blob):
                return
        except OSError:
            pass
        _unlink_view_entry(target, blobs_root)
    try:
        os.link(blob, target)
    except OSError:
        shutil.copyfile(blob, target)


def _unlink_view_entry(target: Path, blobs_root: Path) -> None:
    """Remove one view entry. A path inside ``blobs/`` is left untouched."""
    blobs = blobs_root.resolve()
    parent = target.parent.resolve()
    if parent == blobs or blobs in parent.parents:
        return
    if target.is_dir() and not target.is_symlink():
        return
    target.unlink()
