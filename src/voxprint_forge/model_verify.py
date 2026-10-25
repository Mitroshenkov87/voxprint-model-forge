"""Check installed models and re-download damaged files.

An installed model is one the app manifest already has on disk, or a plug-in
whose ``voxprint-model/1`` file sits in the models folder. Each file is hashed
in chunks. Missing and corrupt files are fetched again from the manifest
source, pinned to the recorded sha256, and hashed once more.

LoRAs (``user-models``), projects, and other user data are never opened for
writing and are not part of the check.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Never

from voxprint_common.download import DownloadError, huggingface_url, modelscope_url
from voxprint_common.mdl.catalog import CatalogError, gdrive_url, kaggle_url
from voxprint_common.mdl.descriptor import Descriptor, DescriptorError, parse_descriptor_text
from voxprint_common.mdl.fetch import Cancelled, FetchError, FetchProgress, fetch_file
from voxprint_common.mdl.status import JobState, Status, Step
from voxprint_common.paths import models_dir

from voxprint_forge.mdl_progress import WindowResult, run_verify_window
from voxprint_forge.model_views import APP_DIR, BLOBS_DIR, ViewError, ensure_view

HASH_CHUNK = 1024 * 1024
MANIFEST_NAME = "voxprint-model.json"
USER_DATA_DIRS = frozenset({"user-models", "projects", "lora"})
WALK_SKIP = USER_DATA_DIRS | frozenset({BLOBS_DIR, APP_DIR})

FileStatus = Literal["ok", "missing", "corrupt"]
RowStatus = Literal["ok", "missing", "corrupt", "checking", "repairing"]
Phase = Literal["verify", "repair"]
MirrorKind = Literal["hf", "kaggle", "gdrive", "modelscope", "url"]
_MIRROR_KINDS = frozenset({"hf", "kaggle", "gdrive", "modelscope", "url"})

Progress = Callable[[list["ModelReport"], Phase], None]


class VerifyError(Exception):
    """The check could not read a manifest or was asked to write outside the model store."""


class VerifyCancelled(Exception):
    """The caller set the cancel event between hashed chunks."""


@dataclass(frozen=True)
class ModelFile:
    """One file named by an installed manifest."""

    rel: str
    size: int
    sha256: str
    urls: tuple[str, ...]
    location: Path
    blob: Path | None


@dataclass(frozen=True)
class InstalledModel:
    """One installed model and the files the check is allowed to touch."""

    id: str
    display_name: str
    source: str
    kind: Literal["app", "plugin"]
    files: tuple[ModelFile, ...]
    models_root: Path
    manifest: Path | None


@dataclass
class FileReport:
    """The check result for one file."""

    path: str
    status: FileStatus
    size: int
    hashed: int


@dataclass
class ModelReport:
    """The check result for one model, including bytes hashed so far."""

    id: str
    display_name: str
    status: RowStatus
    bytes_hashed: int
    bytes_total: int
    source: str
    files: list[FileReport] = field(default_factory=list)
    speed_bps: float = 0.0
    eta_s: float | None = None

    def to_dict(self) -> dict[str, object]:
        """A JSON-ready row."""
        return {
            "id": self.id,
            "display_name": self.display_name,
            "status": self.status,
            "bytes_hashed": self.bytes_hashed,
            "bytes_total": self.bytes_total,
            "source": self.source,
            "files": [
                {"path": item.path, "status": item.status, "size": item.size, "hashed": item.hashed}
                for item in self.files
            ],
        }


def hash_file(
    path: Path,
    *,
    chunk: int = HASH_CHUNK,
    cancel: threading.Event | None = None,
    on_chunk: Callable[[int], None] | None = None,
) -> str:
    """Return the SHA-256 of ``path``, reading ``chunk`` bytes at a time.

    ``on_chunk`` receives how many bytes have been hashed. When ``cancel`` is
    set, the next read raises :class:`VerifyCancelled` and the rest of the
    file stays unread.
    """
    if chunk <= 0:
        raise VerifyError("chunk must be positive")
    digest = hashlib.sha256()
    done = 0
    with path.open("rb") as handle:
        while True:
            if cancel is not None and cancel.is_set():
                raise VerifyCancelled()
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
            done += len(block)
            if on_chunk is not None:
                on_chunk(done)
    return digest.hexdigest()


def source_urls(descriptor: Descriptor, filename: str) -> tuple[str, ...]:
    """HTTPS URLs for ``filename``, in the manifest's source order.

    Plug-in models use the same order: the official ``url`` or repository
    source, then any later mirror.
    """
    found: list[str] = []
    count = len(descriptor.files)
    for mirror in descriptor.mirrors_for(filename):
        kind = _mirror_kind(mirror.kind)
        if kind is None:
            continue
        url = _mirror_url(kind, mirror.ref, mirror.revision, filename, count)
        if url:
            found.append(url)
    return tuple(found)


def discover(manifest: Path, *, root: Path | None = None) -> list[InstalledModel]:
    """Installed app models plus plug-in ``voxprint-model/1`` files under ``root``.

    An app model counts as installed when its view folder or any of its blobs
    exists. A model that was never downloaded is left for Manage models.
    Directories named ``user-models``, ``projects``, and ``lora`` are skipped.
    """
    base = root if root is not None else models_dir()
    app_models = _app_models(manifest, base)
    return app_models + _plugin_models(base)


def verify_models(
    models: Sequence[InstalledModel],
    *,
    cancel: threading.Event | None = None,
    chunk: int = HASH_CHUNK,
    on_progress: Progress | None = None,
    phase: Phase = "verify",
) -> list[ModelReport]:
    """Hash every file of ``models``. Each row is ``ok``, ``missing``, or ``corrupt``."""
    reports: list[ModelReport] = []
    for model in models:
        _raise_if_cancelled(cancel)
        report = ModelReport(
            id=model.id,
            display_name=model.display_name,
            status="checking",
            bytes_hashed=0,
            bytes_total=sum(item.size for item in model.files),
            source=model.source,
        )
        reports.append(report)
        started = time.monotonic()
        for item in model.files:
            _raise_if_cancelled(cancel)
            target = _present_path(model, item)
            if target is None:
                report.files.append(FileReport(item.rel, "missing", item.size, 0))
                _emit(on_progress, reports, phase)
                continue
            before = report.bytes_hashed

            def on_chunk(done: int, *, _before: int = before, _report: ModelReport = report) -> None:
                _report.bytes_hashed = _before + done
                _pace(_report, started)
                _emit(on_progress, reports, phase)

            try:
                digest = hash_file(target, chunk=chunk, cancel=cancel, on_chunk=on_chunk)
            except OSError:
                report.files.append(FileReport(item.rel, "corrupt", item.size, 0))
                _emit(on_progress, reports, phase)
                continue
            status: FileStatus = "ok" if digest == item.sha256 else "corrupt"
            read = target.stat().st_size if target.is_file() else 0
            report.bytes_hashed = before + read
            report.files.append(FileReport(item.rel, status, item.size, read))
            _pace(report, started)
            _emit(on_progress, reports, phase)
        report.status = _overall(report.files)
        report.speed_bps = 0.0
        report.eta_s = None
        _emit(on_progress, reports, phase)
    return reports


def repair_models(
    models: Sequence[InstalledModel],
    reports: Sequence[ModelReport],
    *,
    cancel: threading.Event | None = None,
    chunk: int = HASH_CHUNK,
    on_progress: Progress | None = None,
) -> list[ModelReport]:
    """Re-download missing or corrupt files, then hash those models again.

    A file that already matches its sha256 is left in place. A ``.partial``
    beside the destination is left for the downloader to resume.
    """
    if len(models) != len(reports):
        raise VerifyError("repair reports do not match the installed models")
    for model, report in zip(models, reports, strict=True):
        if report.status == "ok":
            continue
        _raise_if_cancelled(cancel)
        report.status = "repairing"
        _emit(on_progress, list(reports), "repair")
        _repair_model(model, report, cancel=cancel, chunk=chunk, on_progress=on_progress, reports=list(reports))
    return verify_models(models, cancel=cancel, chunk=chunk, on_progress=on_progress, phase="verify")


def download_file(
    destination: Path,
    urls: Sequence[str],
    *,
    sha256: str,
    size: int,
    cancel: threading.Event | None = None,
    on_bytes: Callable[[int, int], None] | None = None,
) -> None:
    """Download ``destination`` from ``urls`` and require ``sha256``.

    The downloader resumes a ``.partial`` file beside ``destination`` when the
    source supports HTTP Range. This function does not delete that partial,
    and it refuses a path inside user data.
    """
    _refuse_user_data(destination)
    if not urls:
        raise VerifyError(f"{destination.name} has no source URL")

    def progress(event: FetchProgress) -> None:
        if on_bytes is not None:
            on_bytes(event.received, event.total)

    try:
        fetch_file(
            destination,
            list(urls),
            sha256=sha256,
            size=size,
            progress=progress,
            cancel=cancel,
            probe_bytes=0,
            mirror_policy="strict-priority",
        )
    except Cancelled as exc:
        raise VerifyCancelled() from exc
    except (FetchError, DownloadError, OSError) as exc:
        raise VerifyError(str(exc)) from exc


def format_table(reports: Sequence[ModelReport]) -> str:
    """A text table: model, status, bytes hashed / total, and problem files."""
    rows = [("model", "status", "checked", "files")]
    if not reports:
        rows.append(("(none)", "OK", "0/0", "-"))
    for report in reports:
        problems = ", ".join(item.path for item in report.files if item.status != "ok") or "-"
        rows.append(
            (
                report.id,
                _status_word(report.status),
                f"{report.bytes_hashed}/{report.bytes_total}",
                problems,
            )
        )
    widths = [max(len(row[index]) for row in rows) for index in range(4)]
    lines: list[str] = []
    for index, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[column]) for column, cell in enumerate(row)).rstrip())
        if index == 0:
            lines.append("  ".join("-" * width for width in widths).rstrip())
    return "\n".join(lines) + "\n"


def snapshot(
    reports: Sequence[ModelReport],
    *,
    phase: Phase,
    done: bool,
    cancelled: bool = False,
) -> Status:
    """A progress-window snapshot. One step per model, with bytes hashed and a status."""
    steps: list[Step] = []
    problems: dict[str, str] = {}
    for report in reports:
        steps.append(
            Step(
                model=report.id,
                size=max(0, report.bytes_total),
                source=report.source,
                speed_bps=report.speed_bps,
                eta_s=report.eta_s,
                state=report.status,
                received=max(0, report.bytes_hashed),
                sha256="",
                apps=["model-forge"],
                item_id=report.id,
                display_name=report.display_name or report.id,
            )
        )
        failed = [item.path for item in report.files if item.status != "ok"]
        if failed:
            problems[report.id] = ", ".join(failed)
    broken = any(report.status in {"missing", "corrupt"} for report in reports)
    if cancelled:
        state = JobState.CANCELLED
    elif not done:
        state = JobState.DOWNLOADING
    elif broken:
        state = JobState.FAILED
    else:
        state = JobState.READY
    status = Status(state=state, steps=steps)
    status.extra = {
        "phase": phase,
        "repairable": bool(done and broken and not cancelled),
        "cancelled": cancelled,
        "empty": bool(done and not reports),
        "problems": problems,
    }
    status.recompute()
    return status


def execute_verify(
    *,
    manifest: Path,
    repair: bool,
    gui: bool,
    root: Path | None = None,
    chunk: int = HASH_CHUNK,
) -> dict[str, object]:
    """Run the check. ``gui`` always opens the progress window, including when every file is OK.

    ``repair`` re-downloads damaged files without asking. The returned
    ``exit_code`` is 0 when every installed model is OK and 4 otherwise.
    """
    models = discover(manifest, root=root)
    holder: dict[str, list[ModelReport]] = {"reports": []}

    def job(cancel: threading.Event, on_status: Callable[[Status], None], do_repair: bool) -> WindowResult:
        def on_progress(reports: list[ModelReport], phase: Phase) -> None:
            holder["reports"] = reports
            on_status(snapshot(reports, phase=phase, done=False))

        try:
            reports = verify_models(models, cancel=cancel, chunk=chunk, on_progress=on_progress)
            if do_repair and any(item.status != "ok" for item in reports):
                reports = repair_models(models, reports, cancel=cancel, chunk=chunk, on_progress=on_progress)
            holder["reports"] = reports
            done = snapshot(reports, phase="verify", done=True)
            on_status(done)
            code = 0 if all(item.status == "ok" for item in reports) else 4
            return WindowResult(code, done)
        except VerifyCancelled:
            reports = holder["reports"]
            done = snapshot(reports, phase="verify", done=True, cancelled=True)
            on_status(done)
            return WindowResult(4, done)

    if gui:
        code = run_verify_window(job, repair=repair)
    else:
        code = job(threading.Event(), lambda _status: None, repair).code
    reports = holder["reports"]
    return {
        "ok": code == 0,
        "exit_code": code,
        "models": [item.to_dict() for item in reports],
        "text": format_table(reports),
    }


def _app_models(manifest: Path, root: Path) -> list[InstalledModel]:
    if not manifest.is_file():
        return []
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise VerifyError(f"{manifest.name} could not be read") from exc
    if not isinstance(document, dict):
        return []
    found: list[InstalledModel] = []
    for model in document.get("models") or []:
        parsed = _app_model(model, root, manifest)
        if parsed is not None:
            found.append(parsed)
    return found


def _app_model(model: object, root: Path, manifest: Path) -> InstalledModel | None:
    if not isinstance(model, dict):
        return None
    model_id = model.get("id")
    if not isinstance(model_id, str) or not model_id:
        return None
    rows = model.get("files")
    if not isinstance(rows, list) or not rows:
        return None
    files: list[ModelFile] = []
    for row in rows:
        item = _app_file(root, model_id, row)
        if item is not None:
            files.append(item)
    if not files or not _app_installed(root, model_id, files):
        return None
    display = model.get("display_name")
    name = display.strip() if isinstance(display, str) and display.strip() else model_id
    return InstalledModel(
        id=model_id,
        display_name=name,
        source=_source_label(files),
        kind="app",
        files=tuple(files),
        models_root=root,
        manifest=manifest,
    )


def _app_file(root: Path, model_id: str, row: object) -> ModelFile | None:
    if not isinstance(row, dict):
        return None
    rel = row.get("path")
    digest = row.get("sha256")
    if not isinstance(rel, str) or not isinstance(digest, str):
        return None
    sha = digest.strip().lower()
    if len(sha) != 64:
        return None
    view_root = root / APP_DIR / model_id
    location = _safe_join(view_root, rel)
    if location is None:
        return None
    size = row.get("size")
    declared = size if isinstance(size, int) and not isinstance(size, bool) and size >= 0 else 0
    urls = tuple(url for url in row.get("mirrors") or [] if isinstance(url, str) and url.startswith("https://"))
    return ModelFile(rel, declared, sha, urls, location, root / BLOBS_DIR / sha)


def _app_installed(root: Path, model_id: str, files: Sequence[ModelFile]) -> bool:
    if (root / APP_DIR / model_id).is_dir():
        return True
    return any(item.blob is not None and item.blob.is_file() for item in files)


def _plugin_models(root: Path) -> list[InstalledModel]:
    if not root.is_dir():
        return []
    found: list[InstalledModel] = []
    for path in _plugin_manifests(root):
        try:
            text = path.read_text(encoding="utf-8")
            descriptor = parse_descriptor_text(text)
        except (OSError, UnicodeError, DescriptorError):
            continue
        model = _plugin_model(descriptor, path, root)
        if model is not None:
            found.append(model)
    return found


def _plugin_manifests(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(name for name in dirnames if name not in WALK_SKIP)
        if MANIFEST_NAME in filenames:
            found.append(Path(dirpath) / MANIFEST_NAME)
    return found


def _plugin_model(descriptor: Descriptor, path: Path, root: Path) -> InstalledModel | None:
    folder = path.parent
    files: list[ModelFile] = []
    for item in descriptor.files:
        location = _safe_join(folder, item.path)
        if location is None or _user_data(location):
            continue
        files.append(
            ModelFile(
                item.path,
                item.size,
                item.sha256,
                source_urls(descriptor, item.path),
                location,
                None,
            )
        )
    if not files:
        return None
    return InstalledModel(
        id=descriptor.id,
        display_name=descriptor.display_name or descriptor.id,
        source=_source_label(files),
        kind="plugin",
        files=tuple(files),
        models_root=root,
        manifest=path,
    )


def _repair_model(
    model: InstalledModel,
    report: ModelReport,
    *,
    cancel: threading.Event | None,
    chunk: int,
    on_progress: Progress | None,
    reports: list[ModelReport],
) -> None:
    for index, item in enumerate(model.files):
        if index >= len(report.files) or report.files[index].status == "ok":
            continue
        _raise_if_cancelled(cancel)
        _repair_file(model, item, report, index, cancel=cancel, chunk=chunk, on_progress=on_progress, reports=reports)
    if model.kind == "app" and model.manifest is not None:
        try:
            ensure_view(model.id, root=model.models_root, manifest=model.manifest)
        except (OSError, ViewError):
            return


def _repair_file(
    model: InstalledModel,
    item: ModelFile,
    report: ModelReport,
    index: int,
    *,
    cancel: threading.Event | None,
    chunk: int,
    on_progress: Progress | None,
    reports: list[ModelReport],
) -> None:
    match model.kind:
        case "app":
            _repair_app_file(item, report, index, cancel=cancel, chunk=chunk, on_progress=on_progress, reports=reports)
        case "plugin":
            _repair_plugin_file(item, report, index, cancel=cancel, on_progress=on_progress, reports=reports)
        case _ as unreachable:
            _never: Never = unreachable
            raise VerifyError(f"unknown model kind {_never}")


def _repair_app_file(
    item: ModelFile,
    report: ModelReport,
    index: int,
    *,
    cancel: threading.Event | None,
    chunk: int,
    on_progress: Progress | None,
    reports: list[ModelReport],
) -> None:
    blob = item.blob
    if blob is None or _user_data(blob) or _user_data(item.location):
        return
    if not _matches(blob, item.sha256, chunk=chunk, cancel=cancel):
        if not item.urls:
            return
        blob.parent.mkdir(parents=True, exist_ok=True)
        _fetch_reporting(blob, item, report, index, cancel=cancel, on_progress=on_progress, reports=reports)
    if _matches(blob, item.sha256, chunk=chunk, cancel=cancel):
        _link_view(item.location, blob)


def _repair_plugin_file(
    item: ModelFile,
    report: ModelReport,
    index: int,
    *,
    cancel: threading.Event | None,
    on_progress: Progress | None,
    reports: list[ModelReport],
) -> None:
    if _user_data(item.location) or not item.urls:
        return
    _fetch_reporting(item.location, item, report, index, cancel=cancel, on_progress=on_progress, reports=reports)


def _fetch_reporting(
    destination: Path,
    item: ModelFile,
    report: ModelReport,
    index: int,
    *,
    cancel: threading.Event | None,
    on_progress: Progress | None,
    reports: list[ModelReport],
) -> None:
    before = sum(entry.hashed for entry in report.files[:index])

    def on_bytes(received: int, _total: int) -> None:
        report.bytes_hashed = before + received
        report.status = "repairing"
        _emit(on_progress, reports, "repair")

    try:
        download_file(
            destination,
            item.urls,
            sha256=item.sha256,
            size=item.size,
            cancel=cancel,
            on_bytes=on_bytes,
        )
    except VerifyCancelled:
        raise
    except VerifyError:
        return


def _link_view(view: Path, blob: Path) -> None:
    """Point ``view`` at ``blob``. A path inside the blob store is not removed."""
    _refuse_user_data(view)
    if not blob.is_file():
        return
    view.parent.mkdir(parents=True, exist_ok=True)
    blobs = blob.parent.resolve()
    if view.exists() or view.is_symlink():
        try:
            if view.is_file() and os.path.samefile(view, blob):
                return
        except OSError:
            pass
        try:
            resolved = view.resolve()
        except OSError:
            return
        if resolved == blobs or blobs in resolved.parents:
            return
        if view.is_dir() and not view.is_symlink():
            return
        view.unlink()
    try:
        os.link(blob, view)
    except OSError:
        shutil.copyfile(blob, view)


def _matches(path: Path, sha256: str, *, chunk: int, cancel: threading.Event | None) -> bool:
    if not path.is_file() or _user_data(path):
        return False
    try:
        return hash_file(path, chunk=chunk, cancel=cancel) == sha256
    except OSError:
        return False


def _present_path(model: InstalledModel, item: ModelFile) -> Path | None:
    view = item.location
    if _user_data(view):
        return None
    if view.is_file():
        return view
    if model.kind != "app" or item.blob is None or _user_data(item.blob):
        return None
    # The view folder is what the app opens. A missing name there is missing
    # even when the blob store still has the bytes; repair links it back.
    if view.parent.is_dir():
        return None
    if item.blob.is_file():
        return item.blob
    return None


def _safe_join(root: Path, rel: str) -> Path | None:
    raw = rel.replace("\\", "/")
    parts = Path(raw).parts
    if not parts or raw.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        return None
    return root.joinpath(*parts)


def _user_data(path: Path) -> bool:
    return any(part in USER_DATA_DIRS for part in path.parts)


def _refuse_user_data(path: Path) -> None:
    if _user_data(path):
        raise VerifyError(f"refusing to change user data: {path}")


def _source_label(files: Sequence[ModelFile]) -> str:
    for item in files:
        for url in item.urls:
            host = urllib.parse.urlparse(url).hostname
            if host:
                return host
    return "local"


def _overall(files: Sequence[FileReport]) -> FileStatus:
    if any(item.status == "corrupt" for item in files):
        return "corrupt"
    if any(item.status == "missing" for item in files):
        return "missing"
    return "ok"


def _status_word(status: RowStatus) -> str:
    match status:
        case "ok":
            return "OK"
        case "missing":
            return "missing"
        case "corrupt":
            return "corrupt"
        case "checking":
            return "checking"
        case "repairing":
            return "repairing"
        case _ as unreachable:
            _never: Never = unreachable
            raise VerifyError(f"unknown status {_never}")


def _pace(report: ModelReport, started: float) -> None:
    elapsed = time.monotonic() - started
    if elapsed <= 0 or report.bytes_hashed <= 0:
        report.speed_bps = 0.0
        report.eta_s = None
        return
    report.speed_bps = report.bytes_hashed / elapsed
    left = report.bytes_total - report.bytes_hashed
    report.eta_s = left / report.speed_bps if report.speed_bps and left > 0 else 0.0


def _emit(on_progress: Progress | None, reports: list[ModelReport], phase: Phase) -> None:
    if on_progress is not None:
        on_progress(reports, phase)


def _raise_if_cancelled(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise VerifyCancelled()


def _mirror_kind(kind: str) -> MirrorKind | None:
    if kind in _MIRROR_KINDS:
        return _as_kind(kind)
    return None


def _as_kind(kind: str) -> MirrorKind:
    match kind:
        case "hf" | "kaggle" | "gdrive" | "modelscope" | "url" as known:
            return known
        case _ as unreachable:
            _never: Never = unreachable
            raise VerifyError(f"unknown mirror kind {_never}")


def _mirror_url(kind: MirrorKind, ref: str, revision: str, filename: str, count: int) -> str:
    try:
        match kind:
            case "hf":
                if not revision.strip():
                    return ""
                return huggingface_url(ref, filename, revision.strip())
            case "modelscope":
                return modelscope_url(ref, filename, revision.strip() or "master")
            case "kaggle":
                owner, _slash, dataset = ref.partition("/")
                if not owner or not dataset or "/" in dataset:
                    return ""
                return kaggle_url(owner, dataset, filename)
            case "gdrive":
                if count != 1:
                    return ""
                return gdrive_url(ref)
            case "url":
                return _join_url(ref, filename)
            case _ as unreachable:
                _never: Never = unreachable
                raise VerifyError(f"unknown mirror kind {_never}")
    except (CatalogError, DownloadError, ValueError):
        return ""


def _join_url(ref: str, filename: str) -> str:
    parsed = urllib.parse.urlparse(ref.strip())
    if parsed.scheme != "https" or not parsed.netloc:
        return ""
    base = ref.strip() if ref.strip().endswith("/") else ref.strip() + "/"
    quoted = "/".join(urllib.parse.quote(part) for part in filename.split("/"))
    return base + quoted
