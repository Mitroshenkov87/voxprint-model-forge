"""The result step's model card: the LoRA file, its trigger, default strength and sha256.

The file is ``lora.safetensors`` when the run wrote one, else the newest named
save-as copy in the project folder. The strength and the hash come from the
``voxprint-model.json`` sidecar next to the file when there is one; the hash is
otherwise computed from the file in a worker thread.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from voxprint_forge.app.text import tr
from voxprint_forge.lora.project import Project

_SIDECAR = "voxprint-model.json"


def lora_file(project: Path) -> Path | None:
    """The trained LoRA of ``project``: ``lora.safetensors``, else the newest named save-as copy."""
    root = Path(project)
    final = root / "lora.safetensors"
    if final.is_file():
        return final
    named = [path for path in root.glob("*.safetensors") if path.is_file()]
    if not named:
        return None
    return max(named, key=lambda path: path.stat().st_mtime)


def sidecar_sha256(project: Path) -> str | None:
    """The hash recorded in the project's ``voxprint-model.json``, or ``None`` when there is none."""
    path = Path(project) / _SIDECAR
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = data.get("sha256") if isinstance(data, dict) else None
    return value if isinstance(value, str) and value else None


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """The sha256 hex of a file, read in chunks (safe for multi-GB LoRAs)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


class _HashSignals(QtCore.QObject):
    done = QtCore.Signal(str)


class _HashWorker(QtCore.QRunnable):
    """One chunked hash run; the result is delivered on the GUI thread."""

    def __init__(self, path: Path, signals: _HashSignals) -> None:
        super().__init__()
        self._path = path
        self._signals = signals

    def run(self) -> None:
        try:
            self._signals.done.emit(sha256_of(self._path))
        except OSError:
            self._signals.done.emit("")


class ResultInfo(QtWidgets.QFrame):
    """Rows for the LoRA file, the trigger, the default strength and the sha256."""

    def __init__(self, project: Path, fallback_strength: float, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("forge.result.info")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self._project = Path(project)
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(12, 10, 12, 10)
        column.setSpacing(8)
        title = QtWidgets.QLabel(tr("forge.app.result.info"))
        title.setObjectName("settingGroup")
        column.addWidget(title)
        rows = QtWidgets.QGridLayout()
        rows.setContentsMargins(0, 0, 0, 0)
        rows.setHorizontalSpacing(12)
        rows.setVerticalSpacing(6)
        column.addLayout(rows)
        self._path_label = self._row(rows, 0, "forge.app.result.file", "forge.result.path")
        self._trigger_label = self._row(rows, 1, "forge.app.subject.trigger", "forge.result.trigger")
        self._strength_label = self._row(rows, 2, "forge.app.result.strength", "forge.result.strength")
        self._sha_label = self._row(rows, 3, "forge.app.result.sha256", "forge.result.sha256")
        self.refresh(fallback_strength)

    def _row(self, rows: QtWidgets.QGridLayout, index: int, key: str, name: str) -> QtWidgets.QLabel:
        title = QtWidgets.QLabel(tr(key))
        title.setObjectName(f"{name}.title")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        value = QtWidgets.QLabel()
        value.setObjectName(name)
        value.setWordWrap(True)
        value.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        rows.addWidget(title, index, 0, QtCore.Qt.AlignmentFlag.AlignTop)
        rows.addWidget(value, index, 1)
        return value

    def refresh(self, fallback_strength: float) -> None:
        """Fill every row from the project and its sidecar; hash the file when no sidecar recorded it."""
        from voxprint_forge.lora.export import recorded_strength  # lazy: lora.export pulls in torch

        target = lora_file(self._project)
        self._path_label.setText(str(target) if target is not None else tr("forge.app.result.none"))
        data: dict[str, object] = {}
        try:
            data = Project(self._project).read()
        except Exception:  # noqa: BLE001 - a missing or broken project.json shows dashes
            data = {}
        subject = data.get("subject") if isinstance(data.get("subject"), dict) else {}
        trigger = str(subject.get("trigger") or "").strip()
        if not trigger:
            trigger = self._sidecar_value("trigger")
        self._trigger_label.setText(trigger or "—")
        strength = recorded_strength(data) if data else None
        if strength is None and target is not None:
            recorded = self._sidecar_value("default_strength")
            try:
                strength = float(recorded) if recorded else None
            except (TypeError, ValueError):
                strength = None
        if strength is None:
            strength = fallback_strength
        self._strength_label.setText(f"{strength:.2f}".rstrip("0").rstrip("."))
        recorded_hash = sidecar_sha256(self._project)
        if recorded_hash:
            self._sha_label.setText(recorded_hash)
        elif target is not None:
            self._start_hash(target)
        else:
            self._sha_label.setText("—")

    def _sidecar_value(self, key: str) -> str:
        path = self._project / _SIDECAR
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        value = data.get(key) if isinstance(data, dict) else None
        return str(value) if value is not None else ""

    def _start_hash(self, target: Path) -> None:
        self._sha_label.setText("…")
        signals = _HashSignals()
        signals.done.connect(self._hash_done)
        pool = QtCore.QThreadPool.globalInstance()
        if pool is None or not pool.maxThreadCount():
            self._hash_done(sha256_of(target))
            return
        pool.start(_HashWorker(target, signals))
        self._hash_signals = signals  # keep alive until delivered

    def _hash_done(self, digest: str) -> None:
        try:
            self._sha_label.setText(digest or "—")
        except RuntimeError:
            return
        self._hash_signals = None

    def compute_sha256_sync(self) -> str:
        """Hash the LoRA file on the calling thread (tests); the label shows the result."""
        target = lora_file(self._project)
        if target is None:
            return ""
        digest = sha256_of(target)
        self._sha_label.setText(digest)
        return digest
