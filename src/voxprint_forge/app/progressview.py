"""The live training view on the Train step (GUI runs and followed terminal runs).

:class:`ProgressView` shows one project's progress the way ``train.json`` and
``vram.csv`` record it: the stage, step N/M, a progress bar, the loss, the
elapsed time, the ETA and the video memory, with the ``previews/`` folder as a
strip that grows as test pictures appear. The composed status sentence and the
bar keep the object names the shell and the tests already use
(``progressText``, ``progress``, ``progressPercent``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from voxprint_forge.app.text import tr
from voxprint_forge.app.thumbs import ThumbService
from voxprint_forge.lora.progress_log import VramSample
from voxprint_forge.lora.project import Project

PREVIEW_SUFFIXES = (".webp", ".png", ".jpg", ".jpeg")
_POLL_MS = 1000

_STAGE_KEYS = {
    "prepare": "forge.app.stage.prepare",
    "encode": "forge.app.stage.encode",
    "train": "forge.app.stage.train",
    "step": "forge.app.stage.train",
    "sample": "forge.app.stage.sample",
    "review_sample": "forge.app.stage.review_sample",
    "paused": "forge.app.stage.paused",
    "done": "forge.app.stage.done",
    "failed": "forge.app.stage.failed",
}


def stage_text(stage: str) -> str:
    """The localized word of one ``train.json`` stage; an unknown stage is shown as recorded."""
    key = _STAGE_KEYS.get(stage.strip().lower())
    return tr(key) if key else stage


class _PreviewStrip(QtWidgets.QListWidget):
    """The project's ``previews/`` as a horizontal strip; a poll adds pictures as they appear."""

    def __init__(self, project: Path, service: ThumbService, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("forge.progress.previews")
        self.setViewMode(QtWidgets.QListView.ViewMode.IconMode)
        self.setFlow(QtWidgets.QListView.Flow.LeftToRight)
        self.setWrapping(False)
        self.setIconSize(QtCore.QSize(96, 96))
        self.setFixedHeight(128)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.setToolTip(tr("forge.app.result.previews.tip"))
        self._service = service
        self._project = Path(project)
        self._known: set[Path] = set()
        self.rescan()
        timer = QtCore.QTimer(self)
        timer.setInterval(_POLL_MS)
        timer.timeout.connect(self.rescan)
        timer.start()
        self._timer = timer

    def rescan(self) -> None:
        """Add thumbnails for pictures that appeared since the last poll."""
        previews = Project(self._project).previews
        if not previews.is_dir():
            return
        for path in sorted(previews.iterdir()):
            if path.suffix.lower() not in PREVIEW_SUFFIXES or not path.is_file() or path in self._known:
                continue
            self._known.add(path)
            self.addItem(_thumb_item(path, self._service))

    def set_project(self, project: Path) -> None:
        self._project = Path(project)
        self.clear()
        self._known.clear()
        self.rescan()


def _thumb_item(path: Path, service: ThumbService) -> QtWidgets.QListWidgetItem:
    """A strip tile whose picture fills in when the thumbnail worker finishes."""
    item = QtWidgets.QListWidgetItem(path.stem)

    def apply(icon: QtGui.QIcon, row: QtWidgets.QListWidgetItem = item) -> None:
        try:
            row.setIcon(icon)
        except RuntimeError:
            return

    pixmap = service.request(path, 96, lambda ready: apply(QtGui.QIcon(ready)))
    item.setIcon(QtGui.QIcon(pixmap))
    item.setToolTip(path.name)
    item.setData(QtCore.Qt.ItemDataRole.UserRole, str(path))
    return item


class ProgressView(QtWidgets.QFrame):
    """Stage, step, bar, loss, elapsed, ETA, VRAM and the growing preview strip of one project."""

    def __init__(
        self,
        project: Path,
        service: ThumbService,
        span: Callable[[float], str],
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("forge.progress")
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_StyledBackground, True)
        self._span = span
        column = QtWidgets.QVBoxLayout(self)
        column.setContentsMargins(12, 10, 12, 10)
        column.setSpacing(8)
        title = QtWidgets.QLabel(tr("forge.progress.title"))
        title.setObjectName("settingGroup")
        column.addWidget(title)
        self._sentence = QtWidgets.QLabel()
        self._sentence.setObjectName("progressText")
        self._sentence.setWordWrap(True)
        column.addWidget(self._sentence)
        self._stage = self._stat(column, "forge.progress.stage", "forge.progress.stage")
        self._step = self._stat(column, "forge.progress.step", "forge.progress.step")
        bar_row = QtWidgets.QHBoxLayout()
        bar_row.setSpacing(8)
        self._bar = QtWidgets.QProgressBar()
        self._bar.setObjectName("progress")
        self._bar.setTextVisible(False)
        self._bar.setRange(0, 1000)
        self._percent = QtWidgets.QLabel()
        self._percent.setObjectName("progressPercent")
        bar_row.addWidget(self._bar, 1)
        bar_row.addWidget(self._percent)
        column.addLayout(bar_row)
        self._loss = self._stat(column, "lora.progress.loss", "forge.train.loss")
        self._elapsed = self._stat(column, "forge.progress.elapsed", "forge.progress.elapsed")
        self._eta = self._stat(column, "forge.progress.eta", "forge.progress.eta")
        self._vram = self._stat(column, "forge.app.train.vram", "forge.train.vram")
        previews_title = QtWidgets.QLabel(tr("forge.progress.previews"))
        previews_title.setObjectName("settingGroup")
        column.addWidget(previews_title)
        self._strip = _PreviewStrip(project, service)
        column.addWidget(self._strip)

    def _stat(self, column: QtWidgets.QVBoxLayout, text_key: str, name: str) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel()
        label.setObjectName(name)
        label.setWordWrap(True)
        label.setProperty("muted", "true")
        label._text_key = text_key  # type: ignore[attr-defined]
        column.addWidget(label)
        return label

    def set_sentence(self, text: str, percent: float | None) -> None:
        """The composed status sentence, with the bar and the percent next to it."""
        self._sentence.setText(text)
        shown = percent is not None
        self._bar.setValue(int((percent or 0) * 10))
        self._bar.setVisible(shown)
        self._percent.setText("" if not shown else f"{int(round(percent or 0))}%")
        self._percent.setVisible(shown)

    def refresh(self, row: dict[str, object] | None, vram: VramSample) -> None:
        """Repaint every stat from one ``train.json`` row and the newest VRAM sample."""
        row = row or {}
        stage = str(row.get("stage") or "")
        self._stage.setText(tr("forge.progress.stage", stage=stage_text(stage)))
        self._stage.setVisible(bool(stage))
        total = int(row.get("total") or 0)
        step = int(row.get("step") or 0)
        self._step.setText(tr("forge.progress.step", step=step, total=total))
        self._step.setVisible(total > 0)
        loss = row.get("loss")
        shown = "—" if not isinstance(loss, int | float) else f"{float(loss):.4f}"
        self._loss.setText(tr("lora.progress.loss", loss=shown))
        self._loss.setVisible(isinstance(loss, int | float))
        elapsed = row.get("elapsed_seconds")
        self._elapsed.setText(tr("forge.progress.elapsed", elapsed=self._span(float(elapsed or 0))))
        self._elapsed.setVisible(isinstance(elapsed, int | float))
        eta = row.get("eta_seconds")
        self._eta.setText(tr("forge.progress.eta", eta=self._span(float(eta))))
        self._eta.setVisible(isinstance(eta, int | float) and not isinstance(eta, bool))
        allocated, reserved = vram
        if allocated is None and reserved is None:
            self._vram.hide()
        else:
            used = "—" if allocated is None else f"{allocated:.1f}"
            held = "—" if reserved is None else f"{reserved:.1f}"
            self._vram.setText(tr("forge.app.train.vram", allocated=used, reserved=held))
            self._vram.show()
