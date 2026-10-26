"""Red/green dataset gate. The shape matches ``voxprint_common.ui.check_gate``.

``mark_dirty`` only turns the indicator red. ``start`` runs the active items on
a worker, one at a time. ``unload`` runs on that worker after every run,
including cancel and error. When ``on_deactivate`` is set and returns, failed
items are handled: one or more passes ends in :attr:`GateState.PASSED`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import assert_never

from PySide6 import QtCore, QtWidgets

from voxprint_forge.app.text import tr


class GateState(Enum):
    """Indicator state. Dirty, failed and cancelled are red; passed is green."""

    DIRTY = "dirty"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class CheckItem:
    """One row the gate can check. Inactive rows are skipped."""

    id: str
    label: str
    active: bool = True


@dataclass(frozen=True)
class ItemResult:
    """The job's answer for one item. ``message`` is a catalog key or plain text."""

    id: str
    ok: bool
    message: str = ""


class CancelToken:
    """Polled by the job. ``cancel`` flips :attr:`is_cancelled`."""

    def __init__(self) -> None:
        self._cancelled = False

    @property
    def is_cancelled(self) -> bool:
        """True after :meth:`cancel` until the next :meth:`CheckGate.start`."""
        return self._cancelled

    def cancel(self) -> None:
        """Ask the worker to stop after the item it is on."""
        self._cancelled = True


class CheckGate(QtWidgets.QWidget):
    """Check button, red/green mark, progress and cancel."""

    stateChanged = QtCore.Signal(object)
    canProceedChanged = QtCore.Signal(bool)
    progress = QtCore.Signal(int, int, str)
    finished = QtCore.Signal(object)

    def __init__(
        self,
        job: Callable[[CheckItem, CancelToken], ItemResult],
        *,
        items: Callable[[], Sequence[CheckItem]],
        unload: Callable[[], None] | None = None,
        on_deactivate: Callable[[Sequence[str]], None] | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._job = job
        self._items = items
        self._unload = unload
        self._on_deactivate = on_deactivate
        self._epoch = 0
        self._token = CancelToken()
        self._results: dict[str, ItemResult] = {}
        self._deactivated: tuple[str, ...] = ()
        self._state = GateState.DIRTY
        self._worker: _CheckWorker | None = None
        self._done = 0
        self._total = 0
        self.setObjectName("forge.photos.gate")
        row = QtWidgets.QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        self._mark = QtWidgets.QLabel("■")
        self._mark.setObjectName("forgeCheckMark")
        self._mark.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._mark.setFixedSize(28, 28)
        self._status = QtWidgets.QLabel()
        self._status.setObjectName("forge.photos.check.status")
        self._status.setWordWrap(True)
        self._bar = QtWidgets.QProgressBar()
        self._bar.setObjectName("forge.photos.check.bar")
        self._bar.setTextVisible(False)
        self._bar.setFixedWidth(160)
        self._bar.setFixedHeight(8)
        self._button = QtWidgets.QPushButton()
        self._button.setObjectName("forge.photos.check")
        self._button.setProperty("gateAction", "true")
        self._button.setAutoDefault(False)
        self._button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._button.clicked.connect(self._on_button)
        row.addWidget(self._mark, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self._status, 1)
        row.addWidget(self._bar, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self._button, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        self._paint()

    @property
    def state(self) -> GateState:
        """The current indicator state."""
        return self._state

    @property
    def can_proceed(self) -> bool:
        """True only while the state is :attr:`GateState.PASSED`."""
        return self._state is GateState.PASSED

    @property
    def results(self) -> Mapping[str, ItemResult]:
        """Results recorded by the run that is current. Empty after :meth:`mark_dirty`."""
        return MappingProxyType(self._results)

    @property
    def deactivated(self) -> tuple[str, ...]:
        """Ids ``on_deactivate`` accepted on the last finished run."""
        return self._deactivated

    def mark_dirty(self) -> None:
        """Turn the indicator red. Does not load a model and does not start a job."""
        self._epoch += 1
        self._token.cancel()
        self._results = {}
        self._deactivated = ()
        self._done = 0
        self._total = 0
        self._set_state(GateState.DIRTY)

    def start(self) -> None:
        """Check the active items on a worker. Ignored while a worker is still running."""
        if self._worker is not None and self._worker.isRunning():
            return
        self._epoch += 1
        epoch = self._epoch
        self._token = CancelToken()
        self._results = {}
        self._deactivated = ()
        self._done = 0
        items = [item for item in self._items() if item.active]
        self._total = len(items)
        self._set_state(GateState.RUNNING)
        worker = _CheckWorker(self._job, items, self._token, self._unload, epoch)
        worker.item_done.connect(self._on_item)
        worker.ended.connect(self._on_ended)
        self._worker = worker
        self._paint()
        worker.start()

    def cancel(self) -> None:
        """Stop after the item the worker is on. The state becomes cancelled when it ends."""
        self._token.cancel()

    def _on_button(self) -> None:
        if self._state is GateState.RUNNING or (self._worker is not None and self._worker.isRunning()):
            self.cancel()
            return
        self.start()

    def _on_item(self, result: object, done: int, total: int, item_id: str, epoch: int) -> None:
        if epoch != self._epoch or not isinstance(result, ItemResult):
            return
        self._results[result.id] = result
        self._done = done
        self._total = total
        self.progress.emit(done, total, item_id)
        self._paint()

    def _on_ended(self, epoch: int) -> None:
        if epoch != self._epoch or self._state is not GateState.RUNNING:
            self._paint()
            return
        worker = self._worker
        error = worker.error if worker is not None else None
        cancelled = self._token.is_cancelled
        passed = [item for item in self._results.values() if item.ok]
        failed = [item.id for item in self._results.values() if not item.ok]
        handled = False
        deactivate_error = False
        if error is None and not cancelled and failed and self._on_deactivate is not None:
            try:
                self._on_deactivate(tuple(failed))
            except Exception:
                deactivate_error = True
            else:
                self._deactivated = tuple(failed)
                handled = True
        if error is not None or deactivate_error:
            state = GateState.FAILED
        elif cancelled:
            state = GateState.CANCELLED
        elif not passed:
            state = GateState.FAILED
        elif failed and not handled:
            state = GateState.FAILED
        else:
            state = GateState.PASSED
        self._set_state(state)
        self.finished.emit(state)

    def _set_state(self, state: GateState) -> None:
        previous = self._state
        self._state = state
        self._paint()
        if previous is not state:
            self.stateChanged.emit(state)
            if (previous is GateState.PASSED) != (state is GateState.PASSED):
                self.canProceedChanged.emit(state is GateState.PASSED)

    def _paint(self) -> None:
        running = self._state is GateState.RUNNING
        self._button.setText(tr("forge.app.photos.check.cancel") if running else tr("forge.app.photos.check"))
        self._button.setToolTip(
            tr("forge.app.photos.check.cancel.tip") if running else tr("forge.app.photos.check.tip")
        )
        self._button.setProperty("running", "true" if running else "false")
        self._button.style().unpolish(self._button)
        self._button.style().polish(self._button)
        self._bar.setVisible(running)
        if self._total:
            self._bar.setRange(0, self._total)
            self._bar.setValue(self._done)
        else:
            self._bar.setRange(0, 1)
            self._bar.setValue(0)
        stopped = self._state in (GateState.DIRTY, GateState.FAILED, GateState.CANCELLED)
        self._mark.setText("■" if stopped or running else "→")
        self._mark.setProperty("gate", "stop" if stopped else "go" if self._state is GateState.PASSED else "run")
        self._mark.style().unpolish(self._mark)
        self._mark.style().polish(self._mark)
        self._status.setText(self._status_text())

    def _status_text(self) -> str:
        state = self._state
        if state is GateState.DIRTY:
            return tr("forge.app.photos.check.dirty")
        if state is GateState.RUNNING:
            return tr("forge.app.photos.check.running", done=self._done, total=self._total)
        if state is GateState.PASSED:
            if self._deactivated:
                return tr("forge.app.photos.check.deactivated", count=len(self._deactivated))
            return tr("forge.app.photos.check.passed")
        if state is GateState.FAILED:
            if self._deactivated:
                return tr("forge.app.photos.check.deactivated", count=len(self._deactivated))
            return tr("forge.app.photos.check.failed")
        if state is GateState.CANCELLED:
            return tr("forge.app.photos.check.cancelled")
        assert_never(state)


class _CheckWorker(QtCore.QThread):
    """Runs items off the UI thread, then unloads."""

    item_done = QtCore.Signal(object, int, int, str, int)
    ended = QtCore.Signal(int)

    def __init__(
        self,
        job: Callable[[CheckItem, CancelToken], ItemResult],
        items: Sequence[CheckItem],
        token: CancelToken,
        unload: Callable[[], None] | None,
        epoch: int,
    ) -> None:
        super().__init__()
        self._job = job
        self._items = list(items)
        self._token = token
        self._unload = unload
        self._epoch = epoch
        self.error: BaseException | None = None

    def run(self) -> None:
        """Check each active item, then always call unload."""
        try:
            total = len(self._items)
            for index, item in enumerate(self._items, start=1):
                if self._token.is_cancelled:
                    break
                result = self._job(item, self._token)
                if self._token.is_cancelled:
                    break
                self.item_done.emit(result, index, total, item.id, self._epoch)
        except Exception as exc:  # noqa: BLE001 - the gate turns a job error into FAILED
            self.error = exc
        finally:
            if self._unload is not None:
                try:
                    self._unload()
                except Exception as exc:  # noqa: BLE001 - unload failure is a failed check
                    if self.error is None:
                        self.error = exc
            self.ended.emit(self._epoch)
