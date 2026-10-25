"""Progress window for checking and repairing installed models.

The vendored ``voxprint_common.mdl.progress_ui`` window is the download
window. Its heading is always “Загрузка моделей”, ``paint`` never shows
``Step.state``, and a successful job closes the window after 800 ms. This
module is the Model Forge adapter: a verify phase, a per-model status, and a
Repair action. It does not change the vendored 0.4.1 wheel.

Components should upstream the following into ``mdl/progress_ui.py``:

1. A verify phase. The heading is always “Загрузка моделей”. ``JobState``
   has no verifying state, and success auto-closes after 800 ms, so a person
   cannot read the per-model results.
2. Per-item status. ``Step.state`` is stored and ``paint()`` never shows it.
   The columns are Model, Size, Source, Speed, and ETA. A check needs a
   status column (OK, missing, or corrupt) and bytes hashed / total on each
   row.
3. A repair action distinct from Retry. Retry starts the same job again.
   Repair re-downloads only missing or corrupt files from the manifest
   source (pinned sha256, HTTP Range resume) and then verifies again.
4. Keep the window usable at 1280 px. The extra column has to stretch inside
   that width, and the action buttons have to stay on screen.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from voxprint_common.i18n import add_catalogs, get_language
from voxprint_common.mdl.progress_ui import format_eta, format_size, format_speed, window_title
from voxprint_common.mdl.status import JobState, Status

from voxprint_forge.app.text import tr

_LOCALES = Path(__file__).resolve().parent / "app" / "locales"
_catalogs_ready = False

VerifyJob = Callable[[threading.Event, Callable[[Status], None], bool], "WindowResult"]
Settled = Callable[[object], None]


@dataclass(frozen=True)
class WindowResult:
    """The exit code and the last status a verify job produced."""

    code: int
    status: Status


def run_verify_window(job: VerifyJob, *, repair: bool = False, on_settled: Settled | None = None) -> int:
    """Open the check window, run ``job``, and return when the window closes.

    The window opens before ``job`` finishes, including when every model is
    already OK. ``repair`` starts the job with repair enabled. The Repair
    button runs ``job`` again with repair enabled. Closing a finished job
    returns its code. Closing a running job cancels it and returns 4.

    PySide6 is imported here so ``models verify`` can run without the GUI extra.
    """
    window, app = _open(None)
    _drive(window, app, job, repair=repair, on_settled=on_settled)
    code = int(app.exec())
    timer = getattr(window, "_timer", None)
    if timer is not None:
        timer.stop()
    return code


def open_verify_window(status: Status | None = None) -> object:
    """Show the check window without starting a job.

    ``status`` is painted immediately so a layout test can read the rows.
    """
    window, _app = _open(status)
    return window


def _open(status: Status | None) -> tuple[object, object]:
    # Qt stays out of the module import so a headless check can import this file.
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    window = _window()
    if status is not None:
        paint = getattr(window, "_paint", None)
        if callable(paint):
            paint(status, busy=False)
    window.show()
    return window, app


def _window() -> object:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import (
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QProgressBar,
        QPushButton,
        QTableWidget,
        QVBoxLayout,
        QWidget,
    )
    from voxprint_common.ui.theme import RED, build_style

    _ensure_catalogs()

    class _VerifyWindow(QWidget):
        def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt API
            closer = getattr(self, "_on_close", None)
            if callable(closer):
                closer()
            accept = getattr(event, "accept", None)
            if callable(accept):
                accept()

    window = _VerifyWindow()
    window.setObjectName("mdlVerify")
    window.setWindowTitle(window_title())
    window.setStyleSheet(build_style(False))
    window.setMinimumWidth(960)
    window.resize(1100, 560)
    layout = QVBoxLayout(window)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(8)
    title = QLabel(window_title())
    title.setObjectName("title")
    heading = QLabel(tr("forge.models.verify.heading"))
    heading.setObjectName("mdlVerifyHeading")
    table = QTableWidget(0, 6)
    table.setObjectName("mdlVerifyTable")
    table.setHorizontalHeaderLabels(
        [
            tr("forge.models.verify.column.model"),
            tr("forge.models.verify.column.checked"),
            tr("forge.models.verify.column.status"),
            tr("forge.models.verify.column.source"),
            tr("forge.models.verify.column.speed"),
            tr("forge.models.verify.column.left"),
        ]
    )
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
    table.setWordWrap(False)
    table.setTextElideMode(Qt.TextElideMode.ElideRight)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    header = table.horizontalHeader()
    header.setStretchLastSection(True)
    header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    header.setMinimumSectionSize(64)
    overall_label = QLabel(tr("forge.models.verify.overall"))
    overall = QProgressBar()
    overall.setRange(0, 1000)
    overall.setTextVisible(False)
    message = QLabel("")
    message.setObjectName("mdlVerifyMessage")
    message.setWordWrap(True)
    buttons = QHBoxLayout()
    repair = QPushButton(tr("forge.models.verify.repair"))
    repair.setObjectName("repair")
    repair.setEnabled(False)
    cancel = QPushButton(tr("forge.models.verify.cancel"))
    cancel.setObjectName("cancel")
    buttons.addStretch(1)
    buttons.addWidget(repair)
    buttons.addWidget(cancel)
    layout.addWidget(title)
    layout.addWidget(heading)
    layout.addWidget(table, stretch=1)
    layout.addWidget(overall_label)
    layout.addWidget(overall)
    layout.addWidget(message)
    layout.addLayout(buttons)

    def paint(current: Status, *, busy: bool) -> None:
        heading.setText(_heading(current))
        rows = current.steps
        table.setRowCount(len(rows))
        problems = current.extra.get("problems")
        tips = problems if isinstance(problems, dict) else {}
        for row, step in enumerate(rows):
            tip = tips.get(step.item_id) or tips.get(step.model) or ""
            tip_text = tip if isinstance(tip, str) else ""
            values = (
                step.display_name or step.model,
                f"{_size(step.received)} / {_size(step.size)}",
                _status_text(step.state),
                step.source or "—",
                _speed(step.speed_bps),
                _eta(step.eta_s),
            )
            for column, text in enumerate(values):
                item = _cell(text, tip_text if column == 0 else tip_text)
                table.setItem(row, column, item)
        if current.bytes_total:
            fraction = min(1.0, current.bytes_done / current.bytes_total)
            overall.setRange(0, 1000)
            overall.setValue(int(fraction * 1000))
            overall_label.setText(f"{tr('forge.models.verify.overall')}: {int(fraction * 100)}%")
        else:
            overall.setRange(0, 1)
            overall.setValue(1 if current.ready else 0)
            overall_label.setText(tr("forge.models.verify.overall"))
        _message(message, current, RED)
        repair.setEnabled(bool(current.extra.get("repairable")) and not busy)

    window._paint = paint  # type: ignore[attr-defined]
    window._repair = repair  # type: ignore[attr-defined]
    window._cancel = cancel  # type: ignore[attr-defined]
    window._table = table  # type: ignore[attr-defined]
    return window


def _drive(
    window: object,
    app: object,
    job: VerifyJob,
    *,
    repair: bool,
    on_settled: Settled | None,
) -> None:
    from PySide6.QtCore import QTimer

    _ensure_catalogs()
    state: dict[str, object] = {"status": None, "result": None, "cancel": None, "busy": False, "seen": None}
    lock = threading.Lock()
    closing = {"value": False}
    repair_button = window._repair  # type: ignore[attr-defined]
    cancel_button = window._cancel  # type: ignore[attr-defined]
    paint = window._paint  # type: ignore[attr-defined]

    def finish(code: int) -> None:
        if closing["value"]:
            return
        closing["value"] = True
        app.exit(code)  # type: ignore[attr-defined]

    def launch(do_repair: bool) -> None:
        if closing["value"]:
            return
        event = threading.Event()
        with lock:
            state["cancel"] = event
            state["result"] = None
            state["busy"] = True
        repair_button.setEnabled(False)
        cancel_button.setText(tr("forge.models.verify.cancel"))

        def run() -> None:
            def on_status(status: Status) -> None:
                with lock:
                    state["status"] = status

            try:
                result = job(event, on_status, do_repair)
            except Exception as exc:
                failed = Status(JobState.FAILED, error=str(exc))
                result = WindowResult(4, failed)
            with lock:
                state["status"] = result.status
                state["result"] = result
                state["busy"] = False

        threading.Thread(target=run, name="voxprint-forge-verify", daemon=True).start()

    def refresh() -> None:
        with lock:
            status = state.get("status")
            result = state.get("result")
            busy = bool(state.get("busy"))
            seen = state.get("seen")
        if isinstance(status, Status):
            paint(status, busy=busy)
        if busy or not isinstance(result, WindowResult) or result is seen or closing["value"]:
            return
        with lock:
            if state.get("seen") is result:
                return
            state["seen"] = result
        cancel_button.setText(tr("forge.models.verify.close"))
        if isinstance(status, Status):
            paint(status, busy=False)
        if on_settled is not None:
            on_settled(window)

    def on_repair() -> None:
        with lock:
            busy = bool(state.get("busy"))
        if busy or closing["value"]:
            return
        launch(True)

    def on_dismiss() -> None:
        if closing["value"]:
            return
        with lock:
            busy = bool(state.get("busy"))
            event = state.get("cancel")
            result = state.get("result")
        if busy:
            if isinstance(event, threading.Event):
                event.set()
            finish(4)
            return
        finish(result.code if isinstance(result, WindowResult) else 0)

    repair_button.clicked.connect(on_repair)
    cancel_button.clicked.connect(on_dismiss)
    window._on_close = on_dismiss  # type: ignore[attr-defined]
    timer = QTimer(window)  # type: ignore[arg-type]
    timer.timeout.connect(refresh)
    timer.start(50)
    window._timer = timer  # type: ignore[attr-defined]
    launch(repair)


def _cell(text: str, tip: str) -> object:
    from PySide6.QtWidgets import QTableWidgetItem

    item = QTableWidgetItem(text)
    if tip:
        item.setToolTip(tip)
    return item


def _message(label: object, status: Status, red: str) -> None:
    cancelled = bool(status.extra.get("cancelled")) or status.state is JobState.CANCELLED
    if cancelled:
        text = tr("forge.models.verify.cancelled")
        style = f"color: {red};"
    elif status.error:
        text = status.error
        style = f"color: {red};"
    elif status.extra.get("empty"):
        text = tr("forge.models.verify.empty")
        style = ""
    elif status.state is JobState.READY:
        text = tr("forge.models.verify.ready")
        style = ""
    elif status.extra.get("repairable") or status.state is JobState.FAILED:
        text = tr("forge.models.verify.repair_needed")
        style = f"color: {red};"
    else:
        text = ""
        style = ""
    label.setText(text)  # type: ignore[attr-defined]
    label.setStyleSheet(style)  # type: ignore[attr-defined]


def _heading(status: Status) -> str:
    if status.extra.get("phase") == "repair":
        return tr("forge.models.verify.heading.repair")
    return tr("forge.models.verify.heading")


def _status_text(state: str) -> str:
    match state:
        case "ok":
            return tr("forge.models.verify.status.ok")
        case "missing":
            return tr("forge.models.verify.status.missing")
        case "corrupt":
            return tr("forge.models.verify.status.corrupt")
        case "checking":
            return tr("forge.models.verify.status.checking")
        case "repairing":
            return tr("forge.models.verify.status.repairing")
        case _:
            return state


def _size(amount: int) -> str:
    if get_language() == "ru":
        return format_size(amount)
    value = float(max(0, amount))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _speed(bps: float) -> str:
    if get_language() == "ru":
        return format_speed(bps)
    if bps <= 0 or math.isnan(bps):
        return "—"
    return f"{_size(int(bps))}/s"


def _eta(seconds: float | None) -> str:
    if get_language() == "ru":
        return format_eta(seconds)
    if seconds is None or seconds < 0 or math.isnan(seconds) or math.isinf(seconds):
        return "—"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours} h {minutes} min"
    if minutes:
        return f"{minutes} min {secs} s"
    return f"{secs} s"


def _ensure_catalogs() -> None:
    global _catalogs_ready
    if _catalogs_ready:
        return
    add_catalogs(_LOCALES)
    _catalogs_ready = True
