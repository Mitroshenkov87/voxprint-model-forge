"""Model Forge window. Buttons call forge --json or a library helper; they do not reimplement a stage."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from voxprint_forge.gui.backdrop import apply_backdrop
from voxprint_forge.gui.client import (
    ForgeRun,
    SubprocessForge,
    forge_args,
    format_checks,
    ssh_status,
    state_badge,
    status_items,
)
from voxprint_forge.gui.gauges import GaugeProbes, disk_total_free, gauge_percent, gauge_text, vram_status
from voxprint_forge.gui.icons import load_icon
from voxprint_forge.gui.secrets import SecretStore, open_store, token_env
from voxprint_forge.gui.theme import FAILED_BG, FAILED_TEXT, build_style
from voxprint_forge.layout import Layout
from voxprint_forge.pipeline import clear_from_stage, retry_stage
from voxprint_forge.preflight import memory_status
from voxprint_forge.version import version_line

SshPoll = Callable[[str, str, str, str], dict[str, object]]
_COLUMNS = ("Model", "Stage", "Progress", "Last activity", "State", "HF", "Kaggle", "Drive", "Retry")
_UPLOAD_COLUMNS = (("hf", 5), ("kaggle", 6), ("drive", 7))


class MainWindow(QMainWindow):
    def __init__(
        self,
        *,
        runner: ForgeRun | None = None,
        ssh: SshPoll | None = None,
        store: SecretStore | None = None,
        probes: GaugeProbes | None = None,
    ) -> None:
        super().__init__()
        self.runner = runner or SubprocessForge()
        self.ssh = ssh or ssh_status
        self.store = store or open_store()
        self._remote = False
        self._about: QDialog | None = None
        self.setWindowTitle("Model Forge")
        self.setObjectName("root")
        self.resize(1100, 760)
        self._build()
        self.probes = probes or GaugeProbes(
            disk=lambda: disk_total_free(self.forge_out_path()),
            ram=memory_status,
            vram=vram_status,
        )
        self._load_remote_fields()
        self.apply_theme()
        self._timer = QTimer(self)
        self._timer.setInterval(2000)
        self._timer.timeout.connect(self.refresh_gauges)
        self._timer.timeout.connect(self.poll_remote)
        self._timer.start()
        self.refresh_gauges()

    def _build(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Model Forge")
        title.setObjectName("title")
        self.version_label = QLabel(version_line())
        self.version_label.setObjectName("footer")
        about = QPushButton("About")
        about.clicked.connect(self.show_about)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.version_label)
        header.addWidget(about)
        layout.addLayout(header)

        paths = QHBoxLayout()
        self.plan_edit = QLineEdit("plans/plan.toml")
        self.root_edit = QLineEdit()
        self.root_edit.setPlaceholderText("repo root")
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("forge-out")
        paths.addWidget(QLabel("Plan"))
        paths.addWidget(self.plan_edit, 2)
        paths.addWidget(QLabel("Root"))
        paths.addWidget(self.root_edit, 1)
        paths.addWidget(QLabel("Output"))
        paths.addWidget(self.out_edit, 1)
        layout.addLayout(paths)

        actions = QHBoxLayout()
        self.check_button = QPushButton("Check before run")
        self.check_button.setObjectName("primary")
        self.check_button.clicked.connect(self.check_before_run)
        self.refresh_button = QPushButton("Refresh status")
        self.refresh_button.setIcon(load_icon("download.svg"))
        self.refresh_button.setToolTip("Refresh status")
        self.refresh_button.clicked.connect(self.refresh_status)
        actions.addWidget(self.check_button)
        actions.addWidget(self.refresh_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.preflight_text = QPlainTextEdit()
        self.preflight_text.setReadOnly(True)
        self.preflight_text.setPlaceholderText(
            "Check before run shows each preflight result and a fix for every failure."
        )
        self.preflight_text.setFixedHeight(108)
        layout.addWidget(self.preflight_text)

        targets = QHBoxLayout()
        upload_icon = QLabel()
        upload_icon.setPixmap(load_icon("export.svg").pixmap(22, 22))
        upload_icon.setToolTip("Upload targets")
        upload_label = QLabel("Upload targets")
        self.hf_box = QCheckBox("Hugging Face")
        self.kaggle_box = QCheckBox("Kaggle")
        self.drive_box = QCheckBox("Google Drive fallback")
        self.hf_box.setChecked(True)
        self.kaggle_box.setChecked(True)
        self.drive_box.setChecked(True)
        targets.addWidget(upload_icon)
        targets.addWidget(upload_label)
        targets.addWidget(self.hf_box)
        targets.addWidget(self.kaggle_box)
        targets.addWidget(self.drive_box)
        targets.addStretch(1)
        layout.addLayout(targets)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        gauges = QHBoxLayout()
        self.disk_label, self.disk_bar = _gauge_widgets()
        self.ram_label, self.ram_bar = _gauge_widgets()
        self.vram_label, self.vram_bar = _gauge_widgets()
        for label, bar in (
            (self.disk_label, self.disk_bar),
            (self.ram_label, self.ram_bar),
            (self.vram_label, self.vram_bar),
        ):
            column = QVBoxLayout()
            column.addWidget(label)
            column.addWidget(bar)
            gauges.addLayout(column)
        layout.addLayout(gauges)

        remote = QHBoxLayout()
        self.host_edit = QLineEdit()
        self.host_edit.setPlaceholderText("host")
        self.user_edit = QLineEdit()
        self.user_edit.setPlaceholderText("user")
        self.key_edit = QLineEdit()
        self.key_edit.setPlaceholderText("private key path")
        self.remote_out_edit = QLineEdit()
        self.remote_out_edit.setPlaceholderText("remote forge-out")
        self.attach_button = QPushButton("Attach to VM")
        self.attach_button.clicked.connect(self.attach_remote)
        remote.addWidget(QLabel("Remote"))
        remote.addWidget(self.host_edit)
        remote.addWidget(self.user_edit)
        remote.addWidget(self.key_edit, 2)
        remote.addWidget(self.remote_out_edit)
        remote.addWidget(self.attach_button)
        layout.addLayout(remote)

        tokens = QHBoxLayout()
        self.token_hf = QLineEdit()
        self.token_kaggle = QLineEdit()
        self.token_drive = QLineEdit()
        for field, placeholder in (
            (self.token_hf, "HF token"),
            (self.token_kaggle, "Kaggle token"),
            (self.token_drive, "Drive token"),
        ):
            field.setPlaceholderText(placeholder)
            field.setEchoMode(QLineEdit.EchoMode.Password)
            tokens.addWidget(field)
        store_button = QPushButton("Store tokens")
        store_button.clicked.connect(self.store_tokens)
        tokens.addWidget(store_button)
        layout.addLayout(tokens)

        self.status_label = QLabel("This machine")
        self.status_label.setObjectName("subtitle")
        layout.addWidget(self.status_label)

    def apply_theme(self) -> str:
        offscreen = os.environ.get("QT_QPA_PLATFORM", "") == "offscreen"
        if offscreen or sys.platform != "win32":
            self.setStyleSheet(build_style(False))
            self.backdrop = "plain"
            return self.backdrop
        self.backdrop = apply_backdrop(int(self.winId()))
        self.setStyleSheet(build_style(self.backdrop == "acrylic"))
        return self.backdrop

    def forge_out_path(self) -> Path:
        text = self.out_edit.text().strip()
        if text:
            return Path(text)
        return Path.home() / "forge-out"

    def selected_upload_targets(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.hf_box.isChecked():
            names.append("hf")
        if self.kaggle_box.isChecked():
            names.append("kaggle")
        if self.drive_box.isChecked():
            names.append("drive")
        return tuple(names)

    def _env(self) -> dict[str, str]:
        return token_env(self.store, self.selected_upload_targets())

    def _args(self, command: str, model: str = "") -> list[str]:
        return forge_args(
            command,
            plan=self.plan_edit.text().strip(),
            root=self.root_edit.text().strip(),
            forge_out=str(self.forge_out_path()),
            model=model,
        )

    def _require_targets(self) -> bool:
        if self.selected_upload_targets():
            return True
        self.preflight_text.setPlainText("Select at least one upload target (Hugging Face, Kaggle, or Google Drive).")
        return False

    def check_before_run(self) -> None:
        if not self._require_targets():
            return
        payload = self.runner(self._args("preflight"), self._env())
        self.preflight_text.setPlainText(format_checks(payload))

    def refresh_status(self) -> None:
        if not self._require_targets():
            return
        self._remote = False
        payload = self.runner(self._args("status"), self._env())
        self._show_items(status_items(payload))
        self.status_label.setText("This machine")

    def retry_model(self, model_id: str) -> None:
        if not model_id or not self._require_targets():
            return
        stage = self._stage_of(model_id)
        command = retry_stage(stage)
        clear_from_stage(Layout(self.forge_out_path()), model_id, command)
        self.runner(self._args(command, model_id), self._env())
        self.refresh_status()

    def store_tokens(self) -> None:
        fields = (
            ("HF_TOKEN", self.token_hf),
            ("KAGGLE_API_TOKEN", self.token_kaggle),
            ("FORGE_DRIVE_TOKEN", self.token_drive),
        )
        for name, field in fields:
            value = field.text().strip()
            if value:
                self.store.set(name, value)
            field.clear()
        self.status_label.setText("Tokens are in the OS keyring.")

    def attach_remote(self) -> None:
        host = self.host_edit.text().strip()
        user = self.user_edit.text().strip()
        key = self.key_edit.text().strip()
        if not host or not user or not key:
            self.status_label.setText("Host, user, and key are required to attach.")
            return
        self.store.set("ssh_host", host)
        self.store.set("ssh_user", user)
        self.store.set("ssh_key", key)
        self._remote = True
        self.poll_remote()

    def poll_remote(self) -> None:
        if not self._remote:
            return
        payload = self.ssh(
            self.store.get("ssh_host"),
            self.store.get("ssh_user"),
            self.store.get("ssh_key"),
            self.remote_out_edit.text().strip(),
        )
        self._show_items(status_items(payload))
        self.status_label.setText("Remote run")

    def refresh_gauges(self) -> None:
        disk_total, disk_free = self.probes.disk()
        _paint_gauge(self.disk_label, self.disk_bar, "Disk", disk_total, disk_total - disk_free)
        ram_total, ram_free = self.probes.ram()
        _paint_gauge(self.ram_label, self.ram_bar, "RAM", ram_total, ram_total - ram_free)
        vram = self.probes.vram()
        if vram is None:
            self.vram_bar.setValue(0)
            self.vram_label.setText("VRAM unavailable")
            return
        vram_total, vram_used = vram
        _paint_gauge(self.vram_label, self.vram_bar, "VRAM", vram_total, vram_used)

    def show_about(self) -> QDialog:
        dialog = QDialog(self)
        dialog.setObjectName("root")
        dialog.setWindowTitle("About Model Forge")
        dialog.setStyleSheet(self.styleSheet())
        body = QVBoxLayout(dialog)
        heading = QLabel("Model Forge")
        heading.setObjectName("title")
        version = QLabel(version_line())
        version.setObjectName("aboutVersion")
        body.addWidget(heading)
        body.addWidget(version)
        dialog.resize(440, 160)
        dialog.show()
        self._about = dialog
        return dialog

    def show_preview(self) -> None:
        """Fill the window with public plan ids for an offscreen screenshot. No tokens or hosts."""
        self.preflight_text.setPlainText(
            "disk: FAILED — need 80.0 GiB for the plan, have 12.0 GiB\n"
            "Fix: Free disk space or remove files from the plan until the free space covers the download."
        )
        self._show_items(
            [
                {
                    "id": "gemma4-31b",
                    "stage": "quantize",
                    "pct": 42,
                    "last_heartbeat": "2026-10-10T18:00:00Z",
                    "state": "RUNNING",
                    "uploads": {"hf": "standby", "kaggle": "standby", "drive": "standby"},
                },
                {
                    "id": "t-lite-it-2.1",
                    "stage": "eval",
                    "pct": 10,
                    "last_heartbeat": "2026-10-10T18:04:00Z",
                    "state": "STAGEFAIL eval",
                    "error": "eval diverged",
                    "uploads": {"hf": "failed", "kaggle": "failed", "drive": "failed"},
                },
            ]
        )
        self.status_label.setText("This machine")

    def _load_remote_fields(self) -> None:
        self.host_edit.setText(self.store.get("ssh_host"))
        self.user_edit.setText(self.store.get("ssh_user"))
        self.key_edit.setText(self.store.get("ssh_key"))

    def _stage_of(self, model_id: str) -> str:
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.text() == model_id:
                stage = item.data(Qt.ItemDataRole.UserRole)
                return str(stage or "")
        return ""

    def _show_items(self, items: list[dict[str, object]]) -> None:
        self.table.setRowCount(0)
        for item in items:
            self._add_row(item)

    def _add_row(self, item: dict[str, object]) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        model_id = str(item.get("id", ""))
        stage = str(item.get("stage", ""))
        model_item = _cell(model_id)
        model_item.setData(Qt.ItemDataRole.UserRole, stage)
        self.table.setItem(row, 0, model_item)
        self.table.setItem(row, 1, _cell(stage))
        bar = QProgressBar()
        pct = int(item.get("pct", 0) or 0)
        bar.setRange(0, 100)
        bar.setValue(max(0, min(100, pct)))
        bar.setFormat(f"{pct}%")
        self.table.setCellWidget(row, 2, bar)
        self.table.setItem(row, 3, _cell(str(item.get("last_heartbeat", ""))))
        label, failed = state_badge(str(item.get("state", "")), str(item.get("error", "")))
        self.table.setCellWidget(row, 4, _badge(label, failed=failed))
        uploads = item.get("uploads")
        upload_map = uploads if isinstance(uploads, dict) else {}
        for name, column in _UPLOAD_COLUMNS:
            text = str(upload_map.get(name, "—"))
            if text == "failed":
                self.table.setCellWidget(row, column, _badge(text, failed=True))
            else:
                self.table.setItem(row, column, _cell(text))
        button = QPushButton("Retry from this stage")
        button.clicked.connect(lambda _checked=False, ident=model_id: self.retry_model(ident))
        self.table.setCellWidget(row, 8, button)
        self.table.setRowHeight(row, 36)


def _gauge_widgets() -> tuple[QLabel, QProgressBar]:
    label = QLabel("—")
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setTextVisible(True)
    return label, bar


def _paint_gauge(label: QLabel, bar: QProgressBar, title: str, total: int, used: int) -> None:
    bar.setValue(gauge_percent(total, used))
    label.setText(gauge_text(title, total, used))


def _badge(text: str, *, failed: bool) -> QLabel:
    label = QLabel(text)
    label.setObjectName("badge")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    if failed:
        label.setProperty("state", "failed")
        label.setStyleSheet(
            f"color: {FAILED_TEXT}; background: {FAILED_BG}; padding: 2px 8px; border-radius: 8px; font-weight: 700;"
        )
    return label


def _cell(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
    return item
