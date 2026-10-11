"""Model Forge window. Buttons call forge --json or a library helper; they do not reimplement a stage."""

from __future__ import annotations

import os
import shlex
import sys
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
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
from voxprint_common.ui.theme import RED, build_style

from voxprint_forge.accounts import LiveAccounts
from voxprint_forge.gui.backdrop import apply_backdrop
from voxprint_forge.gui.client import (
    ForgeRun,
    SubprocessForge,
    forge_args,
    format_checks,
    ssh_status,
    ssh_upload,
    state_badge,
    status_items,
)
from voxprint_forge.gui.gauges import GaugeProbes, disk_total_free, gauge_percent, gauge_text, vram_status
from voxprint_forge.gui.icons import load_icon
from voxprint_forge.gui.secrets import SecretStore, open_store, token_env
from voxprint_forge.layout import Layout
from voxprint_forge.pipeline import clear_from_stage, retry_stage
from voxprint_forge.preflight import memory_status
from voxprint_forge.version import version_line

# Dark text on the suite error red. The failed chip is not a theme token.
FAILED_TEXT = "#2a0a0a"
FAILED_BG = RED
# Human-facing name. Package ids, slugs, and data paths stay lower-case.
DISPLAY_NAME = "VOXPrint AI Model Forge"

SshPoll = Callable[[str, str, str, str], dict[str, object]]
VmUpload = Callable[[str, str, str, str, dict[str, str]], str]
_COLUMNS = ("Model", "Stage", "Progress", "Last activity", "State", "HF", "Kaggle", "Drive", "Retry")
_UPLOAD_COLUMNS = (("hf", 5), ("kaggle", 6), ("drive", 7))


class MainWindow(QMainWindow):
    """The Model Forge window."""
    def __init__(
        self,
        *,
        runner: ForgeRun | None = None,
        ssh: SshPoll | None = None,
        store: SecretStore | None = None,
        probes: GaugeProbes | None = None,
        accounts: LiveAccounts | None = None,
        uploader: VmUpload | None = None,
    ) -> None:
        super().__init__()
        self.runner = runner or SubprocessForge()
        self.ssh = ssh or ssh_status
        self.store = store or open_store()
        self.accounts = accounts or LiveAccounts(self.store)
        self.uploader = uploader or ssh_upload
        self._remote = False
        self._about: QDialog | None = None
        self.setWindowTitle(DISPLAY_NAME)
        self.setObjectName("root")
        self.resize(1100, 760)
        self._build()
        self.probes = probes or GaugeProbes(
            disk=lambda: disk_total_free(self.forge_out_path()),
            ram=memory_status,
            vram=vram_status,
        )
        self._load_remote_fields()
        self._paint_accounts()
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
        title = QLabel(DISPLAY_NAME)
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
        self.upload_vm_button = QPushButton("Upload on VM")
        self.upload_vm_button.setToolTip("Pass keyring tokens only for this upload, then unset them on the VM")
        self.upload_vm_button.clicked.connect(self.upload_on_vm)
        remote.addWidget(QLabel("Remote"))
        remote.addWidget(self.host_edit)
        remote.addWidget(self.user_edit)
        remote.addWidget(self.key_edit, 2)
        remote.addWidget(self.remote_out_edit)
        remote.addWidget(self.attach_button)
        remote.addWidget(self.upload_vm_button)
        layout.addLayout(remote)

        layout.addWidget(QLabel("Accounts"))
        self.token_hf = QLineEdit()
        self.token_kaggle = QLineEdit()
        self.token_drive = QLineEdit()
        self.hf_account = QLabel("not connected")
        self.kaggle_account = QLabel("not connected")
        self.drive_account = QLabel("not connected")
        self._account_labels = {"hf": self.hf_account, "kaggle": self.kaggle_account, "drive": self.drive_account}
        layout.addLayout(
            self._account_row(
                "Hugging Face",
                self.hf_account,
                self.token_hf,
                "HF token",
                "hf",
                browser=True,
            )
        )
        layout.addLayout(
            self._account_row(
                "Kaggle",
                self.kaggle_account,
                self.token_kaggle,
                "Kaggle KGAT token",
                "kaggle",
                browser=False,
            )
        )
        layout.addLayout(
            self._account_row(
                "Google Drive",
                self.drive_account,
                self.token_drive,
                "rclone token JSON",
                "drive",
                browser=True,
            )
        )

        self.status_label = QLabel("This machine")
        self.status_label.setObjectName("subtitle")
        layout.addWidget(self.status_label)

    def apply_theme(self) -> str:
        """Apply the suite theme and return the theme name."""
        offscreen = os.environ.get("QT_QPA_PLATFORM", "") == "offscreen"
        if offscreen or sys.platform != "win32":
            self.setStyleSheet(build_style(False))
            self.backdrop = "plain"
            return self.backdrop
        self.backdrop = apply_backdrop(int(self.winId()))
        self.setStyleSheet(build_style(self.backdrop == "acrylic"))
        return self.backdrop

    def forge_out_path(self) -> Path:
        """Forge-out directory from the path field."""
        text = self.out_edit.text().strip()
        if text:
            return Path(text)
        return Path.home() / "forge-out"

    def selected_upload_targets(self) -> tuple[str, ...]:
        """Upload targets checked in the window."""
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
        """Run preflight and show the report."""
        if not self._require_targets():
            return
        payload = self.runner(self._args("preflight"), self._env())
        self.preflight_text.setPlainText(format_checks(payload))

    def refresh_status(self) -> None:
        """Reload ``forge status`` into the model table."""
        if not self._require_targets():
            return
        self._remote = False
        payload = self.runner(self._args("status"), self._env())
        self._show_items(status_items(payload))
        self.status_label.setText("This machine")

    def retry_model(self, model_id: str) -> None:
        """Clear this model's stage and start that stage again."""
        if not model_id or not self._require_targets():
            return
        stage = self._stage_of(model_id)
        command = retry_stage(stage)
        clear_from_stage(Layout(self.forge_out_path()), model_id, command)
        self.runner(self._args(command, model_id), self._env())
        self.refresh_status()

    def _account_row(
        self,
        title: str,
        name: QLabel,
        field: QLineEdit,
        placeholder: str,
        target: str,
        *,
        browser: bool,
    ) -> QHBoxLayout:
        row = QHBoxLayout()
        row.addWidget(QLabel(title))
        name.setObjectName(f"account-{target}")
        row.addWidget(name)
        field.setPlaceholderText(placeholder)
        field.setEchoMode(QLineEdit.EchoMode.Password)
        row.addWidget(field, 1)
        save = QPushButton("Save token")
        save.clicked.connect(self.store_tokens)
        row.addWidget(save)
        if browser:
            login = QPushButton("Browser login")
            login.clicked.connect(lambda _checked=False, chosen=target: self.browser_login(chosen))
            row.addWidget(login)
        test = QPushButton("Test connection")
        test.clicked.connect(lambda _checked=False, chosen=target: self.test_connection(chosen))
        row.addWidget(test)
        return row

    def _paint_accounts(self) -> None:
        for row in self.accounts.list_accounts():
            target = str(row.get("target", ""))
            label = self._account_labels.get(target)
            if label is None:
                continue
            if row.get("connected"):
                label.setText(str(row.get("name") or "connected"))
            else:
                label.setText("not connected")

    def store_tokens(self) -> None:
        """Save the token fields into the secret store."""
        fields = (
            ("hf", self.token_hf),
            ("kaggle", self.token_kaggle),
            ("drive", self.token_drive),
        )
        for target, field in fields:
            value = field.text().strip()
            if value:
                self.accounts.add_token(target, value)
            field.clear()
        self._paint_accounts()
        self.status_label.setText("Tokens are in the OS keyring.")

    def test_connection(self, target: str) -> None:
        try:
            row = self.accounts.test(target)
        except Exception as exc:
            self.status_label.setText(str(exc))
            return
        self._paint_accounts()
        self.status_label.setText(f"{target} connected as {row.get('name', '')}")

    def browser_login(self, target: str) -> None:
        def on_code(uri: str, code: str) -> None:
            self.status_label.setText(f"Open {uri} and enter {code}")
            app = QApplication.instance()
            if app is not None:
                app.processEvents()

        try:
            row = self.accounts.login_browser(target, on_code)
        except Exception as exc:
            self.status_label.setText(str(exc))
            return
        self._paint_accounts()
        self.status_label.setText(f"{target} connected as {row.get('name', '')}")

    def upload_on_vm(self) -> None:
        row = self.table.currentRow()
        item = self.table.item(row, 0) if row >= 0 else None
        if item is None or not item.text().strip():
            self.status_label.setText("Select a model to upload on the VM.")
            return
        host = self.store.get("ssh_host").strip()
        user = self.store.get("ssh_user").strip()
        key = self.store.get("ssh_key").strip()
        if not host or not user or not key:
            self.status_label.setText("Host, user, and key are required to upload.")
            return
        model = item.text().strip()
        plan = shlex.quote(self.plan_edit.text().strip() or "plans/plan.toml")
        out = shlex.quote(self.remote_out_edit.text().strip() or str(Path.home() / "forge-out"))
        command = f"forge --json --plan {plan} --forge-out {out} upload --model {shlex.quote(model)}"
        secret_names = ("HF_TOKEN", "KAGGLE_API_TOKEN", "FORGE_DRIVE_TOKEN")
        secrets = {name: self.store.get(name) for name in secret_names if self.store.get(name).strip()}
        try:
            self.uploader(host, user, key, command, secrets)
        except Exception as exc:
            self.status_label.setText(str(exc))
            return
        self.status_label.setText("VM upload finished. Remote tokens were unset.")

    def attach_remote(self) -> None:
        """Open the SSH session selected in the window."""
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
        """Read remote status and GPU numbers."""
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
        """Repaint RAM, VRAM, and disk gauges."""
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
        """Open the About dialog and return it."""
        dialog = QDialog(self)
        dialog.setObjectName("root")
        dialog.setWindowTitle(f"About {DISPLAY_NAME}")
        dialog.setStyleSheet(self.styleSheet())
        body = QVBoxLayout(dialog)
        heading = QLabel(DISPLAY_NAME)
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
