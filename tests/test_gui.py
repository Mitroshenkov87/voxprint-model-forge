"""Offscreen PySide6 coverage for the Model Forge window. Forge output is mocked."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PySide6.QtWidgets import QLabel
from voxprint_common.ui.splash import display_name
from voxprint_common.ui.theme import CONTRAST_PAIRS, RED, build_style, contrast_ratio

from voxprint_forge.gui.__main__ import IDENTITY_COLOUR, open_splash, splash_art
from voxprint_forge.gui.backdrop import apply_backdrop
from voxprint_forge.gui.client import ssh_status, state_badge
from voxprint_forge.gui.gauges import GaugeProbes
from voxprint_forge.gui.secrets import MemoryStore
from voxprint_forge.gui.window import DISPLAY_NAME, MainWindow
from voxprint_forge.layout import Layout, write_marker
from voxprint_forge.preflight import memory_status
from voxprint_forge.upload import selected_targets
from voxprint_forge.version import version_line

_GIB = 1024**3
_FIX = "Free disk space or remove files from the plan until the free space covers the download."


class FakeForge:
    def __init__(self, responses: dict[str, dict[str, object]]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[str], dict[str, str]]] = []

    def __call__(self, args: list[str], env: dict[str, str]) -> dict[str, object]:
        self.calls.append((list(args), dict(env)))
        return self.responses[_command(args)]


def _command(args: list[str]) -> str:
    names = ("preflight", "status", "fetch", "abliterate", "quantize", "fit", "eval", "card", "publish-prep", "upload")
    for name in names:
        if name in args:
            return name
    return args[-1]


def _probes(*, vram: tuple[int, int] | None = (16 * _GIB, 4 * _GIB)) -> GaugeProbes:
    return GaugeProbes(
        disk=lambda: (100 * _GIB, 40 * _GIB),
        ram=lambda: (64 * _GIB, 20 * _GIB),
        vram=lambda: vram,
    )


def _window(qtbot: pytest.fixture, runner: FakeForge, **kwargs: object) -> MainWindow:
    store = kwargs.pop("store", MemoryStore())
    probes = kwargs.pop("probes", _probes())
    window = MainWindow(runner=runner, probes=probes, store=store, **kwargs)  # type: ignore[arg-type]
    window._timer.stop()
    qtbot.addWidget(window)
    window.show()
    return window


def _status_row(**extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "id": "t-lite-it-2.1",
        "stage": "eval",
        "pct": 10,
        "last_heartbeat": "2026-10-10T18:04:00Z",
        "state": "STAGEFAIL eval",
        "error": "eval diverged",
        "uploads": {"hf": "failed", "kaggle": "off", "drive": "standby"},
    }
    row.update(extra)
    return {
        "ok": False,
        "command": "status",
        "exit_code": 4,
        "result": {"items": [row], "failed": True, "exit_code": 4},
    }


def test_memory_status_reports_total_and_available() -> None:
    total, available = memory_status()
    assert total > 0
    assert 0 <= available <= total


def test_theme_contrast_meets_wcag_aa() -> None:
    for foreground, background in CONTRAST_PAIRS:
        assert contrast_ratio(foreground, background) >= 4.5, (foreground, background)
    style = build_style(False)
    assert "#17171c" in style
    assert "#202029" in style
    assert "#f5d76e" in style
    assert "#60a5fa" in style
    assert "#f2f2f5" in style
    assert "QPushButton#gear" in style
    assert "QPushButton#disclosure" in style
    assert contrast_ratio("#2a0a0a", RED) >= 4.5


def test_splash_uses_the_approved_art(qtbot: pytest.fixture) -> None:
    art = splash_art()
    assert art.name == "splash.jpg"
    assert art.is_file()
    assert IDENTITY_COLOUR == "#a06020"
    splash = open_splash()
    assert display_name(splash.build_info.app) == DISPLAY_NAME
    splash.set_stage("Opening Model Forge")
    splash.set_progress(1)
    splash.finish()
    del qtbot


def test_window_title_uses_the_display_name(qtbot: pytest.fixture) -> None:
    window = _window(qtbot, FakeForge({}))
    assert window.windowTitle() == DISPLAY_NAME
    title = window.findChild(QLabel, "title")
    assert title is not None
    assert title.text() == DISPLAY_NAME
    dialog = window.show_about()
    assert dialog.windowTitle() == f"About {DISPLAY_NAME}"
    heading = dialog.findChild(QLabel, "title")
    assert heading is not None
    assert heading.text() == DISPLAY_NAME


def test_icons_have_no_floppy_disk() -> None:
    assets = Path("src/voxprint_forge/gui/assets")
    names = {path.name for path in assets.glob("*.svg")}
    assert names == {"download.svg", "export.svg"}
    for path in assets.glob("*.svg"):
        assert "floppy" not in path.read_text(encoding="utf-8").lower()


def test_backdrop_is_plain_off_windows() -> None:
    assert apply_backdrop(1) == "plain"


def test_state_badge_never_says_done() -> None:
    assert state_badge("STAGEFAIL eval", "eval diverged") == ("FAILED", True)
    assert state_badge("done") == ("FAILED", True)
    assert state_badge("RUNNING") == ("RUNNING", False)
    assert state_badge("UPLOADED") == ("UPLOADED", False)


def test_selected_targets_reads_the_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_UPLOAD_TARGETS", "drive, hf")
    assert selected_targets() == ("drive", "hf")


def test_window_uses_the_plain_dark_theme(qtbot: pytest.fixture) -> None:
    window = _window(qtbot, FakeForge({}))
    assert window.backdrop == "plain"
    sheet = window.styleSheet()
    assert "#17171c" in sheet
    assert "rgba(16,16,22,222)" not in sheet


def test_check_before_run_shows_a_fix(qtbot: pytest.fixture) -> None:
    payload = {
        "ok": False,
        "command": "preflight",
        "exit_code": 6,
        "result": {
            "ok": False,
            "checks": [
                {"name": "disk", "ok": False, "detail": "need 80.0 GiB for the plan, have 12.0 GiB", "fix": _FIX},
                {"name": "imports", "ok": True, "detail": "torch, transformers, and gguf import", "fix": ""},
            ],
        },
    }
    forge = FakeForge({"preflight": payload})
    window = _window(qtbot, forge)
    window.check_button.click()
    text = window.preflight_text.toPlainText()
    assert "disk: FAILED" in text
    assert f"Fix: {_FIX}" in text
    assert "imports: ok" in text
    assert forge.calls[0][0][-1] == "preflight"
    assert "HF_TOKEN" not in forge.calls[0][1]


def test_failed_row_is_obvious(qtbot: pytest.fixture) -> None:
    forge = FakeForge({"status": _status_row()})
    window = _window(qtbot, forge)
    window.refresh_button.click()
    assert window.table.item(0, 0).text() == "t-lite-it-2.1"
    assert window.table.item(0, 1).text() == "eval"
    assert window.table.cellWidget(0, 2).value() == 10
    assert window.table.item(0, 3).text() == "2026-10-10T18:04:00Z"
    state = window.table.cellWidget(0, 4)
    assert state.text() == "FAILED"
    assert state.property("state") == "failed"
    assert "done" not in state.text().lower()
    assert window.table.cellWidget(0, 5).text() == "failed"
    assert window.table.item(0, 6).text() == "off"
    assert window.table.item(0, 7).text() == "standby"


def test_retry_from_this_stage(qtbot: pytest.fixture, tmp_path: Path) -> None:
    forge_out = tmp_path / "out"
    layout = Layout(forge_out)
    write_marker(layout.stage("gemma4-31b", "fetch"), {"ok": True})
    write_marker(layout.stage("gemma4-31b", "quantize"), {"ok": True})
    write_marker(layout.stage("gemma4-31b", "eval"), {"ok": True})
    write_marker(layout.stage("gemma4-31b", "upload"), {"ok": True})
    row = _status_row(id="gemma4-31b", stage="eval")
    forge = FakeForge({"status": row, "eval": {"ok": True, "command": "eval", "exit_code": 0, "result": {}}})
    window = _window(qtbot, forge)
    window.out_edit.setText(str(forge_out))
    window.refresh_button.click()
    window.table.cellWidget(0, 8).click()
    assert layout.stage("gemma4-31b", "fetch").is_file()
    assert layout.stage("gemma4-31b", "quantize").is_file()
    assert not layout.stage("gemma4-31b", "eval").is_file()
    assert not layout.stage("gemma4-31b", "upload").is_file()
    retried = [args for args, _env in forge.calls if _command(args) == "eval"]
    assert retried
    assert retried[0][retried[0].index("--model") + 1] == "gemma4-31b"


def test_upload_target_selector(qtbot: pytest.fixture) -> None:
    forge = FakeForge({"status": _status_row()})
    window = _window(qtbot, forge)
    window.kaggle_box.setChecked(False)
    window.drive_box.setChecked(False)
    window.refresh_button.click()
    assert forge.calls[-1][1]["FORGE_UPLOAD_TARGETS"] == "hf"
    assert window.table.cellWidget(0, 5).text() == "failed"


def test_live_gauges(qtbot: pytest.fixture) -> None:
    window = _window(qtbot, FakeForge({}))
    window.refresh_gauges()
    assert window.disk_label.text() == "Disk  40.0 GiB free of 100.0 GiB"
    assert window.disk_bar.value() == 60
    assert window.ram_label.text() == "RAM  20.0 GiB free of 64.0 GiB"
    assert window.vram_label.text() == "VRAM  12.0 GiB free of 16.0 GiB"
    assert window.vram_bar.value() == 25


def test_vram_gauge_when_nvml_is_absent(qtbot: pytest.fixture) -> None:
    window = _window(qtbot, FakeForge({}), probes=_probes(vram=None))
    window.refresh_gauges()
    assert window.vram_label.text() == "VRAM unavailable"
    assert window.vram_bar.value() == 0


def test_remote_run_polls_ssh(qtbot: pytest.fixture, tmp_path: Path) -> None:
    seen: dict[str, str] = {}

    def ssh(host: str, user: str, key: str, forge_out: str) -> dict[str, object]:
        seen.update(host=host, user=user, key=key, forge_out=forge_out)
        return _status_row(id="gemma4-31b", stage="fit", state="RUNNING", error="", pct=5)

    store = MemoryStore()
    window = _window(qtbot, FakeForge({}), ssh=ssh, store=store)
    window.host_edit.setText("vm.example")
    window.user_edit.setText("forge")
    window.key_edit.setText(str(tmp_path / "id_ed25519"))
    window.remote_out_edit.setText("/var/forge-out")
    window.attach_button.click()
    assert seen == {
        "host": "vm.example",
        "user": "forge",
        "key": str(tmp_path / "id_ed25519"),
        "forge_out": "/var/forge-out",
    }
    assert store.get("ssh_host") == "vm.example"
    assert store.get("ssh_key") == str(tmp_path / "id_ed25519")
    assert window.status_label.text() == "Remote run"
    assert window.table.item(0, 1).text() == "fit"
    assert window.table.cellWidget(0, 4).text() == "RUNNING"


def test_ssh_status_argv() -> None:
    seen: dict[str, list[str]] = {}

    def run(argv: list[str]) -> str:
        seen["argv"] = argv
        return json.dumps({"ok": True, "command": "status", "result": {"items": []}})

    ssh_status("vm.example", "forge", "/keys/id_ed25519", "/var/forge-out", run=run)
    argv = seen["argv"]
    assert argv[:6] == ["ssh", "-i", "/keys/id_ed25519", "-o", "BatchMode=yes", "-o"]
    assert "StrictHostKeyChecking=accept-new" in argv
    assert argv[-2] == "forge@vm.example"
    assert argv[-1] == "forge --json --forge-out /var/forge-out status"


def test_tokens_go_to_the_store_only(qtbot: pytest.fixture) -> None:
    token = "hf-gui-only-keyring-token"
    store = MemoryStore()
    forge = FakeForge(
        {"preflight": {"ok": True, "command": "preflight", "exit_code": 0, "result": {"ok": True, "checks": []}}}
    )
    window = _window(qtbot, forge, store=store)
    window.token_hf.setText(token)
    window.token_kaggle.setText("kaggle-gui-only-keyring-token")
    window.store_tokens()
    assert store.get("HF_TOKEN") == token
    assert window.token_hf.text() == ""
    assert window.status_label.text() == "Tokens are in the OS keyring."
    window.check_button.click()
    assert forge.calls[-1][1]["HF_TOKEN"] == token
    for folder in ("src", "plans", "docs", ".github"):
        for path in Path(folder).rglob("*"):
            if path.is_file():
                assert token.encode() not in path.read_bytes()


def test_accounts_show_name_and_test_connection(qtbot: pytest.fixture) -> None:
    store = MemoryStore()
    window = _window(qtbot, FakeForge({}), store=store)
    window.token_hf.setText("hf-account-token")
    window.store_tokens()
    assert store.get("HF_TOKEN") == "hf-account-token"
    assert window.token_hf.text() == ""
    assert window.hf_account.text() == "connected"

    def identify(token: str) -> str:
        assert token == "hf-account-token"
        return "ada"

    window.accounts.probes = {"hf": identify, "kaggle": identify, "drive": identify}
    window.test_connection("hf")
    assert window.hf_account.text() == "ada"
    assert window.status_label.text() == "hf connected as ada"
    window.accounts.remove("hf")
    window._paint_accounts()
    assert window.hf_account.text() == "not connected"
    assert store.get("HF_TOKEN") == ""


def test_vm_upload_sends_tokens_on_stdin_only(qtbot: pytest.fixture) -> None:
    store = MemoryStore()
    store.set("ssh_host", "vm.example")
    store.set("ssh_user", "forge")
    store.set("ssh_key", "/keys/id_ed25519")
    store.set("HF_TOKEN", "hf-vm-upload-token")
    seen: dict[str, object] = {}

    def upload(host: str, user: str, key: str, command: str, secrets: dict[str, str]) -> str:
        seen["host"] = host
        seen["command"] = command
        seen["secrets"] = secrets
        assert "hf-vm-upload-token" not in command
        return ""

    window = _window(qtbot, FakeForge({}), store=store, uploader=upload)
    window.refresh_status = lambda: None  # type: ignore[method-assign]
    payload = _status_row()
    window._show_items(payload["result"]["items"])  # type: ignore[index]
    window.table.selectRow(0)
    window.remote_out_edit.setText("/var/forge-out")
    window.upload_on_vm()
    assert seen["host"] == "vm.example"
    assert "upload --model t-lite-it-2.1" in str(seen["command"])
    assert seen["secrets"] == {"HF_TOKEN": "hf-vm-upload-token"}
    assert window.status_label.text() == "VM upload finished. Remote tokens were unset."


def test_about_shows_the_release_line(qtbot: pytest.fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_RUN_NUMBER", raising=False)
    monkeypatch.delenv("VOXPRINT_BUILD", raising=False)
    window = _window(qtbot, FakeForge({}))
    dialog = window.show_about()
    label = dialog.findChild(QLabel, "aboutVersion")
    assert label is not None
    assert label.text() == version_line()
    assert label.text() == '0.1.0 dev "Charash"'


def test_main_window_screenshot_is_attached() -> None:
    path = Path("docs/screenshots/main-window.png")
    blob = path.read_bytes()
    assert blob.startswith(b"\x89PNG")
    assert len(blob) > 8000
