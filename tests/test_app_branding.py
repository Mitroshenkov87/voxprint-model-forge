"""AppUserModelID, icon path, and the installer shortcut script."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from voxprint_forge.app import branding

ROOT = Path(__file__).resolve().parents[1]
ISS = (ROOT / "installer" / "windows" / "voxprint-model-forge.iss").read_text(encoding="utf-8-sig")
ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def _section(name: str) -> str:
    body = ISS.split(f"[{name}]", 1)[1]
    end = body.find("\n[")
    return body[:end] if end >= 0 else body


def test_app_user_model_id_constant() -> None:
    assert branding.APP_USER_MODEL_ID == "VOXPrint.AISuite.ModelForge"
    assert branding.ICON_NAME == "voxprint-model-forge.ico"


def test_set_app_user_model_id_is_a_noop_off_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(branding.sys, "platform", "linux")
    monkeypatch.setattr(branding, "suite_branding", object())
    assert branding.set_app_user_model_id() is False


def test_windows_uses_suite_branding_when_it_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(branding.sys, "platform", "win32")
    seen: list[str] = []

    class _Suite:
        @staticmethod
        def set_app_user_model_id(app_id: str) -> bool:
            seen.append(app_id)
            return True

    monkeypatch.setattr(branding, "suite_branding", _Suite)
    assert branding.set_app_user_model_id() is True
    assert seen == [branding.APP_USER_MODEL_ID]


def test_windows_falls_back_to_shell32(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(branding.sys, "platform", "win32")
    monkeypatch.setattr(branding, "suite_branding", None)
    seen: list[str] = []

    class _Shell32:
        def SetCurrentProcessExplicitAppUserModelID(self, app_id: str) -> int:
            seen.append(app_id)
            return 0

    class _Windll:
        shell32 = _Shell32()

    monkeypatch.setattr(branding.ctypes, "windll", _Windll(), raising=False)
    monkeypatch.setattr(branding.ctypes, "c_wchar_p", lambda value: value, raising=False)
    assert branding.set_app_user_model_id("VOXPrint.AISuite.ModelForge") is True
    assert seen == ["VOXPrint.AISuite.ModelForge"]


def test_icon_path_prefers_the_install_directory(tmp_path: Path) -> None:
    install = tmp_path / "VOXPrint AI Suite" / "Model Forge"
    exe = install / "venv" / "Scripts" / "pythonw.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    icon = install / "assets" / branding.ICON_NAME
    icon.parent.mkdir()
    icon.write_bytes(b"installed")
    repo = tmp_path / "repo" / "assets" / branding.ICON_NAME
    repo.parent.mkdir(parents=True)
    repo.write_bytes(b"repo")
    assert branding.icon_path(exe, source_root=tmp_path / "repo") == icon


def test_icon_path_falls_back_to_the_repository(tmp_path: Path) -> None:
    exe = tmp_path / "python"
    exe.write_bytes(b"")
    icon = tmp_path / "repo" / "assets" / branding.ICON_NAME
    icon.parent.mkdir(parents=True)
    icon.write_bytes(b"repo")
    assert branding.icon_path(exe, source_root=tmp_path / "repo") == icon


def test_icon_path_is_skipped_when_neither_file_exists(tmp_path: Path) -> None:
    exe = tmp_path / "bin" / "python"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    assert branding.icon_path(exe, source_root=tmp_path / "empty") is None


def test_master_and_icon_frames() -> None:
    master = (ROOT / "assets" / "voxprint-model-forge.png").read_bytes()
    assert master[12:16] == b"IHDR"
    assert struct.unpack(">II", master[16:24]) == (1024, 1024)
    data = (ROOT / "assets" / "voxprint-model-forge.ico").read_bytes()
    reserved, kind, count = struct.unpack("<HHH", data[:6])
    assert (reserved, kind, count) == (0, 1, len(ICON_SIZES))
    sizes = []
    for index in range(count):
        width = data[6 + 16 * index]
        height = data[7 + 16 * index]
        sizes.append(width or 256)
        assert height == width
    assert tuple(sizes) == ICON_SIZES
    for size in ICON_SIZES:
        frame = (ROOT / "assets" / "icon" / "frames" / f"{size}.png").read_bytes()
        assert struct.unpack(">II", frame[16:24]) == (size, size)


def test_installer_shortcuts_follow_the_suite_standard() -> None:
    icons = [line.strip() for line in _section("Icons").splitlines() if line.startswith("Name:")]
    assert icons == [
        'Name: "{group}\\{#MyAppName}"; Filename: "{app}\\voxprint-model-forge.exe"; '
        'WorkingDir: "{app}"; IconFilename: "{app}\\assets\\{#MyIcon}"; AppUserModelID: "{#MyAppId}"',
        'Name: "{autodesktop}\\{#MyAppName}"; Filename: "{app}\\voxprint-model-forge.exe"; '
        'WorkingDir: "{app}"; IconFilename: "{app}\\assets\\{#MyIcon}"; AppUserModelID: "{#MyAppId}"; '
        "Tasks: desktopicon",
    ]
    assert "{uninstallexe}" not in _section("Icons")
    tasks = _section("Tasks")
    assert 'Name: desktopicon; Description: "{cm:CreateDesktopIcon}"' in tasks
    assert "Flags: unchecked" in tasks
    setup = _section("Setup")
    assert "SetupIconFile=..\\..\\assets\\{#MyIcon}" in setup
    assert "UninstallDisplayIcon={app}\\assets\\{#MyIcon}" in setup
    assert "UninstallDisplayName={#MyAppName}" in setup
    deleted = _section("InstallDelete")
    for name in (
        '"{group}\\Uninstall {#MyAppName}.lnk"',
        '"{group}\\Model Forge guide (EN).lnk"',
        '"{group}\\Руководство Model Forge (RU).lnk"',
    ):
        assert name in deleted
    assert "{group}" not in _section("UninstallDelete")
    assert "/target:winexe" in ISS and "/win32icon:" in ISS
    assert "launcher.cs did not compile" in ISS
