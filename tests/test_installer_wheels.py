"""Windows installer wheel selection when an older setup left wheels behind."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL = (ROOT / "installer" / "windows" / "install.ps1").read_text(encoding="utf-8")
ISS = (ROOT / "installer" / "windows" / "voxprint-model-forge.iss").read_text(encoding="utf-8-sig")
SMOKE = (ROOT / "scripts" / "windows_smoke.ps1").read_text(encoding="utf-8")
WORKFLOW = (ROOT / ".github" / "workflows" / "installer.yml").read_text(encoding="utf-8")


def _section(name: str) -> str:
    body = ISS.split(f"[{name}]", 1)[1]
    end = body.find("\n[")
    return body[:end] if end >= 0 else body


def _pinned_common() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    found = set(re.findall(r"voxprint-common(?:\[[^\]]*\])?\s*==\s*([0-9][0-9A-Za-z.]*)", text))
    assert len(found) == 1
    return next(iter(found))


def test_install_ps1_selects_app_wheels_by_exact_version() -> None:
    """App wheels are selected by recorded version, and a count other than one fails."""
    assert not re.search(r"voxprint_common-\*\.whl", INSTALL)
    assert "Select-Object -First 1" not in INSTALL
    assert "exactly one" in INSTALL
    assert _pinned_common() not in INSTALL


def test_stale_wheels_are_removed_only_after_the_copy() -> None:
    """New wheels are staged, and stale ones are deleted only after ssPostInstall."""
    deleted = _section("InstallDelete")
    assert "wheels" not in deleted
    files = _section("Files")
    assert r'Source: "wheels-incoming\*"; DestDir: "{app}\wheels.new"' in files
    wheel_dirs = [line for line in files.splitlines() if "DestDir:" in line and "wheels" in line]
    assert wheel_dirs == [
        'Source: "wheels-incoming\\*"; DestDir: "{app}\\wheels.new"; '
        "Flags: ignoreversion recursesubdirs createallsubdirs"
    ]
    removed = _section("UninstallDelete")
    assert r'Type: filesandordirs; Name: "{app}\wheels"' in removed
    assert r'Type: filesandordirs; Name: "{app}\wheels.new"' in removed
    post = ISS.split("procedure CurStepChanged", 1)[1]
    assert post.index("PromoteWheels") < post.index("install.ps1")
    promote = ISS.split("function PromoteWheels", 1)[1].split("procedure CurStepChanged", 1)[0]
    assert promote.index("FileCopy") < promote.index("DeleteFile")
    assert "wheels.new" in promote


def test_silent_upgrade_exits_20_when_the_app_is_running() -> None:
    """Interactive setup asks to close the app. Silent setup exits 20 first."""
    setup = _section("Setup")
    assert "CloseApplications=force" in setup
    assert "CloseApplicationsFilter=voxprint-model-forge.exe,voxprint-forge-cli.exe,python.exe,pythonw.exe" in setup
    assert "english.AppBusy=Close VOXPrint Model Forge" in ISS
    assert "russian.AppBusy=Закройте VOXPrint Model Forge" in ISS
    assert "AppBusyExitCode = 20" in ISS
    init = ISS.split("function InitializeSetup", 1)[1].split("function PrepareToInstall", 1)[0]
    assert "SilentInstall" in init and "ExitProcess(AppBusyExitCode)" in init
    prepare = ISS.split("function PrepareToInstall", 1)[1].split("procedure ", 1)[0]
    assert prepare.index("RefuseIfAppBusy") < prepare.index("migrate-home.ps1")
    refuse = ISS.split("function RefuseIfAppBusy", 1)[1].split("function InitializeSetup", 1)[0]
    assert "SilentInstall" in refuse and "ExitProcess(AppBusyExitCode)" in refuse
    silent = ISS.split("function SilentInstall", 1)[1].split("function RefuseIfAppBusy", 1)[0]
    assert "WizardSilent" in silent and "/VERYSILENT" in silent
    assert "CustomMessage('AppBusy')" in refuse
    assert "AppBusyExitCode" in SMOKE and "upgrade-busy" in SMOKE
    testing = (ROOT / "docs" / "TESTING.md").read_text(encoding="utf-8")
    assert "upgrade-busy" in WORKFLOW and "exits 20" in testing


def test_install_log_is_replaced_each_run() -> None:
    """install.ps1 deletes setup-work\\install.log before writing it."""
    body = INSTALL.split("New-Item -ItemType Directory -Force $Work", 1)[1].split("Start-Transcript", 1)[0]
    assert "Remove-Item" in body and "install.log" in body


def test_windows_smoke_plants_stale_wheels_and_checks_the_pin() -> None:
    """The installer smoke plants an older common wheel and checks the venv pin."""
    assert "voxprint_common-0.4.0-py3-none-any.whl" in SMOKE
    assert "voxprint_forge-0.0.9-py3-none-any.whl" in SMOKE
    assert "upgrade-wheels" in SMOKE
    assert "importlib.metadata" in SMOKE
    assert "upgrade-wheels" in WORKFLOW
    assert "wheels-incoming" in WORKFLOW
    assert "vendor\\voxprint_common-" + _pinned_common() not in WORKFLOW
    assert "does not pin exactly one voxprint-common version" in WORKFLOW
