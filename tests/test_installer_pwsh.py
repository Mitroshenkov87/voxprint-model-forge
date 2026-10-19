"""The setup prefers PowerShell 7, offers the pinned official MSI, and falls back to Windows PowerShell 5.1."""

from __future__ import annotations

import re
from pathlib import Path

from voxprint_forge.diffusion import guard

ROOT = Path(__file__).resolve().parents[1]
ISS = (ROOT / "installer" / "windows" / "voxprint-model-forge.iss").read_text(encoding="utf-8-sig")
WORKFLOW = (ROOT / ".github" / "workflows" / "installer.yml").read_text(encoding="utf-8")


def _const(name: str) -> str:
    found = re.search(rf"{name} = '([^']+)'", ISS)
    assert found, name
    return found.group(1)


def test_msi_pin_is_the_official_x64_release() -> None:
    url, name, sha = _const("PwshMsiUrl"), _const("PwshMsiName"), _const("PwshMsiSha256")
    found = re.fullmatch(r"https://github\.com/PowerShell/PowerShell/releases/download/v(7\.\d+\.\d+)/(.+)", url)
    assert found and found.group(2) == name == f"PowerShell-{found.group(1)}-win-x64.msi"
    assert re.fullmatch(r"[0-9a-f]{64}", sha)
    assert "DownloadPage.Add(PwshMsiUrl, PwshMsiName, PwshMsiSha256)" in ISS


def test_msi_installs_silently_and_pwsh_is_resolved_again() -> None:
    assert "/qn /norestart ADD_PATH=1" in ISS
    assert "{commonpf64}\\PowerShell\\7\\pwsh.exe" in ISS
    assert "FileSearch('pwsh.exe', GetEnv('PATH'))" in ISS
    assert "PwshExe := FindPwsh;" in ISS.split("function InstallPwsh")[1].split("procedure InitializeWizard")[0]


def test_offer_is_interactive_only_and_falls_back_to_51() -> None:
    offer = ISS.split("function NextButtonClick")[1].split("function PrepareToInstall")[0]
    assert "WizardSilent" in offer and "MB_YESNO" in offer
    assert "PwshFailed" in offer and "PwshDeclined" in offer
    assert "{sys}\\WindowsPowerShell\\v1.0\\powershell.exe" in ISS.split("function ScriptHost")[1]


def test_scripts_run_hidden_through_the_chosen_host() -> None:
    scripts = ISS.split("if PrefetchPage.Values[0]")[0]
    assert "SW_SHOW" not in scripts
    assert "-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File" in ISS
    assert ISS.count("ScriptHost,") == 2


def test_wizard_text_is_in_english_and_russian() -> None:
    for key in ("PwshCaption", "PwshDescription", "PwshInstalling", "PwshOffer", "PwshDeclined", "PwshFailed"):
        assert re.search(rf"^english\.{key}=.+", ISS, re.M), key
        assert re.search(rf"^russian\.{key}=.+", ISS, re.M), key
    assert "english.PwshCaption=Installing PowerShell 7 (Microsoft)" in ISS
    assert "russian.PwshCaption=Установка PowerShell 7 (Microsoft)" in ISS


def test_workflow_checks_the_msi_pin_and_signature() -> None:
    assert "PwshMsiSha256" in WORKFLOW and "Get-FileHash $msi" in WORKFLOW
    assert "Get-AuthenticodeSignature $msi" in WORKFLOW and "O=Microsoft Corporation" in WORKFLOW
    assert "pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\\windows_smoke.ps1" in WORKFLOW
    assert "shell: powershell" not in WORKFLOW


def test_installer_scripts_still_run_on_51() -> None:
    for name in ("installer/windows/install.ps1", "installer/windows/migrate-home.ps1", "scripts/windows_smoke.ps1"):
        assert (ROOT / name).read_text(encoding="utf-8").startswith("#Requires -Version 5.1\n"), name


def test_guard_prefers_pwsh(monkeypatch) -> None:
    monkeypatch.setattr(guard.shutil, "which", lambda name: "C:/pf/pwsh.exe" if name == "pwsh" else None)
    assert guard.powershell_exe() == "C:/pf/pwsh.exe"
    monkeypatch.setattr(guard.shutil, "which", lambda name: None)
    assert guard.powershell_exe() == "powershell"
