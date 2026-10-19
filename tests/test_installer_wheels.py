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


def test_iss_install_delete_removes_stale_wheels() -> None:
    """Stale wheels under {app}\\wheels are removed before the new files are copied."""
    deleted = _section("InstallDelete")
    assert r'Type: files; Name: "{app}\wheels\*.whl"' in deleted


def test_windows_smoke_plants_stale_wheels_and_checks_the_pin() -> None:
    """The installer smoke plants an older common wheel and checks the venv pin."""
    assert "voxprint_common-0.4.0-py3-none-any.whl" in SMOKE
    assert "voxprint_forge-0.0.9-py3-none-any.whl" in SMOKE
    assert "upgrade-wheels" in SMOKE
    assert "importlib.metadata" in SMOKE
    assert "upgrade-wheels" in WORKFLOW
    assert "vendor\\voxprint_common-" + _pinned_common() not in WORKFLOW
    assert "does not pin exactly one voxprint-common version" in WORKFLOW
