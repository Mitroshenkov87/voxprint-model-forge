"""The Windows installer ships pillow-avif-plugin so Canvas AVIF photos decode."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "installer" / "windows" / "requirements-avif.txt"
# PyPI pillow_avif_plugin 1.6.0 cp314-cp314-win_amd64 (not the free-threaded build).
WIN_AMD64_SHA256 = "1686edf1b9462e4950a5f5672ba3ee6a90d600f6a09cb751266614c24309f11d"
PIN = re.compile(
    r"^pillow-avif-plugin==1\.6\.0\s+--hash=sha256:([0-9a-f]{64})\s*$",
    re.MULTILINE,
)


def test_installer_requirement_set_pins_the_avif_plugin_by_hash() -> None:
    text = REQ.read_text(encoding="utf-8").replace("\r\n", "\n")
    match = PIN.search(text)
    assert match is not None
    assert match.group(1) == WIN_AMD64_SHA256
    lowered = text.lower()
    assert "extra-index-url" not in lowered
    assert "git+" not in lowered
    assert "cp314t" not in lowered


def test_installer_installs_the_avif_wheel_with_require_hashes() -> None:
    install = (ROOT / "installer" / "windows" / "install.ps1").read_text(encoding="utf-8")
    assert "requirements-avif.txt" in install
    assert "--require-hashes" in install
    assert "--no-index" in install
    assert "import voxprint_forge, torch, bitsandbytes, pillow_avif" in install
    workflow = (ROOT / ".github" / "workflows" / "installer.yml").read_text(encoding="utf-8")
    assert "requirements-avif.txt" in workflow
    assert "pillow_avif_plugin-1.6.0-cp314-cp314-win_amd64.whl" in workflow
    assert "pillow-avif-plugin sha256 mismatch" in workflow
    assert "cp314t" in workflow


def test_gui_launcher_prints_version_and_help_on_the_parent_console() -> None:
    source = (ROOT / "installer" / "windows" / "launcher.cs").read_text(encoding="utf-8")
    assert "AttachConsole" in source
    assert "BeginOutputReadLine" in source
    assert "BeginErrorReadLine" in source
    assert "python.exe" in source
    assert "--version" in source and "--help" in source
    assert "pythonw.exe" in source
