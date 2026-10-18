"""The Windows installer ships the te-quant extra as a hash-pinned bitsandbytes wheel."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "installer" / "windows" / "requirements-te-quant.txt"
# PyPI bitsandbytes 0.50.2 py3-none-win_amd64 (files.pythonhosted.org).
WIN_AMD64_SHA256 = "c697963c8fda3dcd0d7ebd9b5211ae4067feef7cd06e0350d4e816a434fe683d"
PIN = re.compile(r"^bitsandbytes==0\.50\.2\s+--hash=sha256:([0-9a-f]{64})\s*$", re.MULTILINE)


def test_installer_requirement_set_pins_bitsandbytes_by_hash() -> None:
    """The installer requirement set names bitsandbytes 0.50.2 and its win_amd64 sha256."""
    text = REQ.read_text(encoding="utf-8").replace("\r\n", "\n")
    match = PIN.search(text)
    assert match is not None, "requirements-te-quant.txt must pin bitsandbytes==0.50.2 with a sha256"
    assert match.group(1) == WIN_AMD64_SHA256
    lowered = text.lower()
    assert "extra-index-url" not in lowered
    assert "git+" not in lowered
    assert "https://download.pytorch.org" not in lowered


def test_installer_installs_that_set_with_require_hashes() -> None:
    """install.ps1 hash-checks the requirement set from the bundled wheel, PyPI index only."""
    install = (ROOT / "installer" / "windows" / "install.ps1").read_text(encoding="utf-8")
    assert "requirements-te-quant.txt" in install
    assert "--require-hashes" in install
    assert "--no-index" in install
    workflow = (ROOT / ".github" / "workflows" / "installer.yml").read_text(encoding="utf-8")
    assert "requirements-te-quant.txt" in workflow
    assert "bitsandbytes-0.50.2-py3-none-win_amd64.whl" in workflow
    assert "bitsandbytes sha256 mismatch" in workflow


def test_windows_smoke_imports_bitsandbytes() -> None:
    """The installer smoke checks that the app venv can import bitsandbytes 0.50.2."""
    smoke = (ROOT / "scripts" / "windows_smoke.ps1").read_text(encoding="utf-8")
    assert "import bitsandbytes" in smoke
    assert "0.50.2" in smoke
