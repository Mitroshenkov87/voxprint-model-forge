"""install.ps1 runs each import check bounded; migrate-home.ps1 runs its work bounded."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INSTALL = ROOT / "installer" / "windows" / "install.ps1"
MIGRATE = ROOT / "installer" / "windows" / "migrate-home.ps1"
SMOKE = ROOT / "scripts" / "windows_smoke.ps1"
ISS = ROOT / "installer" / "windows" / "voxprint-model-forge.iss"
PWSH = shutil.which("pwsh")
windows_pwsh = pytest.mark.skipif(os.name != "nt" or PWSH is None, reason="needs pwsh on Windows")


def test_import_check_is_split_into_bounded_steps() -> None:
    text = INSTALL.read_text(encoding="utf-8")
    steps = ("import voxprint_forge", "import torch", "torch.cuda init", "import bitsandbytes", "import mivolo env")
    for name in steps:
        assert f'"{name}"' in text, name
    assert "-RedirectStandardInput $none" in text and "-RedirectStandardOutput $out" in text
    assert "-RedirectStandardError $err" in text
    assert 'PYTHONFAULTHANDLER = "1"' in text and 'PYTHONUNBUFFERED = "1"' in text
    assert "SetErrorMode(0x8003)" in text
    assert "taskkill.exe" in text and "/T /F /PID" in text
    assert "$StepTimeoutSec = 120" in text
    assert "& $Py -c" not in text and "& $MPy -c" not in text


def test_only_bitsandbytes_and_cuda_init_are_optional() -> None:
    lines = [line for line in INSTALL.read_text(encoding="utf-8").splitlines() if "Invoke-PyStep $" in line]
    optional = {line.split('"')[1] for line in lines if "-Optional" in line}
    assert optional == {"torch.cuda init", "import bitsandbytes"}


def test_failure_names_the_step_in_english_and_russian() -> None:
    iss = ISS.read_text(encoding="utf-8-sig")
    assert "english.InstallFailed=" in iss and "russian.InstallFailed=" in iss
    assert "install-error.txt" in iss and "install-error.txt" in INSTALL.read_text(encoding="utf-8")


def test_migration_runs_in_a_bounded_child() -> None:
    text = MIGRATE.read_text(encoding="utf-8")
    assert "[switch]$Inner" in text and "[int]$TimeoutSec = 120" in text
    assert "-RedirectStandardInput $none" in text and "taskkill.exe" in text


def test_smoke_checks_the_steps_and_disables_quickedit() -> None:
    text = SMOKE.read_text(encoding="utf-8")
    assert 'Invoke-Step "import-steps"' in text and "timeout 120 s" in text
    assert "SetConsoleMode" in text and "0x40" in text


def _run_step(tmp_path: Path, code: str, timeout: int, optional: bool) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "run.ps1"
    flag = "-Optional" if optional else ""
    script.write_text(
        f"""$ErrorActionPreference = "Stop"
$tokens = $null; $errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile('{INSTALL}', [ref]$tokens, [ref]$errors)
$kind = [System.Management.Automation.Language.FunctionDefinitionAst]
$fn = $ast.FindAll({{ param($n) $n -is $kind -and $n.Name -eq 'Invoke-PyStep' }}, $true)
. ([scriptblock]::Create($fn[0].Extent.Text))
$Work = '{tmp_path}'
$StepTimeoutSec = {timeout}
$r = Invoke-PyStep '{sys.executable}' "probe" {flag} @'
{code}
'@
"RESULT=$r"
""",
        encoding="utf-8",
    )
    return subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-File", str(script)],
                          capture_output=True, text=True, timeout=120)


@windows_pwsh
def test_step_reads_empty_stdin_and_passes(tmp_path: Path) -> None:
    done = _run_step(tmp_path, "import sys\nprint('stdin', repr(sys.stdin.read()))", 60, False)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "stdin ''" in done.stdout and "step probe: ok in" in done.stdout and "RESULT=True" in done.stdout


@windows_pwsh
def test_hung_optional_step_is_killed_and_setup_continues(tmp_path: Path) -> None:
    start = time.monotonic()
    done = _run_step(tmp_path, "import time\ntime.sleep(600)", 3, True)
    assert time.monotonic() - start < 60
    assert done.returncode == 0, done.stdout + done.stderr
    assert "timed out after 3 s" in done.stdout and "RESULT=False" in done.stdout


@windows_pwsh
def test_failed_required_step_throws(tmp_path: Path) -> None:
    done = _run_step(tmp_path, "raise SystemExit(7)", 60, False)
    assert done.returncode != 0
    assert "step 'probe' exited 7" in done.stdout + done.stderr


@windows_pwsh
def test_migration_child_runs_and_returns_its_code(tmp_path: Path) -> None:
    env = dict(os.environ, LOCALAPPDATA=str(tmp_path), TEMP=str(tmp_path), VOXPRINT_HOME=str(tmp_path / "home"))
    done = subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-File", str(MIGRATE)],
                          capture_output=True, text=True, timeout=120, env=env)
    assert done.returncode == 0, done.stdout + done.stderr
    log = (tmp_path / "voxprint-home-migration.log").read_text(encoding="utf-8-sig")
    assert "VOXPRINT_HOME is set" in log
