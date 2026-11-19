"""Windows installer: torch and torchvision downloads survive a flaky network."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INSTALL = (ROOT / "installer" / "windows" / "install.ps1").read_text(encoding="utf-8")
ISS = (ROOT / "installer" / "windows" / "voxprint-model-forge.iss").read_text(encoding="utf-8-sig")
TORCH = INSTALL.split("function Install-Torch", 1)[1].split("\nfunction ", 1)[0]


def test_uv_http_timeout_and_retries_default_unless_the_user_set_them() -> None:
    """UV_HTTP_TIMEOUT 300 and UV_HTTP_RETRIES 8, read before inherited UV_ settings are cleared."""
    assert '$httpTimeout = if ($env:UV_HTTP_TIMEOUT) { $env:UV_HTTP_TIMEOUT } else { "300" }' in INSTALL
    assert '$httpRetries = if ($env:UV_HTTP_RETRIES) { $env:UV_HTTP_RETRIES } else { "8" }' in INSTALL
    scrub = INSTALL.index('Remove-Item "env:$name"')
    assert INSTALL.index("$httpTimeout = ") < scrub < INSTALL.index("$env:UV_HTTP_TIMEOUT = $httpTimeout")
    assert INSTALL.index("$httpRetries = ") < scrub < INSTALL.index("$env:UV_HTTP_RETRIES = $httpRetries")


def test_both_torch_installs_go_through_the_retry_loop() -> None:
    assert 'Install-Torch $Py $TorchIndex @("torch==2.14.1", "torchvision==0.29.1")' in INSTALL
    assert (
        'Install-Torch $MPy "https://download.pytorch.org/whl/cpu" @("torch==2.5.1", "torchvision==0.20.1")' in INSTALL
    )
    outside = INSTALL.replace(TORCH, "")
    assert not re.search(r"Invoke-Uv[^\n]*download\.pytorch\.org", outside)
    assert '--index-url", $TorchIndex' not in outside


def test_retry_loop_is_five_attempts_with_doubling_backoff() -> None:
    """10s, 20s, 40s, 80s between five attempts."""
    assert "$delay = 10" in TORCH
    assert "$attempt -le 5" in TORCH
    assert "$attempt -lt 5" in TORCH
    assert "Start-Sleep -Seconds $delay" in TORCH
    assert "$delay *= 2" in TORCH
    assert "if ($LASTEXITCODE -eq 0) { return }" in TORCH


def test_retries_reuse_the_uv_cache() -> None:
    """The cache lives under the app folder, is never bypassed, and is removed only after success."""
    assert '$env:UV_CACHE_DIR = Join-Path $Work "uv-cache"' in INSTALL
    assert "--no-cache" not in INSTALL
    assert "cache clean" not in INSTALL
    assert "cache prune" not in INSTALL
    removals = [line for line in INSTALL.splitlines() if "UV_CACHE_DIR" in line and "Remove-Item" in line]
    assert len(removals) == 1
    assert INSTALL.index(removals[0]) > INSTALL.index("compileall failed for venv-mivolo")
    assert INSTALL.index(removals[0]) < INSTALL.index('Write-Host "Model Forge setup finished."')


def test_final_failure_names_the_host_and_suggests_network_firewall_proxy() -> None:
    assert "$hostName = ([uri]$Index).Host" in TORCH
    assert "from $hostName after 5 attempts" in TORCH
    for word in ("network", "firewall", "proxy", "download-r2.pytorch.org", "retry"):
        assert word in TORCH, word


def test_interactive_run_asks_retry_and_silent_run_exits_30() -> None:
    assert "MessageBoxButtons]::RetryCancel" in TORCH
    assert "DialogResult]::Retry" in TORCH
    assert "if (-not (Test-CanPrompt)) { break }" in TORCH
    can_prompt = INSTALL.split("function Test-CanPrompt", 1)[1].split("\n}", 1)[0]
    assert "$NoPrompt" in can_prompt and "UserInteractive" in can_prompt and "-noni" in can_prompt
    assert "param([switch]$NoPrompt)" in INSTALL
    assert "$TorchDownloadExitCode = 30" in INSTALL
    failed = INSTALL.split("} catch {", 1)[1].split("} finally {", 1)[0]
    assert "Set-Content -LiteralPath $TorchErrorPath" in failed
    assert "$code = $TorchDownloadExitCode" in failed


def test_setup_offers_retry_when_install_ps1_exits_30() -> None:
    assert "TorchDownloadExitCode = 30;" in ISS
    post = ISS.split("procedure CurStepChanged", 1)[1]
    loop = post.split("repeat", 1)[1].split("if ResultCode <> 0 then", 1)[0]
    assert "install.ps1')) + ' -NoPrompt'" in loop
    assert "if ResultCode <> TorchDownloadExitCode then" in loop
    assert "torch-download-error.txt" in loop
    # SuppressibleMsgBox answers IDCANCEL in a silent setup, which then exits 30.
    assert "until SuppressibleMsgBox(TorchError, mbError, MB_RETRYCANCEL, IDCANCEL) <> IDRETRY;" in loop
    assert "ExitProcess(ResultCode)" in post


def test_no_unofficial_index_or_mirror() -> None:
    """Only the official PyTorch, PyPI and GitHub URLs; no extra index."""
    urls = set(re.findall(r"https?://[^\s\"')]+", INSTALL))
    allowed = re.compile(
        r"^https://(download\.pytorch\.org/whl/(cu130|cpu)|pypi\.org/simple|github\.com/WildChlamydia/)"
    )
    assert urls and all(allowed.match(url) for url in urls), urls
    lowered = INSTALL.lower()
    assert "extra-index-url" not in lowered
    assert "--index-strategy" not in lowered
    assert "mirror" not in lowered.replace("so no mirror", "")


def test_torch_pins_stay_and_missing_hash_list_is_noted() -> None:
    """Versions stay pinned; no hashes are invented for torch, and the comment says why."""
    assert "the repo has no torch hash list" in INSTALL
    assert not re.search(r"torch[^\n]*--hash=", INSTALL)
