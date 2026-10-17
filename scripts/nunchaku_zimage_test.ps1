# Nunchaku INT4 matrix on Windows Server 2025. Non-interactive.
# Does not prompt and does not install upstream nunchaku.
$ErrorActionPreference = 'Stop'
$ConfirmPreference = 'None'
$env:GIT_TERMINAL_PROMPT = '0'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:PYTHONUNBUFFERED = '1'

$Root = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Set-Location $Root
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$Root\src;$env:PYTHONPATH" } else { "$Root\src" }

$Cache = if ($env:NUNCHAKU_CACHE) { $env:NUNCHAKU_CACHE } else { Join-Path $env:USERPROFILE '.cache\voxprint-forge\nunchaku-zimage' }
$Python = $env:NUNCHAKU_PYTHON
if (-not $Python) {
    $VenvPython = Join-Path $Cache 'venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $VenvPython) {
        $Python = $VenvPython
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        $Python = 'py'
    } else {
        $Python = 'python'
    }
}

if ($Python -eq 'py') {
    & py -3.12 -m voxprint_forge.diffusion.nunchaku_zimage @args
} else {
    & $Python -m voxprint_forge.diffusion.nunchaku_zimage @args
}
if ($null -ne $LASTEXITCODE) {
    exit $LASTEXITCODE
}
exit 2
