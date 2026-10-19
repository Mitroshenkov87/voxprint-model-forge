#Requires -Version 5.1
# VOXPrint AI Model Forge: post-install step, run elevated by the setup.
# Builds <app>\venv (Python 3.14, the app) and <app>\venv-mivolo (Python 3.12,
# the MiVOLO age check, which pins transformers 4.51) with the bundled uv.exe.
# Nothing is written outside <app>. Models are not downloaded here; the app
# asks for them on first launch. Log: <app>\setup-work\install.log
# Each run deletes that log first, so a reader never sees the previous run.
$ErrorActionPreference = "Stop"
$App = Split-Path -Parent $PSCommandPath
$Work = Join-Path $App "setup-work"
New-Item -ItemType Directory -Force $Work | Out-Null
$LogPath = Join-Path $Work "install.log"
if (Test-Path -LiteralPath $LogPath) { Remove-Item -LiteralPath $LogPath -Force }
Start-Transcript -Path $LogPath -Force | Out-Null

function Invoke-Uv([string[]]$ArgList) {
    & $script:Uv @ArgList
    if ($LASTEXITCODE -ne 0) { throw "uv $($ArgList[0..1] -join ' ') failed ($LASTEXITCODE)" }
}

function Resolve-OneWheel([string]$Directory, [string]$Label, [string]$Filter) {
    $found = @(Get-ChildItem -LiteralPath $Directory -Filter $Filter -File -ErrorAction SilentlyContinue)
    if ($found.Count -ne 1) {
        $names = @($found | ForEach-Object { $_.Name }) -join ", "
        throw "expected exactly one $Label wheel ($Filter) in ${Directory}, found $($found.Count): $names"
    }
    return [string]$found[0].FullName
}

function Get-PinnedCommonVersion([string]$WheelPath) {
    # voxprint-common version recorded in the app wheel, from the pyproject.toml pin.
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $zip = [System.IO.Compression.ZipFile]::OpenRead($WheelPath)
    try {
        $entry = $null
        foreach ($item in $zip.Entries) {
            if ($item.FullName -like "*.dist-info/METADATA") { $entry = $item; break }
        }
        if ($null -eq $entry) { throw "the app wheel has no metadata: $WheelPath" }
        $reader = New-Object System.IO.StreamReader($entry.Open())
        try { $text = $reader.ReadToEnd() } finally { $reader.Dispose() }
    } finally {
        $zip.Dispose()
    }
    $versions = @()
    foreach ($line in ($text -split "`n")) {
        $line = $line.TrimEnd("`r").Trim()
        if (-not $line.StartsWith("Requires-Dist:")) { continue }
        $req = $line.Substring($line.IndexOf(":") + 1).Trim()
        $semi = $req.IndexOf(";")
        if ($semi -ge 0) { $req = $req.Substring(0, $semi).Trim() }
        $req = $req -replace "\s", ""
        if ($req -match '^voxprint-common(\[[^\]]*\])?==([0-9][0-9A-Za-z.]*)$') {
            $versions += $Matches[2]
        }
    }
    $unique = @($versions | Select-Object -Unique)
    if ($unique.Count -ne 1) {
        $listed = ($unique -join ", ")
        throw "the app wheel does not pin exactly one voxprint-common version (found $($unique.Count): $listed)"
    }
    return [string]$unique[0]
}

try {
    # Ignore inherited uv, pip, Python and conda settings.
    foreach ($name in (Get-ChildItem env: | ForEach-Object Name)) {
        if ($name -match '^(UV_|PIP_|PYTHON|CONDA_|VIRTUAL_ENV)') { Remove-Item "env:$name" }
    }
    $env:UV_NO_CONFIG = "1"
    $env:UV_MANAGED_PYTHON = "1"
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $App "python"
    $env:UV_PYTHON_BIN_DIR = Join-Path $App "python\bin"
    $env:UV_PYTHON_INSTALL_REGISTRY = "0"
    $env:UV_CACHE_DIR = Join-Path $Work "uv-cache"
    $env:UV_LINK_MODE = "copy"
    $script:Uv = Join-Path $App "uv.exe"
    $Wheels = Join-Path $App "wheels"
    $Version = (Get-Content -Raw (Join-Path $App "BUILD.json") | ConvertFrom-Json).version

    # NVIDIA driver present: CUDA 13.0 wheels. Otherwise CPU wheels (the app still opens).
    $nvsmi = Join-Path $env:SystemRoot "System32\nvidia-smi.exe"
    $TorchIndex = if (Test-Path $nvsmi) { "https://download.pytorch.org/whl/cu130" } else { "https://download.pytorch.org/whl/cpu" }
    Write-Host "torch index: $TorchIndex"

    $Venv = Join-Path $App "venv"
    $Py = Join-Path $Venv "Scripts\python.exe"
    Invoke-Uv @("venv", "--python", "3.14", "--allow-existing", $Venv)
    Invoke-Uv @("pip", "install", "--python", $Py, "--index-url", $TorchIndex, "torch==2.14.1", "torchvision==0.29.1")
    $pins = Join-Path $Work "torch-pins.txt"
    & $script:Uv pip freeze --python $Py | Select-String -Pattern '^(torch|torchvision)==' | ForEach-Object { $_.Line } | Set-Content -Encoding ascii $pins
    # Both wheels by path, so a same-named package on a public index is never used.
    $forgeWheel = Resolve-OneWheel $Wheels "voxprint_forge" "voxprint_forge-$Version-*.whl"
    $commonVersion = Get-PinnedCommonVersion $forgeWheel
    $commonWheel = Resolve-OneWheel $Wheels "voxprint_common" "voxprint_common-${commonVersion}-*.whl"
    Invoke-Uv @("pip", "install", "--python", $Py, "--constraint", $pins, "$commonWheel", "$forgeWheel[gui,compare]")
    # te-quant extra: the bundled PyPI win_amd64 wheel, hash-checked. No extra index.
    $teReq = Join-Path $App "requirements-te-quant.txt"
    Invoke-Uv @("pip", "install", "--python", $Py, "--no-deps", "--require-hashes", "--no-index",
        "--find-links", $Wheels, "-r", $teReq)

    $Mivolo = Join-Path $App "venv-mivolo"
    $MPy = Join-Path $Mivolo "Scripts\python.exe"
    Invoke-Uv @("venv", "--python", "3.12", "--allow-existing", $Mivolo)
    # Two steps, one index each: torch from the PyTorch CPU index only, then the rest
    # from PyPI only with torch held at the installed pins.
    Invoke-Uv @("pip", "install", "--python", $MPy, "--index-url", "https://download.pytorch.org/whl/cpu",
        "torch==2.5.1", "torchvision==0.20.1")
    $mpins = Join-Path $Work "mivolo-torch-pins.txt"
    & $script:Uv pip freeze --python $MPy | Select-String -Pattern '^(torch|torchvision)==' | ForEach-Object { $_.Line } | Set-Content -Encoding ascii $mpins
    $mreq = Join-Path $Work "mivolo-rest.txt"
    Get-Content (Join-Path $App "requirements-mivolo.txt") | Where-Object { $_ -notmatch '^torch(vision)?\b' } | Set-Content -Encoding ascii $mreq
    Invoke-Uv @("pip", "install", "--python", $MPy, "--index-url", "https://pypi.org/simple",
        "--constraint", $mpins, "-r", $mreq)
    Invoke-Uv @("pip", "install", "--python", $MPy, "setuptools<80", "wheel")
    Invoke-Uv @("pip", "install", "--python", $MPy, "--no-deps", "--no-build-isolation",
        "mivolo @ https://github.com/WildChlamydia/MiVOLO/archive/37475e3f8818b5f22448003feec3e64b01bfb188.zip")

    & $Py -c "import voxprint_forge, torch, bitsandbytes; print('forge', voxprint_forge.__name__, 'torch', torch.__version__, 'bitsandbytes', bitsandbytes.__version__); raise SystemExit(0 if bitsandbytes.__version__ == '0.50.2' else 1)"
    if ($LASTEXITCODE -ne 0) { throw "the app environment does not import" }
    & $MPy -c "import mivolo, transformers, cv2; print('mivolo env ok', transformers.__version__)"
    if ($LASTEXITCODE -ne 0) { throw "the MiVOLO environment does not import" }
    # Program Files is read-only for the signed-in user, so bytecode is written at install time.
    # Skip the uv Python trees (Tcl ships a file 3.14 cannot compile), the uv cache, wheels, and
    # venv-mivolo, which is compiled with its own interpreter below.
    $site = Join-Path $Venv "Lib\site-packages"
    $skip = '\\python\\cpython-|\\setup-work\\|\\venv-mivolo\\|\\wheels\\'
    & $Py -m compileall -q -x $skip $site $App
    if ($LASTEXITCODE -ne 0) { throw "compileall failed" }
    & $MPy -m compileall -q (Join-Path $Mivolo "Lib\site-packages")
    if ($LASTEXITCODE -ne 0) { throw "compileall failed for venv-mivolo" }
    Remove-Item -Recurse -Force $env:UV_CACHE_DIR -ErrorAction SilentlyContinue
    Write-Host "Model Forge setup finished."
    $code = 0
} catch {
    Write-Host "Model Forge setup failed: $_"
    $code = 1
} finally {
    Stop-Transcript | Out-Null
}
exit $code
