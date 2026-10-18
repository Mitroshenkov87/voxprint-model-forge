#Requires -Version 5.1
# VOXPrint AI Model Forge: post-install step, run elevated by the setup.
# Builds <app>\venv (Python 3.14, the app) and <app>\venv-mivolo (Python 3.12,
# the MiVOLO age check, which pins transformers 4.51) with the bundled uv.exe.
# Nothing is written outside <app>. Models are not downloaded here; the app
# asks for them on first launch. Log: <app>\setup-work\install.log
$ErrorActionPreference = "Stop"
$App = Split-Path -Parent $PSCommandPath
$Work = Join-Path $App "setup-work"
New-Item -ItemType Directory -Force $Work | Out-Null
Start-Transcript -Path (Join-Path $Work "install.log") -Force | Out-Null
$ErrorFile = Join-Path $Work "install-error.txt"
Remove-Item -LiteralPath $ErrorFile -Force -ErrorAction SilentlyContinue
$StepTimeoutSec = 120

# No Windows error dialogs (child processes inherit the mode) and no QuickEdit pause in a console.
try {
    Add-Type -Namespace VoxprintSetup -Name Native -MemberDefinition @"
[DllImport("kernel32.dll")] public static extern uint SetErrorMode(uint mode);
[DllImport("kernel32.dll")] public static extern System.IntPtr GetStdHandle(int handle);
[DllImport("kernel32.dll")] public static extern bool GetConsoleMode(System.IntPtr handle, out uint mode);
[DllImport("kernel32.dll")] public static extern bool SetConsoleMode(System.IntPtr handle, uint mode);
"@
    [void][VoxprintSetup.Native]::SetErrorMode(0x8003)
    $stdin = [VoxprintSetup.Native]::GetStdHandle(-10)
    $mode = [uint32]0
    if ([VoxprintSetup.Native]::GetConsoleMode($stdin, [ref]$mode)) {
        if ($mode -band 0x40) { $mode = $mode - 0x40 }
        [void][VoxprintSetup.Native]::SetConsoleMode($stdin, [uint32]($mode -bor 0x80))
    }
} catch {
    Write-Host "error mode / console mode not set: $($_.Exception.Message)"
}

function Invoke-Uv([string[]]$ArgList) {
    & $script:Uv @ArgList
    if ($LASTEXITCODE -ne 0) { throw "uv $($ArgList[0..1] -join ' ') failed ($LASTEXITCODE)" }
}

# One Python check in its own process: stdin from an empty file, stdout and stderr to files, a hard
# timeout that kills the process tree. Logs "step <name>: ok in <s> s" or the failure.
# A failed -Optional step is logged as a warning and the setup continues.
function Invoke-PyStep([string]$Python, [string]$Name, [string]$Code, [switch]$Optional) {
    $slug = $Name -replace '[^A-Za-z0-9]+', '-'
    $file = Join-Path $Work "step-$slug.py"
    $out = Join-Path $Work "step-$slug.out.txt"
    $err = Join-Path $Work "step-$slug.err.txt"
    $none = Join-Path $Work "stdin-empty.txt"
    Set-Content -LiteralPath $file -Value $Code -Encoding ascii
    [IO.File]::WriteAllText($none, "")
    $env:PYTHONFAULTHANDLER = "1"
    $env:PYTHONUNBUFFERED = "1"
    Write-Host "step ${Name}: start (timeout $StepTimeoutSec s)"
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process -FilePath $Python -ArgumentList "-X faulthandler -u `"$file`"" -NoNewWindow -PassThru `
        -RedirectStandardInput $none -RedirectStandardOutput $out -RedirectStandardError $err
    [void]$p.Handle
    $problem = ""
    if (-not $p.WaitForExit($StepTimeoutSec * 1000)) {
        Start-Process -FilePath (Join-Path $env:SystemRoot "System32\taskkill.exe") -ArgumentList "/T /F /PID $($p.Id)" -NoNewWindow -Wait
        $problem = "timed out after $StepTimeoutSec s (process tree killed)"
    } else {
        $p.WaitForExit()
        if ($p.ExitCode -ne 0) { $problem = "exited $($p.ExitCode)" }
    }
    $seconds = [math]::Round($clock.Elapsed.TotalSeconds, 1)
    foreach ($part in @($out, $err)) {
        if (Test-Path -LiteralPath $part) { Get-Content -LiteralPath $part | ForEach-Object { Write-Host "  $_" } }
    }
    if (-not $problem) {
        Write-Host "step ${Name}: ok in $seconds s"
        return $true
    }
    if ($Optional) {
        Write-Host "step ${Name}: WARNING $problem after $seconds s (optional, setup continues)"
        return $false
    }
    Write-Host "step ${Name}: FAILED $problem after $seconds s"
    throw "step '$Name' $problem"
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
    $forgeWheel = (Get-ChildItem $Wheels -Filter "voxprint_forge-$Version-*.whl" | Select-Object -First 1).FullName
    $commonWheel = (Get-ChildItem $Wheels -Filter "voxprint_common-*.whl" | Select-Object -First 1).FullName
    if (-not $forgeWheel -or -not $commonWheel) { throw "the app wheels are missing from $Wheels" }
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

    [void](Invoke-PyStep $Py "import voxprint_forge" "import voxprint_forge`nprint('voxprint_forge', voxprint_forge.__file__)")
    [void](Invoke-PyStep $Py "import torch" "import torch`nprint('torch', torch.__version__, 'cuda build', torch.version.cuda)")
    [void](Invoke-PyStep $Py "torch.cuda init" -Optional ("import torch`nok = torch.cuda.is_available()`nprint('cuda available', ok)`n" +
        "if ok:`n    torch.zeros(1, device='cuda')`n    print('device', torch.cuda.get_device_name(0))"))
    # Optional: --te-quant auto falls back when bitsandbytes does not load.
    [void](Invoke-PyStep $Py "import bitsandbytes" -Optional ("import bitsandbytes`nprint('bitsandbytes', bitsandbytes.__version__)`n" +
        "raise SystemExit(0 if bitsandbytes.__version__ == '0.50.2' else 1)"))
    [void](Invoke-PyStep $MPy "import mivolo env" "import mivolo, transformers, cv2`nprint('mivolo env ok', transformers.__version__)")
    Remove-Item -Recurse -Force $env:UV_CACHE_DIR -ErrorAction SilentlyContinue
    Write-Host "Model Forge setup finished."
    $code = 0
} catch {
    Write-Host "Model Forge setup failed: $_"
    try { Set-Content -LiteralPath $ErrorFile -Value "$($_.Exception.Message)" -Encoding ascii } catch { }
    $code = 1
} finally {
    Stop-Transcript | Out-Null
}
exit $code
