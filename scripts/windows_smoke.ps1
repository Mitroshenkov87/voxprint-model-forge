<#
.SYNOPSIS
Windows installer smoke for VOXPrint AI Model Forge: install, --version, preflight,
LoRA dry-run on stub data, uninstall. No GPU work, no model downloads, no tokens.

.DESCRIPTION
Run from an elevated PowerShell (the setup installs to Program Files):

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\windows_smoke.ps1 -Setup .\VoxprintModelForgeSetup.exe

-Setup        The setup exe (CI artifact). Default: dist\VoxprintModelForgeSetup.exe, then the script folder.
-RequireGpu   Also require the preflight CUDA check (use on the laptop; CI has no GPU).
-KeepInstalled  Skip the uninstall step.
-MigrationTest  CI only: when neither data home exists, plant a fake %LOCALAPPDATA%\Voxprint with a dummy
               model first, then check the setup moved it and that a second install is a no-op.

Steps and pass rules:
  0 home       (with -MigrationTest on a clean profile) a fake %LOCALAPPDATA%\Voxprint\models\smoke-dummy file.
  1 install    silent setup exits 0; venv, venv-mivolo, CLI, GUI exe, uv.exe, LICENSE, Start menu shortcut exist;
               the setup holds no model weights (*.safetensors, *.gguf, *.bin, *.pt, *.pth, *.onnx, *.ckpt).
  2 version    voxprint-forge-cli --version exits 0 and prints the BUILD.json version.
  3 preflight  voxprint-forge-cli --json preflight --refset prints a report; diffusers, img2img and the
               MiVOLO env pass. Missing models are listed, not failed (a fresh install has none).
  4 lora       a 12-picture stub project; lora train --dry-run --base z-image --vram 24 returns state dry-run.
  4a migration the setup moved %LOCALAPPDATA%\Voxprint to %LOCALAPPDATA%\VOXPrint AI Suite before any Python ran:
               the old folder is gone, the dummy sits under paths.models_dir() with the same sha256
               (checked on disk first, then through voxprint_common), and a second silent install
               exits 0, logs "no-op" and changes nothing. Without -MigrationTest only the move rule is
               checked against the folders that were there before the install.
  5 uninstall  unins000.exe removes the program folder, the shortcut and the uninstall entry.
               %LOCALAPPDATA%\VOXPrint AI Suite (user data, models) is left in place by design.

Results: %LOCALAPPDATA%\VoxprintForgeSmoke\windows\<timestamp>\result.json and smoke.log.
Exit 0 when every step passes, 1 otherwise, 2 when it cannot start.
#>
[CmdletBinding()]
param(
    [string]$Setup = "",
    [switch]$RequireGpu,
    [switch]$KeepInstalled,
    [switch]$MigrationTest,
    [string]$Results = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$AppId = "{CB5C2217-6254-48EA-99F6-BC3D52260BAB}_is1"
$Dir = Join-Path $env:ProgramFiles "VOXPrint AI Suite\Model Forge"
$Cli = Join-Path $Dir "venv\Scripts\voxprint-forge-cli.exe"
$Py = Join-Path $Dir "venv\Scripts\python.exe"
$Menu = Join-Path $env:ProgramData "Microsoft\Windows\Start Menu\Programs\VOXPrint AI Suite\VOXPrint AI Model Forge.lnk"
$OldHome = Join-Path $env:LOCALAPPDATA "Voxprint"
$NewHome = Join-Path $env:LOCALAPPDATA "VOXPrint AI Suite"
$Dummy = "models\smoke-dummy\smoke-dummy-model.bin"
$Weights = @("*.safetensors", "*.gguf", "*.bin", "*.pt", "*.pth", "*.onnx", "*.ckpt")

if (-not $Results) {
    $Results = Join-Path $env:LOCALAPPDATA ("VoxprintForgeSmoke\windows\" + (Get-Date -Format "yyyyMMdd-HHmmss"))
}
New-Item -ItemType Directory -Force $Results | Out-Null
$Log = Join-Path $Results "smoke.log"
$script:Steps = New-Object System.Collections.ArrayList

function Write-Log([string]$Message) {
    $line = "{0} {1}" -f (Get-Date -Format "o"), $Message
    Write-Host $line
    Add-Content -LiteralPath $Log -Value $line -Encoding utf8
}

function Add-Step([string]$Name, [bool]$Ok, [string]$Detail, [double]$Seconds) {
    [void]$script:Steps.Add([ordered]@{ name = $Name; ok = $Ok; detail = $Detail; seconds = [math]::Round($Seconds, 1) })
    Write-Log ("[{0}] {1}: {2}" -f ($(if ($Ok) { "PASS" } else { "FAIL" })), $Name, $Detail)
}

function Invoke-Step([string]$Name, [scriptblock]$Body) {
    $t = Get-Date
    try {
        $detail = & $Body
        Add-Step $Name $true ([string]$detail) ((Get-Date) - $t).TotalSeconds
        return $true
    } catch {
        Add-Step $Name $false $_.Exception.Message ((Get-Date) - $t).TotalSeconds
        return $false
    }
}

function Invoke-Cli([string[]]$ArgList) {
    # stdout is the JSON; stderr (warnings) goes to the log only. Windows PowerShell 5.1 turns native
    # stderr into error records, so this call must not run under "Stop".
    $ErrorActionPreference = "Continue"
    $err = Join-Path $Results "stderr.txt"
    $out = & $Cli @ArgList 2>> $err | Out-String
    $code = $LASTEXITCODE
    Add-Content -LiteralPath $Log -Value ("> voxprint-forge-cli " + ($ArgList -join " ") + "`n" + $out) -Encoding utf8
    return @{ code = $code; out = $out }
}

function Find-Setup {
    if ($Setup) { return (Resolve-Path $Setup).Path }
    foreach ($candidate in @("dist\VoxprintModelForgeSetup.exe", (Join-Path $PSScriptRoot "VoxprintModelForgeSetup.exe"))) {
        if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
    }
    throw "setup exe not found; pass -Setup"
}

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
try {
    if (-not $admin) { throw "run this from an elevated PowerShell (the setup installs to Program Files)" }
    $SetupExe = Find-Setup
} catch {
    Write-Log "cannot start: $($_.Exception.Message)"
    exit 2
}
Write-Log "setup $SetupExe ($([math]::Round((Get-Item $SetupExe).Length / 1MB, 1)) MB) results $Results"

$Before = @{ old = (Test-Path -LiteralPath $OldHome); new = (Test-Path -LiteralPath $NewHome) }
$Planted = $false
if ($MigrationTest -and -not $Before.old -and -not $Before.new) {
    [void](Invoke-Step "home-plant" {
        $file = Join-Path $OldHome $Dummy
        New-Item -ItemType Directory -Force (Split-Path -Parent $file) | Out-Null
        [IO.File]::WriteAllText($file, "voxprint smoke dummy model, not a real weight file`n")
        $script:DummySha = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash.ToLower()
        $script:Planted = $true
        $Before.old = $true
        "planted $file"
    })
} elseif ($MigrationTest) {
    Write-Log "home-plant skipped: a data home already exists (old=$($Before.old) new=$($Before.new)); real data is never touched"
}

$installed = Invoke-Step "install" {
    $p = Start-Process $SetupExe -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-",
        "/LOG=`"$(Join-Path $Results 'setup.log')`"" -Wait -PassThru
    if ($p.ExitCode -ne 0) {
        $installLog = Join-Path $Dir "setup-work\install.log"
        if (Test-Path $installLog) { Copy-Item $installLog (Join-Path $Results "install.log") }
        throw "setup exited $($p.ExitCode); see install.log"
    }
    foreach ($f in @("venv\Scripts\voxprint-model-forge.exe", "venv\Scripts\voxprint-forge-cli.exe",
            "venv-mivolo\Scripts\python.exe", "uv.exe", "LICENSE", "ACCEPTABLE_USE.md", "unins000.exe")) {
        if (-not (Test-Path (Join-Path $Dir $f))) { throw "missing $f" }
    }
    if (-not (Test-Path $Menu)) { throw "Start menu shortcut missing" }
    $key = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\$AppId"
    if (-not (Test-Path $key)) { throw "uninstall entry missing" }
    $bundled = Get-ChildItem -Recurse -File (Join-Path $Dir "wheels") -Include $Weights -ErrorAction SilentlyContinue
    if ($bundled) { throw "model weights in the setup: $($bundled.Name -join ', ')" }
    & $Py -c "from voxprint_forge.lora.backends import mivolo_python; import sys; sys.exit(0 if 'venv-mivolo' in mivolo_python() else 1)"
    if ($LASTEXITCODE -ne 0) { throw "the app does not find venv-mivolo" }
    "installed to $Dir; uninstall entry $((Get-ItemProperty $key).DisplayName) $((Get-ItemProperty $key).DisplayVersion)"
}

if ($installed) {
    [void](Invoke-Step "version" {
        $want = (Get-Content -Raw (Join-Path $Dir "BUILD.json") | ConvertFrom-Json).version
        $r = Invoke-Cli @("--version")
        if ($r.code -ne 0) { throw "--version exited $($r.code)" }
        if ($r.out -notmatch [regex]::Escape($want)) { throw "--version printed '$($r.out.Trim())', want $want" }
        $r.out.Trim()
    })

    [void](Invoke-Step "preflight" {
        $r = Invoke-Cli @("--json", "preflight", "--refset")
        $start = $r.out.IndexOf("{")
        if ($start -lt 0) { throw "no JSON report (exit $($r.code))" }
        $doc = $r.out.Substring($start) | ConvertFrom-Json
        $report = if ($doc.result) { $doc.result } elseif ($doc.report) { $doc.report } else { $doc }
        $checks = @{}
        foreach ($c in $report.checks) { $checks[$c.name] = $c }
        $need = @("diffusers", "img2img", "mivolo-env")
        if ($RequireGpu) { $need += "cuda" }
        $bad = @($need | Where-Object { -not $checks.ContainsKey($_) -or -not $checks[$_].ok })
        if ($bad.Count) { throw "preflight failed: $($bad -join ', ')" }
        $models = @($report.checks | Where-Object { $_.name -like "model:*" -and -not $_.ok } | ForEach-Object { $_.name })
        "exit $($r.code); env checks pass; models not installed yet: $(if ($models.Count) { $models -join ', ' } else { 'none' })"
    })

    [void](Invoke-Step "lora-dry-run" {
        $project = Join-Path $Results "stub-project"
        $stub = Join-Path $Results "make_stub.py"
        @"
import sys
from pathlib import Path
from PIL import Image
from voxprint_forge.lora.project import Item, Project
root = Path(sys.argv[1]); project = Project(root); project.prepared.mkdir(parents=True, exist_ok=True)
items = []
for i in range(12):
    name = f"prepared/p{i:02d}.png"
    Image.new("RGB", (512, 512), (20 * i % 256, 90, 160)).save(root / name)
    items.append(Item(f"p{i:02d}", name, f"a plain test cube, shade {i}", f"cube {i}", 0))
project.write_items(items)
project.write({"kind": "object", "mode": "simple", "gate_passed": True,
               "subject": {"name": "Smoke cube", "trigger": "smkcube"}})
"@ | Set-Content -LiteralPath $stub -Encoding ascii
        & $Py $stub $project
        if ($LASTEXITCODE -ne 0) { throw "the stub project was not written" }
        $r = Invoke-Cli @("--json", "lora", "train", "--project", $project, "--dry-run", "--base", "z-image",
            "--vram", "24", "--steps", "4", "--no-previews")
        if ($r.code -ne 0) { throw "lora train --dry-run exited $($r.code)" }
        $res = ($r.out.Substring($r.out.IndexOf("{")) | ConvertFrom-Json).result
        if ($res.state -ne "dry-run" -or $res.photos -ne 12) { throw "unexpected result: state=$($res.state) photos=$($res.photos)" }
        "state dry-run, 12 photos, 4 steps, eta $($res.eta_seconds) s, vram need $($res.vram_need_gb) GB"
    })
}

function Get-MigrationLog {
    $log = Join-Path $NewHome "logs\home-migration.log"
    if (Test-Path -LiteralPath $log) { return @(Get-Content -LiteralPath $log) }
    $tmp = Join-Path $env:TEMP "voxprint-home-migration.log"
    if (Test-Path -LiteralPath $tmp) { return @(Get-Content -LiteralPath $tmp) }
    return @()
}

if ($installed) {
    [void](Invoke-Step "home-migration" {
        if ($Before.old -and -not $Before.new) {
            if (Test-Path -LiteralPath $OldHome) { throw "the old home $OldHome is still there" }
            if (-not (Test-Path -LiteralPath $NewHome)) { throw "the new home $NewHome is missing" }
        } elseif ($Before.old -and $Before.new) {
            if (-not (Test-Path -LiteralPath $OldHome)) { throw "both homes existed, but the old one was changed" }
        }
        $detail = "before old=$($Before.old) new=$($Before.new); after old=$(Test-Path -LiteralPath $OldHome) new=$(Test-Path -LiteralPath $NewHome)"
        if ($Planted) {
            # On disk first: no Python has run against the data home yet.
            $moved = Join-Path $NewHome $Dummy
            if (-not (Test-Path -LiteralPath $moved)) { throw "the dummy model is not at $moved" }
            if ((Get-FileHash -LiteralPath $moved -Algorithm SHA256).Hash.ToLower() -ne $script:DummySha) { throw "the dummy model changed" }
            if (-not (Get-MigrationLog | Where-Object { $_ -match " moved " })) { throw "home-migration.log has no move line" }
            # Then through voxprint_common: models_dir() must be <new home>\models.
            $saved = @{}
            foreach ($n in @("VOXPRINT_HOME", "VOXPRINT_MODELS_DIR")) { $saved[$n] = [Environment]::GetEnvironmentVariable($n); [Environment]::SetEnvironmentVariable($n, $null) }
            $md = (& $Py -c "from voxprint_common.paths import models_dir; print(models_dir())" | Out-String).Trim()
            foreach ($n in $saved.Keys) { [Environment]::SetEnvironmentVariable($n, $saved[$n]) }
            if ($md -ne (Join-Path $NewHome "models")) { throw "paths.models_dir() is '$md', want $(Join-Path $NewHome 'models')" }
            if (-not (Test-Path -LiteralPath (Join-Path $md "smoke-dummy\smoke-dummy-model.bin"))) { throw "the dummy is not under models_dir()" }
            $detail += "; dummy under models_dir() $md, sha256 unchanged"
        }
        $detail
    })

    if ($Planted) {
        [void](Invoke-Step "reinstall-noop" {
            $before = @(Get-ChildItem -LiteralPath $NewHome -Recurse -Force -File | Where-Object { $_.FullName -notlike "*\logs\*" } |
                ForEach-Object { "$($_.FullName)|$($_.Length)" })
            $p = Start-Process $SetupExe -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-",
                "/LOG=`"$(Join-Path $Results 'setup-2.log')`"" -Wait -PassThru
            if ($p.ExitCode -ne 0) { throw "the second install exited $($p.ExitCode)" }
            if (Test-Path -LiteralPath $OldHome) { throw "the old home reappeared" }
            $after = @(Get-ChildItem -LiteralPath $NewHome -Recurse -Force -File | Where-Object { $_.FullName -notlike "*\logs\*" } |
                ForEach-Object { "$($_.FullName)|$($_.Length)" })
            if (Compare-Object $before $after) { throw "the second install changed the data home" }
            $last = @(Get-MigrationLog | Where-Object { $_ -match "no-op|moved|REFUSED|FAILED" }) | Select-Object -Last 1
            if ($last -notmatch "no-op") { throw "the second install did not log a no-op: $last" }
            "second install exit 0; data home unchanged; log: $($last.Substring($last.IndexOf(' ') + 1))"
        })
        Remove-Item -LiteralPath (Join-Path $NewHome "models\smoke-dummy") -Recurse -Force -ErrorAction SilentlyContinue
    }
}

if (-not $KeepInstalled -and (Test-Path (Join-Path $Dir "unins000.exe"))) {
    [void](Invoke-Step "uninstall" {
        $u = Start-Process (Join-Path $Dir "unins000.exe") -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART" -Wait -PassThru
        if ($u.ExitCode -ne 0) { throw "uninstaller exited $($u.ExitCode)" }
        for ($i = 0; $i -lt 90 -and (Test-Path $Dir); $i++) { Start-Sleep 2 }
        if (Test-Path $Dir) { throw "uninstall left $Dir" }
        if (Test-Path $Menu) { throw "uninstall left the Start menu shortcut" }
        if (Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\$AppId") { throw "uninstall left the uninstall entry" }
        "program folder, shortcut and uninstall entry removed"
    })
}

$ok = ($script:Steps.Count -gt 0) -and -not ($script:Steps | Where-Object { -not $_.ok })
[ordered]@{
    schema = "voxprint-forge-windows-smoke/1"
    ok = [bool]$ok
    setup = $SetupExe
    require_gpu = [bool]$RequireGpu
    finished = (Get-Date).ToUniversalTime().ToString("o")
    steps = $script:Steps
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $Results "result.json") -Encoding utf8
Write-Log "result $(Join-Path $Results 'result.json') ok=$ok"
if ($ok) { exit 0 } else { exit 1 }
