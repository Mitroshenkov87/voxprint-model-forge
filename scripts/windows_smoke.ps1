#Requires -Version 5.1
<#
.SYNOPSIS
Windows installer smoke for VOXPrint AI Model Forge: install, wheel upgrade, --version, preflight,
LoRA dry-run on stub data, uninstall. No GPU work, no model downloads, no tokens.

.DESCRIPTION
Run from an elevated PowerShell 7 (the setup installs to Program Files). Windows PowerShell 5.1 also works.

    pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\windows_smoke.ps1 -Setup .\VoxprintModelForgeSetup.exe

-Setup        The setup exe (CI artifact). Default: dist\VoxprintModelForgeSetup.exe, then the script folder.
-RequireGpu   Also require the preflight CUDA check (use on the laptop; CI has no GPU).
-KeepInstalled  Skip the uninstall step.
-MigrationTest  CI only: when neither data home exists, plant a fake %LOCALAPPDATA%\Voxprint with a dummy
               model first, then check the setup moved it and that a second install is a no-op.

Steps and pass rules:
  0 home       (with -MigrationTest on a clean profile) a fake %LOCALAPPDATA%\Voxprint\models\smoke-dummy file.
  0w wheels    voxprint_common-0.4.0-py3-none-any.whl and voxprint_forge-0.0.9-py3-none-any.whl planted in {app}\wheels.
  1 install    silent setup exits 0; venv, venv-mivolo, CLI, mdl.json, the compiled launcher and its icon, the icon file,
               uv.exe, LICENSE and the PDF guides in docs\ exist. The Start menu folder holds one Forge shortcut
               (VOXPrint AI Model Forge.lnk): target is the launcher, icon is the app icon, AppUserModelID is
               VOXPrint.AISuite.ModelForge. No Uninstall or guide shortcut, no desktop shortcut. Older Uninstall
               and guide shortcuts are removed. The uninstall entry name, icon and publisher match. No model weights.
  1w upgrade   the planted wheels are gone, and the app venv's voxprint-common version is the pin in the app wheel.
  2 version    voxprint-forge-cli --version exits 0 and prints the BUILD.json version.
  2b help      voxprint-forge-cli lora train --help exits 0 in under 3 seconds.
  2a te-quant  the app venv imports bitsandbytes and its version is 0.50.2.
  3 preflight  voxprint-forge-cli --json preflight --refset prints a report; diffusers, img2img and the
               MiVOLO env pass. Missing models are listed, not failed (a fresh install has none).
  4 lora       a 12-picture stub project; lora train --dry-run --base z-image --vram 24 returns state dry-run.
  4a migration the setup moved %LOCALAPPDATA%\Voxprint to %LOCALAPPDATA%\VOXPrint AI Suite before any Python ran:
               the old folder is gone, the dummy sits under paths.models_dir() with the same sha256
               (checked right after the setup: on disk first, then through voxprint_common), and a second silent install
               exits 0, logs "no-op" and changes nothing. Without -MigrationTest only the move rule is
               checked against the folders that were there before the install.
  5 uninstall  unins000.exe removes the program folder, this app's shortcut and the uninstall entry.
               The shared Start menu folder VOXPrint AI Suite is removed only when it is empty.
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
$Group = Split-Path -Parent $Menu
$GuideLinks = @(
    (Join-Path $Group "Model Forge guide (EN).lnk"),
    (Join-Path $Group ([string][char]0x0420 + [char]0x0443 +
        [char]0x043A + [char]0x043E + [char]0x0432 + [char]0x043E + [char]0x0434 + [char]0x0441 + [char]0x0442 + [char]0x0432 + [char]0x043E +
        " Model Forge (RU).lnk"))
)
# Names this app used to install. Other suite apps may keep their own shortcuts in $Group.
$OldLinks = @((Join-Path $Group "Uninstall VOXPrint AI Model Forge.lnk")) + $GuideLinks
$Desktop = Join-Path $env:PUBLIC "Desktop\VOXPrint AI Model Forge.lnk"
$OldHome = Join-Path $env:LOCALAPPDATA "Voxprint"
$NewHome = Join-Path $env:LOCALAPPDATA "VOXPrint AI Suite"
$Dummy = "models\smoke-dummy\smoke-dummy-model.bin"
$Weights = @("*.safetensors", "*.gguf", "*.bin", "*.pt", "*.pth", "*.onnx", "*.ckpt")
$StaleWheels = @(
    "voxprint_common-0.4.0-py3-none-any.whl",
    "voxprint_forge-0.0.9-py3-none-any.whl"
)

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
    # Run outside any source checkout, as an installed app does.
    Push-Location ([System.IO.Path]::GetTempPath())
    try {
        $out = & $Cli @ArgList 2>> $err | Out-String
        $code = $LASTEXITCODE
    } finally { Pop-Location }
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

[void](Invoke-Step "upgrade-plant" {
    $wheels = Join-Path $Dir "wheels"
    New-Item -ItemType Directory -Force -Path $wheels | Out-Null
    foreach ($name in $StaleWheels) {
        [IO.File]::WriteAllText((Join-Path $wheels $name), "stale wheel")
    }
    "planted $($StaleWheels -join ', ')"
})

$installed = Invoke-Step "install" {
    # No older setup is installed first. Plant the shortcuts that previous builds created so this
    # setup's InstallDelete has something to remove.
    New-Item -ItemType Directory -Force -Path $Group | Out-Null
    foreach ($link in $OldLinks) { [IO.File]::WriteAllText($link, "old shortcut") }
    $p = Start-Process $SetupExe -ArgumentList "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-",
        "/LOG=`"$(Join-Path $Results 'setup.log')`"" -Wait -PassThru
    if ($p.ExitCode -ne 0) {
        $installLog = Join-Path $Dir "setup-work\install.log"
        if (Test-Path $installLog) { Copy-Item $installLog (Join-Path $Results "install.log") }
        throw "setup exited $($p.ExitCode); see install.log"
    }
    foreach ($f in @("voxprint-model-forge.exe", "mdl.json", "assets\voxprint-model-forge.ico",
            "venv\Scripts\voxprint-model-forge.exe", "venv\Scripts\voxprint-forge-cli.exe",
            "venv-mivolo\Scripts\python.exe", "uv.exe", "LICENSE", "ACCEPTABLE_USE.md", "unins000.exe",
            "docs\VOXPrint-Model-Forge-Guide-en.pdf", "docs\VOXPrint-Model-Forge-Guide-ru.pdf")) {
        if (-not (Test-Path (Join-Path $Dir $f))) { throw "missing $f" }
    }
    $owned = @("VOXPrint AI Model Forge.lnk") + @($OldLinks | ForEach-Object { Split-Path -Leaf $_ })
    $present = @(Get-ChildItem -LiteralPath $Group -Filter *.lnk | Where-Object { $owned -contains $_.Name })
    if ($present.Count -ne 1 -or $present[0].Name -ne "VOXPrint AI Model Forge.lnk") {
        throw "Start menu should hold one Forge shortcut, found: $($present.Name -join ', ')"
    }
    foreach ($link in $OldLinks) { if (Test-Path -LiteralPath $link) { throw "old shortcut still present: $link" } }
    $uninstallLeft = @(Get-ChildItem -LiteralPath $Group -Filter "Uninstall*.lnk" | Where-Object { $_.Name -like "*Model Forge*" })
    if ($uninstallLeft) { throw "uninstall shortcut present: $($uninstallLeft.Name -join ', ')" }
    if (Test-Path -LiteralPath $Desktop) { throw "a silent install created a desktop shortcut" }
    $shortcut = (New-Object -ComObject WScript.Shell).CreateShortcut($Menu)
    $launcher = Join-Path $Dir "voxprint-model-forge.exe"
    if ($shortcut.TargetPath -ne $launcher) { throw "shortcut target is $($shortcut.TargetPath)" }
    $wantIcon = Join-Path $Dir "assets\voxprint-model-forge.ico"
    if (-not $shortcut.IconLocation.StartsWith($wantIcon, [StringComparison]::OrdinalIgnoreCase)) {
        throw "IconLocation is $($shortcut.IconLocation)"
    }
    $item = (New-Object -ComObject Shell.Application).Namespace($Group).ParseName("VOXPrint AI Model Forge.lnk")
    $aumid = $item.ExtendedProperty("System.AppUserModel.ID")
    if ($aumid -ne "VOXPrint.AISuite.ModelForge") { throw "AppUserModelID is '$aumid'" }
    Add-Type -AssemblyName System.Drawing
    $drawn = [System.Drawing.Icon]::ExtractAssociatedIcon($launcher)
    if ($null -eq $drawn) { throw "launcher has no icon" }
    $drawn.Dispose()
    $key = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\$AppId"
    if (-not (Test-Path $key)) { throw "uninstall entry missing" }
    $entry = Get-ItemProperty $key
    if ($entry.DisplayName -ne "VOXPrint AI Model Forge") { throw "DisplayName is $($entry.DisplayName)" }
    if ([string]$entry.Publisher -ne "Aleksandr Mitroshenkov") { throw "Publisher is $($entry.Publisher)" }
    $displayIcon = ([string]$entry.DisplayIcon).Trim('"')
    if ($displayIcon -notmatch 'assets\\voxprint-model-forge\.ico(,\d+)?$') { throw "DisplayIcon is $displayIcon" }
    $bundled = Get-ChildItem -Recurse -File (Join-Path $Dir "wheels") -Include $Weights -ErrorAction SilentlyContinue
    if ($bundled) { throw "model weights in the setup: $($bundled.Name -join ', ')" }
    & $Py -c "from voxprint_forge.lora.backends import mivolo_python; import sys; sys.exit(0 if 'venv-mivolo' in mivolo_python() else 1)"
    if ($LASTEXITCODE -ne 0) { throw "the app does not find venv-mivolo" }
    "installed to $Dir; one Start menu launcher; AUMID $aumid; uninstall entry $($entry.DisplayName)"
}

if ($installed) {
    [void](Invoke-Step "upgrade-wheels" {
        foreach ($name in $StaleWheels) {
            $path = Join-Path (Join-Path $Dir "wheels") $name
            if (Test-Path -LiteralPath $path) { throw "stale wheel remains: $name" }
        }
        $ErrorActionPreference = "Continue"
        $err = Join-Path $Results "upgrade-wheels-stderr.txt"
        $checker = Join-Path $Results "check_common_version.py"
        @'
import importlib.metadata as metadata
import json
import re
import sys
import zipfile
from pathlib import Path

app = Path(sys.argv[1])
version = json.loads((app / "BUILD.json").read_text(encoding="utf-8"))["version"]
wheels = sorted((app / "wheels").glob(f"voxprint_forge-{version}-*.whl"))
if len(wheels) != 1:
    raise SystemExit(f"expected one voxprint_forge wheel for {version}, found {len(wheels)}")
with zipfile.ZipFile(wheels[0]) as archive:
    name = next(item for item in archive.namelist() if item.endswith(".dist-info/METADATA"))
    text = archive.read(name).decode("utf-8")
pins = set()
for line in text.splitlines():
    if not line.startswith("Requires-Dist:"):
        continue
    req = "".join(line.split(":", 1)[1].split(";", 1)[0].split())
    match = re.fullmatch(r"voxprint-common(?:\[[^\]]*\])?==([0-9][0-9A-Za-z.]*)", req)
    if match:
        pins.add(match.group(1))
if len(pins) != 1:
    raise SystemExit("expected one voxprint-common pin, found %s" % sorted(pins))
pin = next(iter(pins))
installed = metadata.version("voxprint-common")
if installed != pin:
    raise SystemExit("voxprint-common %s, pin %s" % (installed, pin))
print(installed)
'@ | Set-Content -LiteralPath $checker -Encoding ascii
        $got = (& $Py $checker $Dir 2>> $err | Out-String).Trim()
        if ($LASTEXITCODE -ne 0) {
            $stderr = if (Test-Path -LiteralPath $err) { (Get-Content -Raw -LiteralPath $err).Trim() } else { "" }
            throw "pinned voxprint-common check failed: $stderr"
        }
        "voxprint-common $got"
    })
}

function Get-MigrationLog {
    $log = Join-Path $NewHome "logs\home-migration.log"
    if (Test-Path -LiteralPath $log) { return @(Get-Content -LiteralPath $log) }
    $tmp = Join-Path $env:TEMP "voxprint-home-migration.log"
    if (Test-Path -LiteralPath $tmp) { return @(Get-Content -LiteralPath $tmp) }
    return @()
}

# Right after the setup, before the smoke runs the app.
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
            # On disk first. Only migrate-home.ps1 writes the "moved" line, so it proves the setup did the move.
            $moved = Join-Path $NewHome $Dummy
            if (-not (Test-Path -LiteralPath $moved)) { throw "the dummy model is not at $moved" }
            if ((Get-FileHash -LiteralPath $moved -Algorithm SHA256).Hash.ToLower() -ne $script:DummySha) { throw "the dummy model changed" }
            if (-not (Get-MigrationLog | Where-Object { $_ -match " moved " })) { throw "home-migration.log has no move line" }
            # Then through voxprint_common: models_dir() must be <new home>\models.
            $saved = @{}
            foreach ($n in @("VOXPRINT_HOME", "VOXPRINT_MODELS_DIR", "VOXPRINT_TRT_CACHE")) { $saved[$n] = [Environment]::GetEnvironmentVariable($n); [Environment]::SetEnvironmentVariable($n, $null) }
            $md = (& $Py -c "from voxprint_common.paths import models_dir; print(models_dir())" | Out-String).Trim()
            if ($md -ne (Join-Path $NewHome "models")) { throw "paths.models_dir() is '$md', want $(Join-Path $NewHome 'models')" }
            if (-not (Test-Path -LiteralPath (Join-Path $md "smoke-dummy\smoke-dummy-model.bin"))) { throw "the dummy is not under models_dir()" }
            $trt = (& $Py -c "from voxprint_common.paths import trt_cache_dir; print(trt_cache_dir())" | Out-String).Trim()
            foreach ($n in $saved.Keys) { [Environment]::SetEnvironmentVariable($n, $saved[$n]) }
            if ($trt -ne (Join-Path $NewHome "trt-cache")) { throw "paths.trt_cache_dir() is '$trt', want $(Join-Path $NewHome 'trt-cache')" }
            if (Test-Path -LiteralPath $OldHome) { throw "voxprint_common recreated $OldHome" }
            $detail += "; dummy under models_dir() $md, sha256 unchanged; trt_cache_dir() $trt"
        }
        $detail
    })
}

if ($installed) {
    [void](Invoke-Step "mdl" {
        $mdlJson = Join-Path $Dir "mdl.json"
        if (-not (Test-Path -LiteralPath $mdlJson)) { throw "missing mdl.json" }
        $mdl = Join-Path $Dir "venv\Scripts\voxprint-mdl.exe"
        if (-not (Test-Path -LiteralPath $mdl)) { throw "missing $mdl" }
        $ErrorActionPreference = "Continue"
        $err = Join-Path $Results "mdl-stderr.txt"
        Push-Location ([System.IO.Path]::GetTempPath())
        try {
            $out = & $mdl --json download --dry-run --manifest $mdlJson --model qwen3-vl-4b-instruct --model mivolo-v2 --model yunet-2023mar --model dinov2-base 2>> $err | Out-String
            $code = $LASTEXITCODE
        } finally { Pop-Location }
        Add-Content -LiteralPath $Log -Value ("> voxprint-mdl download --dry-run`n" + $out) -Encoding utf8
        if ($code -ne 0) { throw "voxprint-mdl exited $code" }
        $result = $null
        foreach ($line in ($out -split "`r?`n")) {
            $trim = $line.Trim()
            if ($trim.StartsWith("{")) { $result = $trim | ConvertFrom-Json }
        }
        if ($null -eq $result -or $result.ok -ne $true) { throw "voxprint-mdl result ok is not true" }
        $status = Invoke-Cli @("--json", "models", "status")
        if ($status.code -ne 0) { throw "models status exited $($status.code)" }
        foreach ($id in @("qwen3-vl-4b-instruct", "mivolo-v2", "yunet-2023mar", "dinov2-base")) {
            if ($status.out -notmatch [regex]::Escape($id)) { throw "models status missing $id" }
        }
        "mdl.json; dry-run ok; models status lists qwen3-vl-4b-instruct, mivolo-v2, yunet-2023mar, dinov2-base"
    })

    [void](Invoke-Step "version" {
        $want = (Get-Content -Raw (Join-Path $Dir "BUILD.json") | ConvertFrom-Json).version
        $r = Invoke-Cli @("--version")
        if ($r.code -ne 0) { throw "--version exited $($r.code)" }
        if ($r.out -notmatch [regex]::Escape($want)) { throw "--version printed '$($r.out.Trim())', want $want" }
        $r.out.Trim()
    })

    [void](Invoke-Step "lora-train-help" {
        $sw = [Diagnostics.Stopwatch]::StartNew()
        $r = Invoke-Cli @("lora", "train", "--help")
        $sw.Stop()
        $seconds = $sw.Elapsed.TotalSeconds
        if ($r.code -ne 0) { throw "lora train --help exited $($r.code)" }
        if ($seconds -ge 3) { throw ("lora train --help took {0:N2} s" -f $seconds) }
        "lora train --help {0:N2} s" -f $seconds
    })

    [void](Invoke-Step "te-quant" {
        # stderr stays in the log. Windows PowerShell 5.1 turns native stderr into error records.
        $ErrorActionPreference = "Continue"
        $err = Join-Path $Results "bitsandbytes-stderr.txt"
        & $Py -c "import bitsandbytes; raise SystemExit(0 if bitsandbytes.__version__ == '0.50.2' else 1)" 2>> $err
        if ($LASTEXITCODE -ne 0) { throw "import bitsandbytes 0.50.2 failed" }
        "bitsandbytes 0.50.2"
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

if ($installed) {
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
        if (Test-Path $Dir) {
            $left = @(Get-ChildItem -LiteralPath $Dir -Force | ForEach-Object { $_.Name })
            throw "uninstall left $Dir ($($left -join ', '))"
        }
        if (Test-Path $Menu) { throw "uninstall left the Start menu shortcut" }
        foreach ($link in $OldLinks) { if (Test-Path -LiteralPath $link) { throw "uninstall left $link" } }
        if (Test-Path "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\$AppId") { throw "uninstall left the uninstall entry" }
        # The shared folder stays when another app's shortcut is still in it.
        $left = @()
        if (Test-Path -LiteralPath $Group) {
            $left = @(Get-ChildItem -LiteralPath $Group -Filter *.lnk -ErrorAction SilentlyContinue)
        }
        "program folder, shortcut and uninstall entry removed; suite folder shortcuts left: $($left.Count)"
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
