#Requires -Version 7.0
<#
.SYNOPSIS
Unattended RTX 4090 laptop smoke. No prompts, no tokens, no tty.

Fetch and launch (session drop is fine; the window is hidden):

irm https://raw.githubusercontent.com/Mitroshenkov87/voxprint-model-forge/diffusion-int4-v4-smoke2/scripts/smoke_laptop.ps1 -OutFile "$env:TEMP\smoke_laptop.ps1"
Start-Process pwsh -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File',"$env:TEMP\smoke_laptop.ps1" -WindowStyle Hidden

Results: %LOCALAPPDATA%\VoxprintForgeSmoke\results\<timestamp>\result.json
The same path is printed and written to %LOCALAPPDATA%\VoxprintForgeSmoke\latest.txt.
#>
[CmdletBinding()]
param(
    [switch]$KeepVenv,
    [switch]$Force,
    [string]$Steps = 'smoke-tiny,quantfunc,zimage,calib-quant-timing',
    [int]$Minutes = 90,
    [int]$Blocks = 2
)

$ErrorActionPreference = 'Stop'
$ConfirmPreference = 'None'
$ProgressPreference = 'SilentlyContinue'
$PSNativeCommandUseErrorActionPreference = $false
$env:GIT_TERMINAL_PROMPT = '0'
$env:HF_HUB_ENABLE_HF_TRANSFER = '1'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$env:HF_HUB_DISABLE_PROGRESS_BARS = '1'
$env:PYTHONUNBUFFERED = '1'
Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
Remove-Item Env:HUGGING_FACE_HUB_TOKEN -ErrorAction SilentlyContinue

$Tag = 'diffusion-int4-v4-smoke2'
$RepoUrl = 'https://github.com/Mitroshenkov87/voxprint-model-forge'
$WheelName = 'nunchaku-1.3.0.dev20260629+cu13.0torch2.11-cp312-cp312-win_amd64.whl'
$WheelSha = 'ddc015bfa6fc6b84957532823c23418f5b986cb61e3c02c97a646cfe2a1fd955'
$WheelUrl = 'https://huggingface.co/tonera/vitoom-nunchaku/resolve/85bad5510d409570527b018387491fd0c81e4727/nunchaku-1.3.0.dev20260629+cu13.0torch2.11-cp312-cp312-win_amd64.whl'

$SmokeRoot = Join-Path $env:LOCALAPPDATA 'VoxprintForgeSmoke'
$Stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$Results = Join-Path $SmokeRoot "results\$Stamp"
$Log = Join-Path $Results 'smoke.log'
$Previous = $null
$LatestFile = Join-Path $SmokeRoot 'latest.txt'
if ((-not $Force) -and (Test-Path -LiteralPath $LatestFile)) {
    $Previous = (Get-Content -LiteralPath $LatestFile -Raw).Trim()
}
New-Item -ItemType Directory -Force -Path $Results | Out-Null
$Started = Get-Date
$script:ExitCode = 2

function Write-Log {
    param([string]$Message)
    $line = '{0} {1}' -f (Get-Date -Format 'o'), $Message
    Add-Content -LiteralPath $Log -Value $line -Encoding utf8NoBOM
}

function Write-SetupResult {
    param([string]$Message)
    $payload = @{
        schema      = 'voxprint-forge-smoke-laptop/1'
        ok          = $false
        exit_code   = 2
        error       = $Message
        started     = $Started.ToUniversalTime().ToString('o')
        results_dir = $Results
        steps       = @()
    }
    $payload | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $Results 'result.json') -Encoding utf8NoBOM
    Write-Log "setup failed: $Message"
}

function Invoke-Setup {
    param([string]$File, [string[]]$CommandArgs, [string]$What)
    Write-Log "$What : $File $($CommandArgs -join ' ')"
    & $File @CommandArgs *>> $Log
    if ($LASTEXITCODE -ne 0) {
        Write-SetupResult "$What exited $LASTEXITCODE"
        $script:ExitCode = 2
        exit 2
    }
}

try {
    Write-Log "results $Results"
    Write-Log "steps $Steps minutes $Minutes keep_venv=$KeepVenv"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-Log 'installing uv into the user profile'
        Invoke-Expression (Invoke-RestMethod -Uri 'https://astral.sh/uv/install.ps1')
        $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    }
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uv) {
        Write-SetupResult 'uv is not on PATH after the user-profile install'
        exit 2
    }
    $UvExe = $uv.Source

    $Checkout = $null
    $Parent = Split-Path -Parent $PSScriptRoot
    $Marker = Join-Path $Parent 'src\voxprint_forge\diffusion\smoke_laptop.py'
    if (Test-Path -LiteralPath $Marker) {
        $Checkout = $Parent
        Write-Log "using checkout $Checkout"
    } else {
        $Checkout = Join-Path $SmokeRoot 'repo'
        if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
            Write-SetupResult 'git is required to clone the tagged repo'
            exit 2
        }
        if (-not (Test-Path -LiteralPath (Join-Path $Checkout '.git'))) {
            Invoke-Setup 'git' @('-c', 'advice.detachedHead=false', 'clone', '--depth', '1', '--branch', $Tag, $RepoUrl, $Checkout) 'git clone'
        }
    }

    $Venv = Join-Path $SmokeRoot 'venv'
    $Python = Join-Path $Venv 'Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $Python)) {
        Invoke-Setup $UvExe @('python', 'install', '3.12') 'uv python install'
        Invoke-Setup $UvExe @('venv', '--python', '3.12', $Venv) 'uv venv'
    }
    Invoke-Setup $UvExe @(
        'pip', 'install', '--python', $Python,
        'torch==2.11.*', 'torchvision==0.26.*',
        '--index-url', 'https://download.pytorch.org/whl/cu130'
    ) 'torch cu130'
    Invoke-Setup $UvExe @(
        'pip', 'install', '--python', $Python,
        'diffusers==0.37.1', 'transformers==4.57.6',
        'accelerate>=1.9', 'peft>=0.17', 'pillow', 'safetensors', 'huggingface_hub',
        'hf_xet', 'hf_transfer'
    ) 'diffusers pins'

    $WheelDir = Join-Path $SmokeRoot 'wheels'
    New-Item -ItemType Directory -Force -Path $WheelDir | Out-Null
    $WheelPath = Join-Path $WheelDir $WheelName
    $HaveWheel = $false
    if (Test-Path -LiteralPath $WheelPath) {
        $Actual = (Get-FileHash -LiteralPath $WheelPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Actual -eq $WheelSha) {
            $HaveWheel = $true
        }
    }
    if (-not $HaveWheel) {
        Write-Log "downloading $WheelName"
        Invoke-Setup 'curl.exe' @(
            '-fL', '--retry', '8', '--retry-all-errors', '--retry-delay', '2',
            '-C', '-', '-o', $WheelPath, $WheelUrl
        ) 'wheel download'
        $Actual = (Get-FileHash -LiteralPath $WheelPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($Actual -ne $WheelSha) {
            Remove-Item -LiteralPath $WheelPath -Force -ErrorAction SilentlyContinue
            Write-SetupResult "wheel sha256 $Actual != $WheelSha"
            exit 2
        }
    }
    Invoke-Setup $UvExe @('pip', 'install', '--python', $Python, '--no-deps', $WheelPath) 'nunchaku wheel'

    $env:HF_HOME = Join-Path $SmokeRoot 'hf'
    $env:HUGGINGFACE_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
    $env:PYTHONPATH = Join-Path $Checkout 'src'
    New-Item -ItemType Directory -Force -Path $env:HF_HOME | Out-Null
    $Cache = Join-Path $SmokeRoot 'cache'
    New-Item -ItemType Directory -Force -Path $Cache | Out-Null

    Write-Log "python smoke compute_minutes=$Minutes blocks=$Blocks force=$Force"
    $RunArgs = @(
        '-m', 'voxprint_forge.diffusion.smoke_laptop', 'run',
        '--results', $Results,
        '--repo', $Checkout,
        '--cache', $Cache,
        '--steps', $Steps,
        '--minutes', "$Minutes",
        '--blocks', "$Blocks"
    )
    if ($Force) {
        $RunArgs += '--force'
    }
    if ($Previous) {
        $RunArgs += @('--resume-from', $Previous)
    }
    & $Python @RunArgs *>> $Log
    if ($null -eq $LASTEXITCODE) {
        $script:ExitCode = 2
    } else {
        $script:ExitCode = $LASTEXITCODE
    }
} catch {
    Write-SetupResult "$_"
    $script:ExitCode = 2
} finally {
    if ($script:ExitCode -eq 0) {
        foreach ($name in @('hf', 'cache', 'wheels')) {
            $path = Join-Path $SmokeRoot $name
            if (Test-Path -LiteralPath $path) {
                Write-Log "deleting $path"
                Remove-Item -LiteralPath $path -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    } else {
        Write-Log 'keeping the weight cache so a rerun can resume downloads'
    }
    if (-not $KeepVenv) {
        $venvPath = Join-Path $SmokeRoot 'venv'
        if (Test-Path -LiteralPath $venvPath) {
            Write-Log "deleting $venvPath"
            Remove-Item -LiteralPath $venvPath -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    $latest = Join-Path $SmokeRoot 'latest.txt'
    Set-Content -LiteralPath $latest -Value $Results -Encoding utf8NoBOM
    Write-Log "RESULTS $Results"
    Write-Output "RESULTS $Results"
}

exit $script:ExitCode
