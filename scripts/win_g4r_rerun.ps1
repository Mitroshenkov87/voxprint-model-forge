<#
.SYNOPSIS
  G4R rerun on Windows (RTX 4090, 24 GB): Q8_0 reference rebuilt from the author's pinned safetensors, KL of the
  synth4 tiers against it (full llama-perplexity stdout kept), the fixed refusal probe (thinking off, stock arms
  gated first), verdict. No re-quant, no upload.

.DESCRIPTION
  Setup (both modes): llama.cpp b11476 CUDA 12.4 Windows release binaries and the matching cudart, and the llama.cpp
  source at 988190680 (convert_hf_to_gguf.py), each checked against a pinned sha256; a Python venv under -Work with
  pinned packages and this tree. These are tools, not model weights.
  -DryRun stops after the preflight: no model download, no GGUF hashing, no GPU work.
  Without -DryRun: stock tiers and the author's safetensors (about 62.6 GB) are fetched at pinned revisions and
  sha256-checked, converted to Q8_0, and the safetensors are deleted.

  Exit: 0 PASS, 1 FAIL, 2 usage, 3 probe harness invalid (stock arm > 2/24 empty), 4 preflight,
        5 pinned file mismatch, 6 setup.

.EXAMPLE
  pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\win_g4r_rerun.ps1 -Work D:\g4r -Tiers D:\g4r\tiers `
    -Prompts $env:USERPROFILE\.priv-in\gemma4-refusal-probe.txt -TensorDiff D:\g4r\tensor-diff.json -DryRun
#>
param(
  [Parameter(Mandatory = $true)][string]$Work,
  [Parameter(Mandatory = $true)][string]$Tiers,
  [Parameter(Mandatory = $true)][string]$Prompts,
  [string]$TensorDiff = "",
  [string]$Python = "python",
  [int]$Port = 8097,
  [switch]$DryRun
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

# PINS begin (tests/test_g4r_rerun.py checks these against the release and the module)
$Pins = @(
  @{ Url = "https://github.com/ggml-org/llama.cpp/releases/download/b11476/llama-b11476-bin-win-cuda-12.4-x64.zip";
     Name = "llama-b11476-bin-win-cuda-12.4-x64.zip";
     Sha = "b30289b92274bb7e89200832f4bcfd9e731b716a4056d00b61440f623d4cc8e8"; Dest = "llama-b11476" },
  @{ Url = "https://github.com/ggml-org/llama.cpp/releases/download/b11476/cudart-llama-bin-win-cuda-12.4-x64.zip";
     Name = "cudart-llama-bin-win-cuda-12.4-x64.zip";
     Sha = "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6"; Dest = "llama-b11476" },
  @{ Url = "https://codeload.github.com/ggml-org/llama.cpp/zip/988190680d5a89fce97de3c20df2c2813731fd61";
     Name = "llama.cpp-988190680.zip";
     Sha = "8bcb20c62ffc72e406d0b4db76929559dac9be2578c19ba68d7a35bb33775fcb"; Dest = "llama-src" }
)
# PINS end
$PipPins = @("numpy==2.5.2", "safetensors==0.8.0", "transformers==5.20.1", "sentencepiece==0.2.2",
             "protobuf==5.29.6", "huggingface_hub==1.33.0")
$TorchPin = "torch==2.14.1"

function Write-Log([string]$Text) {
  Write-Host ("[g4r-win {0}] {1}" -f (Get-Date).ToUniversalTime().ToString("HH:mm:ssZ"), $Text)
}
function Stop-Run([string]$Text, [int]$Code) { Write-Log "FAIL: $Text"; exit $Code }

New-Item -ItemType Directory -Force -Path $Work, (Join-Path $Work "downloads") | Out-Null
$env:HF_HOME = Join-Path $Work "hf-home"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONUTF8 = "1"

# 1. Pinned tools.
foreach ($pin in $Pins) {
  $zip = Join-Path $Work ("downloads\" + $pin.Name)
  if (-not (Test-Path $zip)) {
    Write-Log ("download " + $pin.Name)
    & curl.exe -sSfL -o $zip $pin.Url
    if ($LASTEXITCODE -ne 0) { Stop-Run ("download " + $pin.Name) 6 }
  }
  $sha = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLower()
  if ($sha -ne $pin.Sha) { Remove-Item $zip; Stop-Run ($pin.Name + " sha256 $sha is not the pinned " + $pin.Sha) 5 }
  Write-Log ($pin.Name + " sha256 ok")
  $dest = Join-Path $Work $pin.Dest
  $stamp = Join-Path $dest ("." + $pin.Name + ".ok")
  if (-not (Test-Path $stamp)) {
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Expand-Archive -Path $zip -DestinationPath $dest -Force
    Set-Content -Path $stamp -Value $pin.Sha
  }
}
$LlamaBin = Join-Path $Work "llama-b11476"
if (-not (Test-Path (Join-Path $LlamaBin "llama-server.exe"))) {
  $found = Get-ChildItem -Path $LlamaBin -Recurse -Filter "llama-server.exe" | Select-Object -First 1
  if (-not $found) { Stop-Run "llama-server.exe not in the b11476 zip" 6 }
  $LlamaBin = $found.DirectoryName
}
$LlamaSrc = Join-Path $Work "llama-src\llama.cpp-988190680d5a89fce97de3c20df2c2813731fd61"

# 2. Python venv with pinned packages and this tree.
$Venv = Join-Path $Work "venv"
$Py = Join-Path $Venv "Scripts\python.exe"
$VenvStamp = Join-Path $Venv ".g4r-deps.ok"
if (-not (Test-Path $VenvStamp)) {
  Write-Log "venv: create and install pinned packages"
  & $Python -m venv $Venv; if ($LASTEXITCODE -ne 0) { Stop-Run "python -m venv" 6 }
  & $Py -m pip install -q --upgrade pip; if ($LASTEXITCODE -ne 0) { Stop-Run "pip upgrade" 6 }
  & $Py -m pip install -q $TorchPin --index-url https://download.pytorch.org/whl/cpu
  if ($LASTEXITCODE -ne 0) { Stop-Run "pip install torch" 6 }
  & $Py -m pip install -q @PipPins; if ($LASTEXITCODE -ne 0) { Stop-Run "pip install pins" 6 }
  & $Py -m pip install -q (Join-Path $LlamaSrc "gguf-py"); if ($LASTEXITCODE -ne 0) { Stop-Run "pip install gguf-py" 6 }
  & $Py -m pip install -q (Join-Path $Root "vendor\voxprint_common-0.5.2-py3-none-any.whl")
  if ($LASTEXITCODE -ne 0) { Stop-Run "pip install voxprint_common" 6 }
  & $Py -m pip install -q $Root; if ($LASTEXITCODE -ne 0) { Stop-Run "pip install forge tree" 6 }
  Set-Content -Path $VenvStamp -Value (Get-Date).ToUniversalTime().ToString("o")
}
$env:PYTHONPATH = Join-Path $Root "src"

# 3. The rerun itself (same code as scripts/vm_g4r_rerun.sh).
$RunArgs = @("-m", "voxprint_forge.g4r_rerun", "--llama-bin", $LlamaBin, "--work", $Work, "--tiers", $Tiers,
             "--prompts", $Prompts, "--reference", "q8_0-upstream", "--llama-src", $LlamaSrc, "--fit",
             "--private", (Join-Path $Work "private"), "--port", "$Port")
if ($TensorDiff) { $RunArgs += @("--tensor-diff", $TensorDiff) }
if ($DryRun) { $RunArgs += "--dry-run" }
elseif (-not $TensorDiff) { Stop-Run "a real run needs -TensorDiff (synth4 results\tensor-diff.json)" 2 }
& $Py @RunArgs
exit $LASTEXITCODE
