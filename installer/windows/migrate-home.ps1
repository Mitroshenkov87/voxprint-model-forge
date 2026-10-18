#Requires -Version 5.1
# VOXPrint AI Suite: one-time move of the per-user data home (voxprint-common 0.2.5).
#   %LOCALAPPDATA%\Voxprint  ->  %LOCALAPPDATA%\VOXPrint AI Suite
# The setup runs this as the signed-in (original) user from PrepareToInstall, before any
# file is copied and before any Python runs: Windows cannot rename a folder that a running
# Python or app holds open.
#
# Rules (same as voxprint_common.home_migration):
#   old missing                -> nothing to do (exit 0)
#   old present, new missing   -> rename old to new (exit 0); an empty new folder counts as missing
#   both present               -> refuse: nothing is moved or merged, new is used, old is left
#                                 untouched for the user to review (exit 10, not fatal)
#   VOXPRINT_HOME set          -> that folder is the home and is never moved (exit 0)
#   an app or Python is running from the old home or the program folder -> exit 2, nothing moved
#   the rename fails           -> exit 3, old folder unchanged
#   new is a file or a link    -> exit 4, nothing moved
# Running it again after a move is a no-op. Log: <new home>\logs\home-migration.log, or
# %TEMP%\voxprint-home-migration.log while no new home exists.
param([string]$AppDir = "")

$ErrorActionPreference = "Stop"
$Old = Join-Path $env:LOCALAPPDATA "Voxprint"
$New = Join-Path $env:LOCALAPPDATA "VOXPrint AI Suite"
$Lines = New-Object System.Collections.ArrayList

function Say([string]$Message) {
    [void]$Lines.Add(("{0} {1}" -f (Get-Date -Format "o"), $Message))
}

function Finish([int]$Code) {
    Say "exit $Code"
    $dir = if ((Test-Path -LiteralPath $New -PathType Container)) { Join-Path $New "logs" } else { $env:TEMP }
    try {
        New-Item -ItemType Directory -Force $dir | Out-Null
        $name = if ($dir -eq $env:TEMP) { "voxprint-home-migration.log" } else { "home-migration.log" }
        Add-Content -LiteralPath (Join-Path $dir $name) -Value $Lines -Encoding UTF8
    } catch { }
    exit $Code
}

function Test-Link([string]$Path) {
    $item = Get-Item -LiteralPath $Path -Force
    return [bool]($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
}

try {
    Say "home migration: old=$Old new=$New app=$AppDir user=$env:USERNAME"
    if ($env:VOXPRINT_HOME) {
        Say "VOXPRINT_HOME is set; that folder is the home and is not moved"
        Finish 0
    }
    $hasOld = Test-Path -LiteralPath $Old
    $hasNew = Test-Path -LiteralPath $New
    if ($hasNew -and (-not (Test-Path -LiteralPath $New -PathType Container) -or (Test-Link $New))) {
        Say "REFUSED: $New is not a plain folder; nothing moved"
        Finish 4
    }
    if (-not $hasOld) {
        if ($hasNew) { Say "no-op: already at $New; no legacy folder" } else { Say "no-op: no legacy folder and no suite home yet" }
        Finish 0
    }
    if (-not (Test-Path -LiteralPath $Old -PathType Container) -or (Test-Link $Old)) {
        Say "legacy $Old is not a plain folder; left untouched"
        Finish 0
    }
    if ($hasNew -and @(Get-ChildItem -LiteralPath $New -Force).Count -gt 0) {
        Say "REFUSED: both $Old and $New exist. Nothing was moved or merged. The apps use $New; review $Old and move what you still need by hand."
        Finish 10
    }

    # Nothing may run from the old home (shared runtime) or from the program folder.
    $roots = @($Old)
    if ($AppDir) { $roots += $AppDir }
    $busy = @(Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $p = $null
        try { $p = $_.Path } catch { }
        ($p -and @($roots | Where-Object { $p.StartsWith($_ + "\", [StringComparison]::OrdinalIgnoreCase) }).Count -gt 0) -or
        ($_.ProcessName -like "voxprint-*")
    })
    if ($busy.Count -gt 0) {
        $names = ($busy | ForEach-Object { "$($_.ProcessName) ($($_.Id))" }) -join ", "
        Say "REFUSED: close these VOXPrint programs first: $names; nothing moved"
        Finish 2
    }

    if ($hasNew) {
        Remove-Item -LiteralPath $New -Force   # empty folder only (checked above)
        Say "removed the empty $New"
    }
    try {
        [IO.Directory]::Move($Old, $New)
    } catch {
        Say "FAILED: could not rename $Old to ${New}: $($_.Exception.Message); the old folder is unchanged"
        Finish 3
    }
    $count = @(Get-ChildItem -LiteralPath $New -Recurse -Force -File -ErrorAction SilentlyContinue).Count
    Say "moved $Old to $New ($count files)"
    Finish 0
} catch {
    Say "FAILED: $($_.Exception.Message)"
    Finish 3
}
