#!/usr/bin/env bash
# Nunchaku INT4 matrix on Linux or Git Bash. Non-interactive.
# The 4090 host is Windows Server 2025; use scripts/nunchaku_edit_test.ps1 there.
# This script does not prompt and does not install upstream nunchaku.
set +x
set -euo pipefail

export GIT_TERMINAL_PROMPT=0
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

CACHE="${NUNCHAKU_CACHE:-${HOME}/.cache/voxprint-forge/nunchaku-edit}"
PY="${NUNCHAKU_PYTHON:-}"
if [[ -z "$PY" ]]; then
  if [[ -x "${CACHE}/venv/Scripts/python.exe" ]]; then
    PY="${CACHE}/venv/Scripts/python.exe"
  elif [[ -x "${CACHE}/venv/bin/python" ]]; then
    PY="${CACHE}/venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PY=python3
  else
    PY=python
  fi
fi

if [[ ! -x "$PY" ]] && ! command -v "$PY" >/dev/null 2>&1; then
  printf 'python not found. Install Python 3.12 (Windows) or leave a python on PATH. Exit 2.\n' >&2
  exit 2
fi

exec "$PY" -m voxprint_forge.diffusion.nunchaku_edit "$@"
