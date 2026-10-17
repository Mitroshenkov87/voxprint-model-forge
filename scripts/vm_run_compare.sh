#!/usr/bin/env bash
# One synth4 compare job.
# Usage: scripts/vm_run_compare.sh --job D1 [--json]
# Private prompts: set FORGE_PRIVATE_PROMPTS to a directory. Unset skips that set.
# This script does not print tokens, hosts, or prompt text.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORGE_OUT="${FORGE_OUT:-$HOME/forge-out}"
VENV="${FORGE_VENV:-$FORGE_OUT/venv}"
export FORGE_ROOT="$ROOT"
export FORGE_OUT
export HF_HOME="${HF_HOME:-$FORGE_OUT/hf-home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1

JOB=""
JSON=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --job)
      if [[ $# -lt 2 ]]; then
        printf 'usage: vm_run_compare.sh --job ID [--json]\n' >&2
        exit 2
      fi
      JOB="$2"
      shift 2
      ;;
    --json)
      JSON=1
      shift
      ;;
    *)
      printf 'usage: vm_run_compare.sh --job ID [--json]\n' >&2
      exit 2
      ;;
  esac
done

if [[ -z "$JOB" ]]; then
  printf 'usage: vm_run_compare.sh --job ID [--json]\n' >&2
  exit 2
fi
if [[ ! -x "$VENV/bin/forge" ]]; then
  printf 'compare setup: forge venv is missing\n' >&2
  exit 2
fi
if ! "$VENV/bin/python" -c 'import torchmetrics, lpips, PIL' >/dev/null 2>&1; then
  printf 'compare setup: the forge venv needs torchmetrics, lpips, and Pillow\n' >&2
  exit 2
fi

mkdir -p "$FORGE_OUT/logs" "$HF_HOME"
ARGS=(--root "$ROOT" --forge-out "$FORGE_OUT" compare --job "$JOB" --plan "$ROOT/plans/compare-synth4.toml")
if [[ "$JSON" -eq 1 ]]; then
  ARGS=(--json "${ARGS[@]}")
fi
"$VENV/bin/forge" "${ARGS[@]}"
