#!/usr/bin/env bash
# One synth4 KL job on the A100.
# Usage: scripts/vm_run_kl.sh --job L1 [--json]
# Needs FORGE_LLAMA_BIN (llama-perplexity, llama-server) and, for safetensors
# jobs, FORGE_LLAMA_SRC (convert_hf_to_gguf.py from llama.cpp b11476).
# This script does not print tokens.
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
        printf 'usage: vm_run_kl.sh --job ID [--json]\n' >&2
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
      printf 'usage: vm_run_kl.sh --job ID [--json]\n' >&2
      exit 2
      ;;
  esac
done

if [[ -z "$JOB" ]]; then
  printf 'usage: vm_run_kl.sh --job ID [--json]\n' >&2
  exit 2
fi
if [[ ! -x "$VENV/bin/forge" ]]; then
  printf 'kl setup: forge venv is missing\n' >&2
  exit 2
fi

mkdir -p "$FORGE_OUT/logs" "$HF_HOME"
ARGS=(--root "$ROOT" --forge-out "$FORGE_OUT" kl --job "$JOB" --plan "$ROOT/plans/kl-synth4.toml")
if [[ "$JSON" -eq 1 ]]; then
  ARGS=(--json "${ARGS[@]}")
fi
"$VENV/bin/forge" "${ARGS[@]}"
