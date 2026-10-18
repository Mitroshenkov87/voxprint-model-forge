#!/usr/bin/env bash
# G4R rerun on the Linux VM (synth4 layout). Thin wrapper around python -m voxprint_forge.g4r_rerun, the same code
# the Windows entry point (scripts/win_g4r_rerun.ps1) runs. Reference: the synth4 F16 (sha256 pinned). No download
# when the shipped tiers are already in $W/shipped, no re-quant, no upload.
#
# Usage: scripts/vm_g4r_rerun.sh --prompts FILE [--dry-run]
# Exit: 0 PASS, 1 FAIL, 2 usage, 3 probe harness invalid, 4 preflight, 5 pinned file mismatch.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORGE_OUT="${FORGE_OUT:-$HOME/forge-out}"
VENV="${FORGE_VENV:-$FORGE_OUT/venv}"
W="${G4R_WORK:-$FORGE_OUT/gemma4-requant}"
PROMPTS="" EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompts) PROMPTS="${2:-}"; shift 2 ;;
    --dry-run) EXTRA+=(--dry-run); shift ;;
    *) printf 'usage: vm_g4r_rerun.sh --prompts FILE [--dry-run]\n' >&2; exit 2 ;;
  esac
done
[[ -n "$PROMPTS" ]] || { printf 'usage: vm_g4r_rerun.sh --prompts FILE [--dry-run]\n' >&2; exit 2; }
if [[ -n "${G4R_PROMPT_SHA12:-}" ]]; then EXTRA+=(--prompt-sha12 "$G4R_PROMPT_SHA12"); fi
if [[ -n "${G4R_RERUN_DIR:-}" ]]; then EXTRA+=(--results "$G4R_RERUN_DIR"); fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
exec "$VENV/bin/python" -m voxprint_forge.g4r_rerun --llama-bin "${FORGE_LLAMA_BIN:?FORGE_LLAMA_BIN}" \
  --work "$W/rerun-work" --tiers "$W/out" --stock-dir "$W/shipped" --reference f16-local \
  --f16 "$W/out/gemma4-31b-requant-F16.gguf" --tensor-diff "$W/results/tensor-diff.json" \
  --private "${G4R_PRIVATE:-$HOME/.priv-in/g4r-rerun}" --port "${G4R_PORT:-8097}" "${EXTRA[@]}" \
  --prompts "$PROMPTS"
