#!/usr/bin/env bash
# Model work once the toolchain exists. scripts/bootstrap.sh calls this.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORGE_OUT="${FORGE_OUT:-$HOME/forge-out}"
VENV="${FORGE_VENV:-$FORGE_OUT/venv}"
export FORGE_LLAMA_SRC="${FORGE_LLAMA_SRC:-$FORGE_OUT/llama.cpp}"
export FORGE_LLAMA_BIN="${FORGE_LLAMA_BIN:-$FORGE_LLAMA_SRC/build/bin}"
export FORGE_ROOT="$ROOT"
export HF_HOME="${HF_HOME:-$FORGE_OUT/hf-home}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export GIT_TERMINAL_PROMPT=0
mkdir -p "$FORGE_OUT/logs" "$HF_HOME"

"$VENV/bin/forge" run-all --plan "$ROOT/plans/plan.toml" --root "$ROOT" --forge-out "$FORGE_OUT"

# Hub snapshots are copies of weights the work tree already consumed.
rm -rf "$HF_HOME/hub"
