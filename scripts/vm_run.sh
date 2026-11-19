#!/usr/bin/env bash
# Model work once the toolchain exists. scripts/bootstrap.sh calls this.
# Rented-VM logins (FORGE_HOST_KIND=rented-vm) stay for the rental.
# This script leaves HF_TOKEN and KAGGLE_API_TOKEN in place and does not delete token files.
# The hub directory removed at the end is the weight cache, not credentials.
set -euo pipefail

export FORGE_HOST_KIND="${FORGE_HOST_KIND:-local}"

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

export FORGE_PYTHON="${FORGE_PYTHON:-$VENV/bin/python}"
export FORGE_JOBS="${FORGE_JOBS:-2}"

# Global flags precede the subcommand. argparse rejects them after run-all.
"$VENV/bin/forge" --plan "$ROOT/plans/plan.toml" --root "$ROOT" --forge-out "$FORGE_OUT" preflight
"$VENV/bin/forge" --plan "$ROOT/plans/plan.toml" --root "$ROOT" --forge-out "$FORGE_OUT" run-all
printf 'ALL-DONE\n'

# Hub snapshots are copies of weights the work tree already consumed.
rm -rf "$HF_HOME/hub"
