#!/usr/bin/env bash
# One command on a fresh Ubuntu A100 box.
# Tokens stay in the environment. This script does not enable command tracing.
set -euo pipefail

export GIT_TERMINAL_PROMPT=0
FORGE_OUT="${FORGE_OUT:-$HOME/forge-out}"
REPO_URL="${FORGE_REPO_URL:-https://github.com/Mitroshenkov87/voxprint-model-forge.git}"
LLAMA_COMMIT="988190680d5a89fce97de3c20df2c2813731fd61"

prompt_secret() {
  local name="$1"
  if [[ -n "${!name:-}" ]]; then
    return 0
  fi
  if [[ ! -r /dev/tty ]]; then
    printf '%s is not set and there is no terminal to prompt\n' "$name" >&2
    exit 2
  fi
  local value
  printf '%s: ' "$name" >/dev/tty
  IFS= read -r -s value </dev/tty
  printf '\n' >/dev/tty
  printf -v "$name" '%s' "$value"
  export "$name"
}

prompt_secret HF_TOKEN
prompt_secret KAGGLE_API_TOKEN
if [[ -z "${HF_USER:-}" ]]; then
  if [[ ! -r /dev/tty ]]; then
    printf 'HF_USER is not set\n' >&2
    exit 2
  fi
  printf 'HF_USER: ' >/dev/tty
  IFS= read -r HF_USER </dev/tty
  export HF_USER
fi
export KAGGLE_USER="${KAGGLE_USER:-mitroshenkov87}"

if [[ ! -f "${BASH_SOURCE[0]:-}" ]]; then
  prompt_secret GH_TOKEN
  dest="${FORGE_REPO_DIR:-$HOME/voxprint-model-forge}"
  if [[ ! -f "$dest/scripts/bootstrap.sh" ]]; then
    git -c "http.extraHeader=Authorization: Bearer ${GH_TOKEN}" \
      clone --quiet "$REPO_URL" "$dest"
  fi
  exec bash "$dest/scripts/bootstrap.sh"
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$FORGE_OUT/logs" "$FORGE_OUT/stages"
# Scrub before the log file. The prompt above goes only to the terminal.
exec > >(python3 "$ROOT/scripts/logscrub.py" | tee -a "$FORGE_OUT/logs/bootstrap.log") 2>&1

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'nvidia-smi is missing\n' >&2
  exit 9
fi
nvidia-smi

PYTHONPATH="$ROOT/src" python3 -c '
from pathlib import Path
from voxprint_forge.plan import format_estimates, load_plan
print(format_estimates(load_plan(Path("'"$ROOT"'/plans/plan.toml"))))
'

export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

VENV="$FORGE_OUT/venv"
install_torch() {
  local py="$1"
  uv python install "$py"
  uv venv --python "$py" "$VENV" --clear
  uv pip install --python "$VENV" "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu130
}

if ! install_torch 3.14; then
  if ! install_torch 3.13; then
    install_torch 3.12
  fi
fi
uv pip install --python "$VENV" -e "$ROOT" hf_transfer \
  --extra-index-url https://download.pytorch.org/whl/cu130

"$VENV/bin/forge" verify-logins

LLAMA_SRC="$FORGE_OUT/llama.cpp"
if [[ ! -f "$FORGE_OUT/stages/llama-cpp.done" ]]; then
  if [[ ! -d "$LLAMA_SRC/.git" ]]; then
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_SRC"
  fi
  git -C "$LLAMA_SRC" fetch --depth 1 origin "$LLAMA_COMMIT"
  git -C "$LLAMA_SRC" checkout "$LLAMA_COMMIT"
  cmake -S "$LLAMA_SRC" -B "$LLAMA_SRC/build" \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="80;89;120"
  cmake --build "$LLAMA_SRC/build" --parallel "$(nproc)"
  printf '%s\n' "$LLAMA_COMMIT" >"$FORGE_OUT/stages/llama-cpp.done"
fi

FORGE_OUT="$FORGE_OUT" FORGE_VENV="$VENV" bash "$ROOT/scripts/vm_run.sh"

stamp="$(date -u +%F)"
tar -C "$FORGE_OUT" -cf "$FORGE_OUT/forge-results-${stamp}.tar" work manifests
sha256sum "$FORGE_OUT/forge-results-${stamp}.tar" | tee "$FORGE_OUT/forge-results-${stamp}.tar.sha256"
printf 'results: %s\n' "$FORGE_OUT/forge-results-${stamp}.tar"
