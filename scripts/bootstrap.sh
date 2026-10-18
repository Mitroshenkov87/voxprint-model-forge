#!/usr/bin/env bash
# One command on an Ubuntu 24.04 synth4 box, from a bundle with no .git directory.
# This script does not enable command tracing and does not unset tokens.
# FORGE_HOST_KIND=local (default) keeps tokens in the process only.
# FORGE_HOST_KIND=rented-vm logs in once after the venv exists and leaves the logins.
set +x
set -euo pipefail

export GIT_TERMINAL_PROMPT=0
export FORGE_HOST_KIND="${FORGE_HOST_KIND:-local}"
FORGE_OUT="${FORGE_OUT:-$HOME/forge-out}"
export HF_HOME="${HF_HOME:-$FORGE_OUT/hf-home}"

release_bootstrap_lock() {
  set +x
  if [[ -n "${FORGE_OUT:-}" ]]; then
    rm -f "$FORGE_OUT/bootstrap.lock" || true
  fi
}
trap release_bootstrap_lock EXIT INT TERM

# A tmux pty can exist before anyone attaches, and /dev/tty can be readable
# when stdin is not a terminal. On Windows CI, -t 0 is sometimes true while a
# read from /dev/tty still blocks. Never open /dev/tty in those cases.
prompt_secret() {
  local name="$1"
  set +x
  if [[ -n "${!name:-}" ]]; then
    return 0
  fi
  if [[ -n "${CI:-}" || -n "${FORGE_NONINTERACTIVE:-}" || ! -t 0 ]]; then
    printf '%s is not set. Export it before starting. This run is non-interactive, so it will not prompt.\n' "$name" >&2
    exit 2
  fi
  if [[ ! -r /dev/tty ]]; then
    printf '%s is not set. Export it before starting. This run is non-interactive, so it will not prompt.\n' "$name" >&2
    exit 2
  fi
  local value
  printf '%s: ' "$name" >/dev/tty
  IFS= read -r -s value </dev/tty
  printf '\n' >/dev/tty
  printf -v "$name" '%s' "$value"
  export "$name"
}

export HF_USER="${HF_USER:-Mitroshenkov87}"
export KAGGLE_USER="${KAGGLE_USER:-mitroshenkov87}"
prompt_secret HF_TOKEN
prompt_secret KAGGLE_API_TOKEN

if [[ ! -f "${BASH_SOURCE[0]:-}" ]]; then
  printf 'Run bash scripts/bootstrap.sh from the extracted bundle.\n' >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$ROOT/VERSION" ]]; then
  printf 'VERSION is missing. Build the VM bundle with scripts/make_vm_bundle.sh\n' >&2
  exit 9
fi
FORGE_VERSION="$(tr -d '[:space:]' < "$ROOT/VERSION")"
printf 'forge %s\n' "$FORGE_VERSION"

if [[ ! -r /etc/os-release ]]; then
  printf 'This bootstrap targets Ubuntu 24.04\n' >&2
  exit 2
fi
OS_ID="$(. /etc/os-release && printf '%s' "${ID:-}")"
OS_VER="$(. /etc/os-release && printf '%s' "${VERSION_ID:-}")"
if [[ "$OS_ID" != "ubuntu" || "$OS_VER" != "24.04" ]]; then
  printf 'This bootstrap targets Ubuntu 24.04 (found %s %s)\n' "$OS_ID" "$OS_VER" >&2
  exit 2
fi

mkdir -p "$FORGE_OUT/logs" "$FORGE_OUT/stages" "$HF_HOME"
printf '%s\n' "$$" > "$FORGE_OUT/bootstrap.lock"
# Scrub before the log file. The prompt above goes only to the terminal.
exec > >(python3 "$ROOT/scripts/logscrub.py" | tee -a "$FORGE_OUT/logs/bootstrap.log") 2>&1

progress_line() {
  local model="$1" stage="$2" pct="$3" eta="$4"
  local stamp
  stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s  %s  %s  %s%%  eta %s min\n' "$stamp" "$model" "$stage" "$pct" "$eta" >> "$FORGE_OUT/progress.log"
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'nvidia-smi is missing\n' >&2
  exit 9
fi
nvidia-smi

if ! command -v cmake >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get update && apt-get install -y cmake build-essential ninja-build || true
  else
    sudo -n apt-get update && sudo -n apt-get install -y cmake build-essential ninja-build || true
  fi
fi
if ! command -v cmake >/dev/null 2>&1 || ! command -v g++ >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1; then
  printf 'cmake, build-essential, or ninja-build is still missing after apt-get install -y cmake build-essential ninja-build\n' >&2
  exit 9
fi

if [[ -d /usr/local/cuda/bin ]]; then
  export PATH="/usr/local/cuda/bin:${PATH}"
fi
if [[ -z "${CUDACXX:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDACXX=/usr/local/cuda/bin/nvcc
fi
if [[ -z "${LD_LIBRARY_PATH:-}" || "$LD_LIBRARY_PATH" != */usr/local/cuda/* ]]; then
  cuda_lib=""
  if [[ -d /usr/local/cuda/lib64 ]]; then
    cuda_lib="/usr/local/cuda/lib64"
  fi
  if [[ -d /usr/local/cuda/lib ]]; then
    if [[ -n "$cuda_lib" ]]; then
      cuda_lib="${cuda_lib}:/usr/local/cuda/lib"
    else
      cuda_lib="/usr/local/cuda/lib"
    fi
  fi
  if [[ -n "$cuda_lib" ]]; then
    export LD_LIBRARY_PATH="${cuda_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
fi
if ! command -v nvcc >/dev/null 2>&1; then
  printf 'nvcc is missing. Expected /usr/local/cuda/bin/nvcc on PATH (CUDA 13.4).\n' >&2
  exit 9
fi
if ! command -v cmake >/dev/null 2>&1 || ! command -v ninja >/dev/null 2>&1 || ! command -v nvcc >/dev/null 2>&1; then
  printf 'preflight failed: cmake, ninja, and nvcc must be on PATH\n' >&2
  exit 9
fi
printf 'preflight cmake ninja nvcc ok\n'

if ! command -v python3 >/dev/null 2>&1; then
  printf 'python3 is missing\n' >&2
  exit 9
fi

PYTHONPATH="$ROOT/src" python3 -c '
from pathlib import Path
from voxprint_forge.plan import format_estimates, load_plan
print(format_estimates(load_plan(Path("'"$ROOT"'/plans/plan.toml"))))
'

PYTHONPATH="$ROOT/src" python3 -c '
import sys
from pathlib import Path
from voxprint_forge.calibration import assert_calibration, tokens_required
from voxprint_forge.errors import ForgeError
from voxprint_forge.plan import load_plan
plan = load_plan(Path("'"$ROOT"'/plans/plan.toml"))
try:
    count = assert_calibration(plan)
except ForgeError as exc:
    print(exc, file=sys.stderr)
    raise SystemExit(exc.exit_code)
need = tokens_required(plan.imatrix_ctx, plan.imatrix_chunks)
print(f"preflight calibration tokens {count} >= {need}")
'

eval "$(PYTHONPATH="$ROOT/src" python3 -c '
from pathlib import Path
from voxprint_forge.plan import load_plan
plan = load_plan(Path("'"$ROOT"'/plans/plan.toml"))
build = int(plan.estimates_min.get("bootstrap_build", 0))
total = build + sum(model.estimate_min for model in plan.enabled_models())
print(f"FORGE_BUILD_MIN={build}")
print(f"FORGE_TOTAL_MIN={total}")
')"
progress_line bootstrap start 0 "$FORGE_TOTAL_MIN"

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
uv pip install --python "$VENV" "$ROOT/vendor/voxprint_common-0.3.1-py3-none-any.whl"
# Torch came from the cu130 index above; everything else comes from PyPI only.
# Mixing the torch index in here makes uv resolve index-first and pick old
# packages. The constraint keeps the CUDA torch build.
PINS="$FORGE_OUT/torch-pins.txt"
mkdir -p "$FORGE_OUT"
uv pip freeze --python "$VENV" | grep -E '^(torch|torchvision)==' > "$PINS"
uv pip install --python "$VENV" -e "$ROOT" hf_transfer "kaggle>=1.6" --constraint "$PINS"

# shellcheck source=scripts/vm_logins.sh
source "$ROOT/scripts/vm_logins.sh"
persist_rented_vm_logins
"$VENV/bin/forge" verify-logins
progress_line bootstrap install "$(( FORGE_BUILD_MIN * 40 / FORGE_TOTAL_MIN ))" "$(( FORGE_TOTAL_MIN - FORGE_BUILD_MIN * 40 / 100 ))"

LLAMA_COMMIT="$(tr -d '[:space:]' < "$ROOT/scripts/llama.rev")"
LLAMA_SRC="$ROOT/third_party/llama.cpp"
if [[ ! -f "$LLAMA_SRC/CMakeLists.txt" || ! -f "$ROOT/third_party/LLAMA_COMMIT" ]]; then
  printf 'llama.cpp sources are missing from third_party. Use scripts/make_vm_bundle.sh\n' >&2
  exit 9
fi
BUNDLED_LLAMA="$(tr -d '[:space:]' < "$ROOT/third_party/LLAMA_COMMIT")"
if [[ "$BUNDLED_LLAMA" != "$LLAMA_COMMIT" ]]; then
  printf 'bundled llama.cpp commit does not match scripts/llama.rev\n' >&2
  exit 9
fi

export FORGE_PYTHON="${FORGE_PYTHON:-$VENV/bin/python}"
# T-lite-it-2.1 Qwen2 pre-tokenizer. Idempotent. See docs/VM.md.
"$FORGE_PYTHON" -m voxprint_forge.llama_patch "$LLAMA_SRC/conversion/base.py"

ARCHS="${FORGE_CUDA_ARCHS:-80;89;90;120}"
WANT="${LLAMA_COMMIT} ${ARCHS}"
STAMP="$FORGE_OUT/stages/llama-cpp.done"
if [[ ! -f "$STAMP" || "$(tr -d '\n' < "$STAMP")" != "$WANT" ]]; then
  # An older Makefile build tree cannot switch to Ninja in place.
  if [[ -f "$FORGE_OUT/llama-build/CMakeCache.txt" ]] \
    && ! grep -q '^CMAKE_GENERATOR:INTERNAL=Ninja$' "$FORGE_OUT/llama-build/CMakeCache.txt"; then
    rm -rf "$FORGE_OUT/llama-build/CMakeCache.txt" "$FORGE_OUT/llama-build/CMakeFiles"
  fi
  cmake -S "$LLAMA_SRC" -B "$FORGE_OUT/llama-build" \
    -G Ninja \
    -DCMAKE_CUDA_COMPILER="${CUDACXX:-$(command -v nvcc)}" \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCHS"
  cmake --build "$FORGE_OUT/llama-build" --parallel "$(nproc)"
  printf '%s\n' "$WANT" > "$STAMP"
fi
progress_line bootstrap llama "$(( FORGE_BUILD_MIN * 100 / FORGE_TOTAL_MIN ))" "$(( FORGE_TOTAL_MIN - FORGE_BUILD_MIN ))"

export FORGE_LLAMA_SRC="$LLAMA_SRC"
export FORGE_LLAMA_BIN="$FORGE_OUT/llama-build/bin"
FORGE_OUT="$FORGE_OUT" FORGE_VENV="$VENV" bash "$ROOT/scripts/vm_run.sh"

stamp="$(date -u +%F)"
tar -C "$FORGE_OUT" -cf "$FORGE_OUT/forge-results-${stamp}.tar" work manifests
sha256sum "$FORGE_OUT/forge-results-${stamp}.tar" | tee "$FORGE_OUT/forge-results-${stamp}.tar.sha256"
printf 'results: %s\n' "$FORGE_OUT/forge-results-${stamp}.tar"
printf 'copy back: %s\n' "$FORGE_OUT/manifests/${stamp}-a100.json"
printf 'copy back: %s\n' "$FORGE_OUT/manifests/${stamp}-a100.MODELS.md"
progress_line bootstrap done 100 0
printf 'ALL-DONE\n'
