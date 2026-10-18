#!/usr/bin/env bash
# Prepare a synth4 venv and an sm_90 llama.cpp build.
# This script does not log in, does not print tokens, and does not run a job.
set +x
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORGE_OUT="${FORGE_OUT:-${HOME}/forge-out}"

dry=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dry=1
fi

driver_major() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf ''
    return 0
  fi
  nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | cut -d. -f1 || true
}

torch_index() {
  local major
  major="$(driver_major | tr -d '[:space:]')"
  if [[ -n "$major" && "$major" -lt 580 ]]; then
    printf '%s\n' "https://download.pytorch.org/whl/cu128"
    return 0
  fi
  printf '%s\n' "https://download.pytorch.org/whl/cu130"
}

cuda_archs() {
  if [[ -n "${FORGE_CUDA_ARCHS:-}" ]]; then
    printf '%s\n' "$FORGE_CUDA_ARCHS"
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local cap
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '[:space:]' || true)"
    if [[ "$cap" == *.* ]]; then
      printf '%s\n' "${cap//./}"
      return 0
    fi
  fi
  printf '%s\n' "80;89;90;120"
}

if [[ "$dry" == 1 ]]; then
  printf 'python_fallback: 3.14 3.13 3.12\n'
  printf 'torch_index: %s\n' "$(torch_index)"
  printf 'extra: compare\n'
  printf 'cuda_archs: %s\n' "$(cuda_archs)"
  printf 'needs: cmake ninja nvcc (CUDACXX, default /usr/local/cuda/bin/nvcc)\n'
  printf 'env: FORGE_LLAMA_BIN FORGE_LLAMA_SRC FORGE_VENV\n'
  exit 0
fi

# Build tools: install cmake, ninja-build and build-essential when missing,
# put /usr/local/cuda on PATH, and fail before any download if still missing.
need_apt=0
for tool in cmake ninja g++; do
  command -v "$tool" >/dev/null 2>&1 || need_apt=1
done
if [[ "$need_apt" == 1 ]] && command -v apt-get >/dev/null 2>&1; then
  if [[ "$(id -u)" -eq 0 ]]; then
    apt-get update && apt-get install -y cmake ninja-build build-essential || true
  else
    sudo -n apt-get update && sudo -n apt-get install -y cmake ninja-build build-essential || true
  fi
fi
if [[ -d /usr/local/cuda/bin ]]; then
  export PATH="/usr/local/cuda/bin:${PATH}"
fi
if [[ -z "${CUDACXX:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDACXX=/usr/local/cuda/bin/nvcc
fi
missing=()
for tool in cmake ninja g++; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [[ -z "${CUDACXX:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDACXX="$(command -v nvcc)"
    export CUDACXX
  else
    missing+=("nvcc (or set CUDACXX)")
  fi
elif [[ ! -x "$CUDACXX" ]]; then
  missing+=("CUDACXX=$CUDACXX is not executable")
fi
if (( ${#missing[@]} )); then
  printf 'missing build tools: %s\n' "${missing[*]}" >&2
  printf 'install cmake, ninja-build, build-essential and the CUDA toolkit (nvcc); see docs/VM.md\n' >&2
  exit 9
fi

export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

VENV="$FORGE_OUT/venv"
INDEX="$(torch_index)"
install_torch() {
  local py="$1"
  uv python install "$py"
  uv venv --python "$py" "$VENV" --clear
  uv pip install --python "$VENV" "torch==2.14.1" torchvision --index-url "$INDEX"
}

if ! install_torch 3.14; then
  if ! install_torch 3.13; then
    install_torch 3.12
  fi
fi
uv pip install --python "$VENV" "$ROOT/vendor/voxprint_common-0.3.1-py3-none-any.whl"
# Step 2: everything else from PyPI only. Mixing the torch index in makes uv
# resolve index-first and pick old packages (torchmetrics 1.0.3). The constraint
# keeps the CUDA torch and torchvision from step 1.
PINS="$FORGE_OUT/torch-pins.txt"
uv pip freeze --python "$VENV" | grep -E '^(torch|torchvision)==' > "$PINS"
uv pip install --python "$VENV" -e "${ROOT}[compare]" --constraint "$PINS"

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
ARCHS="$(cuda_archs)"
WANT="${LLAMA_COMMIT} ${ARCHS}"
mkdir -p "$FORGE_OUT/stages"
STAMP="$FORGE_OUT/stages/llama-cpp.done"
if [[ ! -f "$STAMP" || "$(tr -d '\n' < "$STAMP")" != "$WANT" ]]; then
  # An older Makefile build tree cannot switch to Ninja in place.
  if [[ -f "$FORGE_OUT/llama-build/CMakeCache.txt" ]] \
    && ! grep -q '^CMAKE_GENERATOR:INTERNAL=Ninja$' "$FORGE_OUT/llama-build/CMakeCache.txt"; then
    rm -rf "$FORGE_OUT/llama-build/CMakeCache.txt" "$FORGE_OUT/llama-build/CMakeFiles"
  fi
  cmake -S "$LLAMA_SRC" -B "$FORGE_OUT/llama-build" \
    -G Ninja \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=OFF \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCHS"
  cmake --build "$FORGE_OUT/llama-build" --parallel "$(nproc)"
  printf '%s\n' "$WANT" > "$STAMP"
fi

cat > "$FORGE_OUT/synth4.env" << EOF
FORGE_LLAMA_BIN=$FORGE_OUT/llama-build/bin
FORGE_LLAMA_SRC=$LLAMA_SRC
FORGE_VENV=$VENV
EOF
printf 'synth4 env: %s\n' "$FORGE_OUT/synth4.env"
