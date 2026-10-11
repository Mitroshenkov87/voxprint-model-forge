#!/usr/bin/env bash
# One resumable diffusion-int4 run for a fresh Ubuntu A100 or H100 80GB
# (immers.cloud teslaa100-1.16.128.320 or h100-1.16.128.320).
# sm_80 is an A100. sm_90 is an H100. The H100 quant step is planned at 1.8x.
# Torch wheels are cu128, which covers sm_80 and sm_90. llama.cpp is built for 80;90.
# FORGE_HOST_KIND=local (default) keeps tokens in the process only.
# FORGE_HOST_KIND=rented-vm logs in once and leaves the logins for the rental.
# This script does not enable command tracing and does not unset tokens.
# --dry-run is the CPU smoke: tiny files, no CUDA, no clones, no token prompt.
# Heartbeats land in $FORGE_OUT/status.json. The job list is ~/watch.json
# unless --watch or FORGE_WATCH is set. Markers are ~/.done/diffusion-int4/<step>.
set +x
set -euo pipefail

export GIT_TERMINAL_PROMPT=0
export FORGE_HOST_KIND="${FORGE_HOST_KIND:-local}"
export HF_HUB_DISABLE_TELEMETRY=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORGE_OUT="${FORGE_OUT:-$HOME/forge-out}"
DONE_DIR="${FORGE_DONE_DIR:-$HOME/.done/diffusion-int4}"
WATCH="${FORGE_WATCH:-$HOME/watch.json}"
DRY=0
QWEN_IMAGE=0
SVD_RANK=256

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY=1
      ;;
    --qwen-image)
      QWEN_IMAGE=1
      ;;
    --svd-rank)
      shift
      SVD_RANK="${1:-}"
      ;;
    --svd-rank=*)
      SVD_RANK="${1#--svd-rank=}"
      ;;
    --forge-out)
      shift
      FORGE_OUT="${1:-}"
      ;;
    --forge-out=*)
      FORGE_OUT="${1#--forge-out=}"
      ;;
    --done-dir)
      shift
      DONE_DIR="${1:-}"
      ;;
    --done-dir=*)
      DONE_DIR="${1#--done-dir=}"
      ;;
    --watch)
      shift
      WATCH="${1:-}"
      ;;
    --watch=*)
      WATCH="${1#--watch=}"
      ;;
    *)
      printf 'unknown argument %s\n' "$1" >&2
      exit 2
      ;;
  esac
  shift
done

export FORGE_OUT FORGE_DONE_DIR="$DONE_DIR" FORGE_WATCH="$WATCH"
export FORGE_ROOT="$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$FORGE_OUT/logs" "$DONE_DIR"

# Bash preflight. The Python disk-floor, which reclaims verified sources first,
# runs once the interpreter can import the package.
df -P "$FORGE_OUT" | awk 'NR==2 { if (($4 / $2) < 0.15) { print "free space is under the 15% floor" > "/dev/stderr"; exit 2 } }'

prompt_secret() {
  local name="$1"
  set +x
  if [[ -n "${!name:-}" ]]; then
    return 0
  fi
  if [[ ! -t 0 || ! -r /dev/tty ]]; then
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

if [[ "$DRY" -eq 0 ]]; then
  export HF_USER="${HF_USER:-Mitroshenkov87}"
  prompt_secret HF_TOKEN
fi

PY="${FORGE_PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  PY=python
fi

disk_floor() {
  "$PY" -m voxprint_forge.diffusion.int4 --root "$ROOT" --forge-out "$FORGE_OUT" disk-floor --done-dir "$DONE_DIR" --path "$FORGE_OUT"
}

# shellcheck source=scripts/vm_logins.sh
source "$ROOT/scripts/vm_logins.sh"

if [[ "$DRY" -eq 1 ]]; then
  printf 'dry-run: skip CUDA, clones, and venvs\n'
  export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
  "$PY" -m voxprint_forge.diffusion.int4 --root "$ROOT" estimates
  disk_floor
  args=(--json --root "$ROOT" --forge-out "$FORGE_OUT" run --dry-run --svd-rank "$SVD_RANK" --done-dir "$DONE_DIR" --watch "$WATCH")
  if [[ "$QWEN_IMAGE" -eq 1 ]]; then
    args+=(--qwen-image)
  fi
  set -o pipefail
  "$PY" -m voxprint_forge.diffusion.int4 "${args[@]}" 2>&1 \
    | "$PY" "$ROOT/scripts/logscrub.py" \
    | tee -a "$FORGE_OUT/logs/diffusion-int4.log"
  printf 'ALL-DONE\n'
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'nvidia-smi is missing. This rental expects an A100 or H100 80GB.\n' >&2
  exit 9
fi
nvidia-smi >/dev/null
detect_gpu() {
  local line cap sm
  line="$(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -n 1 || true)"
  cap="${line##*,}"
  cap="${cap#"${cap%%[![:space:]]*}"}"
  cap="${cap%"${cap##*[![:space:]]}"}"
  if [[ -z "$cap" || "$cap" != *.* ]]; then
    printf 'nvidia-smi did not report a compute capability. This rental expects an A100 or H100 80GB.\n' >&2
    exit 9
  fi
  export FORGE_GPU_NAME="${line%%,*}"
  FORGE_GPU_NAME="${FORGE_GPU_NAME#"${FORGE_GPU_NAME%%[![:space:]]*}"}"
  FORGE_GPU_NAME="${FORGE_GPU_NAME%"${FORGE_GPU_NAME##*[![:space:]]}"}"
  export FORGE_COMPUTE_CAP="$cap"
  sm="sm_${cap//./}"
  case "$cap" in
    8.0) export FORGE_GPU=a100 ;;
    9.0) export FORGE_GPU=h100 ;;
    *) export FORGE_GPU=other ;;
  esac
  printf 'gpu %s %s compute %s (%s)\n' "$FORGE_GPU" "$sm" "$FORGE_COMPUTE_CAP" "$FORGE_GPU_NAME"
  if [[ "$sm" == "sm_90" ]]; then
    printf 'note: the pinned vitoom-nunchaku wheel lists sm_75 sm_80 sm_86 sm_89 sm_120a. sm_90 INT4 eval may fail to launch. Quant files can still be published if eval fails. Sources stay until a mirror is verified.\n'
  fi
}
detect_gpu
if ! command -v nvcc >/dev/null 2>&1; then
  export PATH="/usr/local/cuda/bin:${PATH}"
fi
if ! command -v nvcc >/dev/null 2>&1; then
  printf 'nvcc is missing. Expected /usr/local/cuda/bin/nvcc on PATH.\n' >&2
  exit 9
fi
for tool in git cmake ninja; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    printf '%s is missing.\n' "$tool" >&2
    exit 9
  fi
done
printf 'preflight nvidia-smi nvcc git cmake ninja ok\n'
disk_floor

export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi

SRC="$FORGE_OUT/src"
mkdir -p "$SRC"

clone_pin() {
  local url="$1" dest="$2" rev="$3"
  if [[ ! -d "$dest/.git" ]]; then
    git clone "$url" "$dest"
  fi
  git -C "$dest" fetch --tags origin
  git -C "$dest" checkout "$rev"
  printf '%s\n' "$rev" > "$dest/PIN"
}

clone_pin "https://github.com/mit-han-lab/deepcompressor" "$SRC/deepcompressor" "69f3473f5e1c1504bae35cc50c7858ef900a9b17"
clone_pin "https://github.com/city96/ComfyUI-GGUF" "$SRC/ComfyUI-GGUF" "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
clone_pin "https://github.com/ggml-org/llama.cpp" "$SRC/llama.cpp" "c8c07d658a6cefc5a50cfdf6be7d726503612303"
clone_pin "https://github.com/Wan-Video/Wan2.2" "$SRC/Wan2.2" "1ea34ff48f87168174e12956e200b1d908b1c5ff"

if ! grep -q "Apache License" "$SRC/deepcompressor/LICENSE"; then
  printf 'deepcompressor LICENSE is not Apache-2.0.\n' >&2
  exit 3
fi
if ! grep -q "Version 2.0" "$SRC/deepcompressor/LICENSE"; then
  printf 'deepcompressor LICENSE is not Apache-2.0.\n' >&2
  exit 3
fi

VENV_INT4="$FORGE_OUT/venv-int4"
VENV_WAN="$FORGE_OUT/venv-wan"
if [[ ! -x "$VENV_INT4/bin/python" ]]; then
  uv python install 3.11
  uv venv --python 3.11 "$VENV_INT4"
fi
if [[ ! -x "$VENV_WAN/bin/python" ]]; then
  uv venv --python 3.11 "$VENV_WAN"
fi

# cu128 wheels cover sm_80 (A100) and sm_90 (H100).
uv pip install --python "$VENV_INT4" "torch==2.11.*" "torchvision==0.26.*" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$VENV_INT4" -e "$ROOT" -e "$SRC/deepcompressor"
uv pip install --python "$VENV_INT4" "diffusers==0.37.1" "transformers==4.57.6"
WHEEL_URL="https://huggingface.co/tonera/vitoom-nunchaku/resolve/85bad5510d409570527b018387491fd0c81e4727/nunchaku-1.3.0.dev20260622+cu12.8torch2.11-cp311-cp311-linux_x86_64.whl"
uv pip install --python "$VENV_INT4" "$WHEEL_URL"

uv pip install --python "$VENV_WAN" "torch==2.11.*" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$VENV_WAN" -r "$SRC/Wan2.2/requirements.txt" "gguf>=0.13.0" "safetensors"

if [[ ! -x "$SRC/llama.cpp/build/bin/llama-quantize" ]]; then
  patch="$SRC/ComfyUI-GGUF/tools/lcpp.patch"
  if git -C "$SRC/llama.cpp" apply --check "$patch" 2>/dev/null; then
    git -C "$SRC/llama.cpp" apply "$patch"
  elif git -C "$SRC/llama.cpp" apply --reverse --check "$patch" 2>/dev/null; then
    printf 'lcpp.patch already applied\n'
  else
    printf 'lcpp.patch does not apply to llama.cpp %s\n' "c8c07d658a6cefc5a50cfdf6be7d726503612303" >&2
    exit 6
  fi
  cmake -S "$SRC/llama.cpp" -B "$SRC/llama.cpp/build" -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES="80;90"
  cmake --build "$SRC/llama.cpp/build" --target llama-quantize -j "$(nproc)"
fi

export FORGE_DEEPCOMPRESSOR="$SRC/deepcompressor"
export FORGE_COMFY_GGUF="$SRC/ComfyUI-GGUF"
export FORGE_LLAMA_SRC="$SRC/llama.cpp"
export FORGE_LLAMA_BIN="$SRC/llama.cpp/build/bin"
export FORGE_WAN_REPO="$SRC/Wan2.2"
export FORGE_WAN_PYTHON="$VENV_WAN/bin/python"
export VENV="$VENV_INT4"
export FORGE_PYTHON="$VENV_INT4/bin/python"
PY="$FORGE_PYTHON"

persist_rented_vm_logins

"$PY" -c 'from pathlib import Path; import sys; from voxprint_forge.diffusion.int4.patch import apply_checkout; print(apply_checkout(Path(sys.argv[1])))' "$SRC/deepcompressor"

WEIGHTS="$SRC/weights"
"$PY" -m voxprint_forge.diffusion.int4 --json --root "$ROOT" --forge-out "$FORGE_OUT" fetch --dest "$WEIGHTS" \
  > "$FORGE_OUT/logs/diffusion-int4-fetch.json"
# shellcheck disable=SC1091
source "$WEIGHTS/exports.sh"

disk_floor
"$PY" -m voxprint_forge.diffusion.int4 --root "$ROOT" estimates
args=(--json --root "$ROOT" --forge-out "$FORGE_OUT" run --svd-rank "$SVD_RANK" --done-dir "$DONE_DIR" --watch "$WATCH")
if [[ "$QWEN_IMAGE" -eq 1 ]]; then
  args+=(--qwen-image)
fi
set -o pipefail
"$PY" -m voxprint_forge.diffusion.int4 "${args[@]}" 2>&1 \
  | "$PY" "$ROOT/scripts/logscrub.py" \
  | tee -a "$FORGE_OUT/logs/diffusion-int4.log"
disk_floor
printf 'ALL-DONE\n'
