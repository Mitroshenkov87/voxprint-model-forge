#!/usr/bin/env bash
# One resumable Wan 2.2 diffusion-int4 run for a fresh Ubuntu A100 or H100 80GB.
# sm_80 is an A100. sm_90 is an H100. llama.cpp uses FORGE_CUDA_ARCHS (default 80;89;90;120).
# The forge interpreter is FORGE_PYTHON. When that is unset, the live path
# creates $FORGE_OUT/venv-forge (Python 3.14) before the first disk-floor check.
# Torch comes from the cu130 index when the driver is 580 or newer, else cu128.
# FORGE_HOST_KIND=local (default) keeps tokens in the process only.
# FORGE_HOST_KIND=rented-vm logs in once and leaves the logins for the rental.
# This script does not enable command tracing and does not unset tokens.
# --dry-run is the CPU smoke: tiny files, no CUDA, no clones, no token prompt.
# FORGE_SKIP_WAN=1 omits the Wan 2.2 clones, venv, and steps. The default still runs Wan.
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
SVD_RANK=256
WAN_REV="1ea34ff48f87168174e12956e200b1d908b1c5ff"
COMFY_REV="6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
LLAMA_REV="c8c07d658a6cefc5a50cfdf6be7d726503612303"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY=1
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
if [[ -n "${FORGE_COMMON_SRC:-}" ]]; then
  export PYTHONPATH="${FORGE_COMMON_SRC}${PYTHONPATH:+:$PYTHONPATH}"
fi
mkdir -p "$FORGE_OUT/logs" "$DONE_DIR"

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

disk_floor() {
  "$PY" -m voxprint_forge.diffusion.int4 --root "$ROOT" --forge-out "$FORGE_OUT" disk-floor --done-dir "$DONE_DIR" --path "$FORGE_OUT"
}

torch_index_for_driver() {
  local driver="$1" major
  major="${driver%%.*}"
  major="${major//[^0-9]/}"
  if [[ -n "$major" && "$major" -ge 580 ]]; then
    printf '%s\n' "https://download.pytorch.org/whl/cu130"
    return 0
  fi
  printf '%s\n' "https://download.pytorch.org/whl/cu128"
}

ensure_uv() {
  export PATH="${HOME}/.local/bin:${PATH}"
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
  fi
}

# Live path only. --dry-run does not call this and does not create a venv.
# An operator-supplied FORGE_PYTHON is used as-is. Otherwise every forge
# command, including the first disk-floor, runs under venv-forge.
ensure_forge_python() {
  if [[ -n "${FORGE_PYTHON:-}" ]]; then
    PY="$FORGE_PYTHON"
    export FORGE_PYTHON
    return 0
  fi
  ensure_uv
  local driver index
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1 || true)"
  driver="${driver#"${driver%%[![:space:]]*}"}"
  driver="${driver%"${driver##*[![:space:]]}"}"
  index="$(torch_index_for_driver "$driver")"
  export FORGE_TORCH_INDEX="$index"
  VENV_FORGE="$FORGE_OUT/venv-forge"
  if [[ ! -x "$VENV_FORGE/bin/python" ]]; then
    uv python install 3.14
    uv venv --python 3.14 "$VENV_FORGE"
  fi
  uv pip install --python "$VENV_FORGE" torch --index-url "$index"
  uv pip install --python "$VENV_FORGE" \
    "safetensors" \
    "huggingface_hub>=0.26" \
    "gguf" \
    "jinja2>=3.1"
  if [[ -f "$ROOT/vendor/voxprint_common-0.4.1-py3-none-any.whl" ]]; then
    uv pip install --python "$VENV_FORGE" "$ROOT/vendor/voxprint_common-0.4.1-py3-none-any.whl"
  fi
  uv pip install --python "$VENV_FORGE" -e "$ROOT" --extra-index-url "$index"
  export FORGE_PYTHON="$VENV_FORGE/bin/python"
  export VENV="$VENV_FORGE"
  PY="$FORGE_PYTHON"
  if [[ ! -x "$PY" ]]; then
    printf 'forge venv python is missing at %s\n' "$PY" >&2
    exit 2
  fi
}

# shellcheck source=scripts/vm_logins.sh
source "$ROOT/scripts/vm_logins.sh"

if [[ "$DRY" -eq 1 ]]; then
  printf 'dry-run: skip CUDA, clones, and venvs\n'
  PY="${FORGE_PYTHON:-python3}"
  if ! command -v "$PY" >/dev/null 2>&1; then
    PY=python
  fi
  "$PY" -m voxprint_forge.diffusion.int4 --root "$ROOT" estimates
  disk_floor
  args=(--json --root "$ROOT" --forge-out "$FORGE_OUT" run --dry-run --svd-rank "$SVD_RANK" --done-dir "$DONE_DIR" --watch "$WATCH")
  set -o pipefail
  "$PY" -m voxprint_forge.diffusion.int4 "${args[@]}" 2>&1 \
    | "$PY" "$ROOT/scripts/logscrub.py" \
    | tee -a "$FORGE_OUT/logs/diffusion-int4.log"
  printf 'ALL-DONE\n'
  exit 0
fi

export HF_USER="${HF_USER:-Mitroshenkov87}"
prompt_secret HF_TOKEN

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
ensure_forge_python
disk_floor
ensure_uv

SRC="$FORGE_OUT/src"
mkdir -p "$SRC"
VENV_WAN="$FORGE_OUT/venv-wan"

clone_pin() {
  local url="$1" dest="$2" rev="$3"
  if [[ -f "$dest/PIN" && "$(cat "$dest/PIN")" == "$rev" && "$(git -C "$dest" rev-parse HEAD 2>/dev/null)" == "$rev" ]]; then
    printf 'pinned %s already at %s\n' "$dest" "$rev"
    return 0
  fi
  if [[ ! -d "$dest/.git" ]]; then
    git clone "$url" "$dest"
  fi
  git -C "$dest" fetch --tags origin
  git -C "$dest" checkout "$rev"
  printf '%s\n' "$rev" > "$dest/PIN"
}

if [[ "${FORGE_SKIP_WAN:-0}" != "1" ]]; then
  clone_pin "https://github.com/city96/ComfyUI-GGUF" "$SRC/ComfyUI-GGUF" "$COMFY_REV"
  clone_pin "https://github.com/ggml-org/llama.cpp" "$SRC/llama.cpp" "$LLAMA_REV"
  clone_pin "https://github.com/Wan-Video/Wan2.2" "$SRC/Wan2.2" "$WAN_REV"
  if [[ ! -x "$VENV_WAN/bin/python" ]]; then
    uv venv --python 3.11 "$VENV_WAN"
  fi
  uv pip install --python "$VENV_WAN" "torch==2.11.*" --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python "$VENV_WAN" -r "$SRC/Wan2.2/requirements.txt" "gguf>=0.13.0" "safetensors"
  if [[ ! -x "$SRC/llama.cpp/build/bin/llama-quantize" ]]; then
    patch="$SRC/ComfyUI-GGUF/tools/lcpp.patch"
    if git -C "$SRC/llama.cpp" apply --check "$patch" 2>/dev/null; then
      git -C "$SRC/llama.cpp" apply "$patch"
    elif git -C "$SRC/llama.cpp" apply --reverse --check "$patch" 2>/dev/null; then
      printf 'lcpp.patch already applied\n'
    else
      printf 'lcpp.patch does not apply to llama.cpp %s\n' "$LLAMA_REV" >&2
      exit 6
    fi
    cmake -S "$SRC/llama.cpp" -B "$SRC/llama.cpp/build" -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES="${FORGE_CUDA_ARCHS:-80;89;90;120}"
    cmake --build "$SRC/llama.cpp/build" --target llama-quantize -j "$(nproc)"
  fi
fi

export FORGE_COMFY_GGUF="$SRC/ComfyUI-GGUF"
export FORGE_LLAMA_SRC="$SRC/llama.cpp"
export FORGE_LLAMA_BIN="$SRC/llama.cpp/build/bin"
export FORGE_WAN_REPO="$SRC/Wan2.2"
export FORGE_WAN_PYTHON="$VENV_WAN/bin/python"

persist_rented_vm_logins

WEIGHTS="$SRC/weights"
"$PY" -m voxprint_forge.diffusion.int4 --json --root "$ROOT" --forge-out "$FORGE_OUT" fetch --dest "$WEIGHTS" \
  > "$FORGE_OUT/logs/diffusion-int4-fetch.json"
# shellcheck disable=SC1091
source "$WEIGHTS/exports.sh"

disk_floor
"$PY" -m voxprint_forge.diffusion.int4 --root "$ROOT" estimates
args=(--json --root "$ROOT" --forge-out "$FORGE_OUT" run --svd-rank "$SVD_RANK" --done-dir "$DONE_DIR" --watch "$WATCH")
set -o pipefail
"$PY" -m voxprint_forge.diffusion.int4 "${args[@]}" 2>&1 \
  | "$PY" "$ROOT/scripts/logscrub.py" \
  | tee -a "$FORGE_OUT/logs/diffusion-int4.log"
disk_floor
printf 'ALL-DONE\n'
