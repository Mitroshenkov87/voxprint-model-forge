#!/usr/bin/env bash
# One resumable diffusion-int4 run for a fresh Ubuntu A100 or H100 80GB
# (immers.cloud teslaa100-1.16.128.320 or h100-1.16.128.320).
# sm_80 is an A100. sm_90 is an H100. The H100 quant step is planned at 1.8x.
# Torch wheels are cu128, which covers sm_80 and sm_90. llama.cpp is built for 80;90.
# FORGE_HOST_KIND=local (default) keeps tokens in the process only.
# FORGE_HOST_KIND=rented-vm logs in once and leaves the logins for the rental.
# This script does not enable command tracing and does not unset tokens.
# --own-edit defaults to on: the own Qwen-Edit-2511 INT4 r256 quant always runs.
# --own-edit auto is opt-in and can skip that quant when Phase 0 QuantFunc r128 passes.
# --phase0-result is an eval and card baseline. A missing file does not change the plan.
# --dry-run is the CPU smoke: tiny files, no CUDA, no clones, no token prompt.
# FORGE_SKIP_WAN=1 omits the Wan 2.2 clones, venv, and steps. The default still runs Wan.
# FORGE_WAIT_FOR holds the live GPU stages until that file exists. --dry-run does not wait.
# FORGE_PRIVATE_CALIB is an optional directory or private dataset id. Unset skips that set.
# FORGE_COMMON_SRC is an optional directory on PYTHONPATH for voxprint_common in the cp311 venv.
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
OWN_EDIT="${FORGE_OWN_EDIT:-on}"
PHASE0_RESULT="${FORGE_PHASE0_RESULT:-/workspace/vm-results/model-forge/nunchaku-test-1/result.json}"

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
    --own-edit)
      shift
      OWN_EDIT="${1:-}"
      ;;
    --own-edit=*)
      OWN_EDIT="${1#--own-edit=}"
      ;;
    --phase0-result)
      shift
      PHASE0_RESULT="${1:-}"
      ;;
    --phase0-result=*)
      PHASE0_RESULT="${1#--phase0-result=}"
      ;;
    *)
      printf 'unknown argument %s\n' "$1" >&2
      exit 2
      ;;
  esac
  shift
done

case "$OWN_EDIT" in
  auto|on|off) ;;
  *)
    printf 'own-edit must be auto, on, or off\n' >&2
    exit 2
    ;;
esac

export FORGE_OUT FORGE_DONE_DIR="$DONE_DIR" FORGE_WATCH="$WATCH"
EDIT_FLAGS=(--own-edit "$OWN_EDIT" --phase0-result "$PHASE0_RESULT")
export FORGE_ROOT="$ROOT"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -n "${FORGE_COMMON_SRC:-}" ]]; then
  export PYTHONPATH="${FORGE_COMMON_SRC}${PYTHONPATH:+:$PYTHONPATH}"
fi
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
  "$PY" -m voxprint_forge.diffusion.int4 --root "$ROOT" estimates "${EDIT_FLAGS[@]}"
  disk_floor
  args=(--json --root "$ROOT" --forge-out "$FORGE_OUT" run --dry-run --svd-rank "$SVD_RANK" --done-dir "$DONE_DIR" --watch "$WATCH")
  args+=("${EDIT_FLAGS[@]}")
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

clone_pin "https://github.com/mit-han-lab/deepcompressor" "$SRC/deepcompressor" "69f3473f5e1c1504bae35cc50c7858ef900a9b17"
if [[ "${FORGE_SKIP_WAN:-0}" != "1" ]]; then
  clone_pin "https://github.com/city96/ComfyUI-GGUF" "$SRC/ComfyUI-GGUF" "6ea2651e7df66d7585f6ffee804b20e92fb38b8a"
  clone_pin "https://github.com/ggml-org/llama.cpp" "$SRC/llama.cpp" "c8c07d658a6cefc5a50cfdf6be7d726503612303"
  clone_pin "https://github.com/Wan-Video/Wan2.2" "$SRC/Wan2.2" "1ea34ff48f87168174e12956e200b1d908b1c5ff"
fi

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
if [[ "${FORGE_SKIP_WAN:-0}" != "1" && ! -x "$VENV_WAN/bin/python" ]]; then
  uv venv --python 3.11 "$VENV_WAN"
fi

# cu128 wheels cover sm_80 (A100) and sm_90 (H100).
uv pip install --python "$VENV_INT4" "torch==2.11.*" "torchvision==0.26.*" --index-url https://download.pytorch.org/whl/cu128
# voxprint-forge and voxprint-common declare Python>=3.14. The vitoom wheel is cp311.
# The forge package stays on PYTHONPATH. Set FORGE_COMMON_SRC to the directory that contains voxprint_common.
# deepcompressor's poetry deps include "pyav" (not on PyPI), xformers (would re-pin torch) and LLM/video eval extras.
# Install it with --no-deps plus the diffusion subset it imports.
uv pip install --python "$VENV_INT4" --no-deps -e "$SRC/deepcompressor"
uv pip install --python "$VENV_INT4" "tqdm>=4.66" "torchmetrics>=1.4" "ninja>=1.11.1" "datasets>=3.0,<4" "lm_eval>=0.4.2" jieba fuzzywuzzy rouge "python-Levenshtein>=0.26.1" "sentencepiece>=0.1.99" "omniconfig>=0.1.10" "protobuf>=5.26" "clean-fid>=0.1.35" "dominate>=2.9.1" "opencv-python>=4.10" "einops>=0.8" "timm>=1.0.7" "rotary-embedding-torch>=0.6.4" "ftfy>=6.2" "av>=13" "bs4" "image_reward @ git+https://github.com/THUDM/ImageReward.git@78f8051059b34fef8e0032ccb40561c23a2ddbbb"
uv pip install --python "$VENV_INT4" "huggingface_hub>=0.26" "jinja2>=3.1" "safetensors>=0.4" "accelerate>=1.0" "gguf>=0.10" pillow
uv pip install --python "$VENV_INT4" "diffusers==0.37.1" "transformers==4.57.6"
WHEEL_URL="https://huggingface.co/tonera/vitoom-nunchaku/resolve/85bad5510d409570527b018387491fd0c81e4727/nunchaku-1.3.0.dev20260622+cu12.8torch2.11-cp311-cp311-linux_x86_64.whl"
uv pip install --python "$VENV_INT4" "$WHEEL_URL"

if [[ "${FORGE_SKIP_WAN:-0}" != "1" ]]; then
uv pip install --python "$VENV_WAN" "torch==2.11.*" --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$VENV_WAN" -r "$SRC/Wan2.2/requirements.txt" "gguf>=0.13.0" "safetensors"
fi

if [[ "${FORGE_SKIP_WAN:-0}" != "1" && ! -x "$SRC/llama.cpp/build/bin/llama-quantize" ]]; then
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

# FORGE_WAIT_FOR: hold the GPU stages until another job (Phase 0) writes this file.
if [[ -n "${FORGE_WAIT_FOR:-}" ]]; then
  while [[ ! -f "$FORGE_WAIT_FOR" ]]; do
    printf '%s waiting for %s\n' "$(date -Is)" "$FORGE_WAIT_FOR"
    sleep 60
  done
fi

disk_floor
"$PY" -m voxprint_forge.diffusion.int4 --root "$ROOT" estimates "${EDIT_FLAGS[@]}"
args=(--json --root "$ROOT" --forge-out "$FORGE_OUT" run --svd-rank "$SVD_RANK" --done-dir "$DONE_DIR" --watch "$WATCH")
args+=("${EDIT_FLAGS[@]}")
if [[ "$QWEN_IMAGE" -eq 1 ]]; then
  args+=(--qwen-image)
fi
set -o pipefail
"$PY" -m voxprint_forge.diffusion.int4 "${args[@]}" 2>&1 \
  | "$PY" "$ROOT/scripts/logscrub.py" \
  | tee -a "$FORGE_OUT/logs/diffusion-int4.log"
disk_floor
printf 'ALL-DONE\n'
