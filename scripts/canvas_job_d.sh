#!/usr/bin/env bash
# Part D on the A100, after bootstrap.sh has finished.
# Does not replace scripts/canvas_job.sh or scripts/bootstrap.sh.
# Tokens stay in the process environment. This script does not enable command tracing.
set +x
set -euo pipefail

export GIT_TERMINAL_PROMPT=0
FORGE_OUT="${FORGE_OUT:-$HOME/forge-out}"
export HF_HOME="${HF_HOME:-$FORGE_OUT/hf-home}"

cleanup_secrets() {
  set +x
  local home="${HOME:-/root}"
  rm -f "$home/.cache/huggingface/token" "$home/.huggingface/token" || true
  rm -rf "$home/.cache/huggingface/stored_tokens" "$home/.kaggle" "$home/.cache/kaggle" || true
  if [[ -n "${HF_HOME:-}" ]]; then
    rm -f "$HF_HOME/token" || true
    rm -rf "$HF_HOME/stored_tokens" || true
  fi
  if [[ -n "${FORGE_OUT:-}" ]]; then
    rm -f "$FORGE_OUT/hf-home/token" || true
    rm -rf "$FORGE_OUT/hf-home/stored_tokens" || true
  fi
  local dir
  for dir in "$home/.cache" "$home/.config" /tmp; do
    if [[ -d "$dir" ]]; then
      find "$dir" -type f \( -name 'kaggle.json' -o -name 'access_token' \) -delete 2>/dev/null || true
    fi
  done
  unset HF_TOKEN KAGGLE_API_TOKEN
}
trap cleanup_secrets EXIT INT TERM

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

export HF_USER="${HF_USER:-Mitroshenkov87}"
export KAGGLE_USER="${KAGGLE_USER:-mitroshenkov87}"

FORCE=0
DEADLINE="${FORGE_DEADLINE:-}"
PARTS=""
DEFER=0
case "${FORGE_DEFER_UPLOAD:-}" in
  1|true|TRUE|yes|YES|on|ON) DEFER=1 ;;
esac
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      ;;
    --defer-upload)
      DEFER=1
      ;;
    --deadline)
      shift
      DEADLINE="${1:-}"
      ;;
    --deadline=*)
      DEADLINE="${1#--deadline=}"
      ;;
    --parts)
      shift
      PARTS="${1:-}"
      ;;
    --parts=*)
      PARTS="${1#--parts=}"
      ;;
    *)
      printf 'unknown argument %s\n' "$1" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$DEFER" -eq 0 ]]; then
  prompt_secret HF_TOKEN
  prompt_secret KAGGLE_API_TOKEN
fi

if [[ ! -f "${BASH_SOURCE[0]:-}" ]]; then
  printf 'Run bash scripts/canvas_job_d.sh from the extracted bundle.\n' >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p "$FORGE_OUT/logs" "$FORGE_OUT/stages" "$HF_HOME"

if [[ -f "$ROOT/CANVAS_VERSION" ]]; then
  printf 'canvas %s\n' "$(tr -d '[:space:]' < "$ROOT/CANVAS_VERSION")"
fi
printf 'estimate part qwen 120 min\n'
printf 'estimate part qwen-reuse 280 min\n'
printf 'estimate part qwen-recompute 340 min\n'
printf 'estimate part encoder 15 min\n'
printf 'estimate part edit 55 min\n'

llm_still_running() {
  local lock="${FORGE_OUT}/bootstrap.lock"
  if [[ -f "$lock" ]]; then
    local pid
    pid="$(tr -d '[:space:]' < "$lock")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  if pgrep -f '[s]cripts/bootstrap.sh' >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

if [[ "$FORCE" -eq 0 ]] && llm_still_running; then
  printf 'LLM bootstrap is still running. Refusing to start. Wait until scripts/bootstrap.sh exits, or pass --force.\n' >&2
  exit 2
fi

# canvas_job_d.sh does not match this pattern. The bracket hides the pgrep line itself.
if [[ "$FORCE" -eq 0 ]] && pgrep -f '[s]cripts/canvas_job\.sh([^_]|$)' >/dev/null 2>&1; then
  printf 'scripts/canvas_job.sh is still running. Refusing to start.\n' >&2
  exit 2
fi

if ! command -v nvcc >/dev/null 2>&1; then
  export PATH="/usr/local/cuda/bin:${PATH}"
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
printf 'preflight nvcc ok\n'

VENV="$FORGE_OUT/venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  printf 'venv python is missing at %s. Finish bootstrap before the canvas job.\n' "$VENV" >&2
  exit 9
fi
if ! "$VENV/bin/python" -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
  printf 'torch CUDA is not available. Check the CUDA 13.4 driver and /usr/local/cuda.\n' >&2
  exit 9
fi

export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv >/dev/null 2>&1; then
  printf 'uv is missing. Finish bootstrap before the canvas job.\n' >&2
  exit 9
fi

uv pip install --python "$VENV" -e "$ROOT" --no-deps
uv pip install --python "$VENV" --no-deps \
  "diffusers==0.41.0" \
  "torchao==0.18.0" \
  "lpips==0.1.4" \
  "pillow-avif-plugin==1.6.0"
uv pip install --python "$VENV" --no-deps \
  "torchvision==0.29.1" \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python "$VENV" \
  "safetensors>=0.8" \
  "numpy" \
  "regex" \
  "pillow" \
  "importlib-metadata" \
  "filelock" \
  "tqdm" \
  "requests" \
  "scipy" \
  "sentencepiece" \
  "protobuf"

if ! "$VENV/bin/python" -c 'import torchao, lpips, pillow_avif; from diffusers import QwenImagePipeline, QwenImageEditPlusPipeline, ZImagePipeline, ZImageControlNetPipeline, ZImageControlNetInpaintPipeline'; then
  printf 'QwenImagePipeline or a canvas extra is missing. diffusers==0.41.0 did not import.\n' >&2
  exit 9
fi

progress_line() {
  local model="$1" stage="$2" pct="$3" eta="$4"
  local stamp
  stamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s  %s  %s  %s%%  eta %s min\n' "$stamp" "$model" "$stage" "$pct" "$eta" >> "$FORGE_OUT/progress.log"
}
progress_line canvas-qwen-image-2512 start 0 120

# Positional command, then flags. Omitted --parts selects D, which is retired unless --force.
args=(run "$ROOT/plans/canvas-qwen-image-2512-v1.toml" --json --forge-out "$FORGE_OUT" --root "$ROOT")
if [[ -n "$PARTS" ]]; then
  args+=(--parts "$PARTS")
fi
if [[ "$FORCE" -eq 1 ]]; then
  args+=(--force)
fi
if [[ -n "$DEADLINE" ]]; then
  args+=(--deadline "$DEADLINE")
fi
if [[ "$DEFER" -eq 1 ]]; then
  args+=(--defer-upload)
fi

set -o pipefail
"$VENV/bin/python" -m voxprint_forge.diffusion.qwen_main "${args[@]}" 2>&1 \
  | python3 "$ROOT/scripts/logscrub.py" \
  | tee -a "$FORGE_OUT/logs/canvas-qwen.log"
progress_line canvas-qwen-image-2512 done 100 0
printf 'ALL-DONE\n'
