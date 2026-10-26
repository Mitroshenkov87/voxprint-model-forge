#!/usr/bin/env bash
# Upload canvas parts that were prepared with FORGE_DEFER_UPLOAD=1.
# Tokens are required here. The prepare step does not have them.
# This is the same work as `forge canvas upload PLAN --parts ...`.
# The diffusion module is the entry so an overlay can upload without replacing the LLM cli.
# This script does not enable command tracing and does not unset tokens.
# FORGE_HOST_KIND=local (default) keeps tokens in the process only.
# FORGE_HOST_KIND=rented-vm logs in once and leaves the logins for the rental.
set +x
set -euo pipefail

export GIT_TERMINAL_PROMPT=0
export FORGE_HOST_KIND="${FORGE_HOST_KIND:-local}"
FORGE_OUT="${FORGE_OUT:-$HOME/forge-out}"
export HF_HOME="${HF_HOME:-$FORGE_OUT/hf-home}"

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
PARTS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
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

prompt_secret HF_TOKEN
prompt_secret KAGGLE_API_TOKEN

if [[ ! -f "${BASH_SOURCE[0]:-}" ]]; then
  printf 'Run bash scripts/canvas_upload.sh from the extracted bundle.\n' >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$FORGE_OUT/venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  printf 'venv python is missing at %s. Finish bootstrap before the canvas upload.\n' "$VENV" >&2
  exit 9
fi
export PATH="${HOME}/.local/bin:${PATH}"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$VENV" "$ROOT/vendor/voxprint_common-0.5.1-py3-none-any.whl"
  uv pip install --python "$VENV" -e "$ROOT" --no-deps
fi
# shellcheck source=scripts/vm_logins.sh
source "$ROOT/scripts/vm_logins.sh"
persist_rented_vm_logins

RUN_AC=0
AC_PARTS=""
if [[ -z "$PARTS" ]]; then
  RUN_AC=1
else
  IFS=',' read -ra TOKENS <<< "$PARTS"
  for token in "${TOKENS[@]}"; do
    key="$(printf '%s' "$token" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
    case "$key" in
      a|base)
        RUN_AC=1
        AC_PARTS="${AC_PARTS:+$AC_PARTS,}A"
        ;;
      b|controlnet)
        RUN_AC=1
        AC_PARTS="${AC_PARTS:+$AC_PARTS,}B"
        ;;
      *)
        printf 'unknown part %s\n' "$token" >&2
        exit 2
        ;;
    esac
  done
fi

mkdir -p "$FORGE_OUT/logs"
set -o pipefail
if [[ "$RUN_AC" -eq 1 ]]; then
  args=(upload "$ROOT/plans/canvas-zimage-tiers-v1.toml" --json --forge-out "$FORGE_OUT" --root "$ROOT")
  if [[ -n "$AC_PARTS" ]]; then
    args+=(--parts "$AC_PARTS")
  fi
  if [[ "$FORCE" -eq 1 ]]; then
    args+=(--force)
  fi
  "$VENV/bin/python" -m voxprint_forge.diffusion "${args[@]}" 2>&1 \
    | python3 "$ROOT/scripts/logscrub.py" \
    | tee -a "$FORGE_OUT/logs/canvas-upload.log"
fi
