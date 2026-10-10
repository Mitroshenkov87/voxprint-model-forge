#!/usr/bin/env bash
# Build a tarball for a VM that has no GitHub access and no .git directory.
# Run this on a machine that can reach GitHub. The VM only unpacks the result.
set -euo pipefail
set +x

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse HEAD)"

# CANVAS_OVERLAY_PATHS begin
# Only new canvas files. Untarring this overlay must not replace the running LLM job.
CANVAS_OVERLAY_PATHS=(
  scripts/canvas_job.sh
  scripts/canvas_job_d.sh
  scripts/canvas_upload.sh
  scripts/vm_logins.sh
  src/voxprint_forge/diffusion
  plans/canvas-zimage-tiers-v1.toml
  plans/canvas-qwen-image-2512-v1.toml
  templates/canvas_model_card.md.jinja
  docs/CANVAS.md
)
# CANVAS_OVERLAY_PATHS end

pack_canvas() {
  local sha="$1"
  local out="$ROOT/dist/voxprint-model-forge-canvas-${sha}.tar.gz"
  local stage tree
  stage="$(mktemp -d)"
  tree="$stage/overlay"
  mkdir -p "$tree"
  git -C "$ROOT" archive --format=tar HEAD "${CANVAS_OVERLAY_PATHS[@]}" | tar -x -C "$tree"
  printf '%s\n' "$sha" > "$tree/CANVAS_VERSION"
  mkdir -p "$ROOT/dist"
  tar -C "$tree" -czf "$out" .
  rm -rf "$stage"
  printf 'canvas-bundle %s\n' "$out"
}

CANVAS_ONLY=0
if [[ "${1:-}" == "--canvas-only" ]]; then
  CANVAS_ONLY=1
  shift
fi
if [[ "$CANVAS_ONLY" -eq 1 ]]; then
  pack_canvas "$SHA"
  exit 0
fi

LLAMA_COMMIT="$(tr -d '[:space:]' < "$ROOT/scripts/llama.rev")"
OUT="${1:-$ROOT/dist/voxprint-model-forge-${SHA}.tar.gz}"
STAGE="$(mktemp -d)"
TREE="$STAGE/voxprint-model-forge"

cleanup() {
  rm -rf "$STAGE"
}
trap cleanup EXIT

mkdir -p "$TREE"
git -C "$ROOT" archive --format=tar HEAD | tar -x -C "$TREE"
printf '%s\n' "$SHA" > "$TREE/VERSION"

LLAMA="$TREE/third_party/llama.cpp"
mkdir -p "$LLAMA"
git init -q "$LLAMA"
git -C "$LLAMA" remote add origin https://github.com/ggml-org/llama.cpp
git -C "$LLAMA" fetch --depth 1 -q origin "$LLAMA_COMMIT"
git -C "$LLAMA" checkout -q --detach FETCH_HEAD
rm -rf "$LLAMA/.git"
printf '%s\n' "$LLAMA_COMMIT" > "$TREE/third_party/LLAMA_COMMIT"

mkdir -p "$(dirname "$OUT")"
tar -C "$STAGE" -czf "$OUT" voxprint-model-forge
printf 'bundle %s\nforge %s\n' "$OUT" "$SHA"
pack_canvas "$SHA"
