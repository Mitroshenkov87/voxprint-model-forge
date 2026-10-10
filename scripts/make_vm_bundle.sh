#!/usr/bin/env bash
# Build a tarball for a VM that has no GitHub access and no .git directory.
# Run this on a machine that can reach GitHub. The VM only unpacks the result.
set -euo pipefail
set +x

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHA="$(git -C "$ROOT" rev-parse HEAD)"
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
