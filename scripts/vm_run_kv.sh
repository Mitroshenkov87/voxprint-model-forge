#!/usr/bin/env bash
# Plotweaver llama.cpp KV arms + spec/prefix bench (synth4). Plan: plans/kv-synth4.toml.
# Usage: scripts/vm_run_kv.sh [--preflight-only] [--models "L1 L3 L4"]
# Needs the forge venv and llama.cpp b11476 binaries on PATH (or FORGE_LLAMA_BIN), as for vm_run_kl.sh.
# L2 is not in the default model list. It waits for the L1/L2 eval fix; add it with --models once that fix is merged.
# Output: $FORGE_OUT/kv/<model>/kv.json and $FORGE_OUT/kv/bench-<model>.jsonl. No tokens, hosts, or prompt text are logged.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORGE_OUT="${FORGE_OUT:-$HOME/forge-out}"
VENV="${FORGE_VENV:-$FORGE_OUT/venv}"
PLAN="$ROOT/plans/kv-synth4.toml"
OUT="$FORGE_OUT/kv"
BIN="${FORGE_LLAMA_BIN:-}"
export HF_HOME="${HF_HOME:-$FORGE_OUT/hf-home}" HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
[[ -n "$BIN" ]] && export PATH="$BIN:$PATH"
PREFLIGHT_ONLY=0
MODELS="L1 L3 L4"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --models) MODELS="$2"; shift 2 ;;
    *) printf 'usage: vm_run_kv.sh [--preflight-only] [--models "L1 L3 L4"]\n' >&2; exit 2 ;;
  esac
done
PY="$VENV/bin/python"
mkdir -p "$OUT/gguf" "$OUT/corpus"

llama-server --help > "$OUT/server-help.txt" 2>&1 || true
llama-perplexity --help > "$OUT/ppl-help.txt" 2>&1 || true
GEMMA_ARGS=()
if [[ " $MODELS " == *" L2 "* ]]; then
  "$PY" - "$PLAN" "$OUT/gguf/L2" <<'PY'
import sys, tomllib
from pathlib import Path
from huggingface_hub import snapshot_download
m = next(m for m in tomllib.loads(Path(sys.argv[1]).read_text())["models"] if m["id"] == "L2")
snapshot_download(m["repo"], revision=m["revision"], allow_patterns=[m["pattern"]], local_dir=sys.argv[2])
PY
  GEMMA_ARGS=(--gemma-gguf "$(find "$OUT/gguf/L2" -name '*.gguf' | sort | head -1)")
fi
"$PY" -m voxprint_forge.kl.kv --plan "$PLAN" --preflight --server-help "$OUT/server-help.txt" \
  --ppl-help "$OUT/ppl-help.txt" "${GEMMA_ARGS[@]}"
"$PY" -m voxprint_forge.kl.kv --plan "$PLAN" --dry-run > "$OUT/matrix.json"
if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
  echo "preflight only: $OUT/matrix.json"
  exit 0
fi

for M in $MODELS; do
  "$PY" - "$PLAN" "$M" "$OUT/gguf/$M" <<'PY'
import sys, tomllib
from pathlib import Path
from huggingface_hub import snapshot_download
m = next(m for m in tomllib.loads(Path(sys.argv[1]).read_text())["models"] if m["id"] == sys.argv[2])
snapshot_download(m["repo"], revision=m["revision"], allow_patterns=[m["pattern"]], local_dir=sys.argv[3])
PY
  GGUF="$(find "$OUT/gguf/$M" -name '*.gguf' | sort | head -1)"
  "$PY" -m voxprint_forge.kl.kv --plan "$PLAN" --kv-model "$M" --gguf "$GGUF" --corpus-dir "$OUT/corpus" --work "$OUT"
  if "$PY" -c "import sys,tomllib;sys.exit(0 if sys.argv[2] in tomllib.load(open(sys.argv[1],'rb'))['bench']['models'] else 1)" "$PLAN" "$M"; then
    ARMS=$("$PY" -c "import sys,tomllib,json;b=tomllib.load(open(sys.argv[1],'rb'))['bench'];print(json.dumps({a['id']:a['args'] for a in b['spec_arms']+b['prefix_arms']}))" "$PLAN")
    for ARM in $("$PY" -c "import json,sys;print(' '.join(json.loads(sys.argv[1])))" "$ARMS"); do
      mapfile -t EXTRA < <("$PY" -c "import json,sys;[print(x) for x in json.loads(sys.argv[1])[sys.argv[2]]]" "$ARMS" "$ARM")
      llama-server -m "$GGUF" -c 8192 -ngl 99 -fa on --host 127.0.0.1 --port 8090 -np 1 --fit off --cache-prompt \
        "${EXTRA[@]}" > "$OUT/server-$M-$ARM.log" 2>&1 &
      PID=$!
      READY=0
      for _ in $(seq 1 180); do
        if curl -sf http://127.0.0.1:8090/health >/dev/null; then READY=1; break; fi
        if ! kill -0 "$PID" 2>/dev/null; then break; fi
        sleep 2
      done
      if [[ "$READY" -eq 1 ]]; then
        "$PY" -m voxprint_forge.kl.kv --plan "$PLAN" --bench-model "$M" --cell-arm "$ARM" \
          --url http://127.0.0.1:8090 --corpus-dir "$OUT/corpus" >> "$OUT/bench-$M.jsonl" \
          || echo "{\"model\":\"$M\",\"arm\":\"$ARM\",\"status\":\"error\"}" >> "$OUT/bench-$M.jsonl"
      else
        echo "{\"model\":\"$M\",\"arm\":\"$ARM\",\"status\":\"server_refused\"}" >> "$OUT/bench-$M.jsonl"
      fi
      kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true
    done
  fi
  rm -rf "$OUT/gguf/$M"
done
echo "kv done: $OUT"
