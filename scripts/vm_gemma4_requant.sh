#!/usr/bin/env bash
# GEMMA4-REQUANT: re-quantise gemma-4-31b from the author's ablated safetensors and check that the ablation survived.
# Runs on synth4 after SYNTH-14. Uploads nothing; the new GGUFs stay in $W/out until a PASS is reviewed.
#
# Usage: scripts/vm_gemma4_requant.sh --prompts FILE [--dry-run]
#   --dry-run  preflight only: disk, deps, binaries, GPU, HF reachability, revision sha. No downloads, no GPU work.
# Needs FORGE_LLAMA_BIN (llama-imatrix, llama-quantize, llama-perplexity, llama-server, build 11476) and
# FORGE_LLAMA_SRC (llama.cpp checkout at 988190680d5a89fce97de3c20df2c2813731fd61 for convert_hf_to_gguf.py).
# FILE is the private refusal-probe set (mode 600). Its text is never logged; results carry sha256[:12] ids only.
# Stages leave a marker in $W/state, so a rerun resumes. This script does not print tokens.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORGE_OUT="${FORGE_OUT:-$HOME/forge-out}"
VENV="${FORGE_VENV:-$FORGE_OUT/venv}"
PY="$VENV/bin/python"
W="${G4R_WORK:-$FORGE_OUT/gemma4-requant}"
RES="$W/results"
PORT="${G4R_PORT:-8097}"
export FORGE_ROOT="$ROOT" FORGE_OUT
export HF_HOME="${HF_HOME:-$FORGE_OUT/hf-home}"
export HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1
unset FORGE_IMATRIX_RU  # the shipped IQ3_XXS used the bundled corpus only (quantize.imatrix.dataset, 128 chunks)

AUTHOR_REPO="paperscarecrow/Gemma-4-31B-it-abliterated"
AUTHOR_REV="2491ee58472b82a2372d288f982943b9d5e6a82c"
AUTHOR_DIR="gemma-4-31b-abliterated"
STOCK_URL="https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF/resolve/4fa4fdf38bee237b5c9e8a5b4e72cf39404c9dcc/gemma-4-31B-it-BF16.gguf"
SHIP_REPO="Mitroshenkov87/voxprint-gemma4-31b"
SHIP_REV="0d84d5f0d1ea740c85c16de0a7c7a844a8bee587"
SHIP_IQ3_SHA="c7bb8f9f4bdb396fb40a5beb617de27132cda6727b885e82f635340f353c5340"
SHIP_Q4_SHA="380f2f50f441225e669ab1541220985387bf3933573cd2f4f7e667f3790e73a3"
LLAMA_COMMIT="988190680d5a89fce97de3c20df2c2813731fd61"
TIERS=(IQ3_XXS Q4_K_M)
# Layer 59 attn_output (o_proj) and ffn_down (down_proj) are the only matrices the author's script changed
# (scan of every layer's o_proj, down_proj and q_proj against google/gemma-4-31B-it).
MODIFIED=(blk.59.attn_output.weight blk.59.ffn_down.weight)
CONTROLS=(blk.58.attn_output.weight blk.58.ffn_down.weight blk.59.attn_q.weight)
# L2 on synth4 (2026-10-17): shipped-lineage quants against the stock-identical F16, same corpus/chunks/ctx.
L2_KL='{"IQ3_XXS": {"kl": 2.293518, "top1": 44.314}, "Q4_K_M": {"kl": 0.827463, "top1": 68.366}}'
# Disk: author folder 62.58 GB (deleted after conversion) + new F16 61.4 + new tiers 30.8 + shipped tiers 30.8
# + KL base logits up to 27 + 10% slack. The peak is after conversion, about 150 GB; require 200 GB as asked.
# A resume after the quant stage already holds the F16 and tiers: it needs the shipped tiers + logits + slack.
NEED_GB=200
RESUME_NEED_GB=70

PROMPTS=""
DRY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --prompts) PROMPTS="${2:-}"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) printf 'usage: vm_gemma4_requant.sh --prompts FILE [--dry-run]\n' >&2; exit 2 ;;
  esac
done
[[ -n "$PROMPTS" ]] || { printf 'usage: vm_gemma4_requant.sh --prompts FILE [--dry-run]\n' >&2; exit 2; }

mkdir -p "$W/state" "$RES"
log() { printf '[g4r %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$RES/run.log"; }
die() { log "FAIL: $*"; exit 1; }
done_() { [[ -f "$W/state/$1" ]]; }
mark() { touch "$W/state/$1"; }

manifest() {
  cat <<'EOF'
d1ac1e9509e5492b9924c8f1f6d28d4a4d9c10c0f7f410890dd852647eae8b75  gemma-4-31b-abliterated/model-00001-of-00002.safetensors
cbd223bb460ee1c92fd286f51880acd12955f987ba2cfa83a563c375010f864e  gemma-4-31b-abliterated/model-00002-of-00002.safetensors
a2619fe11b50dbed06ac443c51d757b354d0b62d64baa514404d4e84e6713519  gemma-4-31b-abliterated/tokenizer.json
2dfbfc7d538912f4ea11d29d85b4e25d7bc26386e53f57529f1d707c28b5828c  gemma-4-31b-abliterated/chat_template.jinja
15e57c2d2945536d2c2fef5d33090b73de7291654a6072e69c90f25980e6e5b0  gemma-4-31b-abliterated/config.json
afb77136430424e6c4bfdf6d2006c706dc7a4affb45d7edc3472729baf114b7b  gemma-4-31b-abliterated/gemma4_31b_abliterator.py
5439541d3bf0ba9ade7c1122dc14703f85f32a520150351b344b14f48c574cc2  gemma-4-31b-abliterated/generation_config.json
390b7f9a0a502e4526cf20862d22954c911cc318d8aba2cdb745c6fbd98264f6  gemma-4-31b-abliterated/model.safetensors.index.json
32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c  gemma-4-31b-abliterated/processor_config.json
b9686f02a0bb2d5de25e5bed7949d5858109e62cbb1111ce4cea731885e779e9  gemma-4-31b-abliterated/tokenizer_config.json
EOF
}

preflight() {
  local bad=0 free bin
  free=$(df -BG --output=avail "$W" | tail -1 | tr -dc 0-9)
  if done_ quant; then NEED_GB=$RESUME_NEED_GB; log "resume: quant stage done, F16 and tiers on disk"; fi
  if (( free < NEED_GB )); then log "disk: ${free} GB free under $W, need ${NEED_GB}"; bad=1; else log "disk: ${free} GB free (need ${NEED_GB})"; fi
  for bin in llama-imatrix llama-quantize llama-perplexity llama-server; do
    if [[ ! -x "${FORGE_LLAMA_BIN:-}/$bin" ]]; then log "binary missing: $bin"; bad=1; continue; fi
    # llama-quantize has no --version; it ships in the same build directory as the others.
    [[ "$bin" == llama-quantize ]] && continue
    if ! "$FORGE_LLAMA_BIN/$bin" --version 2>&1 | grep -q "build 11476, commit 9881906"; then log "binary $bin is not b11476"; bad=1; fi
  done
  if [[ ! -f "${FORGE_LLAMA_SRC:-}/convert_hf_to_gguf.py" ]]; then log "convert_hf_to_gguf.py missing"; bad=1
  elif [[ "$(git -C "$FORGE_LLAMA_SRC" rev-parse HEAD 2>/dev/null)" != "$LLAMA_COMMIT" ]]; then log "FORGE_LLAMA_SRC is not $LLAMA_COMMIT"; bad=1
  else log "llama.cpp: build 11476 binaries and convert script at ${LLAMA_COMMIT:0:9}"; fi
  [[ -x "$PY" && -x "$VENV/bin/forge" ]] || { log "forge venv missing at $VENV"; bad=1; }
  if ! "$PY" - <<'EOF' 2>&1 | tee -a "$RES/run.log"; then bad=1; fi
import importlib.metadata as m, sys
need = {"protobuf": "5.29.6"}
ok = True
for name in ("sentencepiece", "protobuf", "gguf", "huggingface_hub", "torch", "numpy", "safetensors", "transformers"):
    try:
        version = m.version(name)
    except m.PackageNotFoundError:
        print(f"dep missing: {name}"); ok = False; continue
    if name in need and version != need[name]:
        print(f"dep {name}=={version}, need {need[name]}"); ok = False
    else:
        print(f"dep {name} {version}")
import voxprint_forge.requant_check  # noqa: F401
sys.exit(0 if ok else 1)
EOF
  if command -v nvidia-smi >/dev/null; then
    log "gpu: $(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader | head -1)"
    [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]] || { log "gpu busy: wait for SYNTH-14 to finish"; bad=1; }
  else log "nvidia-smi missing"; bad=1; fi
  local mode; mode=$(stat -c %a "$PROMPTS" 2>/dev/null || echo none)
  if [[ "$mode" != 600 ]]; then log "prompt set mode is $mode, need 600"; bad=1
  else log "prompt set: $(grep -cv '^#' "$PROMPTS") prompts, sha256[:12] $(sha256sum "$PROMPTS" | cut -c1-12)"; fi
  if ! "$PY" - "$AUTHOR_REPO" "$AUTHOR_REV" "$AUTHOR_DIR" "$SHIP_REPO" "$SHIP_REV" "$STOCK_URL" 2>&1 <<'EOF' | tee -a "$RES/run.log"; then bad=1; fi
import sys, urllib.request
from huggingface_hub import HfApi
author, rev, folder, ship, ship_rev, stock = sys.argv[1:]
api = HfApi()
info = api.model_info(author, revision=rev, files_metadata=True)
files = {s.rfilename: s for s in info.siblings if s.rfilename.startswith(folder + "/")}
total = sum(s.size or 0 for s in files.values())
print(f"hf: {author} @ {info.sha} ({len(files)} files, {total / 1e9:.2f} GB)")
ok = info.sha == rev and len(files) == 10
for name, sha in (("model-00001-of-00002.safetensors", "d1ac1e95"), ("model-00002-of-00002.safetensors", "cbd223bb")):
    entry = files.get(f"{folder}/{name}")
    ok &= bool(entry and entry.lfs and entry.lfs.sha256.startswith(sha))
ship_info = api.model_info(ship, revision=ship_rev)
print(f"hf: {ship} @ {ship_info.sha}")
ok &= ship_info.sha == ship_rev
request = urllib.request.Request(stock, headers={"Range": "bytes=0-3"})
with urllib.request.urlopen(request, timeout=60) as response:
    magic = response.read()
print(f"hf: stock BF16 range read {'ok' if magic == b'GGUF' else 'FAILED'}")
ok &= magic == b"GGUF"
sys.exit(0 if ok else 1)
EOF
  return $bad
}

log "GEMMA4-REQUANT preflight (dry_run=$DRY)"
if ! preflight; then die "preflight"; fi
log "preflight OK"
if [[ "$DRY" -eq 1 ]]; then exit 0; fi

SRC="$W/src" OUT="$W/out" F16="$W/out/gemma4-31b-requant-F16.gguf"
mkdir -p "$SRC" "$OUT"

if ! done_ download; then
  log "download $AUTHOR_REPO @ $AUTHOR_REV ($AUTHOR_DIR/)"
  "$PY" -c 'import sys; from huggingface_hub import snapshot_download as s; s(repo_id=sys.argv[1], revision=sys.argv[2], local_dir=sys.argv[3], allow_patterns=[sys.argv[4] + "/*"])' \
    "$AUTHOR_REPO" "$AUTHOR_REV" "$SRC" "$AUTHOR_DIR"
  (cd "$SRC" && manifest | sha256sum -c --quiet) || die "author file sha256"
  manifest > "$RES/author-files.sha256"
  mark download
fi

if ! done_ convert; then
  log "convert to F16 (llama.cpp ${LLAMA_COMMIT:0:9})"
  "$PY" "$FORGE_LLAMA_SRC/convert_hf_to_gguf.py" "$SRC/$AUTHOR_DIR" --outtype f16 --outfile "$F16" >"$RES/convert.log" 2>&1 || die "convert (see convert.log)"
  mark convert
fi

if ! done_ diff; then
  log "tensor diff vs stock (modified ${MODIFIED[*]})"
  if ! "$PY" -m voxprint_forge.requant_check tensor-diff --local "$F16" --stock-url "$STOCK_URL" \
      --modified "${MODIFIED[@]}" --controls "${CONTROLS[@]}" --out "$RES/tensor-diff.json" >/dev/null; then
    die "tensor diff: the converted F16 does not carry the ablation (tensor-diff.json); stopping before GPU work"
  fi
  mark diff
fi
rm -rf "$SRC"

if ! done_ quant; then
  log "imatrix (bundled corpus, 128 chunks, ctx 512) and tiers ${TIERS[*]}"
  CORPUS=$("$PY" -c 'from voxprint_forge.calibration import ensure_calibration; print(ensure_calibration())')
  "$FORGE_LLAMA_BIN/llama-imatrix" -m "$F16" -f "$CORPUS" -o "$OUT/imatrix.gguf" --output-format gguf \
    --chunks 128 -c 512 -ngl 99 -fa on --no-ppl >"$RES/imatrix.log" 2>&1 || die "imatrix"
  for tier in "${TIERS[@]}"; do
    "$FORGE_LLAMA_BIN/llama-quantize" --imatrix "$OUT/imatrix.gguf" "$F16" "$OUT/$tier.gguf" "$tier" >"$RES/quantize-$tier.log" 2>&1 || die "quantize $tier"
  done
  (cd "$OUT" && sha256sum gemma4-31b-requant-F16.gguf imatrix.gguf "${TIERS[@]/%/.gguf}") > "$RES/new-files.sha256"
  mark quant
fi

if ! done_ kl; then
  log "KL/top-1 of the new tiers against the new F16"
  PLAN="$W/kl-g4r.toml"
  # Local tiers carry source = "local" and a content revision (local-sha256:<12 hex>) from new-files.sha256;
  # the plan is loaded back by the KL loader here, before any GPU work.
  [[ -s "$RES/new-files.sha256" ]] || die "new-files.sha256 missing; rerun the quant stage"
  TIER_ARGS=()
  for tier in "${TIERS[@]}"; do TIER_ARGS+=(--tier "$tier=$OUT/$tier.gguf"); done
  "$PY" -m voxprint_forge.requant_check kl-plan --job G4R --author-repo "$AUTHOR_REPO" --author-rev "$AUTHOR_REV" \
    --f16 "$F16" "${TIER_ARGS[@]}" --sha256-file "$RES/new-files.sha256" --plan-out "$PLAN" \
    --note "Re-quant from the author safetensors at ${AUTHOR_REV:0:8}, scored against its own F16. PPL is not meaningful for Gemma 4 instruct on raw prose; use KL and top-1." \
    2>&1 | tee -a "$RES/run.log" || die "kl plan"
  cp "$PLAN" "$RES/kl-g4r.toml"
  "$VENV/bin/forge" --root "$ROOT" --forge-out "$FORGE_OUT" kl --job G4R --plan "$PLAN" >"$RES/kl.log" 2>&1 || die "kl (see kl.log)"
  cp "$FORGE_OUT/results/kl/G4R/summary.json" "$RES/kl-summary.json"
  cp "$FORGE_OUT/results/kl/G4R/summary.md" "$RES/kl-summary.md"
  mark kl
fi

if ! done_ shipped; then
  log "fetch the shipped tiers (stock arm of the probe)"
  "$PY" -c 'import sys; from huggingface_hub import hf_hub_download as h; h(sys.argv[1], "IQ3_XXS.gguf", revision=sys.argv[2], local_dir=sys.argv[3]); h("paperscarecrow/Gemma-4-31B-it-abliterated", "gemma-4-31b-abliterated-Q4_K_M.gguf", revision="2491ee58472b82a2372d288f982943b9d5e6a82c", local_dir=sys.argv[3])' \
    "$SHIP_REPO" "$SHIP_REV" "$W/shipped"
  printf '%s  IQ3_XXS.gguf\n%s  gemma-4-31b-abliterated-Q4_K_M.gguf\n' "$SHIP_IQ3_SHA" "$SHIP_Q4_SHA" | (cd "$W/shipped" && sha256sum -c --quiet) || die "shipped sha256"
  mark shipped
fi

serve_probe() {  # label model
  local label="$1" model="$2" pid ok=0
  done_ "probe-$label" && return 0
  [[ -n "$label" && -s "$model" ]] || die "probe ${label:-?}: model file missing"
  "$FORGE_LLAMA_BIN/llama-server" -m "$model" -c 8192 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --jinja \
    --host 127.0.0.1 --port "$PORT" >"$RES/server-$label.log" 2>&1 &
  pid=$!
  for _ in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then ok=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 2
  done
  if (( ok )); then
    "$PY" -m voxprint_forge.requant_check probe --server "http://127.0.0.1:$PORT" --prompts "$PROMPTS" \
      --label "$label" --out "$RES/probe-$label.json" >/dev/null || ok=0
  fi
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  (( ok )) || die "probe $label (see server-$label.log; an empty probe set fails here)"
  log "probe $label: $("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("refusal rate %s, %s/%s answered, %s empty" % (d["refusal_rate"], d["answered"], d["n"], d["empty"]))' "$RES/probe-$label.json")"
  mark "probe-$label"
}
serve_probe stock-IQ3_XXS "$W/shipped/IQ3_XXS.gguf"
serve_probe new-IQ3_XXS "$OUT/IQ3_XXS.gguf"
serve_probe stock-Q4_K_M "$W/shipped/gemma-4-31b-abliterated-Q4_K_M.gguf"
serve_probe new-Q4_K_M "$OUT/Q4_K_M.gguf"

set +e
"$PY" -m voxprint_forge.requant_check verdict --results "$RES" --kl-summary "$RES/kl-summary.json" --l2-kl "$L2_KL" \
  --pair IQ3_XXS:stock-IQ3_XXS:new-IQ3_XXS --pair Q4_K_M:stock-Q4_K_M:new-Q4_K_M > "$RES/verdict.json"
status=$?
set -e
verdict=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("verdict") or "")' "$RES/verdict.json" 2>/dev/null || true)
[[ -n "$verdict" ]] || die "verdict did not write verdict.json (exit $status)"
log "verdict: $verdict"
log "new GGUFs kept in $OUT (not uploaded); sync $RES back"
exit $status
