#!/usr/bin/env bash
# MOSS-Image fit job (synth4). Arms x caps from plans/moss-synth4.toml.
# Usage: scripts/vm_run_moss_fit.sh [--preflight-only]
# Separate venv: the vendored pipeline needs diffusers>=0.38, transformers>=5.3 (the forge venv pins <0.38).
# Downloads: the author SFT snapshot at the pinned revision (27.2 GB) and three Comfy-Org reference files (15.5 GB).
# Results: $FORGE_OUT/moss-fit/<arm>/<cap>/result.json + PNGs named by prompt sha256[:12]. No tokens or hosts printed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FORGE_OUT="${FORGE_OUT:-$HOME/forge-out}"
VENV="${MOSS_VENV:-$FORGE_OUT/moss-venv}"
OUT="$FORGE_OUT/moss-fit"
SNAP="$FORGE_OUT/moss-snapshots/sft-fc5de0d5"
COMFY="$FORGE_OUT/comfyui-a639f187"
PLAN="$ROOT/plans/moss-synth4.toml"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="${HF_HOME:-$FORGE_OUT/hf-home}"
export HF_HUB_DISABLE_TELEMETRY=1 PYTHONUNBUFFERED=1 PYTHONPATH="$ROOT/src"
PREFLIGHT_ONLY=0
[[ "${1:-}" == "--preflight-only" ]] && PREFLIGHT_ONLY=1

mkdir -p "$OUT" "$HF_HOME"
if [[ ! -x "$VENV/bin/python" ]]; then
  uv venv -q -p 3.14 "$VENV"
fi
# Same torch as the forge venv (scripts/vm_setup_synth4.sh).
uv pip install -q --python "$VENV" "torch==2.14.1" --index-url "$TORCH_INDEX"
VIRTUAL_ENV="$VENV" uv pip install -q "diffusers==0.41.0" "transformers==5.20.1" \
  "accelerate>=1.0" "safetensors>=0.4" Pillow "huggingface_hub>=0.26" tomli-w
# Fast GDN kernels for the Qwen3.5 text encoder; optional (preflight warns when missing).
VIRTUAL_ENV="$VENV" uv pip install -q flash-linear-attention causal-conv1d || echo "warn: optional GDN kernels not installed"

"$VENV/bin/python" -m voxprint_forge.diffusion.moss_preflight --plan "$PLAN" --vm --disk-path "$FORGE_OUT"
"$VENV/bin/python" -m voxprint_forge.diffusion.moss_fit --plan "$PLAN" --dry-run > "$OUT/matrix.json"
if [[ "$PREFLIGHT_ONLY" -eq 1 ]]; then
  echo "preflight only: matrix at $OUT/matrix.json"
  exit 0
fi

# Author snapshot at the pinned revision; safetensors only, no remote code.
"$VENV/bin/python" - "$SNAP" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("OpenMOSS-Team/MOSS-Image-SFT", revision="fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
                  local_dir=sys.argv[1], allow_patterns=["*.json", "*.jinja", "*.safetensors", "LICENSE"])
PY

"$VENV/bin/python" -m voxprint_forge.diffusion.moss_fit --plan "$PLAN" --arm bf16-te-offload --snapshot "$SNAP" --out "$OUT"
"$VENV/bin/python" -m voxprint_forge.diffusion.moss_fit --plan "$PLAN" --arm c-dq --snapshot "$SNAP" --out "$OUT"

# Comfy INT8 reference: ComfyUI at the pinned commit, its own venv, files sha-checked.
if [[ ! -d "$COMFY/.git" ]]; then
  git clone -q --filter=blob:none https://github.com/Comfy-Org/ComfyUI "$COMFY"
fi
git -C "$COMFY" fetch -q origin a639f187a9f6abfecbda9673034f2045148b6d85
git -C "$COMFY" checkout -q a639f187a9f6abfecbda9673034f2045148b6d85
[[ -x "$COMFY/venv/bin/python" ]] || uv venv -q -p 3.14 "$COMFY/venv"
uv pip install -q --python "$COMFY/venv" "torch==2.14.1" torchvision torchaudio --index-url "$TORCH_INDEX"
VIRTUAL_ENV="$COMFY/venv" uv pip install -q -r "$COMFY/requirements.txt" "comfy-kitchen==0.2.38"
"$VENV/bin/python" - "$COMFY/models" "$PLAN" <<'PY'
import hashlib, sys, tomllib
from pathlib import Path
from huggingface_hub import hf_hub_download
models, plan = Path(sys.argv[1]), tomllib.loads(Path(sys.argv[2]).read_text())
arm = next(a for a in plan["arms"] if a["engine"] == "comfyui")
for rel, want in zip(arm["files"], arm["sha256"]):
    p = Path(hf_hub_download(arm["repo"], rel, revision=arm["revision"], local_dir=models))
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    assert h.hexdigest() == want, f"sha mismatch {rel}"
    print("ok", rel, want[:12])
PY
TOTAL_GB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | awk '{printf "%d", $1/1024}')
for CAP in 16 24; do
  RESERVE=$((TOTAL_GB - CAP))
  "$COMFY/venv/bin/python" "$COMFY/main.py" --listen 127.0.0.1 --port 8199 --disable-auto-launch \
    --reserve-vram "$RESERVE" --output-directory "$OUT/comfy-out" > "$OUT/comfy-$CAP.log" 2>&1 &
  PID=$!
  for _ in $(seq 1 120); do curl -sf http://127.0.0.1:8199/system_stats >/dev/null && break; sleep 2; done
  "$VENV/bin/python" -m voxprint_forge.diffusion.moss_fit --plan "$PLAN" --arm comfy-int8 --cap "$CAP" \
    --comfy-url http://127.0.0.1:8199 --out "$OUT" || echo "comfy cap $CAP failed (see comfy-$CAP.log)"
  kill "$PID"; wait "$PID" 2>/dev/null || true
done
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader > "$OUT/gpu.txt"
echo "moss-fit done: $OUT"
