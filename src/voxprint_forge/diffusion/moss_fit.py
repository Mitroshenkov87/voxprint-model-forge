"""MOSS-Image VRAM/speed fit job: arms x memory caps, 1024², 50 steps, CFG 3.

``--dry-run`` prints the matrix and checks the plan without torch. A real run writes
``<out>/<arm>/<cap>/result.json`` plus PNGs named ``<prompt sha256[:12]>-<seed>.png``.
Memory caps emulate 16/24 GB cards on a larger GPU. diffusers arms use
``torch.cuda.set_per_process_memory_fraction``; the ComfyUI arm is started with
``--reserve-vram``. Emulation is not a real 16/24 GB card.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import tomllib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from voxprint_forge.errors import PlanError


@dataclass(frozen=True)
class Arm:
    """One fit arm from plans/moss-synth4.toml."""

    id: str
    role: str
    engine: str
    weights: str = ""
    te_mode: dict[str, str] = field(default_factory=dict)
    repo: str = ""
    revision: str = ""
    files: tuple[str, ...] = ()
    sha256: tuple[str, ...] = ()
    comfyui_commit: str = ""
    comfy_kitchen: str = ""


@dataclass(frozen=True)
class FitPlan:
    """Defaults, mirrors, and arms."""

    width: int
    height: int
    steps: int
    guidance: float
    uncond_skip_layers: tuple[int, ...]
    seeds: tuple[int, ...]
    warmup: int
    caps_gb: tuple[float, ...]
    prompts: tuple[str, ...]
    model: str
    arms: tuple[Arm, ...]
    mirrors: tuple[dict[str, object], ...]


def load_plan(path: Path) -> FitPlan:
    """Parse and validate the plan."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    d = raw.get("defaults", {})
    arms = []
    for a in raw.get("arms", []):
        arm = Arm(
            id=str(a["id"]),
            role=str(a["role"]),
            engine=str(a["engine"]),
            weights=str(a.get("weights", "")),
            te_mode={str(k): str(v) for k, v in a.get("te_mode", {}).items()},
            repo=str(a.get("repo", "")),
            revision=str(a.get("revision", "")),
            files=tuple(a.get("files", ())),
            sha256=tuple(a.get("sha256", ())),
            comfyui_commit=str(a.get("comfyui_commit", "")),
            comfy_kitchen=str(a.get("comfy_kitchen", "")),
        )
        if arm.engine not in {"diffusers", "comfyui"}:
            raise PlanError(f"arm {arm.id}: engine {arm.engine!r}", code="plan")
        if arm.engine == "comfyui" and (len(arm.files) != len(arm.sha256) or len(arm.revision) != 40):
            raise PlanError(f"arm {arm.id}: pin revision and one sha256 per file", code="plan")
        if arm.engine == "diffusers" and any(m not in {"cpu", "swap"} for m in arm.te_mode.values()):
            raise PlanError(f"arm {arm.id}: te_mode must be cpu or swap", code="plan")
        arms.append(arm)
    plan = FitPlan(
        width=int(d.get("width", 1024)),
        height=int(d.get("height", 1024)),
        steps=int(d.get("steps", 50)),
        guidance=float(d.get("guidance", 3.0)),
        uncond_skip_layers=tuple(int(x) for x in d.get("uncond_skip_layers", [22])),
        seeds=tuple(int(x) for x in d.get("seeds", [0])),
        warmup=int(d.get("warmup", 1)),
        caps_gb=tuple(float(x) for x in d.get("caps_gb", [16.0, 24.0])),
        prompts=tuple(str(x) for x in d.get("prompts", ())),
        model=str(d.get("model", "sft")),
        arms=tuple(arms),
        mirrors=tuple(raw.get("mirrors", ())),
    )
    if not plan.prompts or not plan.arms:
        raise PlanError("plan needs prompts and arms", code="plan")
    if plan.width % 16 or plan.height % 16:
        raise PlanError("width/height must be multiples of 16", code="plan")
    return plan


def prompt_id(prompt: str) -> str:
    """Public id of a prompt: sha256[:12]."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def cap_key(cap: float) -> str:
    """'16' for 16.0."""
    return str(int(cap)) if float(cap).is_integer() else str(cap)


def skip_reason(arm: Arm, env: dict[str, str]) -> str:
    """Why an arm cannot run here, or ''."""
    if arm.weights.startswith("env:"):
        var = arm.weights[4:]
        if not env.get(var):
            return f"{var} is unset (no weights built yet)"
    return ""


def matrix(plan: FitPlan, env: dict[str, str] | None = None) -> list[dict[str, object]]:
    """Every (arm, cap) cell with its mode and skip reason."""
    env = dict(os.environ) if env is None else env
    rows = []
    for arm in plan.arms:
        for cap in plan.caps_gb:
            rows.append({
                "arm": arm.id,
                "role": arm.role,
                "engine": arm.engine,
                "cap_gb": cap,
                "te_mode": arm.te_mode.get(cap_key(cap), "") if arm.engine == "diffusers" else "comfy",
                "images": len(plan.prompts) * len(plan.seeds),
                "skip": skip_reason(arm, env),
            })
    return rows


def _write(path: Path, doc: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _load_fp8_into(pipe: object, folder: Path) -> None:
    from voxprint_forge.diffusion.fp8 import load_dequant_bf16

    for name in ("transformer", "text_encoder"):
        files = sorted((folder / name).glob("*.safetensors"))
        if not files:
            raise PlanError(f"{folder / name}: no voxprint-fp8-v1 files", code="plan")
        state: dict[str, object] = {}
        for f in files:
            state.update(load_dequant_bf16(f))
        getattr(pipe, name).load_state_dict(state, strict=True)


def run_diffusers(plan: FitPlan, arm: Arm, cap: float, snapshot: Path, out: Path) -> dict[str, object]:
    """One diffusers cell. Returns the result row (also written to result.json)."""
    import torch

    from voxprint_forge.diffusion.moss import encode_on_cpu, load_pipeline

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 1024**3 / total))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mode = arm.te_mode.get(cap_key(cap), "cpu")
    t0 = time.perf_counter()
    pipe = load_pipeline(snapshot)
    if arm.weights.startswith("env:"):
        _load_fp8_into(pipe, Path(os.environ[arm.weights[4:]]))
    if mode == "swap":
        pipe.enable_model_cpu_offload()
    else:
        pipe.transformer.to("cuda")
        pipe.vae.to("cuda")
    load_s = time.perf_counter() - t0
    row: dict[str, object] = {"arm": arm.id, "cap_gb": cap, "te_mode": mode, "load_s": round(load_s, 1)}
    times, encode = [], []
    try:
        jobs = [(p, s) for p in plan.prompts for s in plan.seeds]
        for i, (prompt, seed) in enumerate([jobs[0]] * plan.warmup + jobs):
            kwargs: dict[str, object] = {}
            t1 = time.perf_counter()
            if mode == "cpu":
                pe, pm = encode_on_cpu(pipe, prompt)
                ne, nm = encode_on_cpu(pipe, "")
                kwargs = {"prompt_embeds": pe.to("cuda"), "prompt_attention_mask": pm.to("cuda"),
                          "negative_prompt_embeds": ne.to("cuda"), "negative_prompt_attention_mask": nm.to("cuda")}
            else:
                kwargs = {"prompt": prompt}
            t2 = time.perf_counter()
            image = pipe(height=plan.height, width=plan.width, num_inference_steps=plan.steps,
                         guidance_scale=plan.guidance, uncond_skip_layers=plan.uncond_skip_layers,
                         generator=torch.Generator("cuda").manual_seed(seed), **kwargs).images[0]
            t3 = time.perf_counter()
            if i >= plan.warmup:
                times.append(t3 - t1)
                encode.append(t2 - t1)
                image.save(out / f"{prompt_id(prompt)}-{seed}.png")
        row.update(status="ok", s_per_img=round(sum(times) / len(times), 2),
                   encode_s=round(sum(encode) / len(encode), 2))
    except torch.OutOfMemoryError:
        row.update(status="oom")
    row["peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    row["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1e9, 2)
    return row


def comfy_workflow(plan: FitPlan, arm: Arm, prompt: str, seed: int, prefix: str) -> dict[str, object]:
    """ComfyUI API-format graph with core nodes only (PR #17095 at the pinned commit)."""
    unet, te, vae = (Path(f).name for f in arm.files)
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": te, "type": "moss_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["2", 0]}},
        "6": {"class_type": "EmptyFlux2LatentImage",
              "inputs": {"width": plan.width, "height": plan.height, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
            "seed": seed, "steps": plan.steps, "cfg": plan.guidance, "sampler_name": "euler",
            "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }


def comfy_nodes_missing(object_info: dict[str, object], graph: dict[str, object]) -> list[str]:
    """Node classes the graph uses that the server does not register."""
    return sorted({n["class_type"] for n in graph.values()} - set(object_info))


def _http(url: str, data: dict[str, object] | None = None, timeout: float = 30.0) -> dict[str, object]:
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # localhost only
        return json.loads(r.read())


def run_comfy(plan: FitPlan, arm: Arm, cap: float, url: str, out: Path) -> dict[str, object]:
    """One ComfyUI cell against a server the shell script started with --reserve-vram."""
    if not url.startswith("http://127.0.0.1:"):
        raise PlanError("ComfyUI URL must be http://127.0.0.1:<port>", code="plan")
    info = _http(f"{url}/object_info")
    jobs = [(p, s) for p in plan.prompts for s in plan.seeds]
    missing = comfy_nodes_missing(info, comfy_workflow(plan, arm, "x", 0, "x"))
    if missing:
        return {"arm": arm.id, "cap_gb": cap, "status": "missing_nodes", "missing": missing}
    times = []
    for i, (prompt, seed) in enumerate([jobs[0]] * plan.warmup + jobs):
        prefix = f"moss-fit/{arm.id}-{cap_key(cap)}/{prompt_id(prompt)}-{seed}"
        t0 = time.perf_counter()
        pid = _http(f"{url}/prompt", {"prompt": comfy_workflow(plan, arm, prompt, seed, prefix)})["prompt_id"]
        while True:
            hist = _http(f"{url}/history/{pid}")
            if pid in hist and hist[pid].get("status", {}).get("completed") is not None:
                status = hist[pid]["status"]
                break
            time.sleep(0.5)
        if status.get("status_str") != "success":
            msgs = json.dumps(status.get("messages", []))[-400:]
            return {"arm": arm.id, "cap_gb": cap, "status": "oom" if "OutOfMemory" in msgs else "error",
                    "detail": msgs}
        if i >= plan.warmup:
            times.append(time.perf_counter() - t0)
    stats = _http(f"{url}/system_stats")
    return {"arm": arm.id, "cap_gb": cap, "status": "ok", "s_per_img": round(sum(times) / len(times), 2),
            "devices": stats.get("devices", [])}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="python -m voxprint_forge.diffusion.moss_fit", description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--snapshot", type=Path, help="local author snapshot (diffusers arms)")
    ap.add_argument("--arm")
    ap.add_argument("--cap", type=float)
    ap.add_argument("--comfy-url", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    plan = load_plan(args.plan)
    rows = [r for r in matrix(plan) if (not args.arm or r["arm"] == args.arm) and
            (args.cap is None or r["cap_gb"] == args.cap)]
    if args.dry_run:
        print(json.dumps({"model": plan.model, "steps": plan.steps, "cfg": plan.guidance,
                          "size": [plan.width, plan.height], "cells": rows}, indent=2))
        return 0
    if args.out is None:
        ap.error("--out is required unless --dry-run")
    arms = {a.id: a for a in plan.arms}
    for r in rows:
        arm, cap = arms[str(r["arm"])], float(r["cap_gb"])
        cell = args.out / arm.id / cap_key(cap)
        cell.mkdir(parents=True, exist_ok=True)
        if r["skip"]:
            result = {"arm": arm.id, "cap_gb": cap, "status": "skipped", "reason": r["skip"]}
        elif arm.engine == "comfyui":
            if not args.comfy_url:
                continue
            result = run_comfy(plan, arm, cap, args.comfy_url, cell)
        else:
            if args.snapshot is None:
                ap.error("--snapshot is required for diffusers arms")
            result = run_diffusers(plan, arm, cap, args.snapshot, cell)
        _write(cell / "result.json", result)
        print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
