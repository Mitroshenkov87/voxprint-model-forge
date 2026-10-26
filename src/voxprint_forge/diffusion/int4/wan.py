"""Wan 2.2 A14B GGUF, FP8, and a short smoke test."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from voxprint_forge.diffusion.fp8 import quantize_state, save_quantized
from voxprint_forge.diffusion.int4.context import StageContext
from voxprint_forge.diffusion.int4.disk import reclaim_sources
from voxprint_forge.diffusion.int4.pins import (
    COMFY_GGUF_COMMIT,
    LLAMA_CPP_COMMIT,
    WAN_CODE_COMMIT,
    WAN_EXPERTS,
    WAN_LICENCE,
    WAN_QUANTS,
    WAN_SMOKE_FRAMES,
    WAN_SMOKE_SIZE,
    WAN_SMOKE_STEPS,
    WAN_WEIGHT_REPO,
    WAN_WEIGHT_REVISION,
)
from voxprint_forge.diffusion.int4.publish import (
    LocalMirror,
    live_drive,
    live_hub,
    publish_verified,
    render_int4_card,
    write_publish_tree,
)
from voxprint_forge.diffusion.int4.runtime import execute
from voxprint_forge.errors import ToolMissingError
from voxprint_forge.mdl import LLAMA_CC, WAN_LLAMA_TAG, ManifestInput

JOB = "wan-2.2-a14b"


def _python(ctx: StageContext) -> str:
    env = ctx.tool("FORGE_WAN_PYTHON", "venv-wan", "bin", "python")
    if env.is_file():
        return str(env)
    return sys.executable


def merge_shards(paths: list[Path], dest: Path) -> None:
    """Concatenate safetensors shards into one file."""
    tensors: dict[str, torch.Tensor] = {}
    for path in paths:
        tensors.update(load_file(str(path)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(dest))


def gguf_name(expert: str, quant: str) -> str:
    """Published GGUF file name for one expert and quant."""
    label = expert.replace("_", "-")
    return f"wan2.2-t2v-a14b-{label}-{quant}.gguf"


def fp8_name(expert: str) -> str:
    """Published FP8 file name for one expert."""
    label = expert.replace("_", "-")
    return f"wan2.2-t2v-a14b-{label}-fp8.safetensors"


def _write_shards(folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in (1, 2):
        path = folder / f"shard-{index}.safetensors"
        save_file({f"blocks.{index}.weight": torch.ones(4, 4)}, str(path))
        paths.append(path)
    return paths


def _placeholder_gguf(path: Path, quant: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + f"voxprint-dry-run {quant}\n".encode("ascii"))


def convert_commands(ctx: StageContext, expert: str, merged: Path, work: Path) -> list[list[str]]:
    """city96 convert, llama-quantize for both K-quants, then the 5D-tensor fix."""
    comfy = ctx.tool("FORGE_COMFY_GGUF", "ComfyUI-GGUF")
    llama_bin = ctx.tool("FORGE_LLAMA_BIN", "llama.cpp", "build", "bin")
    python = _python(ctx)
    raw = work / "raw" / f"{expert}.gguf"
    commands = [
        [python, str(comfy / "tools" / "convert.py"), "--src", str(merged), "--dst", str(raw)],
    ]
    for quant in WAN_QUANTS:
        dst = work / "gguf" / gguf_name(expert, quant)
        commands.append([str(llama_bin / "llama-quantize"), str(raw), str(dst), quant])
        commands.append(
            [
                python,
                str(comfy / "tools" / "fix_5d_tensors.py"),
                "--src",
                str(dst),
                "--dst",
                str(work / "fixed" / dst.name),
            ]
        )
    return commands


_EXPERT_DIR = {"high_noise": "high_noise_model", "low_noise": "low_noise_model"}


def expert_shards(folder: Path) -> list[Path]:
    """Safetensors shards in one expert directory, preferring the numbered shards."""
    numbered = sorted(folder.glob("diffusion_pytorch_model-*-of-*.safetensors"))
    if numbered:
        return numbered
    single = folder / "diffusion_pytorch_model.safetensors"
    if single.is_file():
        return [single]
    return sorted(path for path in folder.glob("*.safetensors") if path.is_file())


def _expert_shards(ctx: StageContext, expert: str, work: Path) -> list[Path]:
    local = expert_shards(work / "shards" / expert)
    if local:
        return local
    if ctx.dry_run:
        return _write_shards(work / "shards" / expert)
    ckpt = os.environ.get("FORGE_WAN_CKPT", "").strip()
    if ckpt:
        found = expert_shards(Path(ckpt) / _EXPERT_DIR[expert])
        if found:
            return found
    raise ToolMissingError(
        f"no shards for {expert}. Download {WAN_WEIGHT_REPO}@{WAN_WEIGHT_REVISION}.",
        code="tool_missing",
    )


def gguf_step(ctx: StageContext) -> dict[str, object]:
    """Quantize both experts to Q4_K_M and Q5_K_M. Dry-run writes tiny files."""
    work = ctx.job(JOB)
    produced: list[str] = []
    commands: list[list[str]] = []
    for expert in WAN_EXPERTS:
        shards = _expert_shards(ctx, expert, work)
        merged = work / "merged" / f"{expert}.safetensors"
        if shards:
            merge_shards(shards, merged)
        commands.extend(convert_commands(ctx, expert, merged, work))
        for quant in WAN_QUANTS:
            dest = work / "gguf" / gguf_name(expert, quant)
            if ctx.dry_run:
                _placeholder_gguf(dest, quant)
            produced.append(str(dest))
    if not ctx.dry_run:
        for argv in commands:
            execute(argv, cwd=work, disk_path=ctx.forge_out, dry_run=False, ratio=ctx.free_ratio)
    return {
        "files": produced,
        "commands": commands,
        "experts": list(WAN_EXPERTS),
        "quants": list(WAN_QUANTS),
        "llama_cpp": LLAMA_CPP_COMMIT,
        "comfy_gguf": COMFY_GGUF_COMMIT,
        "dry_run": ctx.dry_run,
    }


def fp8_step(ctx: StageContext) -> dict[str, object]:
    """Write a voxprint-fp8-v1 file per expert from the merged weights."""
    work = ctx.job(JOB)
    files: list[str] = []
    for expert in WAN_EXPERTS:
        merged = work / "merged" / f"{expert}.safetensors"
        if not merged.is_file():
            if not ctx.dry_run:
                raise ToolMissingError(f"missing merged weights {merged}", code="tool_missing")
            _write_shards(work / "shards" / expert)
            merge_shards(sorted((work / "shards" / expert).glob("*.safetensors")), merged)
        tensors = load_file(str(merged))
        named = {f"{name}.weight" if not name.endswith(".weight") else name: value for name, value in tensors.items()}
        quantized = quantize_state(named, "diffusion")
        dest = work / "fp8" / fp8_name(expert)
        save_quantized(quantized, dest)
        files.append(str(dest))
    return {"files": files, "format": "voxprint-fp8-v1"}


def smoke_argv(ctx: StageContext, save_file_path: Path) -> list[str]:
    """Pinned Wan2.2 generate.py command: 5 frames, lowest t2v size, 4 steps."""
    repo = ctx.tool("FORGE_WAN_REPO", "Wan2.2")
    ckpt = ctx.tool("FORGE_WAN_CKPT", "wan-weights")
    return [
        _python(ctx),
        str(repo / "generate.py"),
        "--task",
        "t2v-A14B",
        "--size",
        WAN_SMOKE_SIZE,
        "--frame_num",
        str(WAN_SMOKE_FRAMES),
        "--sample_steps",
        str(WAN_SMOKE_STEPS),
        "--offload_model",
        "True",
        "--t5_cpu",
        "--base_seed",
        "0",
        "--ckpt_dir",
        str(ckpt),
        "--prompt",
        "An empty coastal cliff at dawn, pale water, no people.",
        "--save_file",
        str(save_file_path),
    ]


def smoke_step(ctx: StageContext) -> dict[str, object]:
    """Record a few low-resolution frames. Dry-run does not load Wan."""
    work = ctx.job(JOB)
    save = work / "smoke" / "smoke.mp4"
    argv = smoke_argv(ctx, save)
    payload: dict[str, object] = {
        "frames": WAN_SMOKE_FRAMES,
        "size": WAN_SMOKE_SIZE,
        "steps": WAN_SMOKE_STEPS,
        "argv": argv,
        "wan_code": WAN_CODE_COMMIT,
        "dry_run": ctx.dry_run,
    }
    if ctx.dry_run:
        payload["backend"] = "cpu-smoke"
    else:
        execute(
            argv,
            cwd=ctx.tool("FORGE_WAN_REPO", "Wan2.2"),
            disk_path=ctx.forge_out,
            dry_run=False,
            ratio=ctx.free_ratio,
        )
        payload["backend"] = "wan2.2"
        payload["save_file"] = str(save)
    path = work / "smoke" / "smoke.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return payload


def publish_step(ctx: StageContext) -> dict[str, object]:
    """Card, sha256, and upload for the GGUF and FP8 files."""
    work = ctx.job(JOB)
    weights = sorted((work / "gguf").glob("*.gguf")) + sorted((work / "fp8").glob("*.safetensors"))
    if len(weights) < 6:
        raise ToolMissingError(f"expected 4 GGUF and 2 FP8 files, found {len(weights)}", code="wan_files")
    smoke = work / "smoke" / "smoke.json"
    card = render_int4_card(
        ctx.repo,
        title="VOXPrint Wan2.2-T2V-A14B GGUF and FP8",
        base_models=(f"{WAN_WEIGHT_REPO}@{WAN_WEIGHT_REVISION}",),
        pipeline_tag="text-to-video",
        quant_lines=tuple(path.name for path in weights),
        eval_lines=(
            f"Smoke: {WAN_SMOKE_FRAMES} frames at {WAN_SMOKE_SIZE}, {WAN_SMOKE_STEPS} steps.",
        ),
        credits=(
            "Wan-AI, for Wan2.2-T2V-A14B.",
            "city96, for ComfyUI-GGUF convert and the 5D tensor fix.",
            "llama.cpp, for llama-quantize.",
            "Voxprint, for this package. No endorsement is implied.",
        ),
        kind="wan",
    )
    output = work / "publish"
    write_publish_tree(output, [*weights, smoke], card)
    slug = "voxprint-wan2.2-t2v-a14b"
    hub = ctx.hub
    drive = ctx.drive
    if ctx.dry_run and hub is None:
        mirror = LocalMirror(ctx.forge_out / "mirrors")
        hub = mirror.upload_hf
        drive = mirror.upload_drive
    elif hub is None:
        hub = live_hub

        def drive(folder_id: str, files: list) -> list[dict[str, object]]:
            return live_drive(folder_id, files, model_id=slug)
    published = publish_verified(
        output,
        slug=slug,
        card_text=card,
        licence=WAN_LICENCE,
        hub=hub,
        drive=drive,
        manifest=ManifestInput(
            model_id=slug.removeprefix("voxprint-"),
            display_name="VOXPrint Wan2.2-T2V-A14B GGUF and FP8",
            app_ids=("canvas",),
            task="text-to-video",
            license=WAN_LICENCE,
            base_model=WAN_WEIGHT_REPO,
            engine="llama.cpp",
            version_req=WAN_LLAMA_TAG,
            min_vram_gb=16,
            recommended_vram_gb=24,
            gpu_min_cc=LLAMA_CC,
        ),
    )
    sources = [work / "shards", work / "merged", work / "raw", work / "source"]
    return {
        "verified": published.verified,
        "hf": published.hf,
        "drive": published.drive,
        "sources": [str(path) for path in sources],
        "files": list(published.records),
        "slug": slug,
    }


def reclaim_step(sources: list[str], *, verified: bool) -> dict[str, object]:
    """Delete Wan sources after the publish marker says the mirror matched."""
    removed = reclaim_sources(sources, verified=verified)
    return {"removed": removed, "verified": verified}
