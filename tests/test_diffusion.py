"""Canvas add-on: FP8, gates, plan, and the overlay that must not touch bootstrap."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import torch
from huggingface_hub import ModelCard
from safetensors import safe_open
from safetensors.torch import save_file

from voxprint_forge.diffusion.cardrender import CardContext, render_canvas_card
from voxprint_forge.diffusion.deadline import DEFERRED, defer_part, resolve_deadline
from voxprint_forge.diffusion.fp8 import (
    FORMAT,
    dequant_state,
    keep_bf16,
    quantize_state,
    quantize_weight,
    read_format,
    save_quantized,
)
from voxprint_forge.diffusion.guard import llm_job_running
from voxprint_forge.diffusion.names import (
    CONTROL_FULL_BF16,
    CONTROL_LITE_FP8,
    EDIT_ENCODER_FP8,
    EDIT_FP8_PREFIX,
    QWEN3_FP8_NAME,
    TURBO_BF16_PREFIX,
    TURBO_FP8_NAME,
    shard_name,
)
from voxprint_forge.diffusion.planload import load_canvas_plan
from voxprint_forge.diffusion.policy import decide_base, decide_control, decide_edit
from voxprint_forge.diffusion.prompts import BASE_PROMPTS, CONTROL_PROMPTS, CONTROL_TYPES, EDIT_PAIRS, INPAINT_PROMPTS
from voxprint_forge.diffusion.scenes import canny_map, contact_sheet, depth_map, pose_map, render_scene
from voxprint_forge.diffusion.spec import PART_MINUTES
from voxprint_forge.diffusion.weights import plan_shards, write_bf16_shards, write_fp8_file
from voxprint_forge.errors import UsageError

ROOT = Path(__file__).resolve().parents[1]
BANNED = ("child", "minor", "nude", "naked", "kill", "blood", "gun", "weapon", "teen")


def test_fp8_scale_metadata_and_roundtrip(tmp_path: Path) -> None:
    weight = torch.randn(8, 8)
    _quantized, scale = quantize_weight(weight)
    amax = weight.detach().float().abs().amax()
    expect = torch.clamp(amax / 448.0, min=torch.finfo(torch.float32).tiny).to(dtype=torch.bfloat16)
    assert scale.dtype == torch.bfloat16
    assert torch.allclose(scale.float(), expect.float(), rtol=0, atol=1e-3)
    state = quantize_state(
        {
            "layer.weight": weight,
            "layer.bias": torch.randn(8),
            "norm.weight": torch.randn(8, 8),
            "ffn.w1.weight": torch.randn(4, 4),
        },
        "diffusion",
    )
    assert state["layer.weight"].dtype == torch.float8_e4m3fn
    assert state["layer.weight.scale_weight"].dtype == torch.bfloat16
    assert state["layer.bias"].dtype == torch.bfloat16
    assert "layer.bias.scale_weight" not in state
    assert state["norm.weight"].dtype == torch.bfloat16
    assert state["ffn.w1.weight"].dtype == torch.float8_e4m3fn
    path = tmp_path / "tiny.safetensors"
    save_quantized(state, path)
    assert read_format(path) == FORMAT
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        assert handle.metadata()["format"] == "voxprint-fp8-v1"
    restored = dequant_state(state)
    assert restored["layer.weight"].dtype == torch.bfloat16
    assert torch.allclose(restored["layer.weight"].float(), weight.float(), rtol=0.2, atol=0.2)
    assert "layer.weight.scale_weight" not in restored


def test_keep_rules() -> None:
    assert keep_bf16("layers.0.adaLN_modulation.0.weight", "diffusion")
    assert keep_bf16("cap_embedder.weight", "diffusion")
    assert keep_bf16("lm_head.weight", "diffusion")
    assert not keep_bf16("layers.0.attention.to_q.weight", "diffusion")
    assert not keep_bf16("layers.0.ffn.w1.weight", "diffusion")
    assert keep_bf16("zero_conv.weight", "controlnet")
    assert not keep_bf16("zero_conv.weight", "diffusion")
    assert keep_bf16("img_in.weight", "edit_transformer")
    assert keep_bf16("proj_out.weight", "edit_transformer")
    assert not keep_bf16("layers.0.attn.to_q.weight", "edit_transformer")
    assert keep_bf16("visual.blocks.0.attn.qkv.weight", "edit_encoder")
    assert keep_bf16("model.embed_tokens.weight", "edit_encoder")
    assert not keep_bf16("model.layers.0.mlp.down_proj.weight", "edit_encoder")


def test_quality_gates() -> None:
    assert decide_base(0.11, 0.0) == "bf16"
    assert decide_base(0.09, 0.005) == "fp8"
    assert decide_base(0.10, 0.01) == "fp8"
    assert decide_base(0.09, 0.02) == "bf16"
    assert decide_control(0.11) == "bf16"
    assert decide_control(0.09) == "fp8"
    assert decide_control(0.10) == "fp8"
    assert decide_edit(0.13, 0.99) == "bf16"
    assert decide_edit(0.11, 0.94) == "bf16"
    assert decide_edit(0.11, 0.96) == "fp8"
    assert decide_edit(0.12, 0.95) == "fp8"


def test_prompt_sets_are_fixed_and_safe() -> None:
    assert len(BASE_PROMPTS) == 32
    assert len(CONTROL_PROMPTS) == 16
    assert CONTROL_TYPES == ("canny", "depth", "pose")
    assert len(INPAINT_PROMPTS) == 8
    assert len(EDIT_PAIRS) == 24
    blob = "\n".join([*BASE_PROMPTS, *CONTROL_PROMPTS, *INPAINT_PROMPTS, *(item[0] for item in EDIT_PAIRS)]).lower()
    for word in BANNED:
        assert word not in blob
    assert "adult" in blob
    assert "clothed" in blob


def test_plan_estimates_and_pins() -> None:
    plan = load_canvas_plan(ROOT / "plans" / "canvas-zimage-tiers-v1.toml")
    assert plan.id == "canvas-zimage-tiers-v1"
    assert [part.id for part in plan.parts] == ["base", "controlnet", "edit"]
    assert [part.estimate_min for part in plan.parts] == [55, 25, 55]
    assert plan.total_min() == 135
    assert {part.id: part.estimate_min for part in plan.parts} == PART_MINUTES
    base = plan.part("base")
    assert base.hf_repo == "Mitroshenkov87/voxprint-mirror-z-image-turbo"
    assert base.sources[0].revision == "f332072aa78be7aecdf3ee76d5c247082da564a6"
    assert base.sources[1].revision == "04cc4abb7c5069926f75c9bfde9ef43d49423021"
    control = plan.part("controlnet")
    assert control.estimate_min == 25
    assert control.hf_repo == "Mitroshenkov87/voxprint-mirror-z-image-turbo-controlnet-union-2.1"
    assert control.sources[0].revision == "5155fc56d17821007d6f62ac192c09e0f0e72016"
    edit = plan.part("edit")
    assert edit.estimate_min == 55
    assert edit.hf_repo == "Mitroshenkov87/voxprint-mirror-qwen-image-edit-2511"
    assert edit.sources[0].revision == "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
    assert TURBO_FP8_NAME == "zimage-turbo-transformer-fp8.safetensors"
    assert QWEN3_FP8_NAME == "qwen3-4b-text-encoder-fp8.safetensors"
    assert CONTROL_LITE_FP8 == "zimage-turbo-controlnet-union-2.1-lite-fp8.safetensors"
    assert CONTROL_FULL_BF16 == "zimage-turbo-controlnet-union-2.1-bf16.safetensors"
    assert EDIT_ENCODER_FP8 == "qwen2.5-vl-7b-text-encoder-fp8.safetensors"
    assert shard_name(TURBO_BF16_PREFIX, 1, 3) == "zimage-turbo-transformer-bf16-00001-of-00003.safetensors"
    assert shard_name(EDIT_FP8_PREFIX, 2, 4) == "qwen-image-edit-2511-transformer-fp8-00002-of-00004.safetensors"


def test_deadline_defers_only_when_the_estimate_exceeds_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_DEADLINE", raising=False)
    start = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
    deadline = datetime(2026, 10, 10, 12, 25, tzinfo=timezone.utc)
    assert defer_part(25, deadline, start) is False
    assert defer_part(26, deadline, start) is True
    assert defer_part(55, deadline, start) is True
    assert defer_part(55, None, start) is False
    assert resolve_deadline("2026-10-10T18:00:00Z") == datetime(2026, 10, 10, 18, 0, tzinfo=timezone.utc)
    assert resolve_deadline(None) is None
    with pytest.raises(UsageError):
        resolve_deadline("tomorrow")
    assert DEFERRED == "deferred to next rental"


def test_shard_plan_and_tiny_files(tmp_path: Path) -> None:
    assert plan_shards([("a", 2), ("b", 2), ("c", 2)], 5) == [["a", "b"], ["c"]]
    assert plan_shards([("a", 9)], 5) == [["a"]]
    source = tmp_path / "src"
    source.mkdir()
    save_file(
        {"layers.0.attention.to_q.weight": torch.randn(4, 4), "norm.weight": torch.randn(4)},
        source / "a.safetensors",
    )
    shards = write_bf16_shards(source, tmp_path / "bf16", "zimage-turbo-transformer-bf16", max_bytes=10_000)
    assert len(shards) == 1
    assert shards[0].name == "zimage-turbo-transformer-bf16-00001-of-00001.safetensors"
    fp8 = tmp_path / "qwen3-4b-text-encoder-fp8.safetensors"
    write_fp8_file(source, fp8, "diffusion")
    assert read_format(fp8) == FORMAT


def test_synthetic_scenes_are_tensors() -> None:
    scene = render_scene(0, 32)
    assert scene.shape == (3, 32, 32)
    edges = canny_map(scene)
    assert float(edges.max()) > 0
    depth = depth_map(1, 32)
    assert float(depth.max()) > float(depth.min())
    pose = pose_map(2, 48)
    assert float(pose.sum()) > 0
    sheet = contact_sheet([scene, edges], 2)
    assert sheet.shape[2] == 64


def test_canvas_card_front_matter() -> None:
    text = render_canvas_card(
        ROOT,
        CardContext(
            title="Z-Image Turbo voxprint tiers",
            base_models=("Tongyi-MAI/Z-Image-Turbo", "Tongyi-MAI/Z-Image"),
            base_lines=("Quantized from: `Tongyi-MAI/Z-Image-Turbo`",),
            pipeline_tag="text-to-image",
            tags=("voxprint", "fp8", "z-image"),
            language=("en",),
            quant_lines=("turbo transformer: publish fp8",),
            eval_lines=("turbo transformer: LPIPS 0.02, CLIP drop 0.004, decision fp8",),
            credits=("Tongyi-MAI / Alibaba Z-Image",),
            stamped="2026-10-10",
        ),
    )
    card = ModelCard(text)
    data = card.data.to_dict()
    assert card.data.license == "apache-2.0"
    assert list(card.data.base_model) == ["Tongyi-MAI/Z-Image-Turbo", "Tongyi-MAI/Z-Image"]
    assert data["base_model_relation"] == "quantized"
    assert card.data.pipeline_tag == "text-to-image"
    assert list(card.data.tags) == ["voxprint", "fp8", "z-image"]
    assert "This repository is private" not in text
    assert "finetune" not in text.split("---", 2)[1]
    assert "## Made for Voxprint AI Media Suite" in text
    assert "https://github.com/Mitroshenkov87/voxprint-canvas" in text
    assert "dtype conversion" in text.lower() or "Dtype conversion" in text


def test_llm_lock_and_canvas_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_DEADLINE", raising=False)
    (tmp_path / "bootstrap.lock").write_text(f"{os.getpid()}\n")
    assert llm_job_running(tmp_path, processes=lambda: []) is True
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "bootstrap.lock").write_text("99999999\n")
    assert llm_job_running(stale, processes=lambda: []) is False
    assert llm_job_running(stale, processes=lambda: ["python -m voxprint_forge run-all"]) is True
    assert llm_job_running(stale, processes=lambda: ["bash scripts/bootstrap.sh"]) is True
    assert (
        llm_job_running(stale, processes=lambda: ["tmux new-session -s forge bash scripts/bootstrap.sh"])
        is False
    )
    from voxprint_forge.cli import main
    from voxprint_forge.diffusion.cli import main as canvas_main

    code = main(
        [
            "--root",
            str(ROOT),
            "--forge-out",
            str(tmp_path),
            "canvas",
            "run",
            str(ROOT / "plans" / "canvas-zimage-tiers-v1.toml"),
        ]
    )
    assert code == 2
    code = canvas_main(
        [
            "run",
            str(ROOT / "plans" / "canvas-zimage-tiers-v1.toml"),
            "--forge-out",
            str(tmp_path),
            "--root",
            str(ROOT),
            "--json",
        ]
    )
    assert code == 2
    assert "diffusers" not in sys.modules
    assert "voxprint_forge.diffusion.job" not in sys.modules


def test_overlay_and_scripts_stay_off_the_running_llm_job() -> None:
    bundle = (ROOT / "scripts" / "make_vm_bundle.sh").read_text(encoding="utf-8")
    begin = bundle.split("# CANVAS_OVERLAY_PATHS begin", 1)[1].split("# CANVAS_OVERLAY_PATHS end", 1)[0]
    prefixes = ("scripts/", "src/", "plans/", "templates/", "docs/")
    paths = [line.strip() for line in begin.splitlines() if line.strip().startswith(prefixes)]
    assert paths == [
        "scripts/canvas_job.sh",
        "scripts/canvas_job_d.sh",
        "scripts/canvas_upload.sh",
        "src/voxprint_forge/diffusion",
        "plans/canvas-zimage-tiers-v1.toml",
        "plans/canvas-qwen-image-2512-v1.toml",
        "templates/canvas_model_card.md.jinja",
        "docs/CANVAS.md",
    ]
    forbidden = (
        "scripts/bootstrap.sh",
        "src/voxprint_forge/cli.py",
        "src/voxprint_forge/pipeline.py",
        "src/voxprint_forge/upload.py",
        "src/voxprint_forge/progress.py",
        "templates/model_card.md.jinja",
        "plans/plan.toml",
        "pyproject.toml",
    )
    for name in forbidden:
        assert name not in paths
    job = (ROOT / "scripts" / "canvas_job.sh").read_text(encoding="utf-8")
    assert "set +x" in job
    assert "set -x" not in job
    assert "apt-get" not in job
    assert "ninja-build" not in job
    assert "/usr/local/cuda/bin" in job
    assert "LD_LIBRARY_PATH" in job
    assert "torch.cuda.is_available()" in job
    assert "--force" in job
    assert "pgrep" in job
    assert "bootstrap.lock" in job
    assert "FORGE_DEADLINE" in job
    assert "--deadline" in job
    assert "diffusers==0.41.0" in job
    assert "torchao==0.18.0" in job
    assert "lpips==0.1.4" in job
    assert "pillow-avif-plugin==1.6.0" in job
    assert "torchvision==0.29.1" in job
    assert "--no-deps" in job
    assert '"$VENV/bin/python" -m voxprint_forge.diffusion' in job
    assert "estimate part base 55 min" in job
    assert "estimate part controlnet 25 min" in job
    assert "estimate part edit 55 min" in job
    assert "huggingface-cli login" not in job
    assert "hf auth login" not in job
    assert "GH_TOKEN" not in job
    assert "github.com" not in job
    assert "cleanup_secrets" in job
    assert 'export HF_USER="${HF_USER:-Mitroshenkov87}"' in job
    assert "non-interactive" in job
    assert "preflight nvcc ok" in job
    assert "cmake and ninja are not required" in job
    docs = (ROOT / "docs" / "CANVAS.md").read_text(encoding="utf-8")
    assert "55" in docs and "25" in docs and "135" in docs
    assert "deferred to next rental" in docs
    assert "FORGE_DEADLINE=2026-10-10T18:00:00Z" in docs
    assert "tmux new -s canvas 'bash scripts/canvas_job.sh'" in docs
    assert "Mitroshenkov87" in docs
    assert "not a terminal" in docs
    cli = (ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
    assert "forge canvas run plans/canvas-zimage-tiers-v1.toml" in cli
    assert "deferred to next rental" in cli
