"""Part A: Z-Image-Turbo BF16/FP8 tiers. Section 6 of the canvas decision."""

from __future__ import annotations

import shutil
from pathlib import Path

from voxprint_forge.diffusion.cardrender import CardContext, render_canvas_card, today_stamp
from voxprint_forge.diffusion.common import (
    Ctx,
    check_disk,
    delete_intermediates,
    progress,
    run_stage,
    source_dir,
    write_json,
    write_notice,
)
from voxprint_forge.diffusion.generate import evaluate_base
from voxprint_forge.diffusion.names import (
    BASE_GRID,
    BASE_METRICS,
    QWEN3_FP8_NAME,
    TURBO_BF16_PREFIX,
    TURBO_FP8_NAME,
    ZIMAGE_BF16_DIR,
    ZIMAGE_FP8_NAME,
)
from voxprint_forge.diffusion.publish import publish_output
from voxprint_forge.diffusion.sources import fetch_sources, same_trees
from voxprint_forge.diffusion.weights import (
    copy_config,
    has_fp32,
    link_or_copy,
    link_tree,
    passthrough_or_bf16,
    safetensor_files,
    write_bf16_shards,
    write_fp8_file,
)
from voxprint_forge.errors import StageError


def _export_dir(src: Path, dest: Path) -> str:
    if dest.exists():
        shutil.rmtree(dest)
    if not has_fp32(src):
        link_tree(src, dest)
        return "passthrough"
    dest.mkdir(parents=True)
    for path in sorted(item for item in src.rglob("*") if item.is_file() and item.suffix != ".safetensors"):
        link_or_copy(path, dest / path.relative_to(src))
    for path in safetensor_files(src):
        passthrough_or_bf16(path, dest / path.relative_to(src))
    return "cast"


def _fetch(ctx: Ctx) -> dict[str, object]:
    ctx.upstream.mkdir(parents=True, exist_ok=True)
    fetch_sources(ctx.part.sources, ctx.upstream, ctx.token)
    return {"fetched": [source.id for source in ctx.part.sources]}


def _convert(ctx: Ctx) -> dict[str, object]:
    turbo = source_dir(ctx, "turbo")
    base = source_dir(ctx, "base")
    output = ctx.output
    output.mkdir(parents=True, exist_ok=True)
    write_notice(output)
    link_or_copy(ctx.upstream / "upstream-sha256.json", output / "upstream-sha256.json")
    shared_encoder = same_trees(turbo / "text_encoder", base / "text_encoder")
    shared_vae = same_trees(turbo / "vae", base / "vae")
    write_json(output / "shared-components.json", {"text_encoder": shared_encoder, "vae": shared_vae})
    write_bf16_shards(turbo / "transformer", output, TURBO_BF16_PREFIX)
    copy_config(turbo / "transformer", output / "zimage-turbo-transformer-bf16.config.json")
    write_fp8_file(turbo / "transformer", output / TURBO_FP8_NAME, "diffusion")
    mode = _export_dir(base / "transformer", output / ZIMAGE_BF16_DIR)
    write_fp8_file(base / "transformer", output / ZIMAGE_FP8_NAME, "diffusion")
    encoder_mode = _export_dir(turbo / "text_encoder", output / "text_encoder")
    write_fp8_file(turbo / "text_encoder", output / QWEN3_FP8_NAME, "diffusion")
    link_tree(turbo / "vae", output / "vae")
    link_tree(turbo / "tokenizer", output / "tokenizer")
    if not shared_encoder:
        link_tree(base / "text_encoder", output / "zimage-text-encoder")
    if not shared_vae:
        link_tree(base / "vae", output / "zimage-vae")
    return {
        "zimage_transformer": mode,
        "text_encoder": encoder_mode,
        "shared_text_encoder": shared_encoder,
        "shared_vae": shared_vae,
    }


def _drop(path: Path) -> None:
    if path.is_file():
        path.unlink()


def _quality(ctx: Ctx) -> dict[str, object]:
    report = evaluate_base(
        turbo_dir=source_dir(ctx, "turbo"),
        base_dir=source_dir(ctx, "base"),
        turbo_fp8=ctx.output / TURBO_FP8_NAME,
        base_fp8=ctx.output / ZIMAGE_FP8_NAME,
        encoder_fp8=ctx.output / QWEN3_FP8_NAME,
        grid_path=ctx.output / BASE_GRID,
        progress=lambda stage, done, total: progress(
            ctx, stage, int(100 * done / max(total, 1)), ctx.part.estimate_min
        ),
    )
    components = report["components"]
    if not isinstance(components, dict):
        raise StageError("base quality report has no components", code="quality")
    if components["turbo-transformer"]["decision"] != "fp8":
        _drop(ctx.output / TURBO_FP8_NAME)
    if components["zimage-transformer"]["decision"] != "fp8":
        _drop(ctx.output / ZIMAGE_FP8_NAME)
    if components["qwen3-4b-encoder"]["decision"] != "fp8":
        _drop(ctx.output / QWEN3_FP8_NAME)
    write_json(ctx.output / BASE_METRICS, report)
    return {"decisions": components}


def _lines(decisions: dict[str, object]) -> tuple[list[str], list[str]]:
    quant: list[str] = []
    evals: list[str] = []
    for name, body in decisions.items():
        if not isinstance(body, dict):
            continue
        decision = str(body.get("decision", ""))
        quant.append(f"{name}: publish {decision}")
        lpips_mean = body.get("lpips_mean")
        clip_drop = body.get("clip_drop")
        evals.append(f"{name}: LPIPS {lpips_mean}, CLIP drop {clip_drop}, decision {decision}")
    return quant, evals


def _card(ctx: Ctx, quality: dict[str, object]) -> dict[str, object]:
    decisions = quality.get("decisions")
    if not isinstance(decisions, dict):
        raise StageError("base card is missing quality decisions", code="quality")
    quant, evals = _lines(decisions)
    turbo = ctx.part.sources[0]
    base = ctx.part.sources[1]
    card = CardContext(
        title="Z-Image Turbo voxprint tiers",
        base_models=(turbo.repo, base.repo),
        base_lines=(
            f"Quantized from: `{turbo.repo}`",
            f"Turbo revision: `{turbo.revision}`",
            f"Creative revision: `{base.repo}` @ `{base.revision}`",
            "Licence of the weights: Apache-2.0",
            f"Card date: {today_stamp()}",
        ),
        pipeline_tag="text-to-image",
        tags=("voxprint", "fp8", "z-image"),
        language=("en",),
        quant_lines=tuple(quant),
        eval_lines=tuple(evals),
        credits=(
            "Tongyi-MAI / Alibaba Z-Image and Z-Image-Turbo",
            "Qwen3-4B text encoder",
            "diffusers ZImagePipeline",
            "torchao",
        ),
        stamped=today_stamp(),
    )
    (ctx.output / "README.md").write_text(render_canvas_card(ctx.repo, card), encoding="utf-8")
    return {"card": "README.md"}


def _source_for(name: str, ctx: Ctx) -> tuple[str, str]:
    base = ctx.part.sources[1]
    turbo = ctx.part.sources[0]
    if name.startswith("zimage-transformer") or name.startswith("zimage-text-encoder") or name.startswith("zimage-vae"):
        return base.repo, base.revision
    return turbo.repo, turbo.revision


def _tiers(ctx: Ctx) -> dict[str, tuple[str, int | None, str, str]]:
    tiers: dict[str, tuple[str, int | None, str, str]] = {}
    for path in sorted(ctx.output.glob(f"{TURBO_BF16_PREFIX}-*.safetensors")):
        tiers[path.name] = ("24gb", 24, *_source_for(path.name, ctx))
    if (ctx.output / TURBO_FP8_NAME).is_file():
        tiers[TURBO_FP8_NAME] = ("16gb", 16, *_source_for(TURBO_FP8_NAME, ctx))
    if (ctx.output / QWEN3_FP8_NAME).is_file():
        tiers[QWEN3_FP8_NAME] = ("16gb", 16, *_source_for(QWEN3_FP8_NAME, ctx))
    if (ctx.output / ZIMAGE_FP8_NAME).is_file():
        tiers[ZIMAGE_FP8_NAME] = ("creative", 24, *_source_for(ZIMAGE_FP8_NAME, ctx))
    bf16_dir = ctx.output / ZIMAGE_BF16_DIR
    if bf16_dir.is_dir():
        for path in bf16_dir.rglob("*"):
            if path.is_file():
                name = path.relative_to(ctx.output).as_posix()
                tiers[name] = ("creative", 24, *_source_for(name, ctx))
    return tiers


def _variations(ctx: Ctx, decisions: dict[str, object]) -> dict[str, list[str]]:
    shards = sorted(path.name for path in ctx.output.glob(f"{TURBO_BF16_PREFIX}-*.safetensors"))
    variations: dict[str, list[str]] = {"24gb-bf16": shards}
    turbo_decision = decisions.get("turbo-transformer", {})
    encoder_decision = decisions.get("qwen3-4b-encoder", {})
    creative_decision = decisions.get("zimage-transformer", {})
    if isinstance(turbo_decision, dict) and turbo_decision.get("decision") == "fp8":
        names = [TURBO_FP8_NAME]
        if isinstance(encoder_decision, dict) and encoder_decision.get("decision") == "fp8":
            names.append(QWEN3_FP8_NAME)
        variations["16gb-fp8"] = names
    else:
        variations["16gb-bf16"] = shards
    if isinstance(creative_decision, dict) and creative_decision.get("decision") == "fp8":
        variations["creative"] = [ZIMAGE_FP8_NAME]
    else:
        variations["creative"] = [
            path.relative_to(ctx.output).as_posix()
            for path in sorted((ctx.output / ZIMAGE_BF16_DIR).rglob("*.safetensors"))
        ]
    return variations


def _upload(ctx: Ctx, quality: dict[str, object]) -> dict[str, object]:
    decisions = quality.get("decisions")
    if not isinstance(decisions, dict):
        raise StageError("base upload is missing quality decisions", code="quality")
    turbo = ctx.part.sources[0]
    records = publish_output(
        output=ctx.output,
        hf_repo=ctx.part.hf_repo,
        kaggle_slug=ctx.part.kaggle_slug,
        title="Voxprint Z-Image Turbo tiers",
        description="Private Voxprint dtype conversion of Z-Image-Turbo and Z-Image. Apache-2.0.",
        tiers=_tiers(ctx),
        variations=_variations(ctx, decisions),
        repo=turbo.repo,
        revision=turbo.revision,
        staging=ctx.upstream / "kaggle-staging",
    )
    delete_intermediates(ctx.upstream)
    return {"files": records}


def run_base(ctx: Ctx) -> dict[str, object]:
    check_disk(ctx)
    progress(ctx, "start", 0, ctx.part.estimate_min)
    run_stage(ctx, "fetch", lambda: _fetch(ctx))
    run_stage(ctx, "convert", lambda: _convert(ctx))
    quality = run_stage(ctx, "quality", lambda: _quality(ctx))
    run_stage(ctx, "card", lambda: _card(ctx, quality))
    uploaded = run_stage(ctx, "upload", lambda: _upload(ctx, quality))
    progress(ctx, "done", 100, 0)
    return {
        "id": "base",
        "status": "uploaded",
        "estimate_min": ctx.part.estimate_min,
        "hf_repo": ctx.part.hf_repo,
        "kaggle_slug": ctx.part.kaggle_slug,
        "decisions": quality.get("decisions"),
        "files": uploaded.get("files", []),
    }
