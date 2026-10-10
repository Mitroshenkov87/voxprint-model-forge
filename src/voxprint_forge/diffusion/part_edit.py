"""Part C: Qwen-Image-Edit-2511 FP8. Section 8."""

from __future__ import annotations

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
from voxprint_forge.diffusion.names import (
    EDIT_BF16_PREFIX,
    EDIT_ENCODER_FP8,
    EDIT_FP8_PREFIX,
    EDIT_TRANSFORMER_CONFIG,
    I2I_GRID,
    I2I_METRICS,
)
from voxprint_forge.diffusion.publish import publish_output
from voxprint_forge.diffusion.sources import fetch_sources
from voxprint_forge.diffusion.weights import (
    copy_config,
    link_or_copy,
    link_tree,
    write_bf16_shards,
    write_fp8_file,
    write_fp8_shards,
)
from voxprint_forge.errors import StageError


def _fetch(ctx: Ctx) -> dict[str, object]:
    ctx.upstream.mkdir(parents=True, exist_ok=True)
    fetch_sources(ctx.part.sources, ctx.upstream, ctx.token)
    return {"fetched": [source.id for source in ctx.part.sources]}


def _convert(ctx: Ctx) -> dict[str, object]:
    edit = source_dir(ctx, "edit")
    output = ctx.output
    output.mkdir(parents=True, exist_ok=True)
    write_notice(output)
    link_or_copy(ctx.upstream / "upstream-sha256.json", output / "upstream-sha256.json")
    shards = write_fp8_shards(edit / "transformer", output, EDIT_FP8_PREFIX, "edit_transformer")
    copy_config(edit / "transformer", output / EDIT_TRANSFORMER_CONFIG)
    write_fp8_file(edit / "text_encoder", output / EDIT_ENCODER_FP8, "edit_encoder")
    for folder in ("vae", "processor", "tokenizer", "text_encoder"):
        src = edit / folder
        if src.is_dir():
            link_tree(src, output / folder)
    return {"fp8_shards": [path.name for path in shards]}


def _shard_paths(output: Path, prefix: str) -> list[Path]:
    paths = sorted(output.glob(f"{prefix}-*.safetensors"))
    if not paths:
        raise StageError(f"no shards named {prefix}", code="shards")
    return paths


def _quality(ctx: Ctx) -> dict[str, object]:
    # diffusers is optional and must not load until this stage runs.
    from voxprint_forge.diffusion.generate import evaluate_edit

    fp8_paths = _shard_paths(ctx.output, EDIT_FP8_PREFIX)
    report = evaluate_edit(
        edit_dir=source_dir(ctx, "edit"),
        transformer_fp8=fp8_paths,
        encoder_fp8=ctx.output / EDIT_ENCODER_FP8,
        grid_path=ctx.output / I2I_GRID,
        progress=lambda stage, done, total: progress(
            ctx, stage, int(100 * done / max(total, 1)), ctx.part.estimate_min
        ),
    )
    components = report["components"]
    if not isinstance(components, dict):
        raise StageError("edit quality report has no components", code="quality")
    transformer = components.get("edit-transformer", {})
    encoder = components.get("qwen2.5-vl-encoder", {})
    if not isinstance(transformer, dict) or transformer.get("decision") != "fp8":
        write_bf16_shards(source_dir(ctx, "edit") / "transformer", ctx.output, EDIT_BF16_PREFIX)
        for path in fp8_paths:
            path.unlink(missing_ok=True)
    if not isinstance(encoder, dict) or encoder.get("decision") != "fp8":
        fp8 = ctx.output / EDIT_ENCODER_FP8
        if fp8.is_file():
            fp8.unlink()
    write_json(ctx.output / I2I_METRICS, report)
    return {"decisions": components}


def _card(ctx: Ctx, quality: dict[str, object]) -> dict[str, object]:
    decisions = quality.get("decisions")
    if not isinstance(decisions, dict):
        raise StageError("edit card is missing quality decisions", code="quality")
    lines = []
    evals = []
    for name, body in decisions.items():
        if not isinstance(body, dict):
            continue
        lines.append(f"{name}: publish {body.get('decision')}")
        evals.append(
            f"{name}: LPIPS {body.get('lpips_mean')}, CLIP-I {body.get('clip_i')}, decision {body.get('decision')}"
        )
    source = ctx.part.sources[0]
    card = CardContext(
        title="Qwen-Image-Edit-2511 voxprint FP8",
        base_models=(source.repo,),
        base_lines=(
            f"Quantized from: `{source.repo}`",
            f"Revision: `{source.revision}`",
            "Text encoder: Qwen2.5-VL-7B",
            "Licence of the weights: Apache-2.0",
            f"Card date: {today_stamp()}",
        ),
        pipeline_tag="image-to-image",
        tags=("voxprint", "fp8", "qwen-image"),
        language=("en",),
        quant_lines=tuple(lines) + ("img_in, txt_in, norms, adaLN, and proj_out stay BF16",),
        eval_lines=tuple(evals) + ("24 edit pairs x 2 seeds, 40 steps, true CFG 4.0",),
        credits=(
            "Qwen / Alibaba Qwen-Image-Edit-2511",
            "Qwen2.5-VL-7B text encoder",
            "diffusers QwenImageEditPlusPipeline",
            "torchao",
        ),
        stamped=today_stamp(),
    )
    (ctx.output / "README.md").write_text(render_canvas_card(ctx.repo, card), encoding="utf-8")
    return {"card": "README.md"}


def _upload(ctx: Ctx, quality: dict[str, object]) -> dict[str, object]:
    decisions = quality.get("decisions")
    if not isinstance(decisions, dict):
        raise StageError("edit upload is missing quality decisions", code="quality")
    source = ctx.part.sources[0]
    transformer = decisions.get("edit-transformer", {})
    encoder = decisions.get("qwen2.5-vl-encoder", {})
    use_fp8 = isinstance(transformer, dict) and transformer.get("decision") == "fp8"
    prefix = EDIT_FP8_PREFIX if use_fp8 else EDIT_BF16_PREFIX
    names = [path.name for path in _shard_paths(ctx.output, prefix)]
    tier_name = "24gb" if use_fp8 else "24gb-bf16"
    tiers: dict[str, tuple[str, int | None, str, str]] = {
        name: (tier_name, 24, source.repo, source.revision) for name in names
    }
    variation_names = list(names)
    if isinstance(encoder, dict) and encoder.get("decision") == "fp8" and (ctx.output / EDIT_ENCODER_FP8).is_file():
        tiers[EDIT_ENCODER_FP8] = ("24gb", 24, source.repo, source.revision)
        variation_names.append(EDIT_ENCODER_FP8)
    variation = "24gb-fp8" if use_fp8 else "24gb-bf16"
    records = publish_output(
        output=ctx.output,
        hf_repo=ctx.part.hf_repo,
        kaggle_slug=ctx.part.kaggle_slug,
        title="Voxprint Qwen-Image-Edit-2511",
        description="Private Voxprint dtype conversion of Qwen-Image-Edit-2511. Apache-2.0.",
        tiers=tiers,
        variations={variation: variation_names},
        repo=source.repo,
        revision=source.revision,
        staging=ctx.upstream / "kaggle-staging",
    )
    delete_intermediates(ctx.upstream)
    return {"files": records}


def run_edit(ctx: Ctx) -> dict[str, object]:
    check_disk(ctx)
    progress(ctx, "start", 0, ctx.part.estimate_min)
    run_stage(ctx, "fetch", lambda: _fetch(ctx))
    run_stage(ctx, "convert", lambda: _convert(ctx))
    quality = run_stage(ctx, "quality", lambda: _quality(ctx))
    run_stage(ctx, "card", lambda: _card(ctx, quality))
    uploaded = run_stage(ctx, "upload", lambda: _upload(ctx, quality))
    progress(ctx, "done", 100, 0)
    return {
        "id": "edit",
        "status": "uploaded",
        "estimate_min": ctx.part.estimate_min,
        "hf_repo": ctx.part.hf_repo,
        "kaggle_slug": ctx.part.kaggle_slug,
        "decisions": quality.get("decisions"),
        "files": uploaded.get("files", []),
    }
