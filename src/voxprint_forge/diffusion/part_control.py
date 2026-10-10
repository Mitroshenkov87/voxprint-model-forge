"""Part B: Z-Image-Turbo ControlNet Union 2.1. Section 8."""

from __future__ import annotations

from voxprint_forge.diffusion.cardrender import CardContext, render_canvas_card, today_stamp
from voxprint_forge.diffusion.common import (
    Ctx,
    check_disk,
    finish_part,
    progress,
    run_stage,
    shared_sources,
    source_dir,
    write_json,
    write_notice,
)
from voxprint_forge.diffusion.defer import drop_marker, write_local_files
from voxprint_forge.diffusion.names import (
    CONTROL_FULL_BF16,
    CONTROL_FULL_UPSTREAM,
    CONTROL_LITE_BF16,
    CONTROL_LITE_FP8,
    CONTROL_LITE_UPSTREAM,
    I2I_GRID,
    I2I_METRICS,
    INPAINT_GRID,
)
from voxprint_forge.diffusion.publish import publish_output
from voxprint_forge.diffusion.sources import fetch_sources
from voxprint_forge.diffusion.weights import link_or_copy, passthrough_or_bf16, write_fp8_file
from voxprint_forge.errors import StageError


def _fetch(ctx: Ctx) -> dict[str, object]:
    ctx.upstream.mkdir(parents=True, exist_ok=True)
    dest = shared_sources(ctx)
    dest.mkdir(parents=True, exist_ok=True)
    fetch_sources(ctx.part.sources, dest, ctx.token)
    return {"fetched": [source.id for source in ctx.part.sources]}


def _convert(ctx: Ctx) -> dict[str, object]:
    control = source_dir(ctx, "controlnet")
    output = ctx.output
    output.mkdir(parents=True, exist_ok=True)
    write_notice(output)
    link_or_copy(shared_sources(ctx) / "upstream-sha256.json", output / "upstream-sha256.json")
    full_mode = passthrough_or_bf16(control / CONTROL_FULL_UPSTREAM, output / CONTROL_FULL_BF16)
    lite_bf16 = ctx.upstream / "lite-bf16.safetensors"
    lite_mode = passthrough_or_bf16(control / CONTROL_LITE_UPSTREAM, lite_bf16)
    write_fp8_file(lite_bf16, output / CONTROL_LITE_FP8, "controlnet")
    return {"full": full_mode, "lite": lite_mode}


def _quality(ctx: Ctx) -> dict[str, object]:
    # diffusers is optional and must not load until this stage runs.
    from voxprint_forge.diffusion.generate import evaluate_control

    report = evaluate_control(
        turbo_dir=source_dir(ctx, "turbo"),
        lite_bf16=ctx.upstream / "lite-bf16.safetensors",
        lite_fp8=ctx.output / CONTROL_LITE_FP8,
        grid_path=ctx.output / I2I_GRID,
        inpaint_grid=ctx.output / INPAINT_GRID,
        work_dir=ctx.upstream / "previews",
        progress=lambda stage, done, total: progress(
            ctx, stage, int(100 * done / max(total, 1)), ctx.part.estimate_min
        ),
    )
    components = report["components"]
    if not isinstance(components, dict):
        raise StageError("control quality report has no components", code="quality")
    decision = components.get("controlnet-lite", {})
    if not isinstance(decision, dict) or decision.get("decision") != "fp8":
        link_or_copy(ctx.upstream / "lite-bf16.safetensors", ctx.output / CONTROL_LITE_BF16)
        fp8 = ctx.output / CONTROL_LITE_FP8
        if fp8.is_file():
            fp8.unlink()
    report["full"] = {"decision": "bf16", "reason": "full file stays BF16"}
    write_json(ctx.output / I2I_METRICS, report)
    return {"decisions": components}


def _card(ctx: Ctx, quality: dict[str, object]) -> dict[str, object]:
    decisions = quality.get("decisions")
    if not isinstance(decisions, dict):
        raise StageError("control card is missing quality decisions", code="quality")
    body = decisions.get("controlnet-lite", {})
    decision = body.get("decision") if isinstance(body, dict) else "bf16"
    lpips_mean = body.get("lpips_mean") if isinstance(body, dict) else None
    control = next(source for source in ctx.part.sources if source.id == "controlnet")
    turbo = next(source for source in ctx.part.sources if source.id == "turbo")
    card = CardContext(
        title="Z-Image Turbo ControlNet Union 2.1",
        base_models=(control.repo, turbo.repo),
        base_lines=(
            f"Quantized from: `{control.repo}`",
            f"ControlNet revision: `{control.revision}`",
            f"Turbo revision: `{turbo.repo}` @ `{turbo.revision}`",
            "Licence of the weights: Apache-2.0",
            f"Card date: {today_stamp()}",
        ),
        pipeline_tag="image-to-image",
        tags=("voxprint", "fp8", "z-image", "controlnet"),
        language=("en",),
        quant_lines=(
            "Union 2.1 full: BF16 (24 GB tier)",
            f"Union 2.1 lite: publish {decision} (16 GB tier when FP8 passes)",
            "Norms, embeddings, and zero-convs stay BF16",
        ),
        eval_lines=(
            f"lite FP8 vs BF16 mean LPIPS {lpips_mean} (gate 0.10), decision {decision}",
            "6 hard control cases and 4 inpaint cases, one seed, stop when the LPIPS interval decides",
            "Control maps are generated in the job from synthetic scenes",
        ),
        credits=(
            "alibaba-pai Z-Image-Turbo-Fun-Controlnet-Union-2.1",
            "Tongyi-MAI / Alibaba Z-Image-Turbo",
            "diffusers ZImageControlNetPipeline",
        ),
        stamped=today_stamp(),
    )
    (ctx.output / "README.md").write_text(render_canvas_card(ctx.repo, card), encoding="utf-8")
    return {"card": "README.md"}


def _upload(ctx: Ctx, quality: dict[str, object], *, remote: bool = True) -> dict[str, object]:
    decisions = quality.get("decisions")
    if not isinstance(decisions, dict):
        raise StageError("control upload is missing quality decisions", code="quality")
    control = next(source for source in ctx.part.sources if source.id == "controlnet")
    lite = decisions.get("controlnet-lite", {})
    lite_name = CONTROL_LITE_FP8 if isinstance(lite, dict) and lite.get("decision") == "fp8" else CONTROL_LITE_BF16
    lite_tier = "16gb" if lite_name == CONTROL_LITE_FP8 else "16gb-bf16"
    tiers = {
        CONTROL_FULL_BF16: ("24gb", 24, control.repo, control.revision),
        lite_name: (lite_tier, 16, control.repo, control.revision),
    }
    variation = "16gb-fp8" if lite_name == CONTROL_LITE_FP8 else "16gb-bf16"
    if not remote:
        rows = write_local_files(ctx.output, tiers, control.repo, control.revision)
        finish_part(ctx)
        return {"files": rows}
    records = publish_output(
        output=ctx.output,
        hf_repo=ctx.part.hf_repo,
        kaggle_slug=ctx.part.kaggle_slug,
        title="Voxprint Z-Image Turbo ControlNet Union 2.1",
        description="Private Voxprint dtype conversion of ControlNet Union 2.1. Apache-2.0.",
        tiers=tiers,
        variations={"24gb-bf16": [CONTROL_FULL_BF16], variation: [lite_name]},
        repo=control.repo,
        revision=control.revision,
        staging=ctx.upstream / "kaggle-staging",
    )
    finish_part(ctx)
    return {"files": records}


def run_control(ctx: Ctx) -> dict[str, object]:
    """Prepare, score, and publish the ControlNet tier."""
    check_disk(ctx)
    progress(ctx, "start", 0, ctx.part.estimate_min)
    run_stage(ctx, "fetch", lambda: _fetch(ctx))
    run_stage(ctx, "convert", lambda: _convert(ctx))
    quality = run_stage(ctx, "quality", lambda: _quality(ctx))
    run_stage(ctx, "card", lambda: _card(ctx, quality))
    if ctx.defer_upload:
        held = run_stage(ctx, "upload-deferred", lambda: _upload(ctx, quality, remote=False))
        status = "upload-deferred"
        files = held.get("files", [])
    else:
        uploaded = run_stage(ctx, "upload", lambda: _upload(ctx, quality))
        drop_marker(ctx.marker("upload-deferred"))
        status = "uploaded"
        files = uploaded.get("files", [])
    progress(ctx, "done", 100, 0)
    return {
        "id": "controlnet",
        "status": status,
        "estimate_min": ctx.part.estimate_min,
        "hf_repo": ctx.part.hf_repo,
        "kaggle_slug": ctx.part.kaggle_slug,
        "decisions": quality.get("decisions"),
        "files": files,
    }
