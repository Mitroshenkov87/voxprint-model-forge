"""Part D runner. The three-part canvas job is not edited and is not replaced."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Never

from voxprint_forge.diffusion.cardrender import CardContext, render_canvas_card, today_stamp
from voxprint_forge.diffusion.common import Ctx, delete_intermediates, write_json, write_notice
from voxprint_forge.diffusion.deadline import DEFERRED, defer_part
from voxprint_forge.diffusion.defer import drop_marker, write_local_files
from voxprint_forge.diffusion.part_base import run_base
from voxprint_forge.diffusion.part_control import run_control
from voxprint_forge.diffusion.part_edit import run_edit
from voxprint_forge.diffusion.planload import PartId, load_canvas_plan
from voxprint_forge.diffusion.publish import publish_output
from voxprint_forge.diffusion.qwen_common import QwenCtx, check_disk, on_qwen_mirror, progress, run_stage
from voxprint_forge.diffusion.qwen_paths import (
    EncoderStatus,
    assert_no_weight_files,
    classify_encoder,
    edit_encoder_fp8,
    encoder_hashes,
    format_stable_lines,
    link_stable_file,
    refresh_fp8_dir,
    write_encoder_pending,
    write_int4_marker,
)
from voxprint_forge.diffusion.qwen_plan import QwenPlan
from voxprint_forge.diffusion.qwen_select import QwenPart, parse_parts, refuse_retired
from voxprint_forge.diffusion.qwen_spec import (
    ENCODER_REUSE,
    INT4_MARKER_NAME,
    INT4_MODE,
    QWEN_BF16_PREFIX,
    QWEN_ENCODER_FP8,
    QWEN_FP8_PREFIX,
    QWEN_GRID,
    QWEN_METRICS,
    QWEN_SIZE,
    QWEN_STEPS,
    QWEN_TRANSFORMER_CONFIG,
    QWEN_TRUE_CFG,
    ZIMAGE_PLAN,
)
from voxprint_forge.diffusion.retain import release_consumed
from voxprint_forge.diffusion.sources import fetch_sources
from voxprint_forge.diffusion.weights import (
    copy_config,
    link_or_copy,
    link_tree,
    write_bf16_shards,
    write_fp8_shards,
)
from voxprint_forge.errors import ForgeError, StageError, UsageError
from voxprint_forge.layout import Layout, read_marker, write_marker
from voxprint_forge.outcome import state_line, summarise
from voxprint_forge.scrub import log
from voxprint_forge.statusfile import current_stage, update_item

_DONE = frozenset({"reuse", "convert"})


def _unexpected_part(part: Never) -> Never:
    raise UsageError(f"unhandled part {part}", code="parts")


def _unexpected_status(status: Never) -> Never:
    raise StageError(f"unhandled encoder status {status}", code="encoder")


def _unexpected_canvas(part_id: Never) -> Never:
    raise UsageError(f"unhandled canvas part {part_id}", code="parts")


def _fetch(ctx: QwenCtx) -> dict[str, object]:
    ctx.upstream_root.mkdir(parents=True, exist_ok=True)
    fetch_sources((ctx.plan.source,), ctx.upstream_root, ctx.token)
    return {"fetched": [ctx.plan.source.id]}


def _convert(ctx: QwenCtx) -> dict[str, object]:
    source = ctx.source_dir
    output = ctx.output
    output.mkdir(parents=True, exist_ok=True)
    write_notice(output)
    link_or_copy(ctx.upstream_root / "upstream-sha256.json", output / "upstream-sha256.json")
    shards = write_fp8_shards(source / "transformer", output, QWEN_FP8_PREFIX, "edit_transformer")
    copy_config(source / "transformer", output / QWEN_TRANSFORMER_CONFIG)
    for folder in ("vae", "tokenizer", "scheduler", "processor"):
        src = source / folder
        if src.is_dir():
            link_tree(src, output / folder)
    return {"fp8_shards": [path.name for path in shards]}


def _shard_paths(output: Path, prefix: str) -> list[Path]:
    paths = sorted(output.glob(f"{prefix}-*.safetensors"))
    if not paths:
        raise StageError(f"no shards named {prefix}", code="shards")
    return paths


def _drop_published_encoder(output: Path) -> None:
    """A reused encoder is a link outside this directory, never a second weight file."""
    path = output / QWEN_ENCODER_FP8
    if path.is_symlink() or path.is_file():
        path.unlink()
    tree = output / "text_encoder"
    if tree.is_symlink() or tree.is_file():
        tree.unlink()
    elif tree.is_dir():
        shutil.rmtree(tree)


def _hash_map(value: object) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    found: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            return None
        found[key] = item
    return found


def _status_of(value: object) -> EncoderStatus | None:
    if value == "reuse" or value == "reuse-pending" or value == "convert":
        return value
    return None


def _current_hashes(ctx: QwenCtx, existing: dict[str, object] | None) -> dict[str, str]:
    text_encoder = ctx.source_dir / "text_encoder"
    if text_encoder.is_dir():
        return encoder_hashes(text_encoder)
    stored = _hash_map(None if existing is None else existing.get("hashes"))
    if stored:
        return stored
    raise StageError(f"missing text encoder {text_encoder}", code="weights_missing")


def _apply_encoder(ctx: QwenCtx, status: EncoderStatus, edit_fp8: Path) -> None:
    """Link part C's encoder. A 2512 download is never turned into that file."""
    if status == "reuse":
        _drop_published_encoder(ctx.output)
        link_stable_file(ctx.forge_out, "encoder", edit_fp8.resolve())
        return
    if status == "reuse-pending" or status == "convert":
        # "convert" is an old marker. New runs classify only reuse or reuse-pending.
        _drop_published_encoder(ctx.output)
        write_encoder_pending(ctx.forge_out)
        return
    _unexpected_status(status)


def ensure_encoder(ctx: QwenCtx) -> dict[str, object]:
    """Link part C's FP8 when the pins match. Re-check a pending marker after part C."""
    existing = read_marker(ctx.marker("encoder"))
    status = _status_of(None if existing is None else existing.get("status"))
    if existing is not None and status in _DONE:
        return existing
    ctx.output.mkdir(parents=True, exist_ok=True)
    found = _current_hashes(ctx, existing)
    edit_fp8 = edit_encoder_fp8(
        ctx.forge_out,
        ctx.plan.edit_job_id,
        ctx.plan.edit_output_name,
        ctx.plan.edit_encoder_fp8,
    )
    decided = classify_encoder(found, ctx.plan.encoder_pins, edit_fp8)
    _apply_encoder(ctx, decided, edit_fp8)
    payload: dict[str, object] = {
        "stage": "qwen-encoder",
        "status": decided,
        "copied_weights": False,
        "edit_fp8": str(edit_fp8),
        "hashes": found,
    }
    write_marker(ctx.marker("encoder"), payload)
    write_json(ctx.output / ENCODER_REUSE, {"status": decided, "copied_weights": False, "edit_fp8": str(edit_fp8)})
    return payload


def _int4(ctx: QwenCtx) -> dict[str, object]:
    marker = write_int4_marker(ctx.forge_out)
    dest = ctx.output / "int4"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / INT4_MARKER_NAME).write_text(marker.read_text(encoding="utf-8"), encoding="utf-8")
    assert_no_weight_files(dest)
    assert_no_weight_files(marker.parent)
    return {"int4": INT4_MODE, "marker": INT4_MARKER_NAME}


def _quality(ctx: QwenCtx) -> dict[str, object]:
    from voxprint_forge.diffusion.qwen_generate import evaluate_qwen

    fp8_paths = _shard_paths(ctx.output, QWEN_FP8_PREFIX)
    report = evaluate_qwen(
        model_dir=ctx.source_dir,
        transformer_fp8=fp8_paths,
        grid_path=ctx.output / QWEN_GRID,
        progress=lambda stage, done, total: progress(
            ctx, stage, int(100 * done / max(total, 1)), float(ctx.plan.estimate_min)
        ),
    )
    components = report.get("components")
    if not isinstance(components, dict):
        raise StageError("qwen quality report has no components", code="quality")
    transformer = components.get("qwen-transformer")
    if not isinstance(transformer, dict) or transformer.get("decision") != "fp8":
        write_bf16_shards(ctx.source_dir / "transformer", ctx.output, QWEN_BF16_PREFIX)
        for path in fp8_paths:
            path.unlink(missing_ok=True)
    write_json(ctx.output / QWEN_METRICS, report)
    return {"decisions": components}


def _decision(quality: dict[str, object]) -> str:
    decisions = quality.get("decisions")
    if not isinstance(decisions, dict):
        raise StageError("qwen card is missing quality decisions", code="quality")
    body = decisions.get("qwen-transformer")
    if not isinstance(body, dict):
        raise StageError("qwen quality report has no transformer decision", code="quality")
    decision = body.get("decision")
    if decision != "fp8" and decision != "bf16":
        raise StageError(f"qwen decision {decision!r} is not fp8 or bf16", code="quality")
    return decision


def _encoder_status(encoder: dict[str, object]) -> EncoderStatus:
    status = _status_of(encoder.get("status"))
    if status is None:
        raise StageError("encoder marker has no status", code="encoder")
    return status


def _card(ctx: QwenCtx, quality: dict[str, object], encoder: dict[str, object]) -> dict[str, object]:
    decision = _decision(quality)
    status = _encoder_status(encoder)
    source = ctx.plan.source
    decisions = quality.get("decisions")
    body = decisions.get("qwen-transformer") if isinstance(decisions, dict) else None
    lpips = body.get("lpips_mean") if isinstance(body, dict) else None
    clip_drop = body.get("clip_drop") if isinstance(body, dict) else None
    card = CardContext(
        title="Qwen-Image-2512 voxprint FP8",
        base_models=(source.repo,),
        base_lines=(
            f"Quantized from: `{source.repo}`",
            f"Revision: `{source.revision}`",
            "Text encoder: Qwen2.5-VL-7B, converted by part C (Qwen-Image-Edit-2511)",
            f"Encoder: {status}. This part links part C's file and does not write one.",
            "INT4: deferred. No official nunchaku-tech SVDQuant pin. No calibration.",
            "Licence of the weights: Apache-2.0",
            f"Card date: {today_stamp()}",
        ),
        pipeline_tag="text-to-image",
        tags=("voxprint", "fp8", "qwen-image"),
        language=("en",),
        quant_lines=(
            f"qwen-transformer: publish {decision}",
            "img_in, txt_in, norms, adaLN, and proj_out stay BF16",
            "INT4 deferred: directory contains no weights",
        ),
        eval_lines=(
            f"qwen-transformer: LPIPS {lpips}, CLIP drop {clip_drop}, decision {decision}",
            f"8 hard prompts, one seed, {QWEN_STEPS} steps, true CFG {QWEN_TRUE_CFG}, {QWEN_SIZE}px",
        ),
        credits=(
            "Qwen / Alibaba Qwen-Image-2512",
            "Qwen2.5-VL-7B text encoder",
            "diffusers QwenImagePipeline",
            "torchao",
        ),
        stamped=today_stamp(),
    )
    (ctx.output / "README.md").write_text(render_canvas_card(ctx.repo, card), encoding="utf-8")
    return {"card": "README.md", "encoder": status}


def _upload(
    ctx: QwenCtx,
    quality: dict[str, object],
    encoder: dict[str, object],
    *,
    remote: bool = True,
) -> dict[str, object]:
    decision = _decision(quality)
    status = _encoder_status(encoder)
    if status != "convert":
        _drop_published_encoder(ctx.output)
    prefix = QWEN_FP8_PREFIX if decision == "fp8" else QWEN_BF16_PREFIX
    names = [path.name for path in _shard_paths(ctx.output, prefix)]
    extra = [QWEN_METRICS, QWEN_GRID, f"int4/{INT4_MARKER_NAME}", ENCODER_REUSE]
    if status == "convert" and (ctx.output / QWEN_ENCODER_FP8).is_file():
        extra.append(QWEN_ENCODER_FP8)
    source = ctx.plan.source
    tier_name = "24gb" if decision == "fp8" else "24gb-bf16"
    tiers = {name: (tier_name, 24, source.repo, source.revision) for name in names}
    if status == "convert":
        tiers[QWEN_ENCODER_FP8] = ("24gb", 24, source.repo, source.revision)
    variation = "24gb-fp8" if decision == "fp8" else "24gb-bf16"
    if not remote:
        rows = write_local_files(ctx.output, tiers, source.repo, source.revision)
        delete_intermediates(ctx.upstream_root)
        _release_canvas_sources(ctx)
        return {"files": rows, "encoder": status}
    records = publish_output(
        output=ctx.output,
        hf_repo=ctx.plan.hf_repo,
        kaggle_slug=ctx.plan.kaggle_slug,
        title="Voxprint Qwen-Image-2512",
        description=(
            "Private Voxprint dtype conversion of Qwen-Image-2512. Apache-2.0. "
            "INT4 is deferred. The text encoder belongs to part C and is linked, not written here."
        ),
        tiers=tiers,
        variations={variation: [*names, *extra]},
        repo=source.repo,
        revision=source.revision,
        licence=source.licence,
        staging=ctx.upstream_root / "kaggle-staging",
        on_verified=lambda target: _on_qwen_verified(ctx, target),
    )
    delete_intermediates(ctx.upstream_root)
    _release_canvas_sources(ctx)
    return {"files": records, "encoder": status}


def _on_qwen_verified(ctx: QwenCtx, target: str) -> None:
    on_qwen_mirror(ctx, target)
    _release_canvas_sources(ctx)


def _release_canvas_sources(ctx: QwenCtx) -> None:
    """Part D is a consumer of the canvas downloads. Release it; delete only if it was last."""
    if not ctx.plan.consumes_plan or not ctx.plan.consumes_sources:
        return
    release_consumed(ctx.forge_out, ctx.repo, ctx.plan.consumes_plan, ctx.plan.consumes_sources, "qwen")


def _refresh(ctx: QwenCtx, quality: dict[str, object]) -> None:
    prefix = QWEN_FP8_PREFIX if _decision(quality) == "fp8" else QWEN_BF16_PREFIX
    refresh_fp8_dir(ctx.forge_out, _shard_paths(ctx.output, prefix))


def _stable(ctx: QwenCtx) -> list[str]:
    return format_stable_lines(
        ctx.forge_out,
        bf16_encoder=ctx.source_dir / "text_encoder",
        expected_fp8=edit_encoder_fp8(
            ctx.forge_out,
            ctx.plan.edit_job_id,
            ctx.plan.edit_output_name,
            ctx.plan.edit_encoder_fp8,
        ),
    )


def _uploaded_qwen(ctx: QwenCtx, marker: dict[str, object]) -> dict[str, object]:
    files = marker.get("files", [])
    return {
        "id": "qwen",
        "status": "uploaded",
        "estimate_min": ctx.plan.estimate_min,
        "estimate_min_reuse": ctx.plan.estimate_min_reuse,
        "estimate_min_recompute": ctx.plan.estimate_min_recompute,
        "hf_repo": ctx.plan.hf_repo,
        "kaggle_slug": ctx.plan.kaggle_slug,
        "resumed": True,
        "encoder": marker.get("encoder"),
        "files": files if isinstance(files, list) else [],
        "stable": _stable(ctx),
    }


def run_qwen_part(ctx: QwenCtx, deadline: datetime | None, now: datetime) -> dict[str, object]:
    """Prepare and publish the Qwen-Image transformer tier."""
    existing = read_marker(ctx.marker("upload"))
    if existing is not None:
        return _uploaded_qwen(ctx, existing)
    if defer_part(ctx.plan.estimate_min, deadline, now):
        return {
            "id": "qwen",
            "status": DEFERRED,
            "estimate_min": ctx.plan.estimate_min,
            "hf_repo": ctx.plan.hf_repo,
            "files": [],
        }
    check_disk(ctx)
    progress(ctx, "start", 0, float(ctx.plan.estimate_min))
    run_stage(ctx, "fetch", lambda: _fetch(ctx))
    run_stage(ctx, "convert", lambda: _convert(ctx))
    encoder = ensure_encoder(ctx)
    run_stage(ctx, "int4", lambda: _int4(ctx))
    quality = run_stage(ctx, "quality", lambda: _quality(ctx))
    run_stage(ctx, "card", lambda: _card(ctx, quality, encoder))
    _refresh(ctx, quality)
    if ctx.defer_upload:
        held = run_stage(ctx, "upload-deferred", lambda: _upload(ctx, quality, encoder, remote=False))
        status = "upload-deferred"
        files = held.get("files", [])
    else:
        uploaded = run_stage(ctx, "upload", lambda: _upload(ctx, quality, encoder))
        drop_marker(ctx.marker("upload-deferred"))
        status = "uploaded"
        files = uploaded.get("files", [])
    progress(ctx, "done", 100, 0)
    return {
        "id": "qwen",
        "status": status,
        "estimate_min": ctx.plan.estimate_min,
        "estimate_min_reuse": ctx.plan.estimate_min_reuse,
        "estimate_min_recompute": ctx.plan.estimate_min_recompute,
        "hf_repo": ctx.plan.hf_repo,
        "kaggle_slug": ctx.plan.kaggle_slug,
        "encoder": _encoder_status(encoder),
        "int4": INT4_MODE,
        "copied_weights": False,
        "files": files if isinstance(files, list) else [],
        "stable": _stable(ctx),
    }


def run_encoder_part(ctx: QwenCtx, deadline: datetime | None, now: datetime) -> dict[str, object]:
    """Prepare the shared text-encoder files for Qwen-Image."""
    if defer_part(ctx.plan.encoder_estimate_min, deadline, now):
        return {
            "id": "encoder",
            "status": DEFERRED,
            "estimate_min": ctx.plan.encoder_estimate_min,
            "files": [],
        }
    check_disk(ctx)
    progress(ctx, "encoder", 0, float(ctx.plan.encoder_estimate_min))
    run_stage(ctx, "fetch", lambda: _fetch(ctx))
    encoder = ensure_encoder(ctx)
    progress(ctx, "encoder", 100, 0)
    uploaded_encoder = read_marker(ctx.layout.stage(ctx.plan.id, "encoder-upload"))
    if uploaded_encoder is not None:
        files = uploaded_encoder.get("files", [])
        return {
            "id": "encoder",
            "status": "uploaded",
            "estimate_min": ctx.plan.encoder_estimate_min,
            "copied_weights": False,
            "resumed": True,
            "files": files if isinstance(files, list) else [],
            "stable": _stable(ctx),
        }
    if ctx.defer_upload:
        held = _hold_encoder(ctx, encoder)
        files = held.get("files", [])
        return {
            "id": "encoder",
            "status": "upload-deferred",
            "estimate_min": ctx.plan.encoder_estimate_min,
            "copied_weights": False,
            "encoder": _encoder_status(encoder),
            "edit_fp8": encoder.get("edit_fp8"),
            "files": files if isinstance(files, list) else [],
            "stable": _stable(ctx),
        }
    return {
        "id": "encoder",
        "status": _encoder_status(encoder),
        "estimate_min": ctx.plan.encoder_estimate_min,
        "copied_weights": False,
        "edit_fp8": encoder.get("edit_fp8"),
        "files": [],
        "stable": _stable(ctx),
    }


def _hold_encoder(ctx: QwenCtx, encoder: dict[str, object]) -> dict[str, object]:
    """Record the encoder file for a later upload. A reused encoder is not copied."""
    status = _encoder_status(encoder)
    fp8 = ctx.output / QWEN_ENCODER_FP8
    files: list[dict[str, object]] = []
    if status == "convert" and fp8.is_file() and not fp8.is_symlink():
        from voxprint_forge.hashing import sha256_file, write_sha256

        digest = sha256_file(fp8)
        write_sha256(fp8, digest)
        files.append({"name": fp8.name, "sha256": digest, "size": fp8.stat().st_size, "tier": "24gb"})
    payload = {"status": status, "copied_weights": False, "files": files}
    write_marker(
        ctx.layout.stage(ctx.plan.id, "encoder-upload-deferred"),
        {"stage": "encoder-upload-deferred", **payload},
    )
    return payload


def _canvas_part(
    *,
    repo: Path,
    forge_out: Path,
    token: str | None,
    part_id: PartId,
    deadline: datetime | None,
    now: datetime,
    defer_upload: bool,
) -> dict[str, object]:
    plan = load_canvas_plan(repo / "plans" / ZIMAGE_PLAN)
    part = plan.part(part_id)
    if defer_part(part.estimate_min, deadline, now):
        return {
            "id": part.id,
            "status": DEFERRED,
            "estimate_min": part.estimate_min,
            "hf_repo": part.hf_repo,
            "files": [],
        }
    layout = Layout(forge_out)
    existing = read_marker(layout.stage(plan.id, f"{part_id}-upload"))
    if existing is not None:
        files = existing.get("files", [])
        return {
            "id": part.id,
            "status": "uploaded",
            "estimate_min": part.estimate_min,
            "hf_repo": part.hf_repo,
            "resumed": True,
            "files": files if isinstance(files, list) else [],
        }
    ctx = Ctx(
        repo=repo,
        forge_out=forge_out,
        plan=plan,
        part=part,
        token=token,
        layout=layout,
        defer_upload=defer_upload,
    )
    if part_id == "base":
        return run_base(ctx)
    if part_id == "controlnet":
        return run_control(ctx)
    if part_id == "edit":
        return run_edit(ctx)
    return _unexpected_canvas(part_id)


def _write_manifest(forge_out: Path, plan: QwenPlan, parts: list[dict[str, object]]) -> tuple[str, str]:
    day = datetime.now(timezone.utc).date().isoformat()
    directory = forge_out / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "job": plan.id,
        "date": day,
        "int4": plan.int4,
        "retired": list(plan.retired),
        "encoder_owner": "edit",
        "estimates_min": {
            "qwen": plan.estimate_min,
            "qwen_reuse": plan.estimate_min_reuse,
            "qwen_recompute": plan.estimate_min_recompute,
            "encoder": plan.encoder_estimate_min,
        },
        "parts": parts,
    }
    json_path = directory / f"{day}-canvas-qwen-image-2512.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Qwen-Image-2512 MODELS",
        "",
        "Retired. Parts D and E are retired Canvas generators.",
        "The Qwen2.5-VL-7B encoder is part C's file "
        "(qwen2.5-vl-7b-text-encoder-fp8.safetensors from Qwen-Image-Edit-2511).",
        "A forced part D links that file. Part C does not read part D.",
        "INT4 deferred. No weights in the int4 directory.",
        "",
    ]
    for part in parts:
        lines.append(f"## {part.get('id')} — {part.get('status')}")
        lines.append("")
    md_path = directory / f"{day}-canvas-qwen-image-2512.MODELS.md"
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(json_path), str(md_path)


def run_selected(
    *,
    repo: Path,
    plan: QwenPlan,
    forge_out: Path,
    parts: tuple[QwenPart, ...],
    deadline: datetime | None,
    token: str | None,
    now: Callable[[], datetime] | None = None,
    defer_upload: bool = False,
) -> dict[str, object]:
    """Run the selected Qwen-Image parts that still fit before the deadline."""
    clock = now or (lambda: datetime.now(timezone.utc))
    layout = Layout(forge_out)
    forge_out.mkdir(parents=True, exist_ok=True)
    ctx = QwenCtx(
        repo=repo,
        forge_out=forge_out,
        plan=plan,
        token=token,
        layout=layout,
        defer_upload=defer_upload,
    )
    results: list[dict[str, object]] = []
    manifest_json = ""
    manifest_md = ""
    try:
        for part in parts:
            current = clock()
            try:
                if part == "base":
                    results.append(
                        _canvas_part(
                            repo=repo,
                            forge_out=forge_out,
                            token=token,
                            part_id="base",
                            deadline=deadline,
                            now=current,
                            defer_upload=defer_upload,
                        )
                    )
                elif part == "controlnet":
                    results.append(
                        _canvas_part(
                            repo=repo,
                            forge_out=forge_out,
                            token=token,
                            part_id="controlnet",
                            deadline=deadline,
                            now=current,
                            defer_upload=defer_upload,
                        )
                    )
                elif part == "qwen":
                    results.append(run_qwen_part(ctx, deadline, current))
                elif part == "edit":
                    results.append(
                        _canvas_part(
                            repo=repo,
                            forge_out=forge_out,
                            token=token,
                            part_id="edit",
                            deadline=deadline,
                            now=current,
                            defer_upload=defer_upload,
                        )
                    )
                elif part == "encoder":
                    results.append(run_encoder_part(ctx, deadline, current))
                else:
                    _unexpected_part(part)
            except ForgeError as exc:
                stage = current_stage()
                failed = {
                    "id": part,
                    "status": f"STAGEFAIL {stage}",
                    "state": f"STAGEFAIL {stage}",
                    "error": str(exc),
                    "files": [],
                }
                update_item(
                    forge_out,
                    part,
                    stage=stage,
                    pct=0,
                    eta_min=0,
                    state=str(failed["state"]),
                    error=str(exc),
                )
                log(f"{part} STAGEFAIL {stage}: {exc}")
                results.append(failed)
    finally:
        manifest_json, manifest_md = _write_manifest(forge_out, plan, results)
    for item in results:
        update_item(
            forge_out,
            str(item.get("id", "")),
            stage=str(item.get("status", "")),
            pct=100,
            eta_min=0,
            state=state_line(item),
            error=str(item.get("error", "")),
        )
    text, ok = summarise(results)
    log(text)
    return {
        "job": plan.id,
        "int4": plan.int4,
        "parts_selected": list(parts),
        "parts": results,
        "manifest": manifest_json,
        "models": manifest_md,
        "copy_back": [manifest_json, manifest_md],
        "ok": ok,
        "states": text,
        "estimates_min": {
            "qwen": plan.estimate_min,
            "qwen_reuse": plan.estimate_min_reuse,
            "qwen_recompute": plan.estimate_min_recompute,
            "encoder": plan.encoder_estimate_min,
        },
    }


def _qwen_upload_ids(text: str | None) -> tuple[str, ...] | None:
    if text is None or not text.strip():
        return None
    chosen = parse_parts(text)
    allowed = ("qwen", "encoder")
    found = tuple(part for part in allowed if part in chosen)
    if not found or len(found) != len(chosen):
        raise UsageError("qwen upload parts must be D, E or qwen, encoder", code="parts")
    return found


def _release_qwen(ctx: QwenCtx, *, explicit: bool) -> dict[str, object] | None:
    uploaded = read_marker(ctx.marker("upload"))
    deferred = ctx.marker("upload-deferred")
    if uploaded is not None:
        drop_marker(deferred)
        return _uploaded_qwen(ctx, uploaded)
    if read_marker(deferred) is None:
        if not explicit:
            return None
        raise StageError("qwen has no upload-deferred marker", code="upload_deferred")
    quality = read_marker(ctx.marker("quality"))
    encoder = read_marker(ctx.marker("encoder"))
    if quality is None or encoder is None:
        raise StageError("qwen is missing quality or encoder markers", code="quality")
    body = run_stage(ctx, "upload", lambda: _upload(ctx, quality, encoder))
    drop_marker(deferred)
    files = body.get("files", [])
    return {
        "id": "qwen",
        "status": "uploaded",
        "estimate_min": ctx.plan.estimate_min,
        "hf_repo": ctx.plan.hf_repo,
        "encoder": body.get("encoder"),
        "copied_weights": False,
        "files": files if isinstance(files, list) else [],
    }


def _release_encoder(ctx: QwenCtx, *, explicit: bool) -> dict[str, object] | None:
    done_path = ctx.layout.stage(ctx.plan.id, "encoder-upload")
    deferred_path = ctx.layout.stage(ctx.plan.id, "encoder-upload-deferred")
    done = read_marker(done_path)
    if done is not None:
        drop_marker(deferred_path)
        files = done.get("files", [])
        return {
            "id": "encoder",
            "status": "uploaded",
            "resumed": True,
            "copied_weights": False,
            "files": files if isinstance(files, list) else [],
        }
    if read_marker(deferred_path) is None:
        if not explicit:
            return None
        raise StageError("encoder has no upload-deferred marker", code="upload_deferred")
    fp8 = ctx.output / QWEN_ENCODER_FP8
    records: list[dict[str, object]] = []
    if fp8.is_file() and not fp8.is_symlink():
        stage = ctx.forge_out / "work" / ctx.plan.id / "encoder-upload-stage"
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir(parents=True)
        link_or_copy(fp8, stage / fp8.name)
        source = ctx.plan.source
        readme = ctx.output / "README.md"
        card_text = readme.read_text(encoding="utf-8") if readme.is_file() else ""
        records = publish_output(
            output=stage,
            hf_repo=ctx.plan.hf_repo,
            kaggle_slug=ctx.plan.kaggle_slug,
            title="Voxprint Qwen-Image-2512 text encoder",
            description="Voxprint FP8 text encoder. Apache-2.0. A reused encoder is not copied.",
            tiers={fp8.name: ("24gb", 24, source.repo, source.revision)},
            variations={"24gb-fp8": [fp8.name]},
            repo=source.repo,
            revision=source.revision,
            licence=source.licence,
            staging=stage / "kaggle-staging",
            card_text=card_text,
        )
        shutil.rmtree(stage, ignore_errors=True)
    write_marker(done_path, {"stage": "encoder-upload", "files": records, "copied_weights": False})
    drop_marker(deferred_path)
    return {"id": "encoder", "status": "uploaded", "copied_weights": False, "files": records}


def release_deferred(
    *,
    repo: Path,
    plan: QwenPlan,
    forge_out: Path,
    token: str | None,
    parts_text: str | None,
    force: bool = False,
) -> dict[str, object]:
    """Upload deferred D and E. Retired parts refuse unless force is set."""
    selected = _qwen_upload_ids(parts_text)
    considered = ("qwen", "encoder") if selected is None else selected
    refuse_retired(considered, plan.retired, force=force)
    forge_out.mkdir(parents=True, exist_ok=True)
    ctx = QwenCtx(repo=repo, forge_out=forge_out, plan=plan, token=token, layout=Layout(forge_out))
    results: list[dict[str, object]] = []
    if selected is None or "qwen" in selected:
        released = _release_qwen(ctx, explicit=selected is not None)
        if released is not None:
            results.append(released)
    if selected is None or "encoder" in selected:
        released = _release_encoder(ctx, explicit=selected is not None)
        if released is not None:
            results.append(released)
    return {"job": plan.id, "parts": results}
