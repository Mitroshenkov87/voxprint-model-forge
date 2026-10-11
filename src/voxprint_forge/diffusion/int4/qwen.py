"""Qwen-Image-Edit-2511 and optional Qwen-Image INT4 steps."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from voxprint_forge.diffusion.int4.calib import (
    calibration_prompts,
    gate_prompts,
    image_root,
    load_prompts,
    write_prompt_yaml,
)
from voxprint_forge.diffusion.int4.config import collect_argv, convert_argv, quant_argv, write_configs
from voxprint_forge.diffusion.int4.context import StageContext
from voxprint_forge.diffusion.int4.evalrule import Arm, decide_winner, dry_run_arms
from voxprint_forge.diffusion.int4.gpu import device_fp8_report
from voxprint_forge.diffusion.int4.measure import measure_live
from voxprint_forge.diffusion.int4.patch import apply_checkout
from voxprint_forge.diffusion.int4.pins import (
    DEEPCOMPRESSOR_COMMIT,
    DEFAULT_SVD_RANK,
    QUANTFUNC_R32,
    QUANTFUNC_R32_SHA256,
    QUANTFUNC_R128,
    QUANTFUNC_R128_SHA256,
    QUANTFUNC_R256,
    QUANTFUNC_REPO,
    QUANTFUNC_REVISION,
    QWEN_EDIT_REPO,
    QWEN_EDIT_REVISION,
    QWEN_IMAGE_REPO,
    QWEN_IMAGE_REVISION,
    QWEN_LICENCE,
    SVD_GROUP,
    TargetName,
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
from voxprint_forge.errors import ChecksumError, StageError, ToolMissingError
from voxprint_forge.hashing import sha256_file
from voxprint_forge.mdl import CUDA_CC, NUNCHAKU_VERSION_REQ, ManifestInput
from voxprint_forge.reclaim import delete_logged

_ZERO = {"qwen-image-edit-2511": True, "qwen-image": False}
_MIRROR_SLUG = "voxprint-qwen-image-edit-2511-quantfunc-int4"
_MIRROR_METHOD = (
    "These files are the QuantFunc Nunchaku INT4 checkpoints for Qwen-Image-Edit-2511. "
    "r128 is the balance file and r32 is the ultimate-speed file. Voxprint did not re-quantize them."
)
_MIRROR_CALIBRATION = "No Voxprint calibration. These are the upstream QuantFunc files at the pinned revision."
_MIRROR_NOTICE = "Mirrored by Voxprint from QuantFunc. Voxprint did not re-quantize these weights."
_QUANTFUNC_SHA = {QUANTFUNC_R128: QUANTFUNC_R128_SHA256, QUANTFUNC_R32: QUANTFUNC_R32_SHA256}


def _target_facts(target: TargetName) -> tuple[str, str, str, bool]:
    if target == "qwen-image-edit-2511":
        return QWEN_EDIT_REPO, QWEN_EDIT_REVISION, "image-to-image", True
    if target == "qwen-image":
        return QWEN_IMAGE_REPO, QWEN_IMAGE_REVISION, "text-to-image", False
    raise StageError(f"unknown target {target}", code="target")


def _deepcompressor(ctx: StageContext) -> Path:
    return ctx.tool("FORGE_DEEPCOMPRESSOR", "deepcompressor")


def _export_name(target: TargetName, rank: int) -> str:
    slug = target.replace("-", "_")
    return f"nunchaku_{slug}_int4_r{rank}.safetensors"


def _metadata(target: TargetName, rank: int) -> dict[str, str]:
    zero = _ZERO[target]
    return {
        "format": "nunchaku-svdquant",
        "model_class": "NunchakuQwenImageTransformer2DModel",
        "quantization_config": json.dumps(
            {
                "method": "svdquant",
                "weight": "int4",
                "activation": "int4",
                "group_size": SVD_GROUP,
                "rank": rank,
            }
        ),
        "config": json.dumps({"zero_cond_t": zero}),
    }


def prepare_configs(ctx: StageContext, target: TargetName) -> dict[str, Path]:
    """Write configs and the calibration yaml for one Qwen target."""
    work = ctx.job(target)
    prompts = calibration_prompts(ctx.repo)
    configs = write_configs(work, target, rank=ctx.svd_rank)
    write_prompt_yaml(prompts, configs["prompts"], image_root=image_root(ctx.repo))
    return configs


def calib_step(ctx: StageContext, target: TargetName) -> dict[str, object]:
    """Record the calibration command. Dry-run does not collect GPU caches."""
    configs = prepare_configs(ctx, target)
    argv = collect_argv(configs)
    record = execute(
        argv,
        cwd=_deepcompressor(ctx) / "examples" / "diffusion",
        disk_path=ctx.forge_out,
        dry_run=ctx.dry_run,
        ratio=ctx.free_ratio,
    )
    return {"argv": record["argv"], "prompts": str(configs["prompts"]), "rank": ctx.svd_rank}


def quant_step(ctx: StageContext, target: TargetName) -> dict[str, object]:
    """Run fast_int4. Dry-run writes a tiny checkpoint marker instead."""
    work = ctx.job(target)
    configs = write_configs(work, target, rank=ctx.svd_rank)
    save_dir = work / "quant"
    save_dir.mkdir(parents=True, exist_ok=True)
    checkout = _deepcompressor(ctx)
    argv = quant_argv(checkout, configs, save_dir)
    if ctx.dry_run:
        (save_dir / "checkpoint.txt").write_text(f"dry-run rank {ctx.svd_rank}\n", encoding="utf-8")
        source = work / "source"
        source.mkdir(parents=True, exist_ok=True)
        (source / "transformer.safetensors").write_bytes(b"source-weights")
        return {"argv": argv, "dry_run": True, "quant": str(save_dir)}
    if not checkout.is_dir():
        raise ToolMissingError(
            f"deepcompressor checkout missing at {checkout}. Clone {DEEPCOMPRESSOR_COMMIT}.",
            code="tool_missing",
        )
    patched = apply_checkout(checkout)
    execute(argv, cwd=checkout / "examples" / "diffusion", disk_path=ctx.forge_out, dry_run=False, ratio=ctx.free_ratio)
    return {"argv": argv, "patched": patched, "quant": str(save_dir)}


def export_step(ctx: StageContext, target: TargetName) -> dict[str, object]:
    """Export Nunchaku svdquant safetensors. Dry-run writes a real tiny file."""
    work = ctx.job(target)
    output = work / "export"
    output.mkdir(parents=True, exist_ok=True)
    name = _export_name(target, ctx.svd_rank)
    dest = output / name
    argv = convert_argv(work / "quant", output, target)
    if ctx.dry_run:
        save_file({"rank.weight": torch.zeros(2, 2)}, str(dest), metadata=_metadata(target, ctx.svd_rank))
        return {"argv": argv, "file": str(dest), "zero_cond_t": _ZERO[target], "dry_run": True}
    execute(
        argv,
        cwd=_deepcompressor(ctx),
        disk_path=ctx.forge_out,
        dry_run=False,
        ratio=ctx.free_ratio,
    )
    if not dest.is_file():
        produced = sorted(output.glob("*.safetensors"))
        if not produced:
            raise StageError(f"{target} export produced no safetensors", code="export_missing")
        dest = produced[0]
    return {"argv": argv, "file": str(dest), "zero_cond_t": _ZERO[target]}


def eval_step(ctx: StageContext, target: TargetName) -> dict[str, object]:
    """Compare arms and record the winner. A missing winner fails the step."""
    prompts = load_prompts(ctx.repo)
    cases = [str(item["id"]) for item in gate_prompts(prompts)]
    if ctx.measurer is not None:
        arms = ctx.measurer()
    elif ctx.dry_run:
        arms = dry_run_arms()
    else:
        arms = measure_live(ctx, target)
    decision = decide_winner(arms)
    payload: dict[str, object] = {
        "cases": cases,
        "rule": decision.rule,
        "winner": decision.winner,
        "fits_16gb": decision.fits_16gb,
        "fits_24gb": decision.fits_24gb,
        "passed": decision.passed,
        "arms": [_arm_dict(arm) for arm in arms],
        "dry_run": ctx.dry_run,
        "fp8_baseline": device_fp8_report(probe=not ctx.dry_run),
    }
    if target == "qwen-image-edit-2511" and ctx.phase0_baseline is not None:
        payload["phase0_baseline"] = ctx.phase0_baseline
    path = ctx.job(target) / "comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if decision.winner is None:
        raise StageError(f"{target} has no arm inside the VRAM caps", code="no_winner")
    return payload


def _arm_dict(arm: Arm) -> dict[str, object]:
    return {
        "id": arm.arm_id,
        "peak_vram_smi_mib": arm.peak_vram_smi_mib,
        "seconds_per_image": arm.seconds_per_image,
        "identity": arm.identity,
        "hands": arm.hands,
        "text": arm.text,
        "edit_fidelity": arm.edit_fidelity,
        "rank": arm.rank,
    }


def _env_path(env_name: str, name: str) -> Path:
    root = os.environ.get(env_name, "").strip()
    if not root:
        raise StageError(f"{env_name} is not set, so the winner cannot be published", code="winner_file")
    path = Path(root) / name if name else Path(root)
    if not path.exists():
        raise StageError(f"missing winner path {path}", code="winner_file")
    return path


def _winner_files(ctx: StageContext, target: TargetName, winner: str) -> list[Path]:
    """Local export for an own arm, otherwise the comparison file that won."""
    if winner.startswith("own-"):
        matches = sorted((ctx.job(target) / "export").glob("*.safetensors"))
        if not matches:
            raise StageError(f"{target} export is missing", code="export_missing")
        return [matches[0]]
    if winner == "quantfunc-r128":
        return [_env_path("FORGE_QUANTFUNC", QUANTFUNC_R128)]
    if winner == "quantfunc-r256":
        return [_env_path("FORGE_QUANTFUNC", QUANTFUNC_R256)]
    if winner == "fp8":
        root = _env_path("FORGE_FP8_EDIT", "")
        if root.is_file():
            return [root]
        files = sorted(path for path in root.rglob("*.safetensors") if path.is_file())
        if not files:
            raise StageError(f"{root} has no safetensors to publish", code="winner_file")
        return files
    raise StageError(f"{target} winner {winner} has no publish path", code="winner_file")


def publish_step(ctx: StageContext, target: TargetName) -> dict[str, object]:
    """Render the card, upload, and record source paths. Do not delete them here."""
    comparison_path = ctx.job(target) / "comparison.json"
    if not comparison_path.is_file():
        raise StageError(f"{target} has no comparison.json", code="no_winner")
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    winner = str(comparison.get("winner") or "")
    weights = _winner_files(ctx, target, winner)
    repo, revision, pipeline, _zero = _target_facts(target)
    title = f"VOXPrint {target} SVDQuant INT4 r{ctx.svd_rank}"
    card = render_int4_card(
        ctx.repo,
        title=title,
        base_models=(f"{repo}@{revision}",),
        pipeline_tag=pipeline,
        quant_lines=(
            f"fast_int4 SVD rank {ctx.svd_rank}, group {SVD_GROUP}, nunchaku svdquant.",
            f"deepcompressor {DEEPCOMPRESSOR_COMMIT}.",
            f"Winner: {winner}.",
        ),
        eval_lines=_eval_lines(comparison),
        credits=_credits(target),
        kind="int4",
    )
    output = ctx.job(target) / "publish"
    write_publish_tree(output, [*weights, comparison_path], card)
    slug = f"voxprint-{target}-svdquant-int4"
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
        licence=QWEN_LICENCE,
        hub=hub,
        drive=drive,
        manifest=ManifestInput(
            model_id=slug.removeprefix("voxprint-"),
            display_name=title,
            app_ids=("canvas",),
            task=pipeline,
            license=QWEN_LICENCE,
            base_model=repo,
            engine="nunchaku",
            version_req=NUNCHAKU_VERSION_REQ,
            min_vram_gb=16,
            recommended_vram_gb=24,
            gpu_min_cc=CUDA_CC,
        ),
    )
    sources = [ctx.job(target) / "source", ctx.job(target) / "quant", ctx.job(target) / "references"]
    return {
        "verified": published.verified,
        "hf": published.hf,
        "drive": published.drive,
        "winner": winner,
        "sources": [str(path) for path in sources],
        "files": list(published.records),
        "slug": slug,
    }


def _eval_lines(comparison: dict[str, object]) -> tuple[str, ...]:
    lines = [
        f"Rule: {comparison.get('rule')}.",
        f"16 GB fits: {comparison.get('fits_16gb')}.",
        f"24 GB fits: {comparison.get('fits_24gb')}.",
    ]
    if comparison.get("dry_run") is True:
        lines.append("Dry-run scores are synthetic.")
    baseline = comparison.get("fp8_baseline")
    if isinstance(baseline, dict):
        note = baseline.get("note")
        if isinstance(note, str) and note:
            lines.append(note)
    lines.extend(_baseline_eval_lines(comparison.get("phase0_baseline")))
    return tuple(lines)


def _baseline_eval_lines(baseline: object) -> tuple[str, ...]:
    """Card lines for the Phase 0 comparison. They do not name a winner."""
    if not isinstance(baseline, dict):
        return ()
    lines = ["Phase 0 result is a comparison baseline and does not choose the winner."]
    arms = baseline.get("arms")
    if not isinstance(arms, list):
        return tuple(lines)
    for arm in arms:
        if not isinstance(arm, dict):
            continue
        arm_id = str(arm.get("id") or "")
        lines.append(f"{arm_id} cap16 fits_16gb={arm.get('fits_16gb')}.")
        cases = arm.get("cases")
        if not isinstance(cases, list):
            continue
        for case in cases:
            if isinstance(case, dict):
                lines.append(f"{arm_id} {case.get('id')}: {_case_brief(case)}.")
    return tuple(lines)


def _case_brief(case: dict[str, object]) -> str:
    parts = [f"seconds={case.get('seconds')}"]
    scores = case.get("scores")
    if isinstance(scores, dict) and scores:
        rendered = ", ".join(f"{key}={scores[key]}" for key in sorted(scores))
        parts.append(rendered)
    return " ".join(parts)


def _credits(target: TargetName) -> tuple[str, ...]:
    credits = [
        "Qwen, for the base model.",
        "MIT HAN Lab, for SVDQuant and deepcompressor.",
        "Nunchaku, for the svdquant runtime layout.",
        "Voxprint, for the calibration set and this package. No endorsement is implied.",
    ]
    if target == "qwen-image-edit-2511":
        credits.insert(3, "tonera vitoom-nunchaku, for the zero_cond_t inference fork.")
        credits.insert(4, "QuantFunc, for the r128 and r256 comparison weights.")
    return tuple(credits)


def mirror_step(ctx: StageContext) -> dict[str, object]:
    """Publish QuantFunc r128 and r32. The own r256 quant does not run."""
    weights = _mirror_weights(ctx)
    repo, revision, pipeline, _zero = _target_facts("qwen-image-edit-2511")
    reason = ctx.own_edit.reason if ctx.own_edit is not None else "The own quant is skipped."
    eval_lines = (reason, "Wan 2.2 still runs.", *_baseline_eval_lines(ctx.phase0_baseline))
    card = render_int4_card(
        ctx.repo,
        title="VOXPrint mirror of QuantFunc Qwen-Image-Edit-2511 INT4",
        base_models=(f"{repo}@{revision}", f"{QUANTFUNC_REPO}@{QUANTFUNC_REVISION}"),
        pipeline_tag=pipeline,
        quant_lines=(
            f"{QUANTFUNC_R128} (balance, r128).",
            f"{QUANTFUNC_R32} (ultimate speed, r32).",
            "Voxprint did not re-quantize these files.",
        ),
        eval_lines=eval_lines,
        credits=_credits("qwen-image-edit-2511"),
        kind="int4",
        method_text=_MIRROR_METHOD,
        calibration_text=_MIRROR_CALIBRATION,
        notice=_MIRROR_NOTICE,
    )
    output = ctx.job("qwen-image-edit-2511") / "publish"
    write_publish_tree(output, weights, card)
    hub = ctx.hub
    drive = ctx.drive
    if ctx.dry_run and hub is None:
        mirror = LocalMirror(ctx.forge_out / "mirrors")
        hub = mirror.upload_hf
        drive = mirror.upload_drive
    elif hub is None:
        hub = live_hub

        def drive(folder_id: str, files: list) -> list[dict[str, object]]:
            return live_drive(folder_id, files, model_id=_MIRROR_SLUG)
    published = publish_verified(
        output,
        slug=_MIRROR_SLUG,
        card_text=card,
        licence=QWEN_LICENCE,
        hub=hub,
        drive=drive,
        manifest=ManifestInput(
            model_id=_MIRROR_SLUG.removeprefix("voxprint-"),
            display_name="VOXPrint mirror of QuantFunc Qwen-Image-Edit-2511 INT4",
            app_ids=("canvas",),
            task=pipeline,
            license=QWEN_LICENCE,
            base_model=repo,
            engine="nunchaku",
            version_req=NUNCHAKU_VERSION_REQ,
            min_vram_gb=16,
            recommended_vram_gb=24,
            gpu_min_cc=CUDA_CC,
        ),
    )
    return {
        "verified": published.verified,
        "hf": published.hf,
        "drive": published.drive,
        "sources": [str(path) for path in weights],
        "files": list(published.records),
        "slug": _MIRROR_SLUG,
    }


def _mirror_weights(ctx: StageContext) -> list[Path]:
    """QuantFunc r128 and r32. Dry-run writes placeholders when the files are absent."""
    names = (QUANTFUNC_R128, QUANTFUNC_R32)
    root = os.environ.get("FORGE_QUANTFUNC", "").strip()
    if root:
        paths = [Path(root) / name for name in names]
        if all(path.is_file() for path in paths):
            if not ctx.dry_run:
                _require_quantfunc_sha(paths)
            return paths
    if not ctx.dry_run:
        raise StageError("FORGE_QUANTFUNC is missing QuantFunc r128 or r32", code="quantfunc")
    folder = ctx.job("qwen-image-edit-2511") / "quantfunc"
    folder.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in names:
        path = folder / name
        if not path.is_file():
            path.write_bytes(b"quantfunc-mirror")
        written.append(path)
    return written


def _require_quantfunc_sha(paths: list[Path]) -> None:
    for path in paths:
        expected = _QUANTFUNC_SHA[path.name]
        digest = sha256_file(path)
        if digest != expected:
            raise ChecksumError(f"{path.name} sha256 {digest} != {expected}", code="checksum")


def reclaim_sources(sources: list[str], *, verified: bool) -> list[str]:
    """Delete source paths only after a mirror has verified."""
    if not verified:
        raise StageError("refusing to delete sources before a mirror verifies", code="reclaim")
    removed: list[str] = []
    for item in sources:
        path = Path(item)
        if path.exists() or path.is_symlink():
            delete_logged(path)
            removed.append(str(path))
    return removed


def default_rank() -> int:
    """SVD rank used when the operator does not pass one."""
    return DEFAULT_SVD_RANK
