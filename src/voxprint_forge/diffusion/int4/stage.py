"""Run the diffusion-int4 steps in order, with markers, heartbeats, and the disk floor."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from voxprint_forge.diffusion.int4.context import StageContext
from voxprint_forge.diffusion.int4.disk import guard_disk, reclaim_verified
from voxprint_forge.diffusion.int4.gate import OwnEditDecision, decide_own_edit, load_phase0_baseline
from voxprint_forge.diffusion.int4.gpu import planning_gpu, step_minutes, total_minutes
from voxprint_forge.diffusion.int4.markers import load_done, save_done
from voxprint_forge.diffusion.int4.pins import PHASE0_RESULT, SCHEMA, StepEstimate, planned_steps
from voxprint_forge.diffusion.int4.qwen import (
    calib_step,
    eval_step,
    export_step,
    mirror_step,
    publish_step,
    quant_step,
    reclaim_sources,
)
from voxprint_forge.diffusion.int4.wan import fp8_step, gguf_step, smoke_step
from voxprint_forge.diffusion.int4.wan import publish_step as wan_publish_step
from voxprint_forge.diffusion.int4.watch import build_watch, estimates_report, write_watch
from voxprint_forge.errors import StageError, UsageError
from voxprint_forge.statusfile import beating, record_field, update_item

StepFn = Callable[[StageContext], dict[str, object]]


def _reclaim(publish_id: str, done: Path) -> StepFn:
    def step(ctx: StageContext) -> dict[str, object]:
        del ctx
        payload = load_done(done, publish_id)
        if not isinstance(payload, dict) or payload.get("verified") is not True:
            raise StageError(f"{publish_id} is not verified, so sources stay", code="reclaim")
        sources = payload.get("sources")
        if not isinstance(sources, list):
            raise StageError(f"{publish_id} has no source list", code="reclaim")
        removed = reclaim_sources([str(item) for item in sources], verified=True)
        return {"removed": removed, "verified": True, "sources": sources}

    return step


def step_functions(done: Path, *, qwen_image: bool, mirror: bool = False) -> dict[str, StepFn]:
    """Map a step id to the function that performs it."""
    if mirror:
        found: dict[str, StepFn] = {
            "qwen-edit-2511-mirror": mirror_step,
            "qwen-edit-2511-reclaim": _reclaim("qwen-edit-2511-mirror", done),
        }
    else:
        found = {
            "qwen-edit-2511-calib": lambda ctx: calib_step(ctx, "qwen-image-edit-2511"),
            "qwen-edit-2511-quant": lambda ctx: quant_step(ctx, "qwen-image-edit-2511"),
            "qwen-edit-2511-export": lambda ctx: export_step(ctx, "qwen-image-edit-2511"),
            "qwen-edit-2511-eval": lambda ctx: eval_step(ctx, "qwen-image-edit-2511"),
            "qwen-edit-2511-publish": lambda ctx: publish_step(ctx, "qwen-image-edit-2511"),
            "qwen-edit-2511-reclaim": _reclaim("qwen-edit-2511-publish", done),
        }
    found.update(
        {
            "wan-2.2-gguf": gguf_step,
            "wan-2.2-fp8": fp8_step,
            "wan-2.2-smoke": smoke_step,
            "wan-2.2-publish": wan_publish_step,
            "wan-2.2-reclaim": _reclaim("wan-2.2-publish", done),
        }
    )
    if qwen_image:
        found.update(
            {
                "qwen-image-quant": lambda ctx: quant_step(ctx, "qwen-image"),
                "qwen-image-export": lambda ctx: export_step(ctx, "qwen-image"),
                "qwen-image-eval": lambda ctx: eval_step(ctx, "qwen-image"),
                "qwen-image-publish": lambda ctx: publish_step(ctx, "qwen-image"),
                "qwen-image-reclaim": _reclaim("qwen-image-publish", done),
            }
        )
    return found


def run(
    ctx: StageContext,
    *,
    done: Path,
    watch: Path,
    qwen_image: bool = False,
    only: str | None = None,
    own_edit: OwnEditDecision | None = None,
    phase0_result: Path | None = None,
) -> dict[str, object]:
    """Run every selected step, skipping markers that already exist.

    The own edit quant runs unless ``own_edit`` says to mirror. A missing
    Phase 0 file leaves that decision on the own quant.
    """
    path = Path(PHASE0_RESULT) if phase0_result is None else phase0_result
    baseline = load_phase0_baseline(path)
    decision = own_edit if own_edit is not None else decide_own_edit("on", path)
    ctx.own_edit = decision
    ctx.phase0_baseline = baseline
    record_field(ctx.forge_out, "own_edit", decision.as_dict())
    record_field(ctx.forge_out, "phase0_baseline", baseline)
    mirror = decision.action == "mirror"
    steps = planned_steps(qwen_image=qwen_image, mirror=mirror)
    functions = step_functions(done, qwen_image=qwen_image, mirror=mirror)
    if only is not None and only not in functions:
        raise UsageError(f"unknown step {only}", code="usage")
    gpu = planning_gpu()
    finished = {step.step_id for step in steps if load_done(done, step.step_id)}
    _watch(watch, qwen_image=qwen_image, done=finished, running="", gpu=gpu, mirror=mirror)
    rows: list[dict[str, object]] = []
    for step in steps:
        if only is not None and step.step_id != only:
            continue
        if step.step_id in finished:
            rows.append({"id": step.step_id, "skipped": True})
            continue
        _require_previous(steps, finished, step)
        guard_disk(
            ctx.forge_out,
            ratio=ctx.free_ratio,
            reclaim=lambda: reclaim_verified(done),
        )
        eta = float(step_minutes(step, gpu))
        update_item(
            ctx.forge_out,
            step.step_id,
            stage=step.step_id,
            pct=0,
            eta_min=eta,
            state="RUNNING",
        )
        try:
            with beating(ctx.forge_out, step.step_id, step.step_id, pct=0, eta_min=eta):
                payload = functions[step.step_id](ctx)
        except Exception as exc:
            update_item(
                ctx.forge_out,
                step.step_id,
                stage=step.step_id,
                pct=0,
                eta_min=eta,
                state=f"STAGEFAIL {step.step_id}",
                error=str(exc),
            )
            raise
        state = "UPLOADED" if payload.get("verified") is True else "DONE"
        update_item(
            ctx.forge_out,
            step.step_id,
            stage=step.step_id,
            pct=100,
            eta_min=0,
            state=state,
        )
        record = {"id": step.step_id, "skipped": False, **payload}
        save_done(done, step.step_id, record)
        finished.add(step.step_id)
        rows.append(record)
        _watch(watch, qwen_image=qwen_image, done=finished, running="", gpu=gpu, mirror=mirror)
        guard_disk(
            ctx.forge_out,
            ratio=ctx.free_ratio,
            reclaim=lambda: reclaim_verified(done),
        )
    winner = _winner(done)
    return {
        "schema": SCHEMA,
        "ok": True,
        "dry_run": ctx.dry_run,
        "svd_rank": ctx.svd_rank,
        "gpu": gpu,
        "expected_min": total_minutes(steps, gpu),
        "winner": winner,
        "steps": rows,
        "qwen_image": qwen_image,
        "own_edit": decision.as_dict(),
        "phase0_baseline": baseline,
    }


def _require_previous(steps: tuple[StepEstimate, ...], finished: set[str], step: StepEstimate) -> None:
    for earlier in steps:
        if earlier.step_id == step.step_id:
            return
        if earlier.step_id not in finished:
            raise UsageError(
                f"{step.step_id} needs {earlier.step_id} first",
                code="usage",
            )


def _winner(done: Path) -> str | None:
    payload = load_done(done, "qwen-edit-2511-eval")
    if not isinstance(payload, dict):
        return None
    winner = payload.get("winner")
    return str(winner) if winner else None


def _watch(path: Path, *, qwen_image: bool, done: set[str], running: str, gpu: str, mirror: bool) -> None:
    write_watch(path, build_watch(qwen_image=qwen_image, done=done, running=running, gpu=gpu, mirror=mirror))


def format_estimates(*, qwen_image: bool, mirror: bool = False) -> str:
    """A100 and H100 planning tables for the selected steps."""
    return estimates_report(planned_steps(qwen_image=qwen_image, mirror=mirror))
