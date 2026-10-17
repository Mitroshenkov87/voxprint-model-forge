"""Run the diffusion-int4 steps in order, with markers, heartbeats, and the disk floor."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from voxprint_forge.diffusion.int4.context import StageContext
from voxprint_forge.diffusion.int4.disk import guard_disk, reclaim_sources, reclaim_verified
from voxprint_forge.diffusion.int4.gpu import planning_gpu, step_minutes, total_minutes
from voxprint_forge.diffusion.int4.markers import load_done, save_done
from voxprint_forge.diffusion.int4.pins import SCHEMA, StepEstimate, planned_steps
from voxprint_forge.diffusion.int4.wan import fp8_step, gguf_step, smoke_step
from voxprint_forge.diffusion.int4.wan import publish_step as wan_publish_step
from voxprint_forge.diffusion.int4.watch import build_watch, estimates_report, write_watch
from voxprint_forge.errors import StageError, UsageError
from voxprint_forge.statusfile import beating, update_item

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


def step_functions(done: Path) -> dict[str, StepFn]:
    """Map a step id to the function that performs it."""
    return {
        "wan-2.2-gguf": gguf_step,
        "wan-2.2-fp8": fp8_step,
        "wan-2.2-smoke": smoke_step,
        "wan-2.2-publish": wan_publish_step,
        "wan-2.2-reclaim": _reclaim("wan-2.2-publish", done),
    }


def run(
    ctx: StageContext,
    *,
    done: Path,
    watch: Path,
    only: str | None = None,
) -> dict[str, object]:
    """Run every selected step, skipping markers that already exist."""
    steps = planned_steps()
    functions = step_functions(done)
    if only is not None and only not in functions:
        raise UsageError(f"unknown step {only}", code="usage")
    gpu = planning_gpu()
    finished = {step.step_id for step in steps if load_done(done, step.step_id)}
    _watch(watch, done=finished, running="", gpu=gpu)
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
        _watch(watch, done=finished, running="", gpu=gpu)
        guard_disk(
            ctx.forge_out,
            ratio=ctx.free_ratio,
            reclaim=lambda: reclaim_verified(done),
        )
    return {
        "schema": SCHEMA,
        "ok": True,
        "dry_run": ctx.dry_run,
        "svd_rank": ctx.svd_rank,
        "gpu": gpu,
        "expected_min": total_minutes(steps, gpu),
        "steps": rows,
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


def _watch(path: Path, *, done: set[str], running: str, gpu: str) -> None:
    write_watch(path, build_watch(done=done, running=running, gpu=gpu))


def format_estimates() -> str:
    """A100 and H100 planning tables for the selected steps."""
    return estimates_report(planned_steps())
