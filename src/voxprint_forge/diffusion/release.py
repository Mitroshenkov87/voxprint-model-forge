"""Upload parts that were left at `<part>-upload-deferred`. A second run is a no-op."""

from __future__ import annotations

from pathlib import Path
from typing import Never

from voxprint_forge.diffusion.common import Ctx, run_stage
from voxprint_forge.diffusion.defer import drop_marker
from voxprint_forge.diffusion.planload import CanvasPlan, PartId, load_canvas_plan
from voxprint_forge.diffusion.qwen_select import parse_parts
from voxprint_forge.errors import StageError, UsageError
from voxprint_forge.layout import Layout, read_marker


def _unexpected(part_id: Never) -> Never:
    raise UsageError(f"unhandled canvas part {part_id}", code="parts")


def _canvas_ids(text: str | None) -> tuple[PartId, ...] | None:
    """None uploads every deferred canvas part. An explicit list must be A, B, or C."""
    if text is None or not text.strip():
        return None
    chosen = parse_parts(text)
    allowed: tuple[PartId, ...] = ("base", "controlnet", "edit")
    found = tuple(part for part in allowed if part in chosen)
    if not found or len(found) != len(chosen):
        raise UsageError(
            "canvas upload parts must be A, B, C or base, controlnet, edit",
            code="parts",
        )
    return found


def _resumed(part_id: str, estimate_min: int, hf_repo: str, marker: dict[str, object]) -> dict[str, object]:
    files = marker.get("files", [])
    return {
        "id": part_id,
        "status": "uploaded",
        "estimate_min": estimate_min,
        "hf_repo": hf_repo,
        "resumed": True,
        "files": files if isinstance(files, list) else [],
    }


def _publish(ctx: Ctx, quality: dict[str, object]) -> dict[str, object]:
    # Imported here so a deferred run does not need this module's callers to cycle.
    part_id = ctx.part.id
    if part_id == "base":
        from voxprint_forge.diffusion.part_base import _upload

        return _upload(ctx, quality)
    if part_id == "controlnet":
        from voxprint_forge.diffusion.part_control import _upload

        return _upload(ctx, quality)
    if part_id == "edit":
        from voxprint_forge.diffusion.part_edit import _upload

        return _upload(ctx, quality)
    return _unexpected(part_id)


def release_canvas(
    *,
    repo: Path,
    plan: CanvasPlan,
    forge_out: Path,
    token: str | None,
    parts_text: str | None,
) -> dict[str, object]:
    """Upload canvas parts that were deferred."""
    selected = _canvas_ids(parts_text)
    layout = Layout(forge_out)
    results: list[dict[str, object]] = []
    for part in plan.parts:
        if selected is not None and part.id not in selected:
            continue
        uploaded = read_marker(layout.stage(plan.id, f"{part.id}-upload"))
        deferred_path = layout.stage(plan.id, f"{part.id}-upload-deferred")
        if uploaded is not None:
            drop_marker(deferred_path)
            results.append(_resumed(part.id, part.estimate_min, part.hf_repo, uploaded))
            continue
        if read_marker(deferred_path) is None:
            if selected is None:
                continue
            raise StageError(f"{part.id} has no upload-deferred marker", code="upload_deferred")
        quality = read_marker(layout.stage(plan.id, f"{part.id}-quality"))
        if quality is None or "decisions" not in quality:
            raise StageError(f"{part.id} is missing its quality marker", code="quality")
        ctx = Ctx(
            repo=repo,
            forge_out=forge_out,
            plan=plan,
            part=part,
            token=token,
            layout=layout,
            defer_upload=False,
        )
        body = run_stage(ctx, "upload", lambda ctx=ctx, quality=quality: _publish(ctx, quality))
        drop_marker(deferred_path)
        files = body.get("files", [])
        results.append(
            {
                "id": part.id,
                "status": "uploaded",
                "estimate_min": part.estimate_min,
                "hf_repo": part.hf_repo,
                "files": files if isinstance(files, list) else [],
            }
        )
    return {"job": plan.id, "parts": results}


def release_canvas_plan(
    *,
    repo: Path,
    plan_path: Path,
    forge_out: Path,
    token: str | None,
    parts_text: str | None,
) -> dict[str, object]:
    """Load a canvas plan and upload its deferred parts."""
    plan = load_canvas_plan(plan_path)
    return release_canvas(repo=repo, plan=plan, forge_out=forge_out, token=token, parts_text=parts_text)
