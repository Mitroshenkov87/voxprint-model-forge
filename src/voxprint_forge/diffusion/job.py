"""Run the three canvas parts in order. A short deadline defers the rest."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Never

from voxprint_forge.diffusion.common import Ctx
from voxprint_forge.diffusion.deadline import DEFERRED, defer_part
from voxprint_forge.diffusion.part_base import run_base
from voxprint_forge.diffusion.part_control import run_control
from voxprint_forge.diffusion.part_edit import run_edit
from voxprint_forge.diffusion.planload import CanvasPlan
from voxprint_forge.errors import PlanError
from voxprint_forge.layout import Layout, read_marker
from voxprint_forge.progress import append_progress


def _unexpected_part(part_id: Never) -> Never:
    raise PlanError(f"unhandled canvas part {part_id}", code="plan_shape")


def _dispatch(ctx: Ctx) -> dict[str, object]:
    part_id = ctx.part.id
    if part_id == "base":
        return run_base(ctx)
    if part_id == "controlnet":
        return run_control(ctx)
    if part_id == "edit":
        return run_edit(ctx)
    return _unexpected_part(part_id)


def _deferred(part_id: str, estimate_min: int, hf_repo: str) -> dict[str, object]:
    return {
        "id": part_id,
        "status": DEFERRED,
        "estimate_min": estimate_min,
        "hf_repo": hf_repo,
        "files": [],
    }


def _already_uploaded(layout: Layout, plan: CanvasPlan, part_id: str) -> dict[str, object] | None:
    marker = read_marker(layout.stage(plan.id, f"{part_id}-upload"))
    if marker is None:
        return None
    part = plan.part(part_id) if part_id in ("base", "controlnet", "edit") else None
    if part is None:
        return None
    return {
        "id": part.id,
        "status": "uploaded",
        "estimate_min": part.estimate_min,
        "hf_repo": part.hf_repo,
        "resumed": True,
        "files": marker.get("files", []),
    }


def _write_manifest(forge_out: Path, plan: CanvasPlan, parts: list[dict[str, object]]) -> tuple[str, str]:
    day = datetime.now(timezone.utc).date().isoformat()
    directory = forge_out / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "job": plan.id,
        "date": day,
        "estimates_min": {part.id: part.estimate_min for part in plan.parts},
        "total_min": plan.total_min(),
        "parts": parts,
    }
    json_path = directory / f"{day}-canvas-zimage.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Canvas MODELS", ""]
    for part in parts:
        status = str(part.get("status", ""))
        lines.append(f"## {part.get('id')} — {status}")
        lines.append("")
        if status == DEFERRED:
            lines.append(DEFERRED)
            lines.append("")
            continue
        files = part.get("files")
        if isinstance(files, list):
            for item in files:
                if isinstance(item, dict):
                    lines.append(
                        f"- {item.get('name')}  {item.get('size')}  {item.get('sha256')}  {item.get('tier')}"
                    )
            lines.append("")
    md_path = directory / f"{day}-canvas-zimage.MODELS.md"
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(json_path), str(md_path)


def run_job(
    *,
    repo: Path,
    plan: CanvasPlan,
    forge_out: Path,
    deadline: datetime | None,
    token: str | None,
    now: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    clock = now or (lambda: datetime.now(timezone.utc))
    layout = Layout(forge_out)
    forge_out.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []
    manifest_json = ""
    manifest_md = ""
    try:
        for part in plan.parts:
            existing = _already_uploaded(layout, plan, part.id)
            if existing is not None:
                results.append(existing)
                continue
            current = clock()
            if defer_part(part.estimate_min, deadline, current):
                append_progress(
                    forge_out / "progress.log",
                    model=part.id,
                    stage="deferred",
                    pct=0,
                    eta_min=max(0.0, (deadline - current).total_seconds() / 60.0) if deadline else 0,
                )
                results.append(_deferred(part.id, part.estimate_min, part.hf_repo))
                continue
            ctx = Ctx(
                repo=repo,
                forge_out=forge_out,
                plan=plan,
                part=part,
                token=token,
                layout=layout,
            )
            results.append(_dispatch(ctx))
    finally:
        manifest_json, manifest_md = _write_manifest(forge_out, plan, results)
    return {
        "job": plan.id,
        "estimates_min": {part.id: part.estimate_min for part in plan.parts},
        "total_min": plan.total_min(),
        "parts": results,
        "manifest": manifest_json,
        "models": manifest_md,
        "copy_back": [manifest_json, manifest_md],
    }
