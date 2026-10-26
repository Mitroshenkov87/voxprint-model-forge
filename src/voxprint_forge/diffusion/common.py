"""Shared paths, stage markers, and disk checks for one canvas part."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from voxprint_forge.diffusion.names import NOTICE_TEXT
from voxprint_forge.diffusion.planload import CanvasPlan, Part
from voxprint_forge.diffusion.retain import consumers_for, release_consumer, source_store
from voxprint_forge.disk import ensure_free
from voxprint_forge.errors import StageError
from voxprint_forge.layout import Layout, read_marker, write_marker
from voxprint_forge.progress import append_progress
from voxprint_forge.reclaim import delete_logged, hub_cache
from voxprint_forge.statusfile import beating, set_stage, update_item

_SILENT = frozenset({"convert", "quality", "upload", "upload-deferred"})


@dataclass
class Ctx:
    """Paths, plan, and host for one canvas part."""
    repo: Path
    forge_out: Path
    plan: CanvasPlan
    part: Part
    token: str | None
    layout: Layout
    defer_upload: bool = False

    @property
    def upstream(self) -> Path:
        """Scratch directory of upstream weights for this part."""
        return self.forge_out / "work" / self.plan.id / "upstream" / self.part.id

    @property
    def output(self) -> Path:
        """Directory of files this part publishes."""
        return self.forge_out / "work" / self.plan.id / self.part.output_name

    def marker(self, stage: str) -> Path:
        """Marker path for one stage of this part."""
        return self.layout.stage(self.plan.id, f"{self.part.id}-{stage}")

    def done(self, stage: str) -> bool:
        """True when this stage's marker already exists."""
        return read_marker(self.marker(stage)) is not None


def run_stage(ctx: Ctx, name: str, fn: Callable[[], dict[str, object]]) -> dict[str, object]:
    """Run a stage unless its marker says it already finished."""
    set_stage(name)
    existing = read_marker(ctx.marker(name))
    if existing is not None:
        return existing
    beat = (
        beating(ctx.forge_out, ctx.part.id, name, eta_min=float(ctx.part.estimate_min))
        if name in _SILENT
        else nullcontext()
    )
    with beat:
        payload = fn()
    body = {"stage": f"{ctx.part.id}-{name}", **payload}
    write_marker(ctx.marker(name), body)
    return body


def check_disk(ctx: Ctx) -> None:
    """Reclaim leftovers, then raise when the part's disk budget is still not free."""
    set_stage("disk")
    need = ctx.part.min_free_gb * (1024**3)
    ensure_free(
        ctx.forge_out,
        need,
        ctx.part.id,
        forge_out=ctx.forge_out,
        cache_dir=hub_cache(),
        plan=ctx.plan,
    )


def write_notice(output: Path) -> None:
    """Write the licence notice next to published weights."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "NOTICE").write_text(NOTICE_TEXT, encoding="utf-8", newline="\n")


def write_json(path: Path, payload: object) -> None:
    """Write JSON with a trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def progress(ctx: Ctx, stage: str, pct: int, eta_min: float) -> None:
    """Record this stage's percent and remaining minutes."""
    set_stage(stage)
    append_progress(ctx.forge_out / "progress.log", model=ctx.part.id, stage=stage, pct=pct, eta_min=eta_min)
    update_item(ctx.forge_out, ctx.part.id, stage=stage, pct=pct, eta_min=eta_min, state="RUNNING")


def delete_intermediates(path: Path) -> None:
    """Remove a scratch directory if it exists."""
    if path.exists():
        shutil.rmtree(path)


def shared_sources(ctx: Ctx) -> Path:
    return source_store(ctx.forge_out, ctx.plan.id)


def finish_part(ctx: Ctx) -> None:
    """Drop this part's scratch tree. Shared sources stay until every consumer has released."""
    delete_intermediates(ctx.upstream)
    on_mirror_verified(ctx, "kaggle")


def on_mirror_verified(ctx: Ctx, target: str) -> None:
    """Release this part's sources. Drop scratch once staging is no longer in use."""
    for source in ctx.part.sources:
        release_consumer(
            ctx.forge_out,
            ctx.plan.id,
            source.id,
            ctx.part.id,
            consumers_for(ctx.plan, source.id),
        )
    if target == "hf":
        if ctx.upstream.is_dir():
            for child in list(ctx.upstream.iterdir()):
                if child.name == "kaggle-staging":
                    continue
                delete_logged(child)
        return
    delete_logged(ctx.upstream)


def source_dir(ctx: Ctx, source_id: str) -> Path:
    """Shared or legacy directory for one downloaded source."""
    shared = shared_sources(ctx) / source_id
    if shared.is_dir():
        return shared
    path = ctx.upstream / source_id
    if not path.is_dir():
        raise StageError(f"missing download {shared}", code="weights_missing")
    return path


def token_from_env() -> str | None:
    """``HF_TOKEN`` when it is set, otherwise None."""
    value = os.environ.get("HF_TOKEN", "").strip()
    return value or None
