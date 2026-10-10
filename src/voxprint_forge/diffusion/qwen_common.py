"""Stage markers for part D. This is not the three-part Ctx a running canvas job imported."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

from voxprint_forge.diffusion.qwen_plan import QwenPlan
from voxprint_forge.disk import ensure_free
from voxprint_forge.layout import Layout, read_marker, write_marker
from voxprint_forge.progress import append_progress
from voxprint_forge.statusfile import beating, set_stage, update_item

_SILENT = frozenset({"convert", "quality", "upload", "upload-deferred"})


@dataclass
class QwenCtx:
    repo: Path
    forge_out: Path
    plan: QwenPlan
    token: str | None
    layout: Layout
    defer_upload: bool = False

    @property
    def upstream_root(self) -> Path:
        return self.forge_out / "work" / self.plan.id / "upstream"

    @property
    def source_dir(self) -> Path:
        return self.upstream_root / self.plan.source.id

    @property
    def output(self) -> Path:
        return self.forge_out / "work" / self.plan.id / self.plan.output_name

    def marker(self, stage: str) -> Path:
        return self.layout.stage(self.plan.id, f"qwen-{stage}")

    def done(self, stage: str) -> bool:
        return read_marker(self.marker(stage)) is not None


def run_stage(ctx: QwenCtx, name: str, fn: Callable[[], dict[str, object]]) -> dict[str, object]:
    set_stage(name)
    existing = read_marker(ctx.marker(name))
    if existing is not None:
        return existing
    if name in _SILENT:
        beat = beating(ctx.forge_out, "qwen", name, eta_min=float(ctx.plan.estimate_min))
    else:
        beat = nullcontext()
    with beat:
        payload = fn()
    body = {"stage": f"qwen-{name}", **payload}
    write_marker(ctx.marker(name), body)
    return body


def check_disk(ctx: QwenCtx) -> None:
    set_stage("disk")
    need = ctx.plan.min_free_gb * (1024**3)
    ensure_free(ctx.forge_out, need, "qwen")


def progress(ctx: QwenCtx, stage: str, pct: int, eta_min: float) -> None:
    set_stage(stage)
    append_progress(ctx.forge_out / "progress.log", model="qwen", stage=stage, pct=pct, eta_min=eta_min)
    update_item(ctx.forge_out, "qwen", stage=stage, pct=pct, eta_min=eta_min, state="RUNNING")
