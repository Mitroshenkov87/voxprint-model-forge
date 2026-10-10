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
from voxprint_forge.reclaim import delete_logged, hub_cache
from voxprint_forge.statusfile import beating, set_stage, update_item

_SILENT = frozenset({"convert", "quality", "upload", "upload-deferred"})


@dataclass
class QwenCtx:
    """Paths and plan for the Qwen-Image part."""
    repo: Path
    forge_out: Path
    plan: QwenPlan
    token: str | None
    layout: Layout
    defer_upload: bool = False

    @property
    def upstream_root(self) -> Path:
        """Root directory of this part's upstream download."""
        return self.forge_out / "work" / self.plan.id / "upstream"

    @property
    def source_dir(self) -> Path:
        """Shared source directory declared by the plan."""
        return self.upstream_root / self.plan.source.id

    @property
    def output(self) -> Path:
        """Directory of files this part publishes."""
        return self.forge_out / "work" / self.plan.id / self.plan.output_name

    def marker(self, stage: str) -> Path:
        """Marker path for one stage of this part."""
        return self.layout.stage(self.plan.id, f"qwen-{stage}")

    def done(self, stage: str) -> bool:
        """True when this stage's marker already exists."""
        return read_marker(self.marker(stage)) is not None


def run_stage(ctx: QwenCtx, name: str, fn: Callable[[], dict[str, object]]) -> dict[str, object]:
    """Run a stage unless its marker says it already finished."""
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
    """Reclaim leftovers, then raise when the part's disk budget is still not free."""
    set_stage("disk")
    need = ctx.plan.min_free_gb * (1024**3)
    ensure_free(ctx.forge_out, need, "qwen", forge_out=ctx.forge_out, cache_dir=hub_cache())


def on_qwen_mirror(ctx: QwenCtx, target: str) -> None:
    """Delete the Qwen download once a mirror has the finals. Keep Kaggle staging until then."""
    delete_logged(ctx.source_dir)
    delete_logged(ctx.upstream_root / "upstream-sha256.json")
    if target == "hf":
        return
    delete_logged(ctx.upstream_root / "kaggle-staging")


def progress(ctx: QwenCtx, stage: str, pct: int, eta_min: float) -> None:
    """Record this stage's percent and remaining minutes."""
    set_stage(stage)
    append_progress(ctx.forge_out / "progress.log", model="qwen", stage=stage, pct=pct, eta_min=eta_min)
    update_item(ctx.forge_out, "qwen", stage=stage, pct=pct, eta_min=eta_min, state="RUNNING")
