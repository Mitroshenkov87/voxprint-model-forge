"""Shared paths, stage markers, and disk checks for one canvas part."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from voxprint_forge.diffusion.names import NOTICE_TEXT
from voxprint_forge.diffusion.planload import CanvasPlan, Part
from voxprint_forge.disk import ensure_free
from voxprint_forge.errors import StageError
from voxprint_forge.layout import Layout, read_marker, write_marker
from voxprint_forge.progress import append_progress


@dataclass
class Ctx:
    repo: Path
    forge_out: Path
    plan: CanvasPlan
    part: Part
    token: str | None
    layout: Layout
    defer_upload: bool = False

    @property
    def upstream(self) -> Path:
        return self.forge_out / "work" / self.plan.id / "upstream" / self.part.id

    @property
    def output(self) -> Path:
        return self.forge_out / "work" / self.plan.id / self.part.output_name

    def marker(self, stage: str) -> Path:
        return self.layout.stage(self.plan.id, f"{self.part.id}-{stage}")

    def done(self, stage: str) -> bool:
        return read_marker(self.marker(stage)) is not None


def run_stage(ctx: Ctx, name: str, fn: Callable[[], dict[str, object]]) -> dict[str, object]:
    existing = read_marker(ctx.marker(name))
    if existing is not None:
        return existing
    payload = fn()
    body = {"stage": f"{ctx.part.id}-{name}", **payload}
    write_marker(ctx.marker(name), body)
    return body


def check_disk(ctx: Ctx) -> None:
    need = ctx.part.min_free_gb * (1024**3)
    ensure_free(ctx.forge_out, need, ctx.part.id)


def write_notice(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "NOTICE").write_text(NOTICE_TEXT, encoding="utf-8")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def progress(ctx: Ctx, stage: str, pct: int, eta_min: float) -> None:
    append_progress(ctx.forge_out / "progress.log", model=ctx.part.id, stage=stage, pct=pct, eta_min=eta_min)


def delete_intermediates(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def source_dir(ctx: Ctx, source_id: str) -> Path:
    path = ctx.upstream / source_id
    if not path.is_dir():
        raise StageError(f"missing download {path}", code="weights_missing")
    return path


def token_from_env() -> str | None:
    value = os.environ.get("HF_TOKEN", "").strip()
    return value or None
