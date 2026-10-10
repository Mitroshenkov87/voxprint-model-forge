"""Load plans/canvas-zimage-tiers-v1.toml. Parts run in file order."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Never

from voxprint_forge.errors import PlanError

PartId = Literal["base", "controlnet", "edit"]
PART_ORDER: tuple[PartId, ...] = ("base", "controlnet", "edit")


def _unexpected_part(part_id: Never) -> Never:
    raise PlanError(f"unhandled canvas part {part_id}", code="plan_shape")


@dataclass(frozen=True)
class Source:
    id: str
    repo: str
    revision: str
    licence: str
    files: tuple[str, ...]
    allow: tuple[str, ...]


@dataclass(frozen=True)
class Part:
    id: PartId
    estimate_min: int
    output_name: str
    hf_repo: str
    kaggle_slug: str
    min_free_gb: int
    sources: tuple[Source, ...]


@dataclass(frozen=True)
class CanvasPlan:
    id: str
    min_free_gb: int
    parts: tuple[Part, ...]

    def total_min(self) -> int:
        return sum(part.estimate_min for part in self.parts)

    def part(self, part_id: PartId) -> Part:
        if part_id == "base":
            return next(item for item in self.parts if item.id == "base")
        if part_id == "controlnet":
            return next(item for item in self.parts if item.id == "controlnet")
        if part_id == "edit":
            return next(item for item in self.parts if item.id == "edit")
        return _unexpected_part(part_id)


def _require_str(row: dict[str, object], key: str, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"{where} is missing {key}", code="plan_shape")
    return value.strip()


def _require_int(row: dict[str, object], key: str, where: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError(f"{where} {key} must be an integer", code="plan_shape")
    return value


def _strings(row: dict[str, object], key: str) -> tuple[str, ...]:
    value = row.get(key, [])
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise PlanError(f"{key} must be a list of strings", code="plan_shape")
    return tuple(value)


def _source(row: object, where: str) -> Source:
    if not isinstance(row, dict):
        raise PlanError(f"{where} source is not a table", code="plan_shape")
    revision = _require_str(row, "revision", where)
    if len(revision) != 40:
        raise PlanError(f"{where} revision {revision!r} is not a 40-character pin", code="plan_shape")
    return Source(
        id=_require_str(row, "id", where),
        repo=_require_str(row, "repo", where),
        revision=revision,
        licence=_require_str(row, "licence", where),
        files=_strings(row, "files"),
        allow=_strings(row, "allow"),
    )


def _part_id(value: str) -> PartId:
    if value == "base":
        return "base"
    if value == "controlnet":
        return "controlnet"
    if value == "edit":
        return "edit"
    raise PlanError(f"unknown canvas part {value!r}", code="plan_shape")


def _part(row: object) -> Part:
    if not isinstance(row, dict):
        raise PlanError("canvas part is not a table", code="plan_shape")
    part_id = _part_id(_require_str(row, "id", "part"))
    sources_raw = row.get("sources")
    if not isinstance(sources_raw, list) or not sources_raw:
        raise PlanError(f"{part_id} has no sources", code="plan_shape")
    return Part(
        id=part_id,
        estimate_min=_require_int(row, "estimate_min", part_id),
        output_name=_require_str(row, "output_name", part_id),
        hf_repo=_require_str(row, "hf_repo", part_id),
        kaggle_slug=_require_str(row, "kaggle_slug", part_id),
        min_free_gb=_require_int(row, "min_free_gb", part_id),
        sources=tuple(_source(item, part_id) for item in sources_raw),
    )


def load_canvas_plan(path: Path) -> CanvasPlan:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PlanError(f"cannot read {path}: {exc}", code="plan_io") from exc
    except tomllib.TOMLDecodeError as exc:
        raise PlanError(f"invalid TOML in {path}: {exc}", code="plan_toml") from exc
    run = raw.get("run")
    if not isinstance(run, dict):
        raise PlanError("canvas plan is missing [run]", code="plan_shape")
    parts_raw = raw.get("parts")
    if not isinstance(parts_raw, list) or not parts_raw:
        raise PlanError("canvas plan has no [[parts]]", code="plan_shape")
    parts = tuple(_part(item) for item in parts_raw)
    ids = tuple(item.id for item in parts)
    if ids != PART_ORDER:
        raise PlanError(f"canvas parts must be {PART_ORDER}, found {ids}", code="plan_shape")
    return CanvasPlan(
        id=_require_str(run, "id", "run"),
        min_free_gb=_require_int(run, "min_free_gb", "run"),
        parts=parts,
    )
