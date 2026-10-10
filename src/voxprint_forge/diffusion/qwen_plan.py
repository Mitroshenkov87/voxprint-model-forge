"""Load plans/canvas-qwen-image-2512-v1.toml. This is not the three-part canvas plan."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from voxprint_forge.diffusion.planload import Source
from voxprint_forge.diffusion.qwen_spec import INT4_MODE
from voxprint_forge.errors import PlanError


@dataclass(frozen=True)
class QwenPlan:
    """The loaded Qwen-Image plan."""
    id: str
    min_free_gb: int
    int4: str
    estimate_min: int
    estimate_min_reuse: int
    estimate_min_recompute: int
    encoder_estimate_min: int
    source: Source
    output_name: str
    hf_repo: str
    kaggle_slug: str
    edit_job_id: str
    edit_output_name: str
    edit_hf_repo: str
    edit_revision: str
    edit_encoder_fp8: str
    encoder_pins: dict[str, str]
    consumes_plan: str = ""
    consumes_sources: tuple[str, ...] = ()


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


def _revision(value: str, where: str) -> str:
    if len(value) != 40:
        raise PlanError(f"{where} revision {value!r} is not a 40-character pin", code="plan_shape")
    return value


def _sha(value: str, where: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise PlanError(f"{where} sha256 {value!r} is not 64 hex characters", code="plan_shape")
    return value


def _table(raw: dict[str, object], key: str) -> dict[str, object]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise PlanError(f"qwen plan is missing [{key}]", code="plan_shape")
    return value


def _pins(row: dict[str, object]) -> dict[str, str]:
    found: dict[str, str] = {}
    for key, value in row.items():
        if not isinstance(key, str) or not key.startswith("text_encoder/"):
            raise PlanError(f"encoder pin {key!r} must start with text_encoder/", code="plan_shape")
        if not isinstance(value, str):
            raise PlanError(f"encoder pin {key} is not a string", code="plan_shape")
        found[key] = _sha(value.strip(), key)
    if not found:
        raise PlanError("qwen plan has no encoder pins", code="plan_shape")
    return found


def _source(row: dict[str, object]) -> Source:
    revision = _revision(_require_str(row, "revision", "source"), "source")
    licence = _require_str(row, "licence", "source")
    if licence != "apache-2.0":
        raise PlanError(f"source licence {licence!r} is not apache-2.0", code="plan_shape")
    return Source(
        id=_require_str(row, "id", "source"),
        repo=_require_str(row, "repo", "source"),
        revision=revision,
        licence=licence,
        files=(),
        allow=(),
    )


def load_qwen_plan(path: Path) -> QwenPlan:
    """Load and check a Qwen-Image plan TOML file."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PlanError(f"cannot read {path}: {exc}", code="plan_io") from exc
    except tomllib.TOMLDecodeError as exc:
        raise PlanError(f"invalid TOML in {path}: {exc}", code="plan_toml") from exc
    if not isinstance(raw, dict):
        raise PlanError("qwen plan is not a table", code="plan_shape")
    run = _table(raw, "run")
    source_row = _table(raw, "source")
    edit = _table(raw, "edit")
    pins = _pins(_table(raw, "encoder_pins"))
    consumes_raw = raw.get("consumes")
    consumes_plan = ""
    consumes_sources: tuple[str, ...] = ()
    if consumes_raw is not None:
        if not isinstance(consumes_raw, dict):
            raise PlanError("qwen plan [consumes] is not a table", code="plan_shape")
        consumes_plan = _require_str(consumes_raw, "plan", "consumes")
        listed = consumes_raw.get("sources")
        if not isinstance(listed, list) or not listed or not all(isinstance(item, str) and item for item in listed):
            raise PlanError("consumes sources must be a list of source ids", code="plan_shape")
        consumes_sources = tuple(listed)
    int4 = _require_str(run, "int4", "run")
    if int4 != INT4_MODE:
        raise PlanError(
            f"int4 mode {int4!r} is not {INT4_MODE}; this rental does not run calibration",
            code="plan_shape",
        )
    return QwenPlan(
        id=_require_str(run, "id", "run"),
        min_free_gb=_require_int(run, "min_free_gb", "run"),
        int4=int4,
        estimate_min=_require_int(run, "estimate_min", "run"),
        estimate_min_reuse=_require_int(run, "estimate_min_reuse", "run"),
        estimate_min_recompute=_require_int(run, "estimate_min_recompute", "run"),
        encoder_estimate_min=_require_int(run, "encoder_estimate_min", "run"),
        source=_source(source_row),
        output_name=_require_str(source_row, "output_name", "source"),
        hf_repo=_require_str(source_row, "hf_repo", "source"),
        kaggle_slug=_require_str(source_row, "kaggle_slug", "source"),
        edit_job_id=_require_str(edit, "job_id", "edit"),
        edit_output_name=_require_str(edit, "output_name", "edit"),
        edit_hf_repo=_require_str(edit, "hf_repo", "edit"),
        edit_revision=_revision(_require_str(edit, "revision", "edit"), "edit"),
        edit_encoder_fp8=_require_str(edit, "encoder_fp8", "edit"),
        encoder_pins=pins,
        consumes_plan=consumes_plan,
        consumes_sources=consumes_sources,
    )
