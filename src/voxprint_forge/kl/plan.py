"""Load ``plans/kl-synth4.toml``. The file names repos and quants, not prompts."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from voxprint_forge.errors import PlanError

SourceFormat = Literal["safetensors", "gguf"]
_FORMATS = ("safetensors", "gguf")


@dataclass(frozen=True)
class QuantRef:
    """One GGUF to score against the F16 base."""

    quant: str
    repo: str
    revision: str
    pattern: str
    ours: bool


@dataclass(frozen=True)
class KlJob:
    """One L-job: a BF16 (or already-F16) source and the quants to compare."""

    id: str
    bf16_repo: str
    bf16_revision: str
    source_format: SourceFormat
    f16_file: str
    expect_mtp: bool
    index_gap: str
    note: str
    quants: tuple[QuantRef, ...]
    chunks: int
    ctx: int
    fit_ctx: int
    headroom_gb: float
    budgets_gb: tuple[float, ...]
    kv_cache: str


@dataclass(frozen=True)
class KlPlan:
    """Every L-job in the synth4 KL plan."""

    path: Path
    jobs: tuple[KlJob, ...]

    def job(self, job_id: str) -> KlJob:
        """Return one job by id."""
        for item in self.jobs:
            if item.id == job_id:
                return item
        known = ", ".join(item.id for item in self.jobs)
        raise PlanError(f"unknown KL job {job_id} (known: {known})", code="kl_plan")


def load_kl_plan(path: Path) -> KlPlan:
    """Read a KL plan. A missing or malformed file is a plan error."""
    if not path.is_file():
        raise PlanError(f"missing KL plan {path.name}", code="kl_plan")
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise PlanError(f"KL plan is not valid TOML: {exc}", code="kl_plan") from exc
    defaults = payload.get("defaults") if isinstance(payload.get("defaults"), dict) else {}
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise PlanError("KL plan has no jobs", code="kl_plan")
    jobs = tuple(_job(item, defaults) for item in raw_jobs)
    ids = [item.id for item in jobs]
    if len(ids) != len(set(ids)):
        raise PlanError("KL plan repeats a job id", code="kl_plan")
    return KlPlan(path=path, jobs=jobs)


def _job(raw: object, defaults: dict[str, object]) -> KlJob:
    if not isinstance(raw, dict):
        raise PlanError("KL job is not a table", code="kl_plan")
    job_id = _str(raw, "id")
    fmt = _choice(str(raw.get("source_format", "safetensors")), _FORMATS, f"{job_id} source_format")
    quants_raw = raw.get("quants")
    if not isinstance(quants_raw, list) or not quants_raw:
        raise PlanError(f"{job_id} has no quants", code="kl_plan")
    quants = tuple(_quant(item, job_id) for item in quants_raw)
    labels = [item.quant for item in quants]
    if len(labels) != len(set(labels)):
        raise PlanError(f"{job_id} repeats a quant", code="kl_plan")
    f16_file = str(raw.get("f16_file", ""))
    source_format = _format(fmt, job_id)
    if source_format == "gguf" and not f16_file:
        raise PlanError(f"{job_id} gguf source needs f16_file", code="kl_plan")
    budgets = _floats(raw.get("budgets_gb", defaults.get("budgets_gb", [16, 24])), f"{job_id} budgets_gb")
    return KlJob(
        id=job_id,
        bf16_repo=_str(raw, "bf16_repo"),
        bf16_revision=_str(raw, "bf16_revision"),
        source_format=source_format,
        f16_file=f16_file,
        expect_mtp=bool(raw.get("expect_mtp", False)),
        index_gap=str(raw.get("index_gap", "")),
        note=str(raw.get("note", "")),
        quants=quants,
        chunks=_int(raw.get("chunks", defaults.get("chunks", 100)), f"{job_id} chunks"),
        ctx=_int(raw.get("ctx", defaults.get("ctx", 512)), f"{job_id} ctx"),
        fit_ctx=_int(raw.get("fit_ctx", defaults.get("fit_ctx", 8192)), f"{job_id} fit_ctx"),
        headroom_gb=_float(raw.get("headroom_gb", defaults.get("headroom_gb", 3.0)), f"{job_id} headroom_gb"),
        budgets_gb=tuple(budgets),
        kv_cache=str(raw.get("kv_cache", defaults.get("kv_cache", "q8_0"))),
    )


def _quant(raw: object, job_id: str) -> QuantRef:
    if not isinstance(raw, dict):
        raise PlanError(f"{job_id} quant is not a table", code="kl_plan")
    label = _str(raw, "quant")
    ours = raw.get("ours", True)
    if not isinstance(ours, bool):
        raise PlanError(f"{job_id} {label} ours is not a boolean", code="kl_plan")
    return QuantRef(
        quant=label,
        repo=_str(raw, "repo"),
        revision=_str(raw, "revision"),
        pattern=_str(raw, "pattern"),
        ours=ours,
    )


def _format(value: str, job_id: str) -> SourceFormat:
    if value == "safetensors":
        return "safetensors"
    if value == "gguf":
        return "gguf"
    raise PlanError(f"{job_id} source_format must be one of {', '.join(_FORMATS)}", code="kl_plan")


def _choice(value: str, allowed: tuple[str, ...], label: str) -> str:
    if value in allowed:
        return value
    raise PlanError(f"{label} must be one of {', '.join(allowed)}", code="kl_plan")


def _str(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise PlanError(f"KL plan is missing {key}", code="kl_plan")
    return value.strip()


def _int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError(f"{label} must be an integer", code="kl_plan")
    return value


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanError(f"{label} must be a number", code="kl_plan")
    return float(value)


def _floats(value: object, label: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise PlanError(f"{label} must be a list", code="kl_plan")
    return [_float(item, label) for item in value]
