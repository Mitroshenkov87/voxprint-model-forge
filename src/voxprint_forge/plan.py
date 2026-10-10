"""Load plans/plan.toml. Disabled models stay in the file and are not run."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Never

from voxprint_forge.errors import PlanError
from voxprint_forge.fitmath import Arch
from voxprint_forge.licence import assert_repo_allowed, decide_licence

Mode = Literal["requant", "ablate"]
TierKind = Literal["reuse", "quantize"]
SourceFormat = Literal["gguf", "safetensors"]
FileRole = Literal["source", "reuse", "sidecar"]


@dataclass(frozen=True)
class PinnedFile:
    role: FileRole
    quant: str
    path: str
    sha256: str
    size: int
    repo: str
    revision: str


@dataclass(frozen=True)
class Tier:
    name: str
    budget_gb: float
    kind: TierKind
    quants: tuple[str, ...]


@dataclass(frozen=True)
class ModelPlan:
    id: str
    enabled: bool
    optional: bool
    estimate_min: int
    mode: Mode
    repo: str
    revision: str
    licence: str
    params_b: float
    credits: tuple[str, ...]
    base_model: str
    already_abliterated: bool
    source_format: SourceFormat
    arch: Arch
    files: tuple[PinnedFile, ...]
    tiers: tuple[Tier, ...]
    note: str = ""

    def source_files(self) -> tuple[PinnedFile, ...]:
        return tuple(item for item in self.files if item.role == "source")

    def reuse_files(self) -> tuple[PinnedFile, ...]:
        return tuple(item for item in self.files if item.role == "reuse")


@dataclass(frozen=True)
class Plan:
    name: str
    output_dir: Path
    wall_clock_budget_min: int
    headroom_gb: float
    imatrix_chunks: int
    imatrix_ctx: int
    refusal_limit: int
    ppl_chunks: int
    prose_samples: int
    contexts: tuple[int, ...]
    llama_cpp_tag: str
    llama_cpp_commit: str
    estimates_min: dict[str, int]
    artifact_host: str
    models: tuple[ModelPlan, ...]

    def enabled_models(self) -> tuple[ModelPlan, ...]:
        return tuple(model for model in self.models if model.enabled)


def load_plan(path: Path) -> Plan:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PlanError(f"cannot read {path}: {exc}", code="plan_io") from exc
    except tomllib.TOMLDecodeError as exc:
        raise PlanError(f"invalid TOML in {path}: {exc}", code="plan_toml") from exc
    run = raw.get("run")
    if not isinstance(run, dict):
        raise PlanError("plan is missing [run]", code="plan_shape")
    models_raw = raw.get("models")
    if not isinstance(models_raw, list) or not models_raw:
        raise PlanError("plan has no [[models]] entries", code="plan_shape")
    models = tuple(_model(item) for item in models_raw)
    for model in models:
        if model.enabled:
            _validate_enabled(model)
    output = Path(str(run.get("output_dir", "~/forge-out/work"))).expanduser()
    estimates = run.get("estimates_min") or {}
    if not isinstance(estimates, dict):
        raise PlanError("[run.estimates_min] must be a table", code="plan_shape")
    return Plan(
        name=str(run.get("name", path.stem)),
        output_dir=output,
        wall_clock_budget_min=int(run.get("wall_clock_budget_min", 150)),
        headroom_gb=float(run.get("headroom_gb", 3.0)),
        imatrix_chunks=int(run.get("imatrix_chunks", 8)),
        imatrix_ctx=int(run.get("imatrix_ctx", 512)),
        refusal_limit=int(run.get("refusal_limit", 4)),
        ppl_chunks=int(run.get("ppl_chunks", 1)),
        prose_samples=int(run.get("prose_samples", 1)),
        contexts=tuple(int(item) for item in run.get("contexts", [8192, 16384])),
        llama_cpp_tag=str(run.get("llama_cpp_tag", "b11476")),
        llama_cpp_commit=str(run.get("llama_cpp_commit", "")),
        estimates_min={str(key): int(value) for key, value in estimates.items()},
        artifact_host=str(run.get("artifact_host", "")),
        models=models,
    )


def format_estimates(plan: Plan) -> str:
    lines = [
        f"Plan {plan.name}",
        f"Wall-clock budget: {plan.wall_clock_budget_min} min",
        f"Headroom: {plan.headroom_gb:.1f} GB (weights + KV must fit under the card minus this)",
        "Estimates (minutes):",
    ]
    for key, minutes in plan.estimates_min.items():
        lines.append(f"  {key}: {minutes}")
    lines.append("Models:")
    for model in plan.models:
        state = "run" if model.enabled else "disabled"
        if model.optional and model.enabled:
            state = "optional"
        lines.append(
            f"  {model.id}: {state}, {model.mode}, {model.estimate_min} min, "
            f"{model.licence}, {model.repo}@{model.revision[:12]}"
        )
    required = sum(model.estimate_min for model in plan.enabled_models() if not model.optional)
    build = plan.estimates_min.get("bootstrap_build", 0)
    lines.append(f"Required path including build: {required + build} min")
    return "\n".join(lines)


def _model(raw: object) -> ModelPlan:
    if not isinstance(raw, dict):
        raise PlanError("[[models]] entry is not a table", code="plan_shape")
    repo = str(raw.get("repo", ""))
    licence = str(raw.get("licence", ""))
    if repo:
        assert_repo_allowed(repo)
    if raw.get("enabled", True):
        decide_licence(licence, repo=repo)
    mode = _mode(str(raw.get("mode", "")), str(raw.get("id", "")))
    source_format = _source_format(str(raw.get("source_format", "gguf")), str(raw.get("id", "")))
    arch_raw = raw.get("arch") or {}
    if not isinstance(arch_raw, dict):
        raise PlanError(f"{raw.get('id')} [arch] is not a table", code="plan_shape")
    files_raw = raw.get("files") or []
    tiers_raw = raw.get("tiers") or []
    if not isinstance(files_raw, list) or not isinstance(tiers_raw, list):
        raise PlanError(f"{raw.get('id')} files/tiers must be arrays", code="plan_shape")
    credits = raw.get("credits") or []
    if not isinstance(credits, list):
        raise PlanError(f"{raw.get('id')} credits must be a list", code="plan_shape")
    return ModelPlan(
        id=str(raw["id"]),
        enabled=bool(raw.get("enabled", True)),
        optional=bool(raw.get("optional", False)),
        estimate_min=int(raw.get("estimate_min", 0)),
        mode=mode,
        repo=repo,
        revision=str(raw.get("revision", "")),
        licence=licence,
        params_b=float(raw.get("params_b", 0)),
        credits=tuple(str(item) for item in credits),
        base_model=str(raw.get("base_model", "")),
        already_abliterated=bool(raw.get("already_abliterated", False)),
        source_format=source_format,
        arch=Arch(
            full_layers=int(arch_raw.get("full_layers", 0)),
            full_kv_heads=int(arch_raw.get("full_kv_heads", 0)),
            full_head_dim=int(arch_raw.get("full_head_dim", 0)),
            swa_layers=int(arch_raw.get("swa_layers", 0)),
            swa_kv_heads=int(arch_raw.get("swa_kv_heads", 0)),
            swa_head_dim=int(arch_raw.get("swa_head_dim", 0)),
            swa_window=int(arch_raw.get("swa_window", 0)),
        ),
        files=tuple(_file(item, repo, str(raw.get("revision", ""))) for item in files_raw),
        tiers=tuple(_tier(item) for item in tiers_raw),
        note=str(raw.get("note", "")),
    )


def _file(raw: object, default_repo: str, default_revision: str) -> PinnedFile:
    if not isinstance(raw, dict):
        raise PlanError("[[models.files]] entry is not a table", code="plan_shape")
    role = _role(str(raw.get("role", "")))
    repo = str(raw.get("repo", default_repo))
    if repo:
        assert_repo_allowed(repo)
    sha = str(raw.get("sha256", "")).lower()
    if role == "sidecar":
        if sha and not _sha_ok(sha):
            raise PlanError(f"sidecar {raw.get('path')} sha256 is not 64 hex chars", code="plan_sha")
    elif not _sha_ok(sha):
        raise PlanError(f"pinned file {raw.get('path')} needs a sha256", code="plan_sha")
    return PinnedFile(
        role=role,
        quant=str(raw.get("quant", "")),
        path=str(raw["path"]),
        sha256=sha,
        size=int(raw.get("size", 0)),
        repo=repo,
        revision=str(raw.get("revision", default_revision)),
    )


def _tier(raw: object) -> Tier:
    if not isinstance(raw, dict):
        raise PlanError("[[models.tiers]] entry is not a table", code="plan_shape")
    kind = _kind(str(raw.get("kind", "")))
    quants = raw.get("quants") or []
    if not isinstance(quants, list):
        raise PlanError("tier quants must be a list", code="plan_shape")
    return Tier(
        name=str(raw["name"]),
        budget_gb=float(raw["budget_gb"]),
        kind=kind,
        quants=tuple(str(item) for item in quants),
    )


def _sha_ok(sha: str) -> bool:
    return len(sha) == 64 and all(char in "0123456789abcdef" for char in sha)


def _mode(value: str, model_id: str) -> Mode:
    if value == "requant":
        return "requant"
    if value == "ablate":
        return "ablate"
    raise PlanError(f"{model_id} mode must be requant or ablate", code="plan_mode")


def _source_format(value: str, model_id: str) -> SourceFormat:
    if value == "gguf":
        return "gguf"
    if value == "safetensors":
        return "safetensors"
    raise PlanError(f"{model_id} source_format must be gguf or safetensors", code="plan_format")


def _role(value: str) -> FileRole:
    if value == "source":
        return "source"
    if value == "reuse":
        return "reuse"
    if value == "sidecar":
        return "sidecar"
    raise PlanError("file role must be source, reuse, or sidecar", code="plan_role")


def _kind(value: str) -> TierKind:
    if value == "reuse":
        return "reuse"
    if value == "quantize":
        return "quantize"
    raise PlanError("tier kind must be reuse or quantize", code="plan_tier")


def _validate_enabled(model: ModelPlan) -> None:
    if not model.revision:
        raise PlanError(f"{model.id} is missing a revision", code="plan_revision")
    if not model.tiers:
        raise PlanError(f"{model.id} has no tiers", code="plan_tier")
    if model.mode == "requant" and model.source_format != "gguf":
        raise PlanError(f"{model.id} requant source must be gguf", code="plan_format")
    if model.mode == "ablate" and model.source_format != "safetensors":
        raise PlanError(f"{model.id} ablate source must be safetensors", code="plan_format")
    if any(tier.kind == "quantize" for tier in model.tiers) and not model.source_files():
        raise PlanError(f"{model.id} quantize tier needs a source file", code="plan_source")
    for tier in model.tiers:
        if not tier.quants:
            raise PlanError(f"{model.id} tier {tier.name} has no quants", code="plan_tier")
        if tier.kind == "reuse":
            have = {item.quant for item in model.reuse_files()}
            if have.isdisjoint(tier.quants):
                raise PlanError(
                    f"{model.id} tier {tier.name} reuses none of {tier.quants}",
                    code="plan_reuse",
                )
        elif tier.kind == "quantize":
            continue
        else:
            _unexpected(tier.kind)


def _unexpected(value: Never) -> None:
    raise PlanError(f"unhandled value {value}", code="plan_shape")
