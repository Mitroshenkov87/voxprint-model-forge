"""Load ``plans/compare-synth4.toml``. The file names arms, not prompts."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Never, cast

from voxprint_forge.compare.codes import CompareSetup
from voxprint_forge.diffusion.policy import CLIP_DROP_MAX, LPIPS_MAX
from voxprint_forge.hashing import is_sha256

Kind = Literal["turbo", "creative", "control"]
Quant = Literal["fp32", "bf16", "fp8", "int4"]
Compute = Literal["bf16", "bf16-dequant", "int4", "fp8-native"]
PromptSet = Literal["base", "control"]
Role = Literal["reference", "candidate"]

_KINDS = ("turbo", "creative", "control")
_QUANTS = ("fp32", "bf16", "fp8", "int4")
_COMPUTES = ("bf16", "bf16-dequant", "int4", "fp8-native")
_ACT_SCALES = ("dynamic", "unit")
_PROMPTS = ("base", "control")
_ROLES = ("reference", "candidate")


def _unexpected_kind(kind: Never) -> Never:
    raise CompareSetup(f"unhandled compare kind {kind}", code="compare_plan")


@dataclass(frozen=True)
class Arm:
    """One weight variant: repo, revision, quant, and whether it runs.

    ``sha256`` is optional. When set, the one named file in ``files`` is
    checked after download.
    """

    id: str
    role: Role
    repo: str
    revision: str
    quant: Quant
    compute: Compute
    offload: bool
    files: tuple[str, ...]
    enabled: bool
    skip_reason: str
    base_repo: str
    base_revision: str
    reuse_job: str
    reuse_arm: str
    local: bool = False
    fp8_act_scale: str = "dynamic"
    requires_cc: tuple[str, ...] = ()
    sha256: str = ""


@dataclass(frozen=True)
class Job:
    """One synth4 compare job and the arms it scores."""

    id: str
    kind: Kind
    steps: int
    guidance: float
    control_scale: float
    prompts: PromptSet
    controls: tuple[str, ...]
    public_seeds: tuple[int, ...]
    private_sets: tuple[str, ...]
    private_limit: int
    width: int
    height: int
    reference: str
    arms: tuple[Arm, ...]
    lpips_max: float
    clip_drop_max: float
    note: str
    attention: str = "sdpa"
    requires_cc: tuple[str, ...] = ()
    lpips_gate: bool = True
    review: tuple[str, ...] = ()

    def arm(self, arm_id: str) -> Arm:
        """Return one arm by id."""
        for item in self.arms:
            if item.id == arm_id:
                return item
        raise CompareSetup(f"{self.id} has no arm {arm_id}", code="compare_plan")

    def enabled_arms(self) -> tuple[Arm, ...]:
        """Arms that this run will generate."""
        return tuple(item for item in self.arms if item.enabled)

    def reference_arm(self) -> Arm:
        """The enabled reference arm. A disabled reference is a setup error."""
        item = self.arm(self.reference)
        if item.role != "reference":
            raise CompareSetup(f"{self.id} arm {item.id} is not the reference", code="compare_plan")
        if not item.enabled:
            raise CompareSetup(f"{self.id} reference arm {item.id} is disabled", code="compare_plan")
        return item

    def candidates(self) -> tuple[Arm, ...]:
        """Enabled arms scored against the reference."""
        return tuple(item for item in self.enabled_arms() if item.role == "candidate")


@dataclass(frozen=True)
class ComparePlan:
    """Every job in the synth4 compare plan."""

    path: Path
    jobs: tuple[Job, ...]
    queue: tuple[str, ...] = ()

    def job(self, job_id: str) -> Job:
        """Return one job by id."""
        for item in self.jobs:
            if item.id == job_id:
                return item
        known = ", ".join(item.id for item in self.jobs)
        raise CompareSetup(f"unknown compare job {job_id} (known: {known})", code="compare_plan")


def load_compare_plan(path: Path) -> ComparePlan:
    """Read a compare plan. A missing or malformed file is a setup error."""
    if not path.is_file():
        raise CompareSetup(f"missing compare plan {path.name}", code="compare_plan")
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise CompareSetup(f"compare plan is not valid TOML: {exc}", code="compare_plan") from exc
    defaults = payload.get("defaults") if isinstance(payload.get("defaults"), dict) else {}
    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise CompareSetup("compare plan has no jobs", code="compare_plan")
    jobs = tuple(_job(item, defaults) for item in raw_jobs)
    ids = [item.id for item in jobs]
    if len(ids) != len(set(ids)):
        raise CompareSetup("compare plan repeats a job id", code="compare_plan")
    queue = _queue(defaults.get("queue"), ids)
    return ComparePlan(path=path, jobs=jobs, queue=queue)


def _job(raw: object, defaults: dict[str, object]) -> Job:
    if not isinstance(raw, dict):
        raise CompareSetup("compare job is not a table", code="compare_plan")
    job_id = _str(raw, "id")
    kind = _choice(str(raw.get("kind", "")), _KINDS, f"{job_id} kind")
    prompts = _choice(str(raw.get("prompts", "base")), _PROMPTS, f"{job_id} prompts")
    arms_raw = raw.get("arms")
    if not isinstance(arms_raw, list) or not arms_raw:
        raise CompareSetup(f"{job_id} has no arms", code="compare_plan")
    arms = tuple(_arm(item, job_id) for item in arms_raw)
    arm_ids = [item.id for item in arms]
    if len(arm_ids) != len(set(arm_ids)):
        raise CompareSetup(f"{job_id} repeats an arm id", code="compare_plan")
    reference = _str(raw, "reference")
    if reference not in arm_ids:
        raise CompareSetup(f"{job_id} reference {reference} is not an arm", code="compare_plan")
    seeds = _ints(raw.get("public_seeds", defaults.get("public_seeds", [0, 1])), f"{job_id} public_seeds")
    controls = _strs(raw.get("controls", ("canny", "depth", "pose")))
    return Job(
        id=job_id,
        kind=cast(Kind, kind),
        steps=_int(raw.get("steps"), f"{job_id} steps"),
        guidance=_float(raw.get("guidance"), f"{job_id} guidance"),
        control_scale=_float(raw.get("control_scale", defaults.get("control_scale", 0.75)), f"{job_id} control_scale"),
        prompts=cast(PromptSet, prompts),
        controls=tuple(controls),
        public_seeds=tuple(seeds),
        private_sets=tuple(_strs(raw.get("private_sets", ()))),
        private_limit=_int(raw.get("private_limit", defaults.get("private_limit", 50)), f"{job_id} private_limit"),
        width=_int(raw.get("width", defaults.get("width", 1024)), f"{job_id} width"),
        height=_int(raw.get("height", defaults.get("height", 1024)), f"{job_id} height"),
        reference=reference,
        arms=arms,
        lpips_max=_float(raw.get("lpips_max", defaults.get("lpips_max", LPIPS_MAX)), f"{job_id} lpips_max"),
        clip_drop_max=_float(
            raw.get("clip_drop_max", defaults.get("clip_drop_max", CLIP_DROP_MAX)),
            f"{job_id} clip_drop_max",
        ),
        note=str(raw.get("note", "")),
        attention=_attention(raw.get("attention", defaults.get("attention", "sdpa")), job_id),
        requires_cc=_cc_list(raw.get("requires_cc"), f"{job_id} requires_cc"),
        lpips_gate=_bool(raw.get("lpips_gate", defaults.get("lpips_gate", True)), f"{job_id} lpips_gate"),
        review=tuple(_strs(raw.get("review", defaults.get("review", ())))),
    )


def _bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise CompareSetup(f"{label} is not a boolean", code="compare_plan")
    return value


def _arm(raw: object, job_id: str) -> Arm:
    if not isinstance(raw, dict):
        raise CompareSetup(f"{job_id} arm is not a table", code="compare_plan")
    arm_id = _str(raw, "id")
    role = _choice(str(raw.get("role", "")), _ROLES, f"{job_id} arm {arm_id} role")
    quant = _choice(str(raw.get("quant", "bf16")), _QUANTS, f"{job_id} arm {arm_id} quant")
    compute = _choice(str(raw.get("compute", "bf16")), _COMPUTES, f"{job_id} arm {arm_id} compute")
    act_scale = _choice(str(raw.get("fp8_act_scale", "dynamic")), _ACT_SCALES, f"{job_id} arm {arm_id} fp8_act_scale")
    if compute == "fp8-native" and quant != "fp8":
        raise CompareSetup(f"{job_id} arm {arm_id} fp8-native requires quant fp8", code="compare_plan")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise CompareSetup(f"{job_id} arm {arm_id} enabled is not a boolean", code="compare_plan")
    return Arm(
        id=arm_id,
        role=cast(Role, role),
        repo=str(raw.get("repo", "")),
        revision=str(raw.get("revision", "")),
        quant=cast(Quant, quant),
        compute=cast(Compute, compute),
        offload=bool(raw.get("offload", False)),
        files=tuple(_strs(raw.get("files", ()))),
        enabled=enabled,
        skip_reason=str(raw.get("skip_reason", "")),
        base_repo=str(raw.get("base_repo", "")),
        base_revision=str(raw.get("base_revision", "")),
        reuse_job=str(raw.get("reuse_job", "")),
        reuse_arm=str(raw.get("reuse_arm", "")),
        local=_bool(raw.get("local", False), f"{job_id} arm {arm_id} local"),
        fp8_act_scale=act_scale,
        requires_cc=_cc_list(raw.get("requires_cc"), f"{job_id} arm {arm_id} requires_cc"),
        sha256=_optional_sha(raw.get("sha256"), f"{job_id} arm {arm_id} sha256"),
    )


def _optional_sha(value: object, label: str) -> str:
    if value is None or value == "":
        return ""
    if not isinstance(value, str) or not is_sha256(value):
        raise CompareSetup(f"{label} must be 64 lowercase hex digits", code="compare_plan")
    return value


def _queue(value: object, job_ids: list[str]) -> tuple[str, ...]:
    if value is None:
        return tuple(job_ids)
    rows = _strs(value)
    if not rows:
        raise CompareSetup("compare queue is empty", code="compare_plan")
    unknown = [item for item in rows if item not in job_ids]
    if unknown:
        raise CompareSetup(f"compare queue has unknown job {unknown[0]}", code="compare_plan")
    return tuple(rows)


def _attention(value: object, job_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompareSetup(f"{job_id} attention is empty", code="compare_plan")
    return value.strip()


def _cc_list(value: object, label: str) -> tuple[str, ...]:
    cleaned = tuple(item.strip() for item in _strs(value))
    if any(not item for item in cleaned):
        raise CompareSetup(f"{label} has an empty capability", code="compare_plan")
    return cleaned


def _bool(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    raise CompareSetup(f"{label} is not a boolean", code="compare_plan")


def _choice(value: str, allowed: tuple[str, ...], label: str) -> str:
    if value in allowed:
        return value
    raise CompareSetup(f"{label} must be one of {', '.join(allowed)}", code="compare_plan")


def _str(raw: dict[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise CompareSetup(f"compare plan is missing {key}", code="compare_plan")
    return value.strip()


def _int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompareSetup(f"{label} must be an integer", code="compare_plan")
    return value


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompareSetup(f"{label} must be a number", code="compare_plan")
    return float(value)


def _ints(value: object, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise CompareSetup(f"{label} must be a list", code="compare_plan")
    return [_int(item, label) for item in value]


def _strs(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise CompareSetup("compare plan list field is not a list", code="compare_plan")
    rows: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise CompareSetup("compare plan list field has a non-string", code="compare_plan")
        rows.append(item)
    return rows


def kind_label(kind: Kind) -> str:
    """Short label for a job kind."""
    if kind == "turbo":
        return "Z-Image-Turbo"
    if kind == "creative":
        return "Z-Image"
    if kind == "control":
        return "Z-Image-Turbo ControlNet Union 2.1"
    return _unexpected_kind(kind)
