"""Resume file and the public summary. Neither file contains prompt text."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from voxprint_forge.compare.cases import Case
from voxprint_forge.compare.codes import CompareSetup
from voxprint_forge.compare.metrics import PairScore, mean, passes
from voxprint_forge.compare.plan import Arm, Job, kind_label
from voxprint_forge.compare.privacy import assert_mapping_clean, assert_text_clean

PAIRS_NAME = "pairs.jsonl"
TIMINGS_NAME = "timings.jsonl"
SUMMARY_JSON = "summary.json"
SUMMARY_MD = "summary.md"


def pair_key(job_id: str, candidate_id: str, case: Case) -> str:
    """Stable id for one original-versus-candidate image. It contains no prompt text."""
    control = case.control or "-"
    return f"{job_id}:{candidate_id}:{case.ordinal}:{case.prompt_id}:{case.seed}:{case.width}x{case.height}:{control}"


def public_case(case: Case) -> dict[str, object]:
    """Fields of a case that are safe to write down."""
    return {
        "ordinal": case.ordinal,
        "prompt_id": case.prompt_id,
        "seed": case.seed,
        "width": case.width,
        "height": case.height,
        "control": case.control,
        "source": case.source,
        "warmup": case.warmup,
    }


def pair_record(
    *,
    job: Job,
    candidate: Arm,
    case: Case,
    score: PairScore,
    seconds: float | None,
    vram_torch_bytes: int | None,
    vram_smi_bytes: int | None,
) -> dict[str, object]:
    """One JSONL row. Prompt text is not a field."""
    return {
        "pair_id": pair_key(job.id, candidate.id, case),
        "job": job.id,
        "reference_arm": job.reference,
        "candidate_arm": candidate.id,
        "seconds": seconds,
        "vram_torch_bytes": vram_torch_bytes,
        "vram_smi_bytes": vram_smi_bytes,
        "metrics": dict(score),
        **public_case(case),
    }


def timing_id(arm_id: str, ordinal: int) -> str:
    """Stable id for one arm's image. It contains no prompt text."""
    return f"{arm_id}:{ordinal}"


def load_timings(path: Path, secrets: Sequence[str]) -> dict[str, dict[str, object]]:
    """Read per-image timings keyed by arm and ordinal. Prompt text is refused."""
    if not path.is_file():
        return {}
    found: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        assert_text_clean(line, secrets, where=f"{path.name}:{line_number}")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CompareSetup(f"{path.name} line {line_number} is not JSON", code="compare_resume") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("arm"), str):
            raise CompareSetup(f"{path.name} line {line_number} has no arm", code="compare_resume")
        ordinal = payload.get("ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int):
            raise CompareSetup(f"{path.name} line {line_number} has no ordinal", code="compare_resume")
        found[timing_id(str(payload["arm"]), ordinal)] = payload
    return found


def append_timing(path: Path, record: Mapping[str, object], secrets: Sequence[str]) -> None:
    """Append one arm timing. The line is refused when it contains prompt text."""
    assert_mapping_clean(record, secrets, where=path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")


def load_pairs(path: Path, secrets: Sequence[str]) -> dict[str, dict[str, object]]:
    """Read finished pairs keyed by ``pair_id``. A leaked prompt is a setup error."""
    if not path.is_file():
        return {}
    found: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        assert_text_clean(line, secrets, where=f"{path.name}:{line_number}")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CompareSetup(f"{path.name} line {line_number} is not JSON", code="compare_resume") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("pair_id"), str):
            raise CompareSetup(f"{path.name} line {line_number} has no pair_id", code="compare_resume")
        found[str(payload["pair_id"])] = payload
    return found


def append_pair(path: Path, record: Mapping[str, object], secrets: Sequence[str]) -> None:
    """Append one pair. The line is refused when it contains prompt text."""
    assert_mapping_clean(record, secrets, where=path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")


def summarise(
    job: Job,
    pairs: Sequence[Mapping[str, object]],
    *,
    component_bytes: Mapping[str, Mapping[str, int]],
    missing_sets: Sequence[str],
    case_count: int,
    secrets: Sequence[str],
    timings: Sequence[Mapping[str, object]] = (),
    int4_loads: Sequence[Mapping[str, str]] | None = None,
    gpu: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Aggregate non-warmup pairs. New metrics are reported and do not gate."""
    arms: list[dict[str, object]] = []
    failed: list[str] = []
    for candidate in job.candidates():
        rows = [item for item in pairs if item.get("candidate_arm") == candidate.id and item.get("warmup") is not True]
        scored = _scored(rows)
        if scored is None:
            raise CompareSetup(f"{job.id} arm {candidate.id} has no scored pairs", code="compare_empty")
        lpips_mean = scored["lpips_mean"]
        drop_mean = scored["clip_t_drop_mean"]
        if lpips_mean is None or drop_mean is None:
            raise CompareSetup(f"{job.id} arm {candidate.id} has no scored pairs", code="compare_empty")
        ok = passes(
            lpips_mean,
            drop_mean,
            lpips_max=job.lpips_max,
            clip_drop_max=job.clip_drop_max,
            lpips_gate=job.lpips_gate,
        )
        if not ok:
            failed.append(candidate.id)
        arms.append(
            {
                "id": candidate.id,
                "quant": candidate.quant,
                "compute": candidate.compute,
                "offload": candidate.offload,
                "repo": candidate.repo,
                "revision": candidate.revision,
                "passed": ok,
                "samples": len(rows),
                "component_bytes": dict(component_bytes.get(candidate.id, {})),
                **scored,
                "seconds_per_image": _seconds(rows),
                "vram_torch_peak_bytes": _peak(rows, "vram_torch_bytes"),
                "vram_smi_peak_bytes": _peak(rows, "vram_smi_bytes"),
            }
        )
    skipped = [
        {"id": item.id, "reason": item.skip_reason or "disabled"}
        for item in job.arms
        if not item.enabled
    ]
    payload: dict[str, object] = {
        "job": job.id,
        "kind": job.kind,
        "model": kind_label(job.kind),
        "steps": job.steps,
        "guidance": job.guidance,
        "reference_arm": job.reference,
        "reference": _reference_speed(job, timings),
        "passed": not failed,
        "failed_arms": failed,
        "lpips_max": job.lpips_max,
        "clip_drop_max": job.clip_drop_max,
        "lpips_gate": job.lpips_gate,
        "review_required": list(job.review),
        "cases": case_count,
        "pairs": len(pairs),
        "missing_private_sets": list(missing_sets),
        "skipped_arms": skipped,
        "arms": arms,
        "fp8_note": fp8_note(job, _gpu_name(gpu), _gpu_capability(gpu)),
        "int4": [dict(item) for item in int4_loads] if int4_loads else [],
        "prompt_ids_only": True,
        "note": job.note,
    }
    if gpu is not None:
        payload["gpu"] = dict(gpu)
    assert_mapping_clean(payload, secrets, where=SUMMARY_JSON)
    return payload


def write_summary(directory: Path, payload: Mapping[str, object], secrets: Sequence[str]) -> None:
    """Write ``summary.json`` and ``summary.md``."""
    assert_mapping_clean(payload, secrets, where=SUMMARY_JSON)
    text = render_markdown(payload)
    assert_text_clean(text, secrets, where=SUMMARY_MD)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / SUMMARY_JSON).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (directory / SUMMARY_MD).write_text(text, encoding="utf-8", newline="\n")


def _bar_lines(payload: Mapping[str, object]) -> list[str]:
    if payload.get("lpips_gate", True):
        return [
            f"Bar: mean LPIPS <= {payload.get('lpips_max')} and CLIP-T drop <= {payload.get('clip_drop_max')}.",
            "SSIM, PSNR, DINOv2 cosine, and CLIP-I are reported only.",
        ]
    lines = [
        f"Automatic bar (set mean): CLIP-T drop <= {payload.get('clip_drop_max')}.",
        "Same-seed LPIPS, SSIM, PSNR, DINOv2 cosine, and CLIP-I are reported only.",
    ]
    review = payload.get("review_required")
    if isinstance(review, list) and review:
        listed = "; ".join(map(str, review))
        lines.append(f"Also required before a pass is final (not scored by this run): {listed}.")
    return lines


def render_markdown(payload: Mapping[str, object]) -> str:
    """Human summary. Identifiers are prompt hashes, repos, and arm ids."""
    lines = [
        f"# {payload.get('job')}",
        "",
        f"Model: {payload.get('model')}",
        f"Steps: {payload.get('steps')}. Guidance: {payload.get('guidance')}.",
        f"Reference arm: {payload.get('reference_arm')}.",
        _reference_line(payload.get("reference")),
        f"Pass: {'yes' if payload.get('passed') else 'no'}.",
        *_bar_lines(payload),
        "",
    ]
    note = payload.get("fp8_note")
    if isinstance(note, str) and note:
        lines.extend([note, ""])
    lines.append(
        "| arm | quant | LPIPS | CLIP-T drop | SSIM | PSNR | DINOv2 | CLIP-I | s/img | pass |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    arms = payload.get("arms")
    if isinstance(arms, list):
        for arm in arms:
            if not isinstance(arm, dict):
                continue
            lines.append(
                "| {id} | {quant} | {lpips} | {drop} | {ssim} | {psnr} | {dino} | {clip_i} | {speed} | {ok} |".format(
                    id=arm.get("id"),
                    quant=arm.get("quant"),
                    lpips=_fmt(arm.get("lpips_mean")),
                    drop=_fmt(arm.get("clip_t_drop_mean")),
                    ssim=_fmt(arm.get("ssim_mean")),
                    psnr=_fmt(arm.get("psnr_mean")),
                    dino=_fmt(arm.get("dino_cosine_mean")),
                    clip_i=_fmt(arm.get("clip_i_mean")),
                    speed=_fmt(arm.get("seconds_per_image")),
                    ok="yes" if arm.get("passed") else "no",
                )
            )
    lines.append("")
    skipped = payload.get("skipped_arms")
    if isinstance(skipped, list) and skipped:
        lines.append("Skipped arms:")
        for item in skipped:
            if isinstance(item, dict):
                lines.append(f"- {item.get('id')}: {item.get('reason')}")
        lines.append("")
    loaded = payload.get("int4")
    if isinstance(loaded, list) and loaded:
        lines.append("INT4 loads:")
        for item in loaded:
            if isinstance(item, dict):
                lines.append(f"- {item.get('arm')}: nunchaku {item.get('nunchaku')}")
        lines.append("")
    missing = payload.get("missing_private_sets")
    if isinstance(missing, list) and missing:
        lines.append(
            "Private sets not loaded (FORGE_PRIVATE_PROMPTS unset, or no <set>/manifest.json; "
            "the log names the path tried): " + ", ".join(str(item) for item in missing) + "."
        )
        lines.append("")
    lines.append(f"Cases: {payload.get('cases')}. Pairs: {payload.get('pairs')}.")
    lines.append("Outputs use sha256 prompt ids. Prompt text is not recorded.")
    extra = payload.get("note")
    if isinstance(extra, str) and extra:
        lines.extend(["", extra])
    return "\n".join(lines).rstrip() + "\n"


def _reference_speed(job: Job, timings: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rows = [item for item in timings if item.get("arm") == job.reference and item.get("warmup") is not True]
    return {
        "id": job.reference,
        "samples": len(rows),
        "seconds_per_image": _seconds(rows),
        "vram_torch_peak_bytes": _peak(rows, "vram_torch_bytes"),
        "vram_smi_peak_bytes": _peak(rows, "vram_smi_bytes"),
    }


def _reference_line(reference: object) -> str:
    if not isinstance(reference, dict):
        return "Reference s/img: ."
    return (
        f"Reference s/img: {_fmt(reference.get('seconds_per_image'))} "
        f"over {reference.get('samples')} scored images."
    )


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return ""
    return str(value)


def fp8_note(job: Job, gpu_name: str, capability: str) -> str:
    """One sentence per enabled FP8 arm. The detected GPU name is used, never a fixed A100 label."""
    device = gpu_name or "this datacenter GPU"
    sm = _sm_label(capability)
    lines: list[str] = []
    for arm in job.enabled_arms():
        if arm.compute == "bf16-dequant":
            lines.append(f"{arm.id}: FP8 weights expanded to BF16; s/img and VRAM are BF16 numbers.")
        elif arm.compute == "fp8-native":
            lines.append(
                f"{arm.id}: FP8 E4M3 weights and dynamic per-tensor FP8 activations on FP8 tensor cores "
                f"(sm_{sm}). s/img is {device}, not an RTX 40/50 number; resident VRAM is representative."
            )
    return " ".join(lines)


def _gpu_name(gpu: Mapping[str, object] | None) -> str:
    if not isinstance(gpu, dict):
        return ""
    value = gpu.get("name")
    return value if isinstance(value, str) else ""


def _gpu_capability(gpu: Mapping[str, object] | None) -> str:
    if not isinstance(gpu, dict):
        return ""
    value = gpu.get("compute_capability")
    return value if isinstance(value, str) else ""


def _sm_label(capability: str) -> str:
    if not capability or "." not in capability:
        return capability or "unknown"
    major, minor = capability.split(".", 1)
    return f"{major}{minor}"


def _metric(row: Mapping[str, object], key: str) -> float | None:
    metrics = row.get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _scored(rows: Sequence[Mapping[str, object]]) -> dict[str, float | None] | None:
    if not rows:
        return None
    lpips = [_required(_metric(row, "lpips")) for row in rows]
    drops = [_required(_metric(row, "clip_t_drop")) for row in rows]
    lpips_mean = mean(lpips)
    drop_mean = mean(drops)
    if lpips_mean is None or drop_mean is None:
        return None
    return {
        "lpips_mean": lpips_mean,
        "clip_t_drop_mean": drop_mean,
        "ssim_mean": _optional_mean(rows, "ssim"),
        "psnr_mean": _optional_mean(rows, "psnr"),
        "dino_cosine_mean": _optional_mean(rows, "dino_cosine"),
        "clip_i_mean": _optional_mean(rows, "clip_i"),
        "clip_t_reference_mean": _optional_mean(rows, "clip_t_reference"),
        "clip_t_candidate_mean": _optional_mean(rows, "clip_t_candidate"),
    }


def _required(value: float | None) -> float:
    if value is None:
        raise CompareSetup("a scored pair is missing LPIPS or CLIP-T", code="compare_metric")
    return value


def _optional_mean(rows: Sequence[Mapping[str, object]], key: str) -> float | None:
    values = [item for item in (_metric(row, key) for row in rows) if item is not None]
    return mean(values)


def _seconds(rows: Sequence[Mapping[str, object]]) -> float | None:
    values: list[float] = []
    for row in rows:
        value = row.get("seconds")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return mean(values)


def _peak(rows: Sequence[Mapping[str, object]], key: str) -> int | None:
    peak: int | None = None
    for row in rows:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        peak = value if peak is None else max(peak, value)
    return peak
