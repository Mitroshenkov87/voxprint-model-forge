"""Run one compare job: generate each arm, score pairs, resume, reclaim."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from PIL import Image

from voxprint_forge.compare.cases import Case, Schedule, build_schedule
from voxprint_forge.compare.codes import EXIT_FAIL, EXIT_PASS, CompareNoGpu, CompareSetup
from voxprint_forge.compare.engine import fetch_arm, make_generator
from voxprint_forge.compare.metrics import PairScore
from voxprint_forge.compare.plan import Arm, Job
from voxprint_forge.compare.privacy import assert_text_clean, secret_strings
from voxprint_forge.compare.reclaim import component_bytes, reclaim_weights
from voxprint_forge.compare.results import (
    PAIRS_NAME,
    TIMINGS_NAME,
    append_pair,
    append_timing,
    load_pairs,
    load_timings,
    pair_key,
    pair_record,
    summarise,
    timing_id,
    write_summary,
)
from voxprint_forge.compare.score import make_scorer
from voxprint_forge.compare.vram import SmiSampler, reset_torch_peak, torch_peak_bytes

GenerateFn = Callable[[Arm, Case, Job], Image.Image]
ScoreFn = Callable[[Image.Image, Image.Image, Case], PairScore]
FetchFn = Callable[[Arm, Path], None]
SmiFn = Callable[[], int | None]


def cuda_available() -> bool:
    """True when this process can see a CUDA device."""
    import torch

    return bool(torch.cuda.is_available())


def run_compare(
    job: Job,
    results: Path,
    *,
    private_root: Path | None,
    generate: GenerateFn | None = None,
    score: ScoreFn | None = None,
    fetch: FetchFn | None = None,
    smi: SmiFn | None = None,
    cuda: bool | None = None,
) -> dict[str, object]:
    """Score every enabled candidate against the reference.

    ``generate`` and ``score`` are the live GPU path when omitted. A machine
    without CUDA then exits 3 before any download. The reference arm runs
    first and its PNGs stay on disk. Each later arm loads only after the
    previous pipeline is released. Scorers load after that release. Weights
    under ``results/weights`` are deleted at the end. Review images stay.
    """
    job = gate_local_arms(job)
    reference = job.reference_arm()
    candidates = job.candidates()
    if not candidates:
        skipped = ", ".join(item.id for item in job.arms if not item.enabled) or "none"
        raise CompareSetup(f"{job.id} has no runnable candidate arm (skipped: {skipped})", code="compare_plan")
    schedule = build_schedule(job, private_root)
    if not schedule.cases:
        raise CompareSetup(f"{job.id} has no prompts", code="compare_empty")
    secrets = secret_strings(
        (item.prompt for item in schedule.cases),
        (item.negative for item in schedule.cases),
    )
    if not any(not item.warmup for item in schedule.cases):
        raise CompareSetup(f"{job.id} has no scored image after warm-up", code="compare_empty")
    live = generate is None
    if live and not (cuda_available() if cuda is None else cuda):
        raise CompareNoGpu("compare needs a CUDA GPU", code="compare_gpu")
    results.mkdir(parents=True, exist_ok=True)
    weights = results / "weights"
    images = results / "images"
    pairs_path = results / PAIRS_NAME
    done = load_pairs(pairs_path, secrets)
    sizes = _prepare_weights(job, weights, fetch, live)
    _log(f"{job.id} cases={len(schedule.cases)} resumed={len(done)}", secrets)
    try:
        if live:
            generate, score = _live_hooks(job, weights)
        elif score is None:
            raise CompareSetup(f"{job.id} has no scorer", code="compare_setup")
        if generate is None:
            raise CompareSetup(f"{job.id} has no generator", code="compare_setup")
        timings_path = results / TIMINGS_NAME
        timings = load_timings(timings_path, secrets)
        _generate_pairs(
            job,
            reference,
            candidates,
            schedule,
            images,
            pairs_path,
            done,
            timings,
            timings_path,
            secrets,
            generate,
            score,
            smi,
        )
        payload = summarise(
            job,
            list(done.values()),
            component_bytes=sizes,
            missing_sets=schedule.missing_sets,
            case_count=len(schedule.cases),
            timings=list(timings.values()),
            secrets=secrets,
            int4_loads=list(getattr(generate, "int4_notes", ())),
        )
        write_summary(results, payload, secrets)
    finally:
        reclaim_weights(weights)
    exit_code = EXIT_PASS if payload["passed"] else EXIT_FAIL
    _log(f"{job.id} exit={exit_code}", secrets)
    return {
        "job": job.id,
        "exit_code": exit_code,
        "passed": payload["passed"],
        "pairs": payload["pairs"],
        "cases": payload["cases"],
        "summary": str(results / "summary.json"),
        "prompt_ids_only": True,
        "failed_arms": payload["failed_arms"],
        "missing_private_sets": payload["missing_private_sets"],
    }


def gate_local_arms(job: Job) -> Job:
    """Drop local FP8 arms when ``FORGE_FP8_DIR`` is unset. Other arms still run."""
    raw = os.environ.get("FORGE_FP8_DIR", "").strip()
    if raw and Path(raw).is_dir():
        return job
    arms = tuple(
        replace(item, enabled=False, skip_reason="FORGE_FP8_DIR is unset") if item.local and item.enabled else item
        for item in job.arms
    )
    return replace(job, arms=arms)


def _live_hooks(job: Job, weights: Path) -> tuple[GenerateFn, None]:
    """Build the generator now. The scorer waits until every pipeline is gone."""
    return make_generator(job, weights), None


def _prepare_weights(
    job: Job,
    weights: Path,
    fetch: FetchFn | None,
    live: bool,
) -> dict[str, dict[str, int]]:
    weights.mkdir(parents=True, exist_ok=True)
    sizes: dict[str, dict[str, int]] = {}
    arms = (job.reference_arm(), *job.candidates())
    for arm in arms:
        folder = weights / arm.id
        folder.mkdir(parents=True, exist_ok=True)
        if arm.reuse_job:
            continue
        if fetch is not None:
            fetch(arm, folder)
        elif live:
            fetch_arm(arm, folder)
        sizes[arm.id] = component_bytes(folder)
    return sizes


def _generate_pairs(
    job: Job,
    reference: Arm,
    candidates: Sequence[Arm],
    schedule: Schedule,
    images: Path,
    pairs_path: Path,
    done: dict[str, dict[str, object]],
    timings: dict[str, dict[str, object]],
    timings_path: Path,
    secrets: Sequence[str],
    generate: GenerateFn,
    score: ScoreFn | None,
    smi: SmiFn | None,
) -> None:
    """Render the reference, then each candidate, then score from the PNGs."""
    cache: dict[tuple[str, int], Image.Image] = {}
    shots: dict[tuple[str, int], tuple[Image.Image, float | None, int | None, int | None]] = {}
    for arm in (reference, *candidates):
        try:
            contaminated = _open_arm(generate, arm, job, schedule, images)
            for case in schedule.cases:
                image, seconds, torch_bytes, smi_bytes = _image_for(
                    arm, case, job, images, cache, generate, smi, secrets
                )
                if contaminated and seconds is not None:
                    seconds = None
                    contaminated = False
                shots[(arm.id, case.ordinal)] = (image, seconds, torch_bytes, smi_bytes)
                _store_timing(timings, timings_path, arm, case, seconds, torch_bytes, smi_bytes, secrets)
        finally:
            _release_generator(generate)
    scorer = score if score is not None else make_scorer()
    for case in schedule.cases:
        reference_image = shots[(reference.id, case.ordinal)][0]
        for candidate in candidates:
            key = pair_key(job.id, candidate.id, case)
            if key in done:
                continue
            candidate_image, seconds, torch_bytes, smi_bytes = shots[(candidate.id, case.ordinal)]
            metrics = _score_pair(scorer, reference_image, candidate_image, case, job, candidate, secrets)
            record = pair_record(
                job=job,
                candidate=candidate,
                case=case,
                score=metrics,
                seconds=seconds,
                vram_torch_bytes=torch_bytes,
                vram_smi_bytes=smi_bytes,
            )
            append_pair(pairs_path, record, secrets)
            done[key] = record
            _log(f"{job.id} arm={candidate.id} prompt_id={case.prompt_id} seed={case.seed}", secrets)


def _open_arm(
    generate: GenerateFn,
    arm: Arm,
    job: Job,
    schedule: Schedule,
    images: Path,
) -> bool:
    """Load one arm outside the image timer.

    The return value is true when this generator has no separate load step,
    so the first image it actually renders still includes load time.
    """
    if arm.reuse_job or not _arm_pending(arm, schedule, images):
        return False
    load_arm = getattr(generate, "load_arm", None)
    if load_arm is None:
        return True
    load_arm(arm, job)
    return False


def _arm_pending(arm: Arm, schedule: Schedule, images: Path) -> bool:
    return any(not _image_file(images, arm, case).is_file() for case in schedule.cases)


def _release_generator(generate: GenerateFn) -> None:
    release = getattr(generate, "release", None)
    if release is not None:
        release()


def _score_pair(
    score: ScoreFn,
    reference_image: Image.Image,
    candidate_image: Image.Image,
    case: Case,
    job: Job,
    candidate: Arm,
    secrets: Sequence[str],
) -> PairScore:
    try:
        return score(reference_image, candidate_image, case)
    except CompareSetup:
        raise
    except Exception as exc:
        message = assert_or_scrub(str(exc), secrets)
        raise CompareSetup(
            f"{job.id} arm {candidate.id} metrics failed: {message}",
            code="compare_metric",
        ) from exc


def _store_timing(
    timings: dict[str, dict[str, object]],
    path: Path,
    arm: Arm,
    case: Case,
    seconds: float | None,
    torch_bytes: int | None,
    smi_bytes: int | None,
    secrets: Sequence[str],
) -> None:
    key = timing_id(arm.id, case.ordinal)
    if key in timings:
        return
    if seconds is None and torch_bytes is None and smi_bytes is None:
        return
    record: dict[str, object] = {
        "arm": arm.id,
        "ordinal": case.ordinal,
        "prompt_id": case.prompt_id,
        "seed": case.seed,
        "width": case.width,
        "height": case.height,
        "control": case.control,
        "warmup": case.warmup,
        "seconds": seconds,
        "vram_torch_bytes": torch_bytes,
        "vram_smi_bytes": smi_bytes,
    }
    append_timing(path, record, secrets)
    timings[key] = record


def _image_for(
    arm: Arm,
    case: Case,
    job: Job,
    images: Path,
    cache: dict[tuple[str, int], Image.Image],
    generate: GenerateFn,
    smi: SmiFn | None,
    secrets: Sequence[str],
) -> tuple[Image.Image, float | None, int | None, int | None]:
    cached = cache.get((arm.id, case.ordinal))
    if cached is not None:
        return cached, None, None, None
    path = _image_file(images, arm, case)
    reused = _reused_file(images, arm, case)
    source = reused if reused is not None else path
    if source.is_file():
        image = Image.open(source).convert("RGB")
        cache[(arm.id, case.ordinal)] = image
        return image, None, None, None
    if arm.reuse_job:
        raise CompareSetup(
            f"{job.id} arm {arm.id} is missing reused images from {arm.reuse_job}",
            code="compare_reuse",
        )
    image, seconds, torch_bytes, smi_bytes = _timed(arm, case, job, generate, smi, secrets)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG")
    cache[(arm.id, case.ordinal)] = image
    return image, seconds, torch_bytes, smi_bytes


def _timed(
    arm: Arm,
    case: Case,
    job: Job,
    generate: GenerateFn,
    smi: SmiFn | None,
    secrets: Sequence[str],
) -> tuple[Image.Image, float, int | None, int | None]:
    reset_torch_peak()
    started = time.perf_counter()
    with SmiSampler(smi) as sampler:
        try:
            image = generate(arm, case, job)
        except CompareSetup:
            raise
        except CompareNoGpu:
            raise
        except Exception as exc:
            message = assert_or_scrub(str(exc), secrets)
            raise CompareSetup(f"{job.id} arm {arm.id} failed: {message}", code="compare_generate") from exc
    seconds = time.perf_counter() - started
    return image, seconds, torch_peak_bytes(), sampler.peak_bytes


def assert_or_scrub(message: str, secrets: Sequence[str]) -> str:
    """Replace prompt text in an exception before it can reach a log."""
    cleaned = message
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, "[prompt]")
    return cleaned


def _image_file(images: Path, arm: Arm, case: Case) -> Path:
    control = case.control or "none"
    name = f"{case.prompt_id}-{case.seed}-{case.width}x{case.height}-{control}.png"
    return images / arm.id / name


def _reused_file(images: Path, arm: Arm, case: Case) -> Path | None:
    if not arm.reuse_job:
        return None
    control = case.control or "none"
    name = f"{case.prompt_id}-{case.seed}-{case.width}x{case.height}-{control}.png"
    arm_id = arm.reuse_arm or arm.id
    return images.parent.parent / arm.reuse_job / "images" / arm_id / name


def _log(message: str, secrets: Sequence[str]) -> None:
    assert_text_clean(message, secrets, where="log")
    print(message, file=sys.stderr)
