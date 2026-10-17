"""Build the prompt and seed schedule for one compare job.

Public prompts come from ``diffusion/prompts.py``. Private prompts come from
manifests under ``FORGE_PRIVATE_PROMPTS``. Steps and CFG always come from the
job, never from a manifest.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Never

from voxprint_forge.compare.codes import CompareSetup
from voxprint_forge.compare.manifests import ManifestItem, find_manifest, load_manifest
from voxprint_forge.compare.plan import Job, PromptSet
from voxprint_forge.compare.privacy import PRIVATE_ENV, prompt_id
from voxprint_forge.diffusion.prompts import BASE_PROMPTS, CONTROL_PROMPTS, CONTROL_TYPES

Source = Literal["public", "private"]


@dataclass(frozen=True)
class Case:
    """One generation. ``prompt`` is for the model call and is not serialised."""

    ordinal: int
    prompt_id: str
    prompt: str
    negative: str
    seed: int
    width: int
    height: int
    control: str
    source: Source
    warmup: bool


@dataclass(frozen=True)
class Schedule:
    """Cases for a job, plus private set names that were not on disk yet."""

    cases: tuple[Case, ...]
    missing_sets: tuple[str, ...]


def build_schedule(job: Job, private_root: Path | None) -> Schedule:
    """Public cases first, then up to ``private_limit`` rows from each named set.

    The first case is the warm-up image for every arm. Each arm renders it,
    and the speed mean leaves it out.
    """
    public = _public_cases(job)
    private, missing = _private_cases(job, private_root, start=len(public))
    rows = public + private
    if not rows:
        return Schedule(cases=(), missing_sets=missing)
    marked = [_mark(item, index == 0) for index, item in enumerate(rows)]
    return Schedule(cases=tuple(marked), missing_sets=missing)


def _mark(case: Case, warmup: bool) -> Case:
    if case.warmup == warmup:
        return case
    return Case(
        ordinal=case.ordinal,
        prompt_id=case.prompt_id,
        prompt=case.prompt,
        negative=case.negative,
        seed=case.seed,
        width=case.width,
        height=case.height,
        control=case.control,
        source=case.source,
        warmup=warmup,
    )


def _unexpected_prompts(prompts: Never) -> Never:
    raise CompareSetup(f"unhandled prompt set {prompts}", code="compare_plan")


def _public_cases(job: Job) -> list[Case]:
    prompts: PromptSet = job.prompts
    if prompts == "base":
        texts = BASE_PROMPTS
        controls: tuple[str, ...] = ("",)
    elif prompts == "control":
        texts = CONTROL_PROMPTS
        controls = job.controls or CONTROL_TYPES
    else:
        _unexpected_prompts(prompts)
    rows: list[Case] = []
    for text in texts:
        for control in controls:
            for seed in job.public_seeds:
                rows.append(
                    Case(
                        ordinal=len(rows),
                        prompt_id=prompt_id(text),
                        prompt=text,
                        negative="",
                        seed=seed,
                        width=job.width,
                        height=job.height,
                        control=control,
                        source="public",
                        warmup=False,
                    )
                )
    return rows


def _private_cases(job: Job, root: Path | None, *, start: int) -> tuple[list[Case], tuple[str, ...]]:
    if not job.private_sets:
        return [], ()
    if root is None:
        _warn(f"{PRIVATE_ENV} is not set in this process; private sets skipped: {', '.join(job.private_sets)}")
        return [], job.private_sets
    rows: list[Case] = []
    missing: list[str] = []
    for name in job.private_sets:
        folder = root / name
        manifest = find_manifest(folder) if folder.is_dir() else None
        if manifest is None:
            _warn(
                f"private set {name}: no manifest at {folder / 'manifest.json'} "
                f"or {folder / 'dataset' / 'manifest.json'}"
            )
            missing.append(name)
            continue
        items = load_manifest(manifest)[: job.private_limit]
        for item in items:
            rows.append(_from_item(item, ordinal=start + len(rows), width=job.width, height=job.height))
    return rows, tuple(missing)


def _warn(message: str) -> None:
    """Local diagnostic on stderr. It names paths, never prompt text, and is not written to results."""
    print(f"compare: {message}", file=sys.stderr, flush=True)


def _from_item(item: ManifestItem, *, ordinal: int, width: int, height: int) -> Case:
    """Keep the manifest seed. A missing size uses the job width and height.

    The manifest size is what the source image used. The harness regenerates at
    that size when it is present, so a 1024 row stays 1024. Width or height 0
    means the row did not set a size.
    """
    return Case(
        ordinal=ordinal,
        prompt_id=prompt_id(item.prompt),
        prompt=item.prompt,
        negative=item.negative,
        seed=item.seed,
        width=item.width or width,
        height=item.height or height,
        control="",
        source="private",
        warmup=False,
    )
