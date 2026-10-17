"""Build a training set from 1-10 references (``lora refset round|finish``).

The user gives a few reference pictures. Z-Image-Turbo img2img makes a larger
set of candidates from them; the user keeps or rejects each one, and the next
round starts from the kept ones plus the references. When the user says the set
is right, :func:`finish` copies the approved pictures into the project and the
normal ``lora prepare`` runs on them (with its own checks).

Every reference and every candidate goes through the same input safety gate as
``lora prepare`` (:mod:`voxprint_forge.lora.gate`) before it is used or shown.
A reference that does not pass blocks the loop. A candidate that does not pass
is deleted without a trace in the review; a clear minor signal blocks the loop.
There is no setting that changes this.

Stages run one at a time and each model is unloaded before the next one loads:
gate (references and their prompts), generator, gate (candidates), DINOv2
near-duplicate check. ``lora prepare`` and the trainer run in later processes.

Files live in ``<project>/refset/``: ``state.json`` (ids, seeds, settings,
verdicts), ``prompts.json`` (the prompt per reference; local only), ``refs/``,
``round-<n>/`` and ``approved/``. Pictures are named by ``sha256[:12]`` of
their bytes; nothing else identifies them.
"""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from voxprint_forge.lora.consent import Subject
from voxprint_forge.lora.errors import LoraBlocked, LoraStageError, LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import RULE_VERSION, Gate
from voxprint_forge.lora.photos import EXPERT_CAP, EXPERT_FLOOR, MIN_SIDE, RECOMMENDED_RANGE, open_photo, photo_id
from voxprint_forge.lora.project import Project

SCHEMA = "voxprint-refset/1"
REF_RANGE = (1, 10)
CANDIDATE_RANGE = (8, 64)
DEFAULT_CANDIDATES = 24
STRENGTH_MIN, STRENGTH_MAX = 0.25, 0.9
STRENGTHS = {"low": 0.35, "medium": 0.5, "high": 0.7}
DEFAULT_STRENGTH = STRENGTHS["medium"]
STEPS_RANGE = (4, 16)
DEFAULT_STEPS = 9
LONG_SIDE = 1024
NEAR_DUPLICATE = 0.95
VERDICTS = ("pending", "keep", "reject")

# The teacher that makes candidates. Same repo and revision as the Turbo pin in
# diffusion/nunchaku_zimage.py (a test keeps them equal). Installed by MDL as
# <models>/model-forge/z-image-turbo.
TEACHER_ID = "z-image-turbo"
TEACHER_REPO = "Tongyi-MAI/Z-Image-Turbo"
TEACHER_REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
TEACHER_LICENCE = "Apache-2.0"
TEACHER_PIPELINE = "ZImageImg2ImgPipeline"
EMBEDDER_ID = "dinov2-base"
PROMPT_SYSTEM = "You write factual image descriptions."
PROMPT_ASK = (
    "Describe this photo as a prompt for an image model in two or three plain sentences: subject, clothing, "
    "pose, setting, light. Do not guess names."
)


class Generator(Protocol):
    """Makes one candidate from a source picture."""

    name: str

    def generate(self, image: Any, prompt: str, *, seed: int, strength: float, steps: int) -> Any:
        """A PIL image."""
        ...

    def unload(self) -> None:
        """Free graphics memory."""
        ...


class Embedder(Protocol):
    """Image embeddings for the near-duplicate check."""

    name: str

    def embed(self, images: Sequence[Any]) -> list[list[float]]:
        """One unit-length vector per image."""
        ...

    def unload(self) -> None:
        """Free graphics memory."""
        ...


@dataclass(frozen=True)
class Settings:
    """One round's settings, already checked."""

    count: int = DEFAULT_CANDIDATES
    strength: float = DEFAULT_STRENGTH
    steps: int = DEFAULT_STEPS
    seed: int | None = None


def check_settings(count: int, strength: float, steps: int, seed: int | None = None) -> Settings:
    """Clamp-free validation: out-of-range values are refused, never silently changed."""
    low, high = CANDIDATE_RANGE
    if not low <= count <= high:
        raise LoraUsageError("lora.err.refset_count", low=low, high=high)
    if not STRENGTH_MIN <= strength <= STRENGTH_MAX:
        raise LoraUsageError("lora.err.refset_strength", low=STRENGTH_MIN, high=STRENGTH_MAX)
    if not STEPS_RANGE[0] <= steps <= STEPS_RANGE[1]:
        raise LoraUsageError("lora.err.refset_steps", low=STEPS_RANGE[0], high=STEPS_RANGE[1])
    if int(steps * strength) < 1:
        raise LoraUsageError("lora.err.refset_steps", low=STEPS_RANGE[0], high=STEPS_RANGE[1])
    return Settings(count, float(strength), int(steps), seed)


# ----- files ---------------------------------------------------------------------------------------
def folder(project: Project | Path) -> Path:
    """``<project>/refset``."""
    root = project.root if isinstance(project, Project) else Path(project)
    return root / "refset"


def state_file(project: Project | Path) -> Path:
    """``refset/state.json``."""
    return folder(project) / "state.json"


def prompts_file(project: Project | Path) -> Path:
    """``refset/prompts.json``: ``{ref_id: prompt}``. Local only; never copied into a model or a log."""
    return folder(project) / "prompts.json"


def empty_state() -> dict[str, Any]:
    """A loop that has not started."""
    return {"schema": SCHEMA, "refs": [], "refs_checked": False, "blocked": False, "rounds": [], "finished": False}


def load_state(project: Project | Path) -> dict[str, Any]:
    """``state.json``, or an empty state."""
    path = state_file(project)
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return empty_state()
    return data


def save_state(project: Project | Path, state: dict[str, Any]) -> None:
    """Write ``state.json`` atomically."""
    path = state_file(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_prompts(project: Project | Path) -> dict[str, str]:
    """The prompt per reference id."""
    path = prompts_file(project)
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        data = {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


def save_prompts(project: Project | Path, prompts: dict[str, str]) -> None:
    """Write ``prompts.json`` (Expert caption edits use this too)."""
    path = prompts_file(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prompts, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prompt_id(prompt: str) -> str:
    """``sha256[:12]`` of a prompt; metadata records only this."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


def candidates(state: dict[str, Any], round_no: int | None = None) -> list[dict[str, Any]]:
    """Shown candidates of one round (default: the latest)."""
    rounds = state.get("rounds") or []
    if not rounds:
        return []
    chosen = rounds[-1] if round_no is None else rounds[round_no]
    return list(chosen.get("candidates") or [])


def kept(state: dict[str, Any]) -> list[dict[str, Any]]:
    """Every kept candidate of every round, oldest first."""
    return [c for r in state.get("rounds") or [] for c in r.get("candidates") or [] if c.get("verdict") == "keep"]


def approved_count(state: dict[str, Any]) -> int:
    """References plus kept candidates: the size of the set ``finish`` would hand to prepare."""
    return len(state.get("refs") or []) + len(kept(state))


def set_verdict(project: Project | Path, candidate_id: str, verdict: str) -> dict[str, Any]:
    """Keep, reject or reset one candidate. Changing a verdict reopens a finished loop."""
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    state = load_state(project)
    for round_ in state.get("rounds") or []:
        for item in round_.get("candidates") or []:
            if item.get("id") == candidate_id:
                item["verdict"] = verdict
                state["finished"] = False
    save_state(project, state)
    return state


def ref_ids(files: Sequence[Path]) -> list[str]:
    """``sha256[:12]`` of each reference file, in order."""
    return [photo_id(Path(path)) for path in files]


def same_refs(state: dict[str, Any], files: Sequence[Path]) -> bool:
    """True when ``files`` are exactly the references this state was built from."""
    return [r.get("id") for r in state.get("refs") or []] == ref_ids(files)


def reset(project: Project | Path) -> None:
    """Delete the loop (new references start a new loop)."""
    shutil.rmtree(folder(project), ignore_errors=True)


def restart(project: Project | Path) -> bool:
    """"Start over" with the same references. A blocked loop stays blocked: returns False and keeps it."""
    if load_state(project).get("blocked"):
        return False
    reset(project)
    return True


# ----- stages -------------------------------------------------------------------------------------
def release_stage() -> None:
    """Free graphics memory between stages (diffusion/stage_free.py on CUDA, a GC pass elsewhere)."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        from voxprint_forge.diffusion.stage_free import free_eval_stage

        free_eval_stage(None)
        torch.cuda.ipc_collect()


def _unload_gate(gate: Gate) -> None:
    for backend in (gate.judge, gate.estimator):
        unload = getattr(backend, "unload", None)
        if callable(unload):
            unload()
    release_stage()


def _block(project: Project | Path, state: dict[str, Any], progress: Progress) -> LoraBlocked:
    state["blocked"] = True
    state["finished"] = False
    save_state(project, state)
    progress.emit("refset.safety", "lora.progress.blocked", reason_key="lora.reason.refset_check", state="blocked")
    return LoraBlocked("lora.err.refset_blocked")


def check_refs(
    project: Project | Path,
    files: Sequence[Path],
    subject: Subject,
    *,
    load_gate: Callable[[], Gate],
    progress: Progress,
    describe: Callable[[Any], str] | None = None,
) -> dict[str, Any]:
    """Gate every reference and write the prompt for each. Any reference that does not pass blocks the loop.

    The prompts come from the judge VLM of the gate while it is loaded (``describe`` overrides this in
    tests). The gate is unloaded before this returns.
    """
    low, high = REF_RANGE
    if not low <= len(files) <= high:
        raise LoraUsageError("lora.err.refset_refs")
    state = load_state(project)
    if state.get("blocked") and same_refs(state, files):
        raise _block(project, state, progress)
    if not same_refs(state, files):
        reset(project)
        state = empty_state()
    images = []
    for path in files:
        image = open_photo(Path(path))
        if image is None or min(image.size) < MIN_SIDE:
            raise LoraUsageError("lora.err.refset_unreadable")
        images.append((photo_id(Path(path)), image))
    gate = load_gate()
    if not gate.available:
        _unload_gate(gate)
        progress.emit("refset.safety", "lora.progress.blocked", reason_key="lora.reason.refset_check",
                      state="blocked")
        raise LoraBlocked("lora.err.gate_missing")
    prompts = load_prompts(project)
    try:
        for index, (rid, image) in enumerate(images, 1):
            progress.emit("refset.refs", "lora.progress.refset_refs", {"n": index, "total": len(images)},
                          "lora.reason.refset_check", done=index, total=len(images))
            if not gate.check(image).keep:
                state["refs"] = [{"id": i} for i, _ in images]
                raise _block(project, state, progress)
        for index, (rid, image) in enumerate(images, 1):
            if prompts.get(rid):
                continue
            progress.emit("refset.prompt", "lora.progress.refset_prompt", {"n": index, "total": len(images)},
                          done=index, total=len(images))
            prompts[rid] = _describe(gate, image, describe)
    finally:
        _unload_gate(gate)
    refs_dir = folder(project) / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    for rid, image in images:
        image.save(refs_dir / f"{rid}.png")
    save_prompts(project, prompts)
    state.update({
        "refs": [{"id": rid, "file": f"refs/{rid}.png"} for rid, _ in images],
        "refs_checked": True,
        "blocked": False,
        "subject_kind": subject.kind,
        "gate_rule": RULE_VERSION,
    })
    save_state(project, state)
    return state


def _describe(gate: Gate, image: Any, describe: Callable[[Any], str] | None) -> str:
    if describe is not None:
        return describe(image).strip()
    ask = getattr(gate.judge, "ask", None)
    if callable(ask):
        return str(ask(image, PROMPT_SYSTEM, PROMPT_ASK, 160)).strip()
    return ""


def plan_round(state: dict[str, Any], settings: Settings, prompts: dict[str, str]) -> list[dict[str, Any]]:
    """The jobs of the next round: sources (references plus kept candidates) in turn, one seed each."""
    sources: list[dict[str, Any]] = [
        {"id": r["id"], "file": r["file"], "ref": r["id"]} for r in state.get("refs") or []
    ]
    sources += [{"id": c["id"], "file": c["file"], "ref": c.get("ref", "")} for c in kept(state)]
    if not sources:
        raise LoraUsageError("lora.err.refset_refs")
    round_no = len(state.get("rounds") or [])
    base = settings.seed if settings.seed is not None else 1000 * round_no
    jobs = []
    for index in range(settings.count):
        source = sources[index % len(sources)]
        prompt = prompts.get(source["ref"], "")
        jobs.append({
            "index": index,
            "parent": source["id"],
            "parent_file": source["file"],
            "ref": source["ref"],
            "seed": int(base + index),
            "strength": settings.strength,
            "steps": settings.steps,
            "prompt": prompt,
            "prompt_id": prompt_id(prompt),
        })
    return jobs


def target_size(size: tuple[int, int], long_side: int = LONG_SIDE) -> tuple[int, int]:
    """Width and height for the candidate: the long side ``long_side``, both multiples of 16."""
    width, height = size
    scale = long_side / max(width, height)
    return max(16, round(width * scale / 16) * 16), max(16, round(height * scale / 16) * 16)


def run_round(
    project: Project | Path,
    files: Sequence[Path],
    subject: Subject,
    settings: Settings,
    *,
    load_gate: Callable[[], Gate],
    load_generator: Callable[[], Generator | None],
    load_embedder: Callable[[], Embedder | None] = lambda: None,
    progress: Progress,
    describe: Callable[[Any], str] | None = None,
) -> dict[str, Any]:
    """One round: check the references (first time), make candidates, gate them, mark near-duplicates."""
    state = load_state(project)
    if state.get("blocked") or not state.get("refs_checked") or not same_refs(state, files):
        state = check_refs(project, files, subject, load_gate=load_gate, progress=progress, describe=describe)
    prompts = load_prompts(project)
    jobs = plan_round(state, settings, prompts)
    root = folder(project)
    round_no = len(state.get("rounds") or [])
    pending = root / f"round-{round_no}" / "pending"
    shutil.rmtree(pending, ignore_errors=True)
    pending.mkdir(parents=True, exist_ok=True)
    progress.emit("refset.load", "lora.progress.refset_load", reason_key="lora.reason.refset_generate")
    generator = load_generator()
    if generator is None:
        raise LoraStageError("lora.err.refset_teacher_missing")
    made: list[tuple[dict[str, Any], Path]] = []
    try:
        for job in jobs:
            progress.emit("refset.generate", "lora.progress.refset_generate",
                          {"n": job["index"] + 1, "total": len(jobs)}, "lora.reason.refset_generate",
                          done=job["index"] + 1, total=len(jobs))
            source = open_photo(root / job["parent_file"])
            if source is None:
                continue
            image = generator.generate(source, job["prompt"], seed=job["seed"], strength=job["strength"],
                                       steps=job["steps"])
            path = pending / f"{job['index']:03d}.png"
            image.convert("RGB").save(path)
            made.append((job, path))
    finally:
        progress.emit("refset.unload", "lora.progress.refset_unload", reason_key="lora.reason.refset_unload")
        generator.unload()
        generator = None
        release_stage()
    shown = _gate_candidates(project, state, made, round_no, load_gate=load_gate, progress=progress)
    shutil.rmtree(pending, ignore_errors=True)
    _near_duplicates(project, state, shown, load_embedder=load_embedder, progress=progress)
    state.setdefault("rounds", []).append({
        "round": round_no,
        "count": settings.count,
        "strength": settings.strength,
        "steps": settings.steps,
        "seed_base": jobs[0]["seed"] if jobs else None,
        "candidates": shown,
    })
    state["finished"] = False
    save_state(project, state)
    progress.emit("refset.done", "lora.progress.refset_done",
                  {"round": round_no + 1, "shown": len(shown)}, state="done")
    return {"state": "done", "round": round_no, "shown": len(shown), "kept_so_far": len(kept(state)),
            "approved": approved_count(state)}


def _gate_candidates(
    project: Project | Path,
    state: dict[str, Any],
    made: list[tuple[dict[str, Any], Path]],
    round_no: int,
    *,
    load_gate: Callable[[], Gate],
    progress: Progress,
) -> list[dict[str, Any]]:
    """Gate every candidate before it can be shown. Non-passing ones are deleted; a minor signal blocks."""
    root = folder(project)
    out_dir = root / f"round-{round_no}"
    gate = load_gate()
    if not gate.available:
        _unload_gate(gate)
        for _job, path in made:
            path.unlink(missing_ok=True)
        progress.emit("refset.safety", "lora.progress.blocked", reason_key="lora.reason.refset_check",
                      state="blocked")
        raise LoraBlocked("lora.err.gate_missing")
    shown: list[dict[str, Any]] = []
    minor = False
    try:
        for index, (job, path) in enumerate(made, 1):
            progress.emit("refset.check", "lora.progress.refset_check", {"n": index, "total": len(made)},
                          "lora.reason.refset_check", done=index, total=len(made))
            image = open_photo(path)
            decision = gate.check(image) if image is not None else None
            if decision is None or not decision.keep:
                minor = minor or bool(decision is not None and decision.minor)
                path.unlink(missing_ok=True)
                continue
            cid = photo_id(path)
            if any(item["id"] == cid for item in shown):
                path.unlink(missing_ok=True)
                continue
            target = out_dir / f"{cid}.png"
            path.replace(target)
            shown.append({
                "id": cid,
                "file": f"round-{round_no}/{cid}.png",
                "round": round_no,
                "parent": job["parent"],
                "ref": job["ref"],
                "seed": job["seed"],
                "strength": job["strength"],
                "steps": job["steps"],
                "prompt_id": job["prompt_id"],
                "verdict": "pending",
            })
    finally:
        _unload_gate(gate)
    if minor:
        for item in shown:
            (root / item["file"]).unlink(missing_ok=True)
        raise _block(project, state, progress)
    return shown


def _near_duplicates(
    project: Project | Path,
    state: dict[str, Any],
    shown: list[dict[str, Any]],
    *,
    load_embedder: Callable[[], Embedder | None],
    progress: Progress,
) -> None:
    """Mark new candidates that look almost the same as a reference, a kept one or an earlier new one."""
    if not shown:
        return
    embedder = load_embedder()
    if embedder is None:
        state["near_duplicate_check"] = False
        return
    root = folder(project)
    earlier = [{"id": r["id"], "file": r["file"]} for r in state.get("refs") or []] + kept(state)
    items = earlier + shown
    progress.emit("refset.similar", "lora.progress.refset_similar")
    try:
        pictures = [open_photo(root / item["file"]) for item in items]
        usable = [(item, picture) for item, picture in zip(items, pictures, strict=True) if picture is not None]
        vectors = embedder.embed([picture for _item, picture in usable])
    finally:
        embedder.unload()
        embedder = None
        release_stage()
    state["near_duplicate_check"] = True
    by_id = {item["id"]: vector for (item, _p), vector in zip(usable, vectors, strict=True)}
    seen: list[str] = [item["id"] for item in earlier if item["id"] in by_id]
    for item in shown:
        vector = by_id.get(item["id"])
        if vector is None:
            continue
        match = next((other for other in seen if _cosine(vector, by_id[other]) >= NEAR_DUPLICATE), None)
        if match is not None:
            item["near_duplicate_of"] = match
        seen.append(item["id"])


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def near_duplicates(state: dict[str, Any], ids: Sequence[str] | None = None) -> int:
    """How many candidates (of ``ids``, default the kept ones) are marked as near-duplicates."""
    chosen = set(ids) if ids is not None else {c["id"] for c in kept(state)}
    return sum(1 for r in state.get("rounds") or [] for c in r.get("candidates") or []
               if c["id"] in chosen and c.get("near_duplicate_of"))


def finish(
    project: Project | Path,
    progress: Progress,
    *,
    mode: str = "simple",
    base: str = "z-image",
) -> dict[str, Any]:
    """"This is it": copy the references and kept candidates into ``refset/approved`` and record how."""
    state = load_state(project)
    if state.get("blocked"):
        raise LoraBlocked("lora.err.refset_blocked")
    if not state.get("refs_checked"):
        raise LoraUsageError("lora.err.refset_not_started")
    chosen = kept(state)
    if not chosen:
        raise LoraUsageError("lora.err.refset_none_kept")
    total = approved_count(state)
    low, high = (RECOMMENDED_RANGE if mode == "simple" else (EXPERT_FLOOR, EXPERT_CAP))
    if total < low:
        raise LoraUsageError("lora.err.too_few", need=low - total)
    if total > high:
        raise LoraUsageError("lora.err.too_many", extra=total - high, high=high)
    root = folder(project)
    approved = root / "approved"
    shutil.rmtree(approved, ignore_errors=True)
    approved.mkdir(parents=True)
    progress.emit("refset.finish", "lora.progress.refset_finish", {"count": total})
    rows: list[dict[str, Any]] = []
    for ref in state.get("refs") or []:
        shutil.copy2(root / ref["file"], approved / f"{ref['id']}.png")
        rows.append({"id": ref["id"], "origin": "reference"})
    for item in chosen:
        shutil.copy2(root / item["file"], approved / f"{item['id']}.png")
        rows.append({key: item[key] for key in ("id", "round", "parent", "seed", "strength", "steps", "prompt_id")}
                    | {"origin": "candidate"})
    metadata = {
        "schema": "voxprint-refset-metadata/1",
        "base": base,
        "teacher": teacher_info(),
        "gate_rule": RULE_VERSION,
        "rounds": [
            {key: r.get(key) for key in ("round", "count", "strength", "steps", "seed_base")}
            | {"seeds": [c["seed"] for c in r.get("candidates") or []],
               "kept": sum(1 for c in r.get("candidates") or [] if c.get("verdict") == "keep")}
            for r in state.get("rounds") or []
        ],
        "approved": rows,
        "near_duplicates": near_duplicates(state),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    proj = project if isinstance(project, Project) else Project(Path(project))
    try:
        data = proj.read()
    except LoraUsageError:
        data = {}
    proj.write({**data, "refset": {key: metadata[key] for key in ("base", "teacher", "gate_rule", "rounds")}
                | {"approved": len(rows)}})
    state["finished"] = True
    save_state(project, state)
    progress.emit("refset.done", "lora.progress.done", state="done")
    files = [str(approved / f"{row['id']}.png") for row in rows]
    return {"state": "done", "approved": len(rows), "files": files, "near_duplicates": metadata["near_duplicates"]}


def teacher_info() -> dict[str, str]:
    """The candidate model and its licence, as recorded in the metadata."""
    return {
        "mdl_id": TEACHER_ID,
        "repo": TEACHER_REPO,
        "revision": TEACHER_REVISION,
        "licence": TEACHER_LICENCE,
        "pipeline": TEACHER_PIPELINE,
        "precision": "bf16",
        "offload": "model-cpu-offload",
    }


def approved_files(project: Project | Path) -> list[Path]:
    """The approved pictures after :func:`finish`, references first."""
    state = load_state(project)
    if not state.get("finished"):
        return []
    root = folder(project) / "approved"
    ids = [r["id"] for r in state.get("refs") or []] + [c["id"] for c in kept(state)]
    return [root / f"{i}.png" for i in ids if (root / f"{i}.png").is_file()]


# ----- real backends (GPU path; CI uses stubs) ----------------------------------------------------
class TurboImg2Img:
    """Z-Image-Turbo img2img (diffusers ``ZImageImg2ImgPipeline``), BF16 with the text encoder offloaded.

    ``enable_model_cpu_offload`` keeps each component on the CPU until it runs, so the text encoder
    is never on the GPU next to the transformer. ``guidance_scale`` is 0 (Turbo is distilled);
    diffusers runs ``int(steps * strength)`` denoising steps.
    """

    name = TEACHER_ID

    def __init__(self, weights: Path, device: str = "cuda") -> None:
        import torch
        from diffusers import ZImageImg2ImgPipeline

        self.pipe = ZImageImg2ImgPipeline.from_pretrained(str(weights), torch_dtype=torch.bfloat16)
        if device == "cuda":
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def generate(self, image: Any, prompt: str, *, seed: int, strength: float, steps: int) -> Any:
        """One candidate at the reference's aspect, long side 1024."""
        import torch

        width, height = target_size(image.size)
        source = image.convert("RGB").resize((width, height))
        generator = torch.Generator("cpu").manual_seed(int(seed))
        out = self.pipe(prompt=prompt, image=source, strength=float(strength), num_inference_steps=int(steps),
                        guidance_scale=0.0, width=width, height=height, generator=generator)
        return out.images[0]

    def unload(self) -> None:
        """Drop the offload hooks and the pipeline."""
        pipe = getattr(self, "pipe", None)
        if pipe is not None:
            remove = getattr(pipe, "remove_all_hooks", None)
            if callable(remove):
                remove()
        self.pipe = None


class DinoEmbedder:
    """DINOv2-base (Apache-2.0) class-token embeddings, from MDL (``dinov2-base``) at the pinned revision."""

    name = EMBEDDER_ID

    def __init__(self, source: Path | str, device: str = "cuda", revision: str | None = None) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        kwargs: dict[str, Any] = {} if isinstance(source, Path) else {"revision": revision, "local_files_only": True}
        self.processor = AutoImageProcessor.from_pretrained(str(source), **kwargs)
        self.model = AutoModel.from_pretrained(str(source), **kwargs).to(device)
        self.model.eval()
        self.device = device
        self._torch = torch

    def embed(self, images: Sequence[Any]) -> list[list[float]]:
        """Unit-length class-token vectors, eight images at a time."""
        torch = self._torch
        out: list[list[float]] = []
        for start in range(0, len(images), 8):
            batch = [image.convert("RGB") for image in images[start:start + 8]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            with torch.no_grad():
                hidden = self.model(**inputs).last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden.float(), dim=-1)
            out += hidden.cpu().tolist()
        return out

    def unload(self) -> None:
        """Free graphics memory."""
        self.model = None


def load_teacher(device: str = "cuda") -> Generator | None:
    """Turbo img2img from MDL, or ``None`` when it is not installed."""
    from voxprint_forge.lora.backends import installed_dir

    weights = installed_dir(TEACHER_ID)
    if weights is None:
        return None
    return TurboImg2Img(weights, device)


def load_embedder(device: str = "cuda") -> Embedder | None:
    """DINOv2-base from MDL, else the local Hugging Face cache at the pinned revision, else ``None``.

    The near-duplicate mark is a hint; without the model the round still runs and shows no marks.
    """
    from voxprint_forge.lora.backends import installed_dir

    weights = installed_dir(EMBEDDER_ID)
    try:
        if weights is not None:
            return DinoEmbedder(weights, device)
        from voxprint_forge.compare.score import DINO_REPO, DINO_REVISION

        return DinoEmbedder(DINO_REPO, device, DINO_REVISION)
    except Exception:  # noqa: BLE001 - no embedder means no near-duplicate marks, never a failed round
        return None
