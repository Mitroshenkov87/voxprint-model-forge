"""LoRA training loop: train, extend, pause/resume and dry-run (plan §3-§6)."""

from __future__ import annotations

import math
import os
import random
import shutil
import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from voxprint_forge.lora.bases import BaseAdapter, weights_dir
from voxprint_forge.lora.errors import LoraBlocked, LoraStageError, LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.layers import inject, load_lora_state, lora_parameters, lora_state
from voxprint_forge.lora.memory import MemoryPlan, check_plan, vram_cap_gb
from voxprint_forge.lora.presets import TrainConfig
from voxprint_forge.lora.progress_log import ProgressWriter, read_progress, write_row
from voxprint_forge.lora.project import Item, Project
from voxprint_forge.lora.sample_prompts import build_sample_prompts, caption_rows, read_attire, store_attire
from voxprint_forge.lora.review import (
    DEFAULT_FIRST_ROUND_SAMPLES,
    DEFAULT_FIRST_ROUND_STEPS,
    DEFAULT_SAMPLE_STEPS,
    DEFAULT_SAMPLES,
    HumanReviewDecider,
    ReviewDecider,
    apply_decider,
    build_sets,
    check_review_round_count,
    check_round_options,
    first_pictures_eta,
    load_state,
    passes_output_check,
    plan_samples,
    sample_steps_of,
    samples_for_round,
    save_image,
    save_state,
    steps_completed,
    steps_for_stage,
    summary,
    total_steps_of,
    transition,
)
from voxprint_forge.mdl_app import missing_id_text

MIN_PHOTOS = {"simple": 10, "expert": 5}
STATE_FILE = "state.pt"


def _check_ready(project: Project, mode_floor: bool = True) -> tuple[dict[str, object], list[Item]]:
    data = project.read()
    if data.get("gate_passed") is not True:
        ids = missing_id_text()
        if not ids:
            raise LoraBlocked("lora.err.not_prepared")
        raise LoraBlocked("lora.err.gate_missing", ids=ids)
    items = project.items()
    floor = MIN_PHOTOS.get(str(data.get("mode", "simple")), 10)
    if mode_floor and len(items) < floor:
        raise LoraUsageError("lora.err.too_few", need=floor - len(items))
    return data, items


def estimate(project: Project, config: TrainConfig, adapter: BaseAdapter) -> dict[str, object]:
    """Plan and ETA without training (``--dry-run``)."""
    _, items = _check_ready(project)
    seconds = adapter.seconds_per_step(config) * config.steps
    result: dict[str, object] = {
        "state": "dry-run",
        "base": adapter.info.mdl_id,
        "photos": len(items),
        "steps": config.steps,
        "eta_seconds": int(round(seconds)),
        "vram_need_gb": adapter.vram_need_gb(config),
        "config": config.to_dict(),
    }
    plan = adapter.memory_plan(config, len(items))
    if plan is not None:
        result["memory_plan"] = plan.to_dict()
        cap = vram_cap_gb()
        if cap is not None:
            result["vram_cap_gb"] = cap
            result["plan_fits_cap"] = plan.peak_gb <= cap - 1.0
    return result


def finish_review(project: Project, adapter: BaseAdapter, progress: Progress) -> dict[str, object]:
    """Save the current LoRA, draw a final preview, and record the steps it got.

    The review stays open until the checkpoint is on disk and the preview has been
    drawn, so a failed close does not look like a finished run.
    """
    return _seal_finished(project, load_state(project), adapter, progress)


def discard_run(project: Project) -> dict[str, object]:
    """Stop and discard: remove temporary checkpoints and samples. Photos stay.

    Nothing is written for export. ``prepared/``, ``photos/``, captions and the
    subject record stay. A link that points outside the project is left alone.
    """
    root = project.root.resolve()
    for path in (
        project.checkpoints,
        project.previews,
        project.root / "review",
        project.root / "train.json",
        project.root / "train.log",
        project.root / "vram.csv",
        project.pause,
    ):
        _remove_inside(root, path)
    data = project.read()
    cleaned = {key: value for key, value in data.items() if key not in {"train", "review"}}
    project.write(cleaned)
    return {"state": "discarded", "saved": False}


def _remove_inside(root: Path, path: Path) -> None:
    """Delete ``path`` when it stays inside ``root``. An outside link is skipped."""
    if not path.exists() and not path.is_symlink():
        return
    try:
        resolved = path.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return
    if resolved == root:
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    shutil.rmtree(path)


def lora_base(path: Path) -> dict[str, str]:
    """``voxprint.lora.*`` header metadata of a LoRA file."""
    with safe_open(str(path), framework="pt") as handle:
        meta = handle.metadata() or {}
    return {key: value for key, value in meta.items() if key.startswith("voxprint.lora.")}


def select_items(items: Sequence[Item], replay: float, rng: random.Random) -> list[Item]:
    """For an extend: every photo of the newest round plus ``replay`` share of the older ones."""
    if not items:
        return []
    newest = max(item.round for item in items)
    new = [item for item in items if item.round == newest]
    old = [item for item in items if item.round < newest]
    keep = math.ceil(replay * len(old))
    return new + (rng.sample(old, keep) if keep else [])


def _device(adapter: BaseAdapter, config: TrainConfig, plan: MemoryPlan | None = None) -> str:
    if not adapter.requires_gpu:
        return "cpu"
    if not torch.cuda.is_available():
        raise LoraStageError("lora.err.no_gpu")
    card = torch.cuda.get_device_properties(0).total_memory / 2**30
    have = card
    cap = vram_cap_gb()
    if cap is not None and cap < have:
        # Emulates a smaller card on a big one (calibration): the allocator stops at the cap.
        torch.cuda.set_per_process_memory_fraction(cap / have, 0)
        have = cap
    if plan is not None:
        # Fail fast: the projected peak must fit before any weight reaches the card.
        check_plan(plan, cap, card)
        return "cuda"
    need = adapter.vram_need_gb(config)
    if have + 0.5 < need:
        raise LoraStageError("lora.err.vram", have=int(have), need=int(math.ceil(need)))
    return "cuda"


class VramPhases:
    """Peak allocated and reserved GiB per training phase (load, encode, resident, step, preview).

    Each mark records the peak since the previous one and resets it, so an OOM names its
    phase and a passing run shows which phase sets the overall peak. No-op without CUDA.
    """

    def __init__(self) -> None:
        self.peaks: dict[str, dict[str, float]] = {}
        self.plan: dict[str, float] | None = None
        self.phase = "load"
        self.on = torch.cuda.is_available()
        if self.on:
            torch.cuda.reset_peak_memory_stats()

    def mark(self, done: str, next_phase: str) -> None:
        if self.on:
            alloc = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            reserved = round(torch.cuda.max_memory_reserved() / 2**30, 2)
            old = self.peaks.get(done, {"alloc_gb": 0.0, "reserved_gb": 0.0})
            self.peaks[done] = {
                "alloc_gb": max(alloc, old["alloc_gb"]),
                "reserved_gb": max(reserved, old["reserved_gb"]),
            }
            torch.cuda.reset_peak_memory_stats()
        self.phase = next_phase


def _schedule(config: TrainConfig):
    def factor(step: int) -> float:
        if config.warmup and step < config.warmup:
            return (step + 1) / config.warmup
        if config.scheduler == "cosine":
            span = max(1, config.steps - config.warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - config.warmup) / span)))
        return 1.0

    return factor


def _images(project: Project, items: Sequence[Item], resolution: int) -> list[object]:
    from PIL import Image

    out = []
    for item in items:
        with Image.open(project.root / item.file) as image:
            image = image.convert("RGB")
            if image.size != (resolution, resolution):
                image = image.resize((resolution, resolution), Image.Resampling.LANCZOS)
            out.append(image)
    return out


def _caption_key(config: TrainConfig, rng: random.Random) -> str:
    if rng.random() < config.caption_dropout:
        return ""
    if config.caption_mode == "both":
        return rng.choice(("long", "short"))
    return config.caption_mode


def _save_checkpoint(project: Project, layers, step: int, optimizer, rng: random.Random, meta: dict) -> Path:
    project.checkpoints.mkdir(parents=True, exist_ok=True)
    path = project.checkpoints / f"step-{step:06d}.safetensors"
    save_file(lora_state(layers), str(path), metadata={"voxprint.lora.base": meta["base"], "step": str(step)})
    torch.save(
        {
            "step": step,
            "optimizer": optimizer.state_dict(),
            "rng": rng.getstate(),
            "torch_rng": torch.get_rng_state(),
            **meta,
        },
        project.checkpoints / STATE_FILE,
    )
    return path


def train(
    project: Project,
    config: TrainConfig,
    adapter: BaseAdapter,
    progress: Progress,
    *,
    resume: bool = False,
    from_lora: Path | None = None,
    stop: threading.Event | None = None,
    preview_prompts: Sequence[str] = (),
    base_dir: Path | None = None,
    review_rounds: int = 0,
    samples_per_round: int = DEFAULT_SAMPLES,
    first_round_steps: int = DEFAULT_FIRST_ROUND_STEPS,
    first_round_samples: int = DEFAULT_FIRST_ROUND_SAMPLES,
    sample_steps: int = DEFAULT_SAMPLE_STEPS,
    review_continue: bool = False,
    no_more_reviews: bool = False,
    decider: ReviewDecider | None = None,
    sample_attire: str | None = None,
) -> dict[str, object]:
    """Train (or extend, with ``from_lora``) and return the run summary.

    ``stop`` or a ``PAUSE`` file in the project folder pauses with a
    checkpoint after the current step. A ``PAUSE`` file left from before is
    refused at the start (``lora.err.pause_file``), so a resume never stops at once.
    ``review_rounds`` of 1 to 6 pauses after each planned stage so the user can
    mark samples. ``0`` trains straight through. The first stage is
    ``first_round_steps``; later stages share the preset steps that remain.
    ``review_continue`` trains the next stage. ``no_more_reviews`` trains every
    remaining preset step and then finishes. ``decider`` writes like and dislike
    marks before that next stage. The default reads marks already stored by the
    GUI or ``lora review``.
    """
    data, items = _check_ready(project)
    if project.pause.exists():
        raise LoraUsageError("lora.err.pause_file")
    stop = stop or threading.Event()
    saved: dict | None = None
    if resume and not review_continue:
        state_path = project.checkpoints / STATE_FILE
        if not state_path.is_file():
            raise LoraUsageError("lora.err.no_checkpoint")
        saved = torch.load(state_path, weights_only=False)
        config = TrainConfig.from_dict(saved["config"])
        from_lora = Path(saved["from"]) if saved.get("from") else None
        if saved["base"] != adapter.info.mdl_id:
            raise LoraUsageError("lora.err.base_mismatch")
    if review_continue:
        pending = load_state(project)
        checkpoint = str(pending.get("checkpoint") or "")
        if not checkpoint:
            raise LoraUsageError("lora.err.no_checkpoint")
        from_lora = project.root / checkpoint
        saved_config = pending.get("config")
        if isinstance(saved_config, dict) and saved_config:
            config = TrainConfig.from_dict(saved_config)
    if from_lora is not None:
        meta = lora_base(from_lora)
        if meta.get("voxprint.lora.base") != adapter.info.mdl_id:
            raise LoraUsageError("lora.err.base_mismatch")
        rank = int(meta.get("voxprint.lora.rank", config.rank))
        alpha = int(meta.get("voxprint.lora.alpha", config.alpha))
        config = TrainConfig.from_dict({**config.to_dict(), "rank": rank, "alpha": alpha})
    review_state = _review_plan(
        project,
        config,
        adapter,
        data,
        preview_prompts,
        review_rounds=review_rounds,
        samples_per_round=samples_per_round,
        first_round_steps=first_round_steps,
        first_round_samples=first_round_samples,
        sample_steps=sample_steps,
        review_continue=review_continue,
        no_more_reviews=no_more_reviews,
        resume=resume,
        decider=decider,
        sample_attire=sample_attire,
    )
    if review_state is not None and review_state.get("custom_prompts") is False:
        built = [str(prompt) for prompt in review_state.get("prompts") or [] if str(prompt).strip()]
        if built:
            preview_prompts = built
        stored = project.read().get("sample_attire")
        if isinstance(stored, str):
            data["sample_attire"] = stored
    if review_state is not None and not (resume and not review_continue):
        stage = steps_for_stage(review_state)
        if review_state.get("no_more_reviews") and stage < 1:
            return _seal_finished(project, review_state, adapter, progress)
        config = replace(config, steps=max(1, stage) if stage > 0 else 0)
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    negatives: list[Item] = []
    if review_state is not None:
        sets = build_sets(items, review_state, supports_negatives=bool(adapter.supports_negatives))
        chosen = list(sets.include)
        if adapter.supports_negatives and callable(getattr(adapter, "negative_loss", None)):
            negatives = list(sets.negatives)
    elif from_lora is not None:
        chosen = select_items(items, config.replay, rng)
    else:
        chosen = list(items)
    # Grow segments instead of fragmenting fixed blocks; only read when CUDA first allocates.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    plan = adapter.memory_plan(config, len(chosen), base_dir)
    device = _device(adapter, config, plan)

    phases = VramPhases()
    phases.plan = plan.to_dict() if plan is not None else None
    try:
        return _train(
            project,
            config,
            adapter,
            progress,
            data,
            chosen,
            saved,
            from_lora,
            rng,
            stop,
            preview_prompts,
            base_dir,
            device,
            phases,
            negatives,
            review_state,
        )
    except torch.OutOfMemoryError as exc:
        phases.mark(phases.phase, phases.phase)
        planned = f" planned_peak_gb={phases.plan['peak_gb']}" if phases.plan else ""
        raise torch.OutOfMemoryError(f"{exc} [phase={phases.phase} vram_phases={phases.peaks}{planned}]") from exc


def _train(
    project,
    config,
    adapter,
    progress,
    data,
    chosen,
    saved,
    from_lora,
    rng,
    stop,
    preview_prompts,
    base_dir,
    device,
    phases: VramPhases,
    negatives: Sequence[Item],
    review_state: dict[str, object] | None,
) -> dict[str, object]:
    writer = ProgressWriter(project.root)
    clock = _PictureClock(review_state, adapter, config)
    progress.emit("train.load_base", "lora.progress.load_base", reason_key="lora.reason.load_base")
    _report(writer, stage="load", step=0, total=config.steps, loss=None, eta=None, extra=clock.extra(config.steps))
    base = base_dir or weights_dir(adapter.info.mdl_id)
    bundle = list(chosen) + list(negatives)
    captions = [{"long": i.long, "short": i.short} for i in bundle]
    if getattr(adapter, "encodes_before_load", False):
        # Encoders first, then free the text encoder, then read the DiT: keeps RAM to one big model.
        adapter.load_encoders(base, config, device)
        images = _images(project, bundle, config.resolution)
        phases.mark("load", "encode")
        samples, neg_samples = _split_encoded(adapter.encode(images, captions, config), len(chosen))
        if config.previews and preview_prompts:
            adapter.prepare_previews(list(preview_prompts))
        adapter.free_text_encoder()
        model = adapter.load(base, config, device)
        layers = inject(model, adapter.target_modules, config.rank, config.alpha, dtype=adapter.lora_dtype)
        if from_lora is not None:
            load_lora_state(layers, load_file(str(from_lora)))
    else:
        model = adapter.load(base, config, device)
        layers = inject(model, adapter.target_modules, config.rank, config.alpha, dtype=adapter.lora_dtype)
        if from_lora is not None:
            load_lora_state(layers, load_file(str(from_lora)))
        images = _images(project, bundle, config.resolution)
        phases.mark("load", "encode")
        samples, neg_samples = _split_encoded(adapter.encode(images, captions, config), len(chosen))
        if config.previews and preview_prompts:
            adapter.prepare_previews(list(preview_prompts))
    phases.mark("encode", "resident")
    _report(writer, stage="encode", step=0, total=config.steps, loss=None, eta=None, extra=clock.extra(config.steps))
    adapter.release_encoders()
    phases.mark("resident", "step")

    params = lora_parameters(layers)
    optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _schedule(config))
    start = 0
    if saved is not None:
        last = project.checkpoints / f"step-{int(saved['step']):06d}.safetensors"
        load_lora_state(layers, load_file(str(last)))
        optimizer.load_state_dict(saved["optimizer"])
        rng.setstate(saved["rng"])
        torch.set_rng_state(saved["torch_rng"])
        start = int(saved["step"])
        scheduler.last_epoch = start
        for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs, strict=True):
            group["lr"] = base_lr * _schedule(config)(start)
    meta = {
        "base": adapter.info.mdl_id,
        "config": config.to_dict(),
        "from": str(from_lora) if from_lora is not None else None,
    }
    last_good: Path | None = None
    if saved is not None:
        last_good = project.checkpoints / f"step-{start:06d}.safetensors"
    loss_value = float("nan")
    model.train()
    step_mark = time.perf_counter()
    for step in range(start + 1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(config.grad_accum):
            batch = [rng.choice(samples) for _ in range(config.batch)]
            keys = [_caption_key(config, rng) for _ in batch]
            loss = adapter.loss(model, batch, keys, rng) / config.grad_accum
            extra = getattr(adapter, "negative_loss", None)
            if neg_samples and callable(extra):
                neg_batch = [rng.choice(neg_samples) for _ in range(config.batch)]
                loss = loss + extra(model, neg_batch, keys, rng) / config.grad_accum
            loss.backward()
            total += float(loss.detach())
        if not math.isfinite(total):
            if last_good is not None:
                load_lora_state(layers, load_file(str(last_good)))
            progress.emit("train.step", "lora.progress.failed", state="failed")
            raise LoraStageError("lora.err.diverged")
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        loss_value = total
        event = progress.emit(
            "train.step",
            "lora.progress.train",
            {"step": step, "steps": config.steps},
            "lora.reason.train",
            done=step,
            total=config.steps,
            offset=start,
        )
        clock.note_step(time.perf_counter() - step_mark)
        step_mark = time.perf_counter()
        step_extra = clock.extra(config.steps - step)
        _report(writer, stage="step", step=step, total=config.steps, loss=loss_value, eta=event.eta_seconds,
                extra=step_extra)
        if step % config.save_every == 0 or step == config.steps:
            progress.emit("train.checkpoint", "lora.progress.checkpoint", reason_key="lora.reason.checkpoint")
            last_good = _save_checkpoint(project, layers, step, optimizer, rng, meta)
            _report(writer, stage="checkpoint", step=step, total=config.steps, loss=loss_value, eta=event.eta_seconds,
                    extra=step_extra)
        if config.previews and preview_prompts and step % config.preview_every == 0:
            phases.mark("step", "preview")
            _previews(project, adapter, model, preview_prompts, step, config, progress)
            _report(writer, stage="preview", step=step, total=config.steps, loss=loss_value, eta=event.eta_seconds,
                    extra=step_extra)
            phases.mark("preview", "step")
        if (stop.is_set() or project.pause.exists()) and step < config.steps:
            if last_good is None or not last_good.name.endswith(f"{step:06d}.safetensors"):
                last_good = _save_checkpoint(project, layers, step, optimizer, rng, meta)
            progress.emit("train.paused", "lora.progress.paused", state="paused")
            _report(writer, stage="paused", step=step, total=config.steps, loss=loss_value, eta=event.eta_seconds,
                    extra=step_extra)
            phases.mark(phases.phase, phases.phase)
            return _finish(project, data, adapter, config, "paused", step, last_good, loss_value, from_lora, phases)
    if review_state is not None and review_state.get("no_more_reviews"):
        return _complete_full(
            project, data, adapter, config, last_good, loss_value, from_lora, phases, progress, writer, review_state,
        )
    if review_state is not None:
        return _open_review(
            project, data, adapter, config, model, last_good, loss_value, from_lora, phases, progress, writer,
            review_state, clock,
        )
    done = progress.emit("train.done", "lora.progress.done", state="done", done=config.steps, total=config.steps)
    _report(
        writer, stage="done", step=config.steps, total=config.steps, loss=loss_value, eta=done.eta_seconds,
        extra={"steps": config.steps, "steps_planned": config.steps},
    )
    phases.mark(phases.phase, phases.phase)
    return _finish(
        project, data, adapter, config, "done", config.steps, last_good, loss_value, from_lora, phases,
        steps_planned=config.steps,
    )


def _review_plan(
    project: Project,
    config: TrainConfig,
    adapter: BaseAdapter,
    data: dict[str, object],
    preview_prompts: Sequence[str],
    *,
    review_rounds: int,
    samples_per_round: int,
    first_round_steps: int,
    first_round_samples: int,
    sample_steps: int,
    review_continue: bool,
    no_more_reviews: bool,
    resume: bool,
    decider: ReviewDecider | None = None,
    sample_attire: str | None = None,
) -> dict[str, object] | None:
    """The review state for this stage, or ``None`` when the run does not pause for review."""
    if review_continue:
        state = apply_decider(load_state(project), decider if decider is not None else HumanReviewDecider())
        action = "train_full" if no_more_reviews else "continue"
        state = transition(state, action)
        if not no_more_reviews and state.get("custom_prompts") is False:
            state = _with_sample_prompts(project, data, state, sample_attire)
        save_state(project, state)
        return state
    check_review_round_count(review_rounds)
    if review_rounds < 1:
        return None
    check_round_options(
        first_round_steps=first_round_steps,
        first_round_samples=first_round_samples,
        samples_per_round=samples_per_round,
        sample_steps=sample_steps,
    )
    state = load_state(project)
    if state.get("phase") == "review":
        raise LoraUsageError("lora.err.review_pending")
    if resume and state.get("phase") == "training" and int(state.get("rounds") or 0) > 0:
        return state
    explicit = _explicit_prompts(preview_prompts, data)
    if explicit is None:
        count = first_round_samples if first_round_samples > 0 else samples_per_round
        prompts = _sample_prompt_list(project, data, config.seed, 1, count, sample_attire)
        custom = False
    else:
        prompts = explicit
        custom = True
    state = transition(
        state,
        "start",
        rounds=review_rounds,
        samples_per_round=samples_per_round,
        first_round_steps=first_round_steps,
        first_round_samples=first_round_samples,
        sample_steps=sample_steps,
        total_steps=config.steps,
        prompts=prompts,
        custom_prompts=custom,
        seed=config.seed,
        base=adapter.info.mdl_id,
        config=config.to_dict(),
    )
    save_state(project, state)
    return state


def _explicit_prompts(preview_prompts: Sequence[str], data: dict[str, object]) -> list[str] | None:
    """Prompt lines from a file. The trigger-only fallback is not an explicit list."""
    given = [str(prompt).strip() for prompt in preview_prompts if str(prompt).strip()]
    trigger = _subject_trigger(data)
    if not given or given == ([trigger] if trigger else []):
        return None
    return given


def _subject_trigger(data: dict[str, object]) -> str:
    subject = data.get("subject") if isinstance(data.get("subject"), dict) else {}
    return str(subject.get("trigger") or "")


def _sample_prompt_list(
    project: Project,
    data: dict[str, object],
    seed: int,
    round_no: int,
    count: int,
    sample_attire: str | None,
) -> list[str]:
    mode = sample_attire if isinstance(sample_attire, str) and sample_attire else read_attire(data)
    mode = store_attire(project, mode)
    return build_sample_prompts(
        _subject_trigger(data),
        count=count,
        seed=seed,
        round_no=round_no,
        mode=mode,
        captions=caption_rows(project),
    )


def _with_sample_prompts(
    project: Project,
    data: dict[str, object],
    state: dict[str, object],
    sample_attire: str | None,
) -> dict[str, object]:
    """Replace the stored prompts for the round that is about to be drawn."""
    round_no = int(state.get("index") or 0) + 1
    prompts = _sample_prompt_list(
        project,
        data,
        int(state.get("seed") or 0),
        round_no,
        samples_for_round(state, round_no),
        sample_attire,
    )
    return {**state, "prompts": prompts}


def _open_review(
    project, data, adapter, config, model, last_good, loss_value, from_lora, phases, progress, writer, review_state,
    clock: _PictureClock,
) -> dict[str, object]:
    """Draw the round's samples at the training resolution, then pause with the review on disk."""
    planned = plan_samples(review_state)
    steps = sample_steps_of(review_state)
    saved: list[dict[str, object]] = []
    model.eval()
    try:
        for index, spec in enumerate(planned, 1):
            progress.emit(
                "train.review_sample",
                "lora.progress.review_sample",
                {"n": index, "total": len(planned)},
                "lora.reason.review",
                done=index,
                total=max(1, len(planned)),
            )
            started = time.perf_counter()
            image = adapter.preview(model, str(spec["prompt"]), int(spec["seed"]), config, steps=steps)
            clock.note_sample(time.perf_counter() - started)
            _report(
                writer, stage="review_sample", step=config.steps, total=config.steps, loss=loss_value, eta=None,
                extra=clock.extra(0, samples_left=len(planned) - index),
            )
            if not passes_output_check(image):
                continue
            path = project.root / str(spec["file"])
            save_image(image, path)
            saved.append(spec)
    finally:
        model.train()
    rel = "" if last_good is None else str(Path(last_good).relative_to(project.root))
    state = transition(review_state, "pause", checkpoint=rel, samples=saved)
    save_state(project, state)
    event = progress.emit(
        "train.review",
        "lora.progress.review",
        {"round": int(state["index"]), "rounds": int(state["rounds"])},
        "lora.reason.review",
        state="done",
    )
    _report(writer, stage="review", step=config.steps, total=config.steps, loss=loss_value, eta=event.eta_seconds)
    phases.mark(phases.phase, phases.phase)
    return _finish(
        project, data, adapter, config, "review", config.steps, last_good, loss_value, from_lora, phases,
        review_summary=summary(state),
    )


def _complete_full(
    project, data, adapter, config, last_good, loss_value, from_lora, phases, progress, writer, review_state,
) -> dict[str, object]:
    """Finish a train-fully stage: no more samples and no more pauses."""
    rel = "" if last_good is None else str(Path(last_good).relative_to(project.root))
    finished = transition({**review_state, "checkpoint": rel}, "finish")
    save_state(project, finished)
    planned = total_steps_of(finished)
    done = progress.emit("train.done", "lora.progress.done", state="done", done=planned, total=planned)
    _report(
        writer, stage="done", step=planned, total=planned, loss=loss_value, eta=done.eta_seconds,
        extra={"steps": planned, "steps_planned": planned},
    )
    phases.mark(phases.phase, phases.phase)
    return _finish(
        project, data, adapter, config, "done", planned, last_good, loss_value, from_lora, phases,
        review_summary=summary(finished), steps_planned=planned,
    )


def _split_encoded(encoded: list[object], positive: int) -> tuple[list[object], list[object]]:
    """Positive samples first, disliked negatives after them."""
    return list(encoded[:positive]), list(encoded[positive:])


def _report(
    writer: ProgressWriter,
    *,
    stage: str,
    step: int,
    total: int,
    loss: float | None,
    eta: int | None,
    extra: dict[str, object] | None = None,
) -> None:
    """One progress report. Non-finite loss is stored as unknown."""
    number = None if loss is None or not math.isfinite(loss) else float(loss)
    writer.note(step=step, total=total, loss=number, eta=eta, stage=stage, extra=extra)


class _PictureClock:
    """Step and sample times for the round-1 first-pictures ETA."""

    def __init__(self, state: dict[str, object] | None, adapter: BaseAdapter, config: TrainConfig) -> None:
        self.state = state
        self.sample_steps = sample_steps_of(state) if state else DEFAULT_SAMPLE_STEPS
        self.samples = len(plan_samples(state)) if self.active() and state is not None else 0
        self.fallback = float(adapter.seconds_per_step(config))
        self.step_seconds: list[float] = []
        self.sample_seconds: list[float] = []

    def active(self) -> bool:
        """True only while round 1 is still training or drawing its pictures."""
        state = self.state
        if state is None or state.get("no_more_reviews"):
            return False
        return int(state.get("index") or 0) == 0 and int(state.get("rounds") or 0) > 0

    def note_step(self, seconds: float) -> None:
        """Record one optimizer step."""
        if self.active() and seconds >= 0:
            self.step_seconds.append(seconds)

    def note_sample(self, seconds: float) -> None:
        """Record one review picture."""
        if self.active() and seconds >= 0:
            self.sample_seconds.append(seconds)

    def extra(self, steps_left: int, samples_left: int | None = None) -> dict[str, object] | None:
        """The ``train.json`` fields for the first-pictures line, or ``None`` after round 1."""
        if not self.active():
            return None
        per_step = sum(self.step_seconds) / len(self.step_seconds) if self.step_seconds else self.fallback
        per_sample = None if not self.sample_seconds else sum(self.sample_seconds) / len(self.sample_seconds)
        left = self.samples if samples_left is None else samples_left
        payload = first_pictures_eta(
            steps_left=steps_left,
            seconds_per_step=per_step,
            samples_left=left,
            sample_steps=self.sample_steps,
            seconds_per_sample=per_sample,
        )
        payload["round"] = 1
        payload["seconds_per_step"] = round(per_step, 3)
        return payload


def _seal_finished(
    project: Project, state: dict[str, object], adapter: BaseAdapter, progress: Progress,
) -> dict[str, object]:
    """Draw the final preview, then store a finished, exportable LoRA and the steps it got."""
    phase = state.get("phase")
    if phase != "finished" and not (phase == "review" or (phase == "training" and state.get("no_more_reviews"))):
        transition(state, "finish")
    checkpoint = _review_checkpoint(project, state)
    done_steps = steps_completed(state)
    planned = total_steps_of(state)
    config = _stored_config(project, state)
    _final_preview(project, adapter, config, state, checkpoint, progress)
    if state.get("phase") != "finished":
        state = transition(state, "finish")
        save_state(project, state)
    data = project.read()
    trained = data.get("train")
    record = dict(trained) if isinstance(trained, dict) else {}
    record.update({
        "base": record.get("base") or adapter.info.mdl_id,
        "base_revision": record.get("base_revision") or adapter.info.revision,
        "state": "done",
        "step": done_steps,
        "steps_planned": planned,
        "checkpoint": checkpoint.name,
        "config": record.get("config") if isinstance(record.get("config"), dict) else config.to_dict(),
        "trained_on": record.get("trained_on") or datetime.now(UTC).date().isoformat(),
    })
    project.write({**data, "train": record, "review": summary(state)})
    previous = read_progress(project.root) or {}
    loss = previous.get("loss")
    elapsed = previous.get("elapsed_seconds")
    write_row(project.root, {
        "step": done_steps,
        "total": planned,
        "steps": done_steps,
        "steps_planned": planned,
        "stage": "done",
        "loss": loss if isinstance(loss, int | float) and not isinstance(loss, bool) else None,
        "elapsed_seconds": elapsed if isinstance(elapsed, int | float) and not isinstance(elapsed, bool) else 0,
        "eta_seconds": 0,
    })
    return {
        "state": "done",
        "step": done_steps,
        "steps": done_steps,
        "steps_planned": planned,
        "checkpoint": checkpoint.name,
        "review": summary(state),
    }


def _review_checkpoint(project: Project, state: dict[str, object]) -> Path:
    """The checkpoint the open review saved. Missing files are not a finished LoRA."""
    raw = str(state.get("checkpoint") or "")
    if not raw:
        raise LoraUsageError("lora.err.no_checkpoint")
    path = Path(raw)
    if not path.is_absolute():
        path = project.root / path
    if not path.is_file():
        alt = project.checkpoints / Path(raw).name
        path = alt if alt.is_file() else path
    if not path.is_file():
        raise LoraUsageError("lora.err.no_checkpoint")
    return path


def _stored_config(project: Project, state: dict[str, object]) -> TrainConfig:
    """The config the run was trained with, so the final preview matches that LoRA."""
    data = project.read()
    trained = data.get("train")
    raw = trained.get("config") if isinstance(trained, dict) else None
    if not isinstance(raw, dict) or "rank" not in raw:
        stored = state.get("config")
        raw = stored if isinstance(stored, dict) else {}
    return TrainConfig.from_dict(raw)


def _final_preview(
    project: Project,
    adapter: BaseAdapter,
    config: TrainConfig,
    state: dict[str, object],
    checkpoint: Path,
    progress: Progress,
) -> None:
    """Draw the result pictures from the saved LoRA. The name matches the checkpoint step."""
    prompts = [str(prompt) for prompt in state.get("prompts") or [] if str(prompt).strip()]
    if not prompts:
        prompts = [""]
    device = _device(adapter, config)
    model = adapter.load(weights_dir(adapter.info.mdl_id), config, device)
    layers = inject(model, adapter.target_modules, config.rank, config.alpha, dtype=adapter.lora_dtype)
    load_lora_state(layers, load_file(str(checkpoint)))
    file_step = _checkpoint_step(checkpoint.name)
    steps = sample_steps_of(state)
    progress.emit("train.preview", "lora.progress.preview", reason_key="lora.reason.preview")
    model.eval()
    for index, prompt in enumerate(prompts):
        image = adapter.preview(model, prompt, config.seed + index, config, steps=steps)
        if not passes_output_check(image):
            continue
        project.previews.mkdir(parents=True, exist_ok=True)
        save_image(image, project.previews / f"step-{file_step:06d}-{index}.png")


def _checkpoint_step(name: str) -> int:
    """The step number in a ``step-NNNNNN`` checkpoint name."""
    stem = Path(name).stem
    if stem.startswith("step-") and stem.removeprefix("step-").isdigit():
        return int(stem.removeprefix("step-"))
    return 0


def _previews(project, adapter, model, prompts, step, config, progress) -> None:
    progress.emit("train.preview", "lora.progress.preview", reason_key="lora.reason.preview")
    model.eval()
    try:
        for index, prompt in enumerate(prompts):
            image = adapter.preview(model, prompt, config.seed + index, config)
            if image is not None:
                project.previews.mkdir(parents=True, exist_ok=True)
                image.save(project.previews / f"step-{step:06d}-{index}.png")
    finally:
        model.train()


def _finish(
    project, data, adapter, config, state, step, checkpoint, loss, from_lora, phases=None, review_summary=None,
    steps_planned: int | None = None,
) -> dict[str, object]:
    record = {
        "base": adapter.info.mdl_id,
        "base_revision": adapter.info.revision,
        "config": config.to_dict(),
        "step": step,
        "state": state,
        "checkpoint": checkpoint.name if checkpoint else None,
        "trained_on": datetime.now(UTC).date().isoformat(),
        "extended": from_lora is not None,
    }
    if steps_planned is not None:
        record["steps_planned"] = int(steps_planned)
    payload = {**data, "train": record}
    if review_summary is not None:
        payload["review"] = review_summary
    project.write(payload)
    result = {"state": state, "step": step, "steps": config.steps, "checkpoint": record["checkpoint"], "loss": loss}
    if review_summary is not None:
        result["review"] = review_summary
    if torch.cuda.is_available():
        peaks = phases.peaks.values() if phases is not None and phases.peaks else []
        result["peak_vram_gb"] = max(
            (p["alloc_gb"] for p in peaks), default=round(torch.cuda.max_memory_allocated() / 2**30, 2)
        )
        result["peak_reserved_gb"] = max(
            (p["reserved_gb"] for p in peaks), default=round(torch.cuda.max_memory_reserved() / 2**30, 2)
        )
        if phases is not None:
            result["vram_phases_gb"] = phases.peaks
            if phases.plan is not None:
                result["memory_plan"] = phases.plan
        cap = vram_cap_gb()
        if cap is not None:
            result["vram_cap_gb"] = cap
    return result
