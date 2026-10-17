"""LoRA training loop: train, extend, pause/resume and dry-run (plan §3-§6)."""

from __future__ import annotations

import math
import os
import random
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from voxprint_forge.lora.bases import BaseAdapter, weights_dir
from voxprint_forge.lora.errors import LoraBlocked, LoraStageError, LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.layers import inject, load_lora_state, lora_parameters, lora_state
from voxprint_forge.lora.presets import TrainConfig
from voxprint_forge.lora.project import Item, Project

MIN_PHOTOS = {"simple": 10, "expert": 5}
STATE_FILE = "state.pt"


def _check_ready(project: Project, mode_floor: bool = True) -> tuple[dict[str, object], list[Item]]:
    data = project.read()
    if data.get("gate_passed") is not True:
        raise LoraBlocked("lora.err.gate_missing")
    items = project.items()
    floor = MIN_PHOTOS.get(str(data.get("mode", "simple")), 10)
    if mode_floor and len(items) < floor:
        raise LoraUsageError("lora.err.too_few", need=floor - len(items))
    return data, items


def estimate(project: Project, config: TrainConfig, adapter: BaseAdapter) -> dict[str, object]:
    """Plan and ETA without training (``--dry-run``)."""
    _, items = _check_ready(project)
    seconds = adapter.seconds_per_step(config) * config.steps
    return {
        "state": "dry-run",
        "base": adapter.info.mdl_id,
        "photos": len(items),
        "steps": config.steps,
        "eta_seconds": int(round(seconds)),
        "vram_need_gb": adapter.vram_need_gb(config),
        "config": config.to_dict(),
    }


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


def _device(adapter: BaseAdapter, config: TrainConfig) -> str:
    if not adapter.requires_gpu:
        return "cpu"
    if not torch.cuda.is_available():
        raise LoraStageError("lora.err.no_gpu")
    have = torch.cuda.get_device_properties(0).total_memory / 2**30
    cap = vram_cap_gb()
    if cap is not None and cap < have:
        # Emulates a smaller card on a big one (calibration): the allocator stops at the cap.
        torch.cuda.set_per_process_memory_fraction(cap / have, 0)
        have = cap
    need = adapter.vram_need_gb(config)
    if have + 0.5 < need:
        raise LoraStageError("lora.err.vram", have=int(have), need=int(math.ceil(need)))
    return "cuda"


def vram_cap_gb() -> float | None:
    """``FORGE_VRAM_CAP_GB`` (GiB), a test cap that makes a big card behave like a small one; unset means none."""
    raw = os.environ.get("FORGE_VRAM_CAP_GB", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if value > 0 else None


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
) -> dict[str, object]:
    """Train (or extend, with ``from_lora``) and return the run summary.

    ``stop`` or a ``PAUSE`` file in the project folder pauses with a
    checkpoint after the current step. A ``PAUSE`` file left from before is
    refused at the start (``lora.err.pause_file``), so a resume never stops at once.
    """
    data, items = _check_ready(project)
    if project.pause.exists():
        raise LoraUsageError("lora.err.pause_file")
    stop = stop or threading.Event()
    saved: dict | None = None
    if resume:
        state_path = project.checkpoints / STATE_FILE
        if not state_path.is_file():
            raise LoraUsageError("lora.err.no_checkpoint")
        saved = torch.load(state_path, weights_only=False)
        config = TrainConfig.from_dict(saved["config"])
        from_lora = Path(saved["from"]) if saved.get("from") else None
        if saved["base"] != adapter.info.mdl_id:
            raise LoraUsageError("lora.err.base_mismatch")
    if from_lora is not None:
        meta = lora_base(from_lora)
        if meta.get("voxprint.lora.base") != adapter.info.mdl_id:
            raise LoraUsageError("lora.err.base_mismatch")
        rank = int(meta.get("voxprint.lora.rank", config.rank))
        alpha = int(meta.get("voxprint.lora.alpha", config.alpha))
        config = TrainConfig.from_dict({**config.to_dict(), "rank": rank, "alpha": alpha})
    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    chosen = select_items(items, config.replay, rng) if from_lora is not None else list(items)
    device = _device(adapter, config)

    progress.emit("train.load_base", "lora.progress.load_base", reason_key="lora.reason.load_base")
    model = adapter.load(base_dir or weights_dir(adapter.info.mdl_id), config, device)
    layers = inject(model, adapter.target_modules, config.rank, config.alpha)
    if from_lora is not None:
        load_lora_state(layers, load_file(str(from_lora)))
    images = _images(project, chosen, config.resolution)
    samples = adapter.encode(images, [{"long": i.long, "short": i.short} for i in chosen], config)
    if config.previews and preview_prompts:
        adapter.prepare_previews(list(preview_prompts))
    adapter.release_encoders()

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
    for step in range(start + 1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(config.grad_accum):
            batch = [rng.choice(samples) for _ in range(config.batch)]
            keys = [_caption_key(config, rng) for _ in batch]
            loss = adapter.loss(model, batch, keys, rng) / config.grad_accum
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
        progress.emit(
            "train.step",
            "lora.progress.train",
            {"step": step, "steps": config.steps},
            "lora.reason.train",
            done=step,
            total=config.steps,
            offset=start,
        )
        if step % config.save_every == 0 or step == config.steps:
            progress.emit("train.checkpoint", "lora.progress.checkpoint", reason_key="lora.reason.checkpoint")
            last_good = _save_checkpoint(project, layers, step, optimizer, rng, meta)
        if config.previews and preview_prompts and step % config.preview_every == 0:
            _previews(project, adapter, model, preview_prompts, step, config, progress)
        if (stop.is_set() or project.pause.exists()) and step < config.steps:
            if last_good is None or not last_good.name.endswith(f"{step:06d}.safetensors"):
                last_good = _save_checkpoint(project, layers, step, optimizer, rng, meta)
            progress.emit("train.paused", "lora.progress.paused", state="paused")
            return _finish(project, data, adapter, config, "paused", step, last_good, loss_value, from_lora)
    progress.emit("train.done", "lora.progress.done", state="done", done=config.steps, total=config.steps)
    return _finish(project, data, adapter, config, "done", config.steps, last_good, loss_value, from_lora)


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


def _finish(project, data, adapter, config, state, step, checkpoint, loss, from_lora) -> dict[str, object]:
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
    project.write({**data, "train": record})
    result = {"state": state, "step": step, "steps": config.steps, "checkpoint": record["checkpoint"], "loss": loss}
    if torch.cuda.is_available():
        result["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        result["peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        cap = vram_cap_gb()
        if cap is not None:
            result["vram_cap_gb"] = cap
    return result
