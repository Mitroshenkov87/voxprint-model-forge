"""``forge lora prepare|train|extend|export``: the LoRA trainer without the GUI."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

from voxprint_forge.lora import bases
from voxprint_forge.lora.errors import LoraUsageError
from voxprint_forge.lora.events import Event, Progress, json_line, sentence
from voxprint_forge.lora.i18n import current_lang
from voxprint_forge.lora.presets import CAPTION_MODES, LIKENESS, PRECISIONS, PRESETS, SCHEDULERS, build_config
from voxprint_forge.lora.project import Item, Project


def add_command(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``lora`` and its subcommands."""
    lora = sub.add_parser("lora", help="Train a LoRA from your photos: prepare, train, extend, export.")
    lora_sub = lora.add_subparsers(dest="lora_command", required=True)
    train = lora_sub.add_parser("train", help="Train a LoRA on the prepared photos of a project.")
    extend = lora_sub.add_parser("extend", help="Continue an existing LoRA with the newest photos of a project.")
    for command in (train, extend):
        _training_flags(command)
    train.add_argument("--base", default=bases.DEFAULT_BASE, help="Base model MDL id. Default: z-image.")
    train.add_argument("--rank", type=int, default=None, help="Adapter size (Expert).")
    train.add_argument("--alpha", type=int, default=None, help="Adapter scale, usually equal to rank (Expert).")
    train.add_argument("--resolution", type=int, default=None, help="Training picture size, a multiple of 64.")
    train.add_argument("--train-text-encoder", action="store_true", default=None, help="Also train the text encoder.")
    extend.add_argument("--from", dest="from_lora", type=Path, required=True, help="Existing LoRA .safetensors.")
    extend.add_argument("--replay", type=float, default=None, help="Share of older photos kept, 0.0-1.0.")


def _training_flags(command: argparse.ArgumentParser) -> None:
    command.add_argument("--project", type=Path, required=True, help="Project folder.")
    command.add_argument("--preset", choices=sorted(PRESETS), default="balanced", help="Ready-made settings.")
    command.add_argument("--likeness", choices=sorted(LIKENESS), default="balanced", help="How close to the photos.")
    command.add_argument("--vram", choices=("auto", "16", "24"), default="auto", help="Graphics memory tier in GB.")
    command.add_argument("--steps", type=int, default=None, help="Training steps (Expert).")
    command.add_argument("--lr", type=float, default=None, help="Learning rate (Expert).")
    command.add_argument("--scheduler", choices=SCHEDULERS, default=None, help="Learning-rate schedule (Expert).")
    command.add_argument("--warmup", type=int, default=None, help="Warm-up steps (Expert).")
    command.add_argument("--batch", type=int, default=None, help="Photos per step (Expert).")
    command.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation (Expert).")
    command.add_argument("--precision", choices=PRECISIONS, default=None, help="Number format (Expert).")
    command.add_argument("--grad-checkpointing", action="store_true", default=None, help="Save memory, slower.")
    command.add_argument("--caption-mode", choices=CAPTION_MODES, default=None, help="Which descriptions to use.")
    command.add_argument("--caption-dropout", type=float, default=None, help="Share of steps without a description.")
    command.add_argument("--captions", type=Path, default=None, help="JSONL caption overrides: id, long, short.")
    command.add_argument("--seed", type=int, default=None, help="Seed for repeatable runs.")
    command.add_argument("--save-every", type=int, default=None, help="Save a version every N steps.")
    command.add_argument("--previews", action=argparse.BooleanOptionalAction, default=None, help="Draw test pictures.")
    command.add_argument("--preview-every", type=int, default=None, help="Draw test pictures every N steps.")
    command.add_argument("--preview-prompts", type=Path, default=None, help="Text file, one test prompt per line.")
    command.add_argument("--resume", action="store_true", help="Continue a paused run from its last version.")
    command.add_argument("--dry-run", action="store_true", help="Print the plan and time estimate; train nothing.")


_CONFIG_FLAGS = (
    "rank alpha steps lr scheduler warmup batch grad_accum resolution precision grad_checkpointing "
    "train_text_encoder caption_mode caption_dropout seed save_every previews preview_every replay"
).split()


def execute(args: argparse.Namespace) -> dict[str, object]:
    """Run one ``lora`` subcommand and return its result."""
    if args.lora_command in ("train", "extend"):
        return _train(args)
    raise LoraUsageError("lora.err.not_prepared")


def printer(json_mode: bool, lang: str | None = None):
    """An event sink: JSON lines with ``--json``, plain sentences otherwise."""

    def sink(event: Event) -> None:
        line = json_line(event) if json_mode else sentence(event, lang or current_lang())
        print(line, file=sys.stdout, flush=True)

    return sink


def _train(args: argparse.Namespace) -> dict[str, object]:
    from voxprint_forge.lora.train import estimate, lora_base, train

    project = Project(args.project)
    data = project.read()
    if args.captions is not None:
        _apply_captions(project, args.captions)
    overrides = {name: getattr(args, name, None) for name in _CONFIG_FLAGS}
    try:
        config = build_config(args.preset, args.likeness, args.vram, **overrides)
    except ValueError as exc:
        raise _bad(exc) from exc
    from_lora = getattr(args, "from_lora", None)
    if args.lora_command == "extend":
        base_id = lora_base(from_lora).get("voxprint.lora.base") or ""
    else:
        base_id = args.base
    adapter = bases.get_adapter(base_id)
    if args.dry_run:
        return estimate(project, config, adapter)
    prompts = _prompts(args.preview_prompts, data)
    stop = threading.Event()
    previous = signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        return train(
            project,
            config,
            adapter,
            Progress(printer(args.json)),
            resume=args.resume,
            from_lora=from_lora,
            stop=stop,
            preview_prompts=prompts,
        )
    finally:
        signal.signal(signal.SIGINT, previous)


def _bad(exc: ValueError) -> Exception:
    from voxprint_forge.errors import UsageError

    return UsageError(str(exc), code="usage")


def _prompts(path: Path | None, data: dict[str, object]) -> list[str]:
    if path is not None:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    subject = data.get("subject") if isinstance(data.get("subject"), dict) else {}
    trigger = str(subject.get("trigger") or "")  # type: ignore[union-attr]
    return [trigger] if trigger else []


def _apply_captions(project: Project, path: Path) -> None:
    edits = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            edits[str(row["id"])] = row
    items = [
        Item(i.id, i.file, str(edits[i.id].get("long", i.long)), str(edits[i.id].get("short", i.short)), i.round)
        if i.id in edits
        else i
        for i in project.items()
    ]
    project.write_items(items)
