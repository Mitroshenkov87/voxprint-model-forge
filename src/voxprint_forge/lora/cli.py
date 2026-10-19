"""``forge lora prepare|train|extend|export``: the LoRA trainer without the GUI."""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from pathlib import Path

from voxprint_forge.lora import DEFAULT_BASE
from voxprint_forge.lora.captioner import DEFAULT as CAPTION_DEFAULT
from voxprint_forge.lora.captioner import FALLBACK as CAPTION_FALLBACK
from voxprint_forge.lora.consent import CONSENT_METHODS, KINDS, Subject, check_subject
from voxprint_forge.lora.errors import LoraBlocked, LoraUsageError
from voxprint_forge.lora.events import Event, Progress, json_line, sentence
from voxprint_forge.lora.i18n import current_lang
from voxprint_forge.lora.limits import EXPERT_RANGE, RECOMMENDED_RANGE
from voxprint_forge.lora.photos import BLUR_THRESHOLD, DUP_THRESHOLD, MIN_SIDE, collect
from voxprint_forge.lora.presets import CAPTION_MODES, LIKENESS, PRECISIONS, PRESETS, SCHEDULERS, build_config
from voxprint_forge.lora.project import Item, Project
from voxprint_forge.mdl_app import fetch, missing_ids

_REC = "{}-{}".format(*RECOMMENDED_RANGE)  # lora.limits: the one source of the set-size numbers
_EXPERT = "{}-{}".format(*EXPERT_RANGE)


def add_command(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register ``lora`` and its subcommands."""
    lora = sub.add_parser("lora", help="Train a LoRA from your photos: prepare, train, extend, export.")
    lora_sub = lora.add_subparsers(dest="lora_command", required=True)
    _prepare_flags(lora_sub.add_parser("prepare", help="Check, crop and describe photos into a project."))
    export = lora_sub.add_parser("export", help="Save the LoRA as one .safetensors file with voxprint-model.json.")
    export.add_argument("--project", type=Path, required=True, help="Project folder.")
    export.add_argument("--checkpoint", type=int, default=None, help="Step of the version to save. Default: last.")
    export.add_argument("--out", type=Path, default=None, help="Target .safetensors. Default: <project>/export/.")
    export.add_argument("--install-to-mdl", action="store_true", help="Also add it to your models for Canvas.")
    export.add_argument("--tag", action="append", default=[], help="Plain-word tag for Canvas. Repeat as needed.")
    train = lora_sub.add_parser("train", help="Train a LoRA on the prepared photos of a project.")
    extend = lora_sub.add_parser("extend", help="Continue an existing LoRA with the newest photos of a project.")
    for command in (train, extend):
        _training_flags(command)
    train.add_argument("--base", default=DEFAULT_BASE, help="Base model MDL id. Default: z-image.")
    train.add_argument("--rank", type=int, default=None, help="Adapter size (Expert).")
    train.add_argument("--alpha", type=int, default=None, help="Adapter scale, usually equal to rank (Expert).")
    train.add_argument("--resolution", type=int, default=None, help="Training picture size, a multiple of 64.")
    train.add_argument("--train-text-encoder", action="store_true", default=None, help="Also train the text encoder.")
    extend.add_argument("--from", dest="from_lora", type=Path, required=True, help="Existing LoRA .safetensors.")
    extend.add_argument("--replay", type=float, default=None, help="Share of older photos kept, 0.0-1.0.")
    _refset_flags(lora_sub.add_parser("refset", help="Build a photo set from 1-10 references, round by round."))


def _refset_flags(command: argparse.ArgumentParser) -> None:
    from voxprint_forge.lora import refset

    steps = command.add_subparsers(dest="refset_command", required=True)
    round_ = steps.add_parser("round", help="Check the references (first time), then make and check candidates.")
    round_.add_argument("--project", type=Path, required=True, help="Project folder.")
    round_.add_argument("--ref", type=Path, action="append", default=[], help="One reference picture. 1-10.")
    _subject_flags(round_)
    low, high = refset.CANDIDATE_RANGE
    round_.add_argument("--candidates", type=int, default=refset.DEFAULT_CANDIDATES,
                        help=f"Candidates per round, {low}-{high}. Default: {refset.DEFAULT_CANDIDATES}.")
    round_.add_argument("--strength", type=float, default=refset.DEFAULT_STRENGTH,
                        help=f"Variation strength (img2img denoise), {refset.STRENGTH_MIN}-{refset.STRENGTH_MAX}. "
                        f"Default: {refset.DEFAULT_STRENGTH}.")
    round_.add_argument("--steps", type=int, default=refset.DEFAULT_STEPS,
                        help=f"Turbo steps, {refset.STEPS_RANGE[0]}-{refset.STEPS_RANGE[1]}. "
                        f"Default: {refset.DEFAULT_STEPS}.")
    round_.add_argument("--seed", type=int, default=None, help="First seed of the round. Default: 1000 x round.")
    _fetch_flags(round_)
    finish = steps.add_parser("finish", help="Copy the references and kept candidates into refset/approved.")
    finish.add_argument("--project", type=Path, required=True, help="Project folder.")
    finish.add_argument("--mode", choices=("simple", "expert"), default="simple",
                        help=f"Set size: {_REC}, or {_EXPERT} in Expert.")
    finish.add_argument("--base", default=DEFAULT_BASE, help="Base model the set is for. Default: z-image.")
    _fetch_flags(finish)


def _subject_flags(command: argparse.ArgumentParser) -> None:
    command.add_argument("--kind", choices=KINDS, default=None, help="What the photos show.")
    command.add_argument("--name", default="", help="Name of the subject.")
    command.add_argument("--stage-name", default="", help="Optional alias; becomes the trigger word.")
    command.add_argument("--trigger", default="", help="Word that calls the model in a prompt.")
    command.add_argument("--consent-obtained", action="store_true", help="Person: consent was obtained.")
    command.add_argument("--subject-adult", action="store_true", help="Person: the subject is 18 or older.")
    command.add_argument(
        "--subject-age",
        type=int,
        default=None,
        help="Optional whole number from 18 to 99. Only for a person confirmed as 18 or older.",
    )
    command.add_argument("--rights-confirmed", action="store_true", help="Object or style: you have the rights.")
    command.add_argument("--release", type=Path, default=None, help="Optional model release file.")
    command.add_argument(
        "--consent-method",
        choices=CONSENT_METHODS,
        default=None,
        help="Person: verbal (no file) or file (needs --release). Default: file with --release, else verbal.",
    )
    command.add_argument(
        "--adult-content", action="store_true", help="Person, 18+ only: the photos contain adult content (a label)."
    )
    command.add_argument("--provenance-note", default="", help="Optional note on where the photos come from.")


def _fetch_flags(command: argparse.ArgumentParser) -> None:
    group = command.add_mutually_exclusive_group()
    group.add_argument("--fetch-missing", action="store_true", help="Fetch missing model ids before this command.")
    group.add_argument("--no-fetch", action="store_true", help="Do not fetch model ids.")


def _wants_fetch(args: argparse.Namespace) -> bool:
    """``--fetch-missing`` when set, else ``--no-fetch`` off, else a TTY stdin."""
    if args.fetch_missing:
        return True
    if args.no_fetch:
        return False
    return sys.stdin.isatty()


def _fetch_missing() -> None:
    """Run ``models fetch --required`` when required ids are absent."""
    missing = missing_ids()
    if not missing:
        return
    text = ", ".join(missing)
    print(text, file=sys.stderr)
    print("voxprint-forge-cli models fetch --required", file=sys.stderr)
    result = fetch(missing, gui=False)
    if not result.get("ok"):
        still = ", ".join(missing_ids() or missing)
        if not still:
            raise LoraBlocked("lora.err.not_prepared")
        raise LoraBlocked("lora.err.gate_missing", ids=still)


def _prepare_flags(command: argparse.ArgumentParser) -> None:
    command.add_argument("--project", type=Path, required=True, help="Project folder.")
    command.add_argument("--photos", type=Path, default=None, help="Folder of JPG, PNG or WebP photos.")
    command.add_argument("--photo", type=Path, action="append", default=[], help="One photo. Repeat as needed.")
    command.add_argument("--exclude", type=Path, action="append", default=[], help="Leave this photo out.")
    _subject_flags(command)
    command.add_argument("--mode", choices=("simple", "expert"), default="simple",
                         help=f"Photo limits: {_REC}, or {_EXPERT} in Expert.")
    command.add_argument(
        "--allow-count-outside-range", action="store_true",
        help=f"Expert limits: warn outside {_REC}, allowed {_EXPERT}."
    )
    command.add_argument("--crop", choices=("subject", "whole"), default="subject", help="How photos are cropped.")
    command.add_argument("--captions", type=Path, default=None, help="JSONL caption overrides: id, long, short.")
    command.add_argument(
        "--caption-model", choices=(CAPTION_DEFAULT, CAPTION_FALLBACK), default=CAPTION_DEFAULT, help="Captioner."
    )
    command.add_argument("--min-side", type=int, default=MIN_SIDE, help="Smallest short side in pixels (Expert).")
    command.add_argument("--blur-threshold", type=float, default=BLUR_THRESHOLD, help="Blur limit (Expert).")
    command.add_argument("--dup-threshold", type=int, default=DUP_THRESHOLD, help="Duplicate limit (Expert).")
    _fetch_flags(command)


def _training_flags(command: argparse.ArgumentParser) -> None:
    command.add_argument("--project", type=Path, required=True, help="Project folder.")
    command.add_argument("--preset", choices=sorted(PRESETS), default="balanced", help="Ready-made settings.")
    command.add_argument("--likeness", choices=sorted(LIKENESS), default="balanced", help="How close to the photos.")
    command.add_argument("--vram", choices=("auto", "16", "24"), default="auto", help="Graphics memory tier in GB.")
    command.add_argument(
        "--te-quant", choices=("auto", "none", "int8", "nf4"), default=None,
        help="MOSS text encoder format (Expert). Default auto: bf16 on 24 GB, int8 on 16 GB.",
    )
    command.add_argument(
        "--te-device", choices=("gpu", "cpu"), default=None,
        help="Where the MOSS text encoder runs (Expert). Default gpu; cpu is a slower fallback.",
    )
    command.add_argument("--steps", type=int, default=None, help="Training steps (Expert).")
    command.add_argument("--lr", type=float, default=None, help="Learning rate (Expert).")
    command.add_argument("--scheduler", choices=SCHEDULERS, default=None, help="Learning-rate schedule (Expert).")
    command.add_argument("--warmup", type=int, default=None, help="Warm-up steps (Expert).")
    command.add_argument("--batch", type=int, default=None, help="Photos per step (Expert).")
    command.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation (Expert).")
    command.add_argument("--precision", choices=PRECISIONS, default=None, help="Number format (Expert).")
    command.add_argument(
        "--grad-checkpointing", action=argparse.BooleanOptionalAction, default=None, help="Save memory, slower."
    )
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
    _fetch_flags(command)


_CONFIG_FLAGS = (
    "rank alpha steps lr scheduler warmup batch grad_accum resolution precision grad_checkpointing "
    "train_text_encoder caption_mode caption_dropout seed save_every previews preview_every replay te_quant te_device"
).split()


def execute(args: argparse.Namespace) -> dict[str, object]:
    """Run one ``lora`` subcommand and return its result."""
    if args.lora_command == "export":
        from voxprint_forge.lora.export import export

        return export(
            Project(args.project),
            Progress(printer(args.json)),
            checkpoint=args.checkpoint,
            out=args.out,
            install_to_mdl=args.install_to_mdl,
            tags=tuple(args.tag),
        )
    if args.lora_command == "prepare":
        return _prepare(args)
    if args.lora_command in ("train", "extend"):
        return _train(args)
    if args.lora_command == "refset":
        return _refset(args)
    raise LoraUsageError("lora.err.not_prepared")


def printer(json_mode: bool, lang: str | None = None):
    """An event sink: JSON lines with ``--json``, plain sentences otherwise."""

    def sink(event: Event) -> None:
        line = json_line(event) if json_mode else sentence(event, lang or current_lang())
        print(line, file=sys.stdout, flush=True)

    return sink


def _train(args: argparse.Namespace) -> dict[str, object]:
    from voxprint_forge.lora.bases import get_adapter
    from voxprint_forge.lora.train import estimate, lora_base, train

    if not args.dry_run and _wants_fetch(args):
        _fetch_missing()
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
    adapter = get_adapter(base_id, training=True)
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


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_gate():
    from voxprint_forge.lora.backends import load_gate

    return load_gate(_device())


def _load_captioner(choice: str):
    from voxprint_forge.lora.captioner import load_captioner

    return load_captioner(choice)


def subject_from(args: argparse.Namespace) -> Subject:
    """The checked :class:`Subject` from ``lora prepare`` flags (the GUI builds the same flags)."""
    return check_subject(
        args.kind or "",
        name=args.name,
        stage_name=args.stage_name,
        trigger=args.trigger,
        consent_obtained=args.consent_obtained,
        subject_adult=args.subject_adult,
        rights_confirmed=args.rights_confirmed,
        provenance_note=args.provenance_note,
        release=args.release,
        consent_method=args.consent_method,
        adult_content=args.adult_content,
        subject_age=args.subject_age,
    )


def _prepare(args: argparse.Namespace) -> dict[str, object]:
    from voxprint_forge.lora.prepare import prepare

    project = Project(args.project)
    stored = project.read() if project.file.is_file() else {}
    if args.kind is None and isinstance(stored.get("subject"), dict):
        subject = Subject(**stored["subject"])  # type: ignore[arg-type]
    else:
        subject = subject_from(args)
    files = collect(args.photos, args.photo, args.exclude)
    if _wants_fetch(args):
        _fetch_missing()
    mode = "expert" if args.allow_count_outside_range else args.mode
    result = prepare(
        project,
        files,
        subject,
        gate=_load_gate(),
        load_captioner=_load_captioner,
        progress=Progress(printer(args.json)),
        mode=mode,
        crop_mode=args.crop,
        caption_model=args.caption_model,
        min_side=args.min_side,
        blur_threshold=args.blur_threshold,
        dup_threshold=args.dup_threshold,
        release=args.release,
    )
    if args.captions is not None:
        _apply_captions(project, args.captions)
    return result


def _refset(args: argparse.Namespace) -> dict[str, object]:
    from voxprint_forge.lora import refset

    project = Project(args.project)
    progress = Progress(printer(args.json))
    if _wants_fetch(args):
        _fetch_missing()
    if args.refset_command == "finish":
        return refset.finish(project, progress, mode=args.mode, base=args.base)
    settings = refset.check_settings(args.candidates, args.strength, args.steps, args.seed)
    subject = subject_from(args)
    device = _device()
    return refset.run_round(
        project,
        list(args.ref),
        subject,
        settings,
        load_gate=_load_gate,
        load_generator=lambda: refset.load_teacher(device),
        load_embedder=lambda: refset.load_embedder(device),
        progress=progress,
    )
