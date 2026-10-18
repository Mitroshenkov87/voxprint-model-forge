"""Thin adapter between the GUI and ``voxprint-forge-cli lora ... --json``.

The GUI never trains, filters or gates photos itself. It builds the argument
list from the plan §3 flag table, runs the CLI as a child process and reads
one ``{"event": {...}}`` line per progress event, then the closing JSON
envelope (``{"ok": ..., "result": ..., "error": {"code", "message"}}``).
:func:`commands` reports which ``lora`` subcommands the installed CLI has, so
a step whose command is not in this build says so in plain words instead of
pretending to work.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from voxprint_forge.app.draft import Draft
from voxprint_forge.lora.project import Project

# Expert values that map one-to-one onto CLI flags (plan §3).
FLAG_FOR = {
    "lora.rank": "--rank",
    "lora.alpha": "--alpha",
    "lora.steps": "--steps",
    "lora.lr": "--lr",
    "lora.scheduler": "--scheduler",
    "lora.warmup": "--warmup",
    "lora.batch": "--batch",
    "lora.grad_accum": "--grad-accum",
    "lora.resolution": "--resolution",
    "lora.precision": "--precision",
    "lora.caption_mode": "--caption-mode",
    "lora.caption_dropout": "--caption-dropout",
    "lora.seed": "--seed",
    "lora.save_every": "--save-every",
    "lora.preview_every": "--preview-every",
}
SWITCH_FOR = {"lora.grad_checkpointing": "--grad-checkpointing", "lora.train_text_encoder": "--train-text-encoder"}
TRAIN_ONLY = {"lora.rank", "lora.alpha", "lora.resolution", "lora.train_text_encoder"}


@dataclass(frozen=True)
class EngineEvent:
    """One plan §6 event."""

    step: str
    message_key: str
    params: dict[str, object] = field(default_factory=dict)
    reason_key: str | None = None
    percent: float | None = None
    eta_seconds: int | None = None
    state: str = "running"

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> EngineEvent:
        """Build from the JSON object; unknown keys are ignored."""
        params = data.get("params")
        percent = data.get("percent")
        eta = data.get("eta_seconds")
        return cls(
            step=str(data.get("step", "")),
            message_key=str(data.get("message_key", "")),
            params=dict(params) if isinstance(params, Mapping) else {},
            reason_key=str(data["reason_key"]) if data.get("reason_key") else None,
            percent=float(percent) if isinstance(percent, int | float) else None,
            eta_seconds=int(eta) if isinstance(eta, int | float) else None,
            state=str(data.get("state", "running")),
        )


@dataclass(frozen=True)
class Outcome:
    """The closing envelope: ``ok``, the result and, on failure, the error key."""

    ok: bool
    result: object = None
    error_code: str = ""
    error_message: str = ""
    exit_code: int = 0


def cli_command() -> list[str]:
    """The CLI as a child process of this interpreter (works in a frozen build too)."""
    if getattr(sys, "frozen", False):
        return [str(Path(sys.executable).with_name("voxprint-forge-cli.exe"))]
    return [sys.executable, "-m", "voxprint_forge.cli"]


def commands() -> frozenset[str]:
    """The ``lora`` subcommands this build of the CLI has (empty when it has none)."""
    try:
        from voxprint_forge.cli import build_parser

        parser = build_parser()
    except Exception:  # noqa: BLE001 - a broken CLI means no engine
        return frozenset()
    for action in parser._subparsers._group_actions:  # type: ignore[union-attr]
        lora = getattr(action, "choices", {}).get("lora")
        if lora is None:
            continue
        for sub in lora._subparsers._group_actions:  # type: ignore[union-attr]
            return frozenset(getattr(sub, "choices", {}))
    return frozenset()


def prepare_args(draft: Draft, project: Path) -> list[str]:
    """``lora prepare`` for the photos, the kind and the subject (steps 1-3)."""
    args = ["--json", "lora", "prepare", "--project", str(project), "--mode", "expert" if draft.expert else "simple"]
    photos = draft.training_photos()
    if draft.expert and not 10 <= len(photos) <= 30:
        args.append("--allow-count-outside-range")
    for photo in photos:
        args += ["--photo", str(photo)]
    args += ["--kind", draft.kind, "--crop", draft.crop]
    args += _subject_args(draft)
    return args


def refset_round_args(draft: Draft, project: Path, *, count: int, strength: float, steps: int) -> list[str]:
    """``lora refset round``: the references, the subject and the round settings."""
    args = ["--json", "lora", "refset", "round", "--project", str(project)]
    for ref in draft.photos:
        args += ["--ref", str(ref)]
    args += ["--kind", draft.kind]
    args += _subject_args(draft)
    args += ["--candidates", str(int(count)), "--strength", f"{float(strength):.2f}", "--steps", str(int(steps))]
    return args


def refset_finish_args(draft: Draft, project: Path, base: str) -> list[str]:
    """``lora refset finish``: "This is it"."""
    args = ["--json", "lora", "refset", "finish", "--project", str(project)]
    args += ["--mode", "expert" if draft.expert else "simple"]
    if base:
        args += ["--base", base]
    return args


def train_args(
    project: Path,
    values: Mapping[str, object],
    *,
    extend_from: Path | None = None,
    resume: bool = False,
    dry_run: bool = False,
    expert: bool = False,
) -> list[str]:
    """``lora train`` (or ``extend``) from the live settings; Expert values only in Expert mode."""
    command = "extend" if extend_from is not None else "train"
    args = ["--json", "lora", command, "--project", str(project)]
    args += ["--preset", str(values.get("lora.preset", "balanced"))]
    args += ["--likeness", str(values.get("lora.likeness", "balanced"))]
    args += ["--vram", str(values.get("lora.vram", "auto"))]
    if command == "train" and values.get("lora.base"):
        args += ["--base", str(values["lora.base"])]
    args.append("--previews" if values.get("lora.previews", True) else "--no-previews")
    if extend_from is not None:
        args += ["--from", str(extend_from)]
    if expert:
        for key, flag in FLAG_FOR.items():
            if key in values and values[key] is not None and not (command == "extend" and key in TRAIN_ONLY):
                args += [flag, str(values[key])]
        for key, flag in SWITCH_FOR.items():
            if command == "extend" and key in TRAIN_ONLY:
                continue
            if values.get(key):
                args.append(flag)
            elif key == "lora.grad_checkpointing" and values.get(key) is False:
                args.append("--no-grad-checkpointing")  # on by default for both memory tiers
        if extend_from is not None and values.get("lora.mix_old") is not None:
            args += ["--replay", str(values["lora.mix_old"])]
    if resume:
        args.append("--resume")
    if dry_run:
        args.append("--dry-run")
    return args


def export_args(
    draft: Draft, project: Path, out: Path | None, *, checkpoint: int | None = None, install: bool = False
) -> list[str]:
    """``lora export``: Save (``--out``) or Open in Canvas (``--install-to-mdl``)."""
    args = ["--json", "lora", "export", "--project", str(project)]
    if checkpoint is not None:
        args += ["--checkpoint", str(checkpoint)]
    if out is not None:
        args += ["--out", str(out)]
    if install:
        args.append("--install-to-mdl")
    return args  # export reads the subject from the project; it takes no subject flags


def _subject_args(draft: Draft) -> list[str]:
    args: list[str] = []
    if draft.name:
        args += ["--name", draft.name]
    if draft.stage_name:
        args += ["--stage-name", draft.stage_name]
    if draft.trigger:
        args += ["--trigger", draft.trigger]
    if draft.kind == "person":
        if draft.consent:
            args.append("--consent-obtained")
        args += ["--consent-method", draft.consent_method()]
        if draft.adult:
            args.append("--subject-adult")
            if draft.adult_content:
                args.append("--adult-content")
            if isinstance(draft.attr, int) and not isinstance(draft.attr, bool):
                args += ["--subject-age", str(draft.attr)]
    elif draft.rights:
        args.append("--rights-confirmed")
    if draft.release is not None and not (draft.kind == "person" and draft.verbal):
        args += ["--release", str(draft.release)]
    if draft.note:
        args += ["--provenance-note", draft.note]
    return args


class LineReader:
    """Feed stdout text in chunks; get events as they arrive and the envelope at the end."""

    def __init__(self) -> None:
        self._partial = ""
        self._rest: list[str] = []

    def feed(self, chunk: str) -> list[EngineEvent]:
        """Parse complete lines; return the events among them."""
        events: list[EngineEvent] = []
        lines = (self._partial + chunk).split("\n")
        self._partial = lines.pop()
        for line in lines:
            event = _event(line)
            if event is not None:
                events.append(event)
            else:
                self._rest.append(line)
        return events

    def finish(self, exit_code: int) -> Outcome:
        """The closing envelope. A missing or broken one is a failure, never a success."""
        tail = self._partial
        self._partial = ""
        if tail and _event(tail) is None:
            self._rest.append(tail)
        text = "\n".join(self._rest).strip()
        start = text.find("{")
        try:
            payload = json.loads(text[start:]) if start >= 0 else None
        except json.JSONDecodeError:
            payload = None
        if not isinstance(payload, dict):
            return Outcome(False, None, "forge.app.err.engine_output", "", exit_code)
        error = payload.get("error") if isinstance(payload.get("error"), dict) else {}
        ok = bool(payload.get("ok")) and exit_code == 0
        return Outcome(
            ok,
            payload.get("result"),
            str(error.get("code", "")) if not ok else "",  # type: ignore[union-attr]
            str(error.get("message", "")) if not ok else "",  # type: ignore[union-attr]
            int(payload.get("exit_code", exit_code)),  # type: ignore[call-overload]
        )


def _event(line: str) -> EngineEvent | None:
    line = line.strip()
    if not line.startswith('{"event"'):
        return None
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    event = data.get("event") if isinstance(data, dict) else None
    return EngineEvent.from_dict(event) if isinstance(event, dict) else None


def prepared_from(project: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """``(kept, drops)`` as ``lora prepare`` wrote them: ``captions.jsonl`` and ``drops.jsonl``.

    Kept rows carry ``id``, an absolute ``file``, ``long`` and ``short``. Drops
    carry ``name`` and a ``lora.drop.*`` reason key that never says what was found.
    """
    from voxprint_forge.lora.project import Project

    folder = Project(project)
    kept = [
        {"id": item.id, "file": str(project / item.file), "long": item.long, "short": item.short}
        for item in folder.items()
    ]
    drops: dict[str, dict[str, object]] = {}
    if folder.drops.is_file():
        for line in folder.drops.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                drops[str(row.get("id") or row.get("name") or len(drops))] = row
    return kept, list(drops.values())


def pause_file(project: Path) -> Path:
    """``<project>/PAUSE`` (docs/LORA.md, pause file contract). The engine only reads it."""
    return Project(project).pause


def request_pause(project: Path) -> None:
    """Pause: create ``PAUSE``; the engine saves a version and exits 0 with ``state: paused``."""
    path = pause_file(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def clear_pause(project: Path) -> None:
    """Delete ``PAUSE`` before ``train``/``extend`` start or ``--resume`` (else ``lora.err.pause_file``)."""
    pause_file(project).unlink(missing_ok=True)
