"""``lora prepare``: import, filter, safety gate, crop and caption (plan §4 step 3)."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

from voxprint_forge.lora.captioner import DEFAULT, Captioner
from voxprint_forge.lora.consent import Subject
from voxprint_forge.lora.errors import LoraBlocked, LoraStageError, LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import RULE_VERSION, Decision, Gate
from voxprint_forge.lora.photos import (
    BLUR_THRESHOLD,
    DUP_THRESHOLD,
    MIN_SIDE,
    Drop,
    check_count,
    crop,
    dhash,
    filter_reason,
    open_photo,
    photo_id,
)
from voxprint_forge.lora.project import Item, Project
from voxprint_forge.lora.train import MIN_PHOTOS


def prepare(
    project: Project,
    files: Sequence[Path],
    subject: Subject,
    *,
    gate: Gate,
    load_captioner: Callable[[str], Captioner | None],
    progress: Progress,
    mode: str = "simple",
    crop_mode: str = "subject",
    caption_model: str = DEFAULT,
    min_side: int = MIN_SIDE,
    blur_threshold: float = BLUR_THRESHOLD,
    dup_threshold: int = DUP_THRESHOLD,
    release: Path | None = None,
    trace: Path | None = None,
) -> dict[str, object]:
    """Prepare new photos into ``project``. Photos already in the project keep their round.

    ``trace`` (operator diagnostics) gets one JSON line per checked photo: id, path, reason codes and
    each gate stage's result or exception type. No image content, file names or model text.
    """
    if mode not in MIN_PHOTOS:
        raise LoraUsageError("lora.err.not_prepared")
    warning = check_count(len(files), mode)
    if warning:
        progress.emit("prepare.count", warning, {"count": len(files)})
    if not gate.available:
        progress.emit("prepare.safety", "lora.progress.blocked", reason_key="lora.reason.safety", state="blocked")
        raise LoraBlocked("lora.err.gate_missing")
    existing = project.items()
    known = {item.id for item in existing}
    round_no = max((item.round for item in existing), default=-1) + 1 if existing else 0
    seen: list[int] = []
    for item in existing:
        image = open_photo(project.root / item.file)
        if image is not None:
            seen.append(dhash(image))
    drops: list[Drop] = []
    kept: list[tuple[str, object]] = []
    total = len(files)
    for index, path in enumerate(files, 1):
        progress.emit(
            "prepare.check",
            "lora.progress.check_photos",
            {"n": index, "total": total},
            "lora.reason.check_photos",
            done=index,
            total=total,
        )
        pid = photo_id(path)
        if pid in known or pid in {k for k, _ in kept}:
            continue
        image = open_photo(path)
        reason = filter_reason(
            image, seen, min_side=min_side, blur_threshold=blur_threshold, dup_threshold=dup_threshold
        )
        if reason:
            drops.append(Drop(pid, Path(path).name, reason))
        else:
            kept.append((pid, image))
    names = {photo_id(p): Path(p).name for p in files}
    safe: list[tuple[str, object]] = []
    minor = False
    for index, (pid, image) in enumerate(kept, 1):
        progress.emit(
            "prepare.safety",
            "lora.progress.safety",
            {"n": index, "total": len(kept)},
            "lora.reason.safety",
            done=index,
            total=len(kept),
        )
        decision = gate.check(image)
        if decision.shadow is not None:
            _write_shadow(project, pid, decision)
        _log_stage_errors(pid, decision)
        if trace is not None:
            _write_trace(trace, pid, decision)
        if decision.keep:
            safe.append((pid, image))
        else:
            drops.append(Drop(pid, names.get(pid, pid), "lora.drop.safety"))
            minor = minor or decision.minor
    for backend in (gate.judge, gate.estimator):
        unload = getattr(backend, "unload", None)
        if callable(unload):
            unload()
    _write_drops(project, drops)
    if minor:
        project.write({**_base(project), "subject": subject.to_dict(), "mode": mode, "gate_passed": False})
        progress.emit("prepare.safety", "lora.progress.blocked", reason_key="lora.reason.safety", state="blocked")
        raise LoraBlocked("lora.err.blocked")
    project.prepared.mkdir(parents=True, exist_ok=True)
    for index, (pid, image) in enumerate(safe, 1):
        progress.emit(
            "prepare.crop",
            "lora.progress.crop",
            {"n": index, "total": len(safe)},
            "lora.reason.crop",
            done=index,
            total=len(safe),
        )
        crop(image, crop_mode).save(project.prepared / f"{pid}.png")
    new_items: list[Item] = []
    if safe:
        progress.emit("prepare.load_captioner", "lora.progress.load_captioner", reason_key="lora.reason.load_captioner")
        captioner = load_captioner(caption_model)
        if captioner is None:
            raise LoraStageError("lora.err.captioner_missing")
        try:
            for index, (pid, image) in enumerate(safe, 1):
                progress.emit(
                    "prepare.caption",
                    "lora.progress.caption",
                    {"n": index, "total": len(safe)},
                    "lora.reason.caption",
                    done=index,
                    total=len(safe),
                )
                text = captioner.caption(crop(image, crop_mode))
                new_items.append(
                    Item(pid, f"prepared/{pid}.png", text.get("long", ""), text.get("short", ""), round_no)
                )
        finally:
            progress.emit(
                "prepare.unload_captioner", "lora.progress.unload_captioner", reason_key="lora.reason.unload_captioner"
            )
            captioner.unload()
        caption_name = captioner.name
    else:
        caption_name = ""
    items = existing + new_items
    project.write_items(items)
    if release is not None:
        (project.root / "release").mkdir(parents=True, exist_ok=True)
        shutil.copy2(release, project.root / "release" / Path(release).name)
    data = _base(project)
    project.write(
        {
            **data,
            "subject": subject.to_dict(),
            "kind": subject.kind,
            "mode": mode,
            "crop": crop_mode,
            "gate_passed": True,
            "gate_rule": RULE_VERSION,
            "captioner": caption_name or data.get("captioner", ""),
            "rounds": round_no + 1,
        }
    )
    floor = MIN_PHOTOS[mode]
    summary = {
        "state": "done",
        "prepared": len(items),
        "new": len(new_items),
        "set_aside": len(drops),
        "round": round_no,
    }
    if len(items) < floor:
        raise LoraUsageError("lora.err.too_few", need=floor - len(items))
    progress.emit("prepare.done", "lora.progress.done", state="done")
    return summary


def _base(project: Project) -> dict[str, object]:
    try:
        return project.read()
    except LoraUsageError:
        return {}


def _log_stage_errors(pid: str, decision: Decision) -> None:
    """One stderr line per photo whose gate stage raised: id, stage and exception type only."""
    import sys

    for part in ("judge", "estimator"):
        info = decision.trace.get(part, {})
        if isinstance(info, dict) and "error" in info:
            code = f" code={info['code']}" if "code" in info else ""
            print(f"safety gate {info.get('stage', part)} failed on {pid}: {info['error']}{code}", file=sys.stderr)


def _write_trace(path: Path, pid: str, decision: Decision) -> None:
    row = {
        "id": pid,
        "keep": decision.keep,
        "minor": decision.minor,
        "path": decision.path,
        "reasons": list(decision.reasons),
        **decision.trace,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_shadow(project: Project, pid: str, decision: Decision) -> None:
    """Shadow v3: what it would have decided next to the strict decision (sha12 id, codes, numbers; no image/text)."""
    project.root.mkdir(parents=True, exist_ok=True)
    row = {"id": pid, "strict_keep": decision.keep, "shadow": decision.shadow}
    with (project.root / "gate_v3_shadow.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_drops(project: Project, drops: Sequence[Drop]) -> None:
    project.root.mkdir(parents=True, exist_ok=True)
    with project.drops.open("a", encoding="utf-8") as handle:
        for drop in drops:
            handle.write(
                json.dumps({"id": drop.id, "name": drop.name, "reason": drop.reason}, ensure_ascii=False) + "\n"
            )
