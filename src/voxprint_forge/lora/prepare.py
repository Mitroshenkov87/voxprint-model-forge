"""``lora prepare``."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

from voxprint_forge.lora.captioner import DEFAULT, Captioner
from voxprint_forge.lora.consent import Subject
from voxprint_forge.lora.errors import LoraBlocked, LoraStageError, LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import AVG_MIN, RULE_VERSION, Decision, Gate, child_keyword
from voxprint_forge.lora.photos import (
    BLUR_THRESHOLD,
    DUP_THRESHOLD,
    MIN_SIDE,
    Drop,
    caption_marks_collage,
    check_count,
    crop,
    dhash,
    filter_reason,
    open_photo,
    photo_id,
)
from voxprint_forge.lora.project import Item, Project
from voxprint_forge.lora.refusal import announce, outside_simple, report_from_drops, scan_files
from voxprint_forge.lora.train import MIN_PHOTOS
from voxprint_forge.lora.trigger import prepend_trigger
from voxprint_forge.mdl_app import missing_id_text


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
) -> dict[str, object]:
    if mode not in MIN_PHOTOS:
        raise LoraUsageError("lora.err.not_prepared")
    if mode == "simple" and files and outside_simple(len(files)):
        announce(
            progress,
            scan_files(files, min_side=min_side, blur_threshold=blur_threshold, dup_threshold=dup_threshold),
        )
    warning = check_count(len(files), mode)
    if warning:
        progress.emit("prepare.count", warning, {"count": len(files)})
    attested = subject.kind == "person" and subject.adult_confirmed and subject.consent_obtained
    if subject.kind == "person" and not attested:
        raise LoraUsageError("lora.err.no_consent")
    if not gate.available:
        progress.emit("prepare.safety", "lora.progress.blocked", reason_key="lora.reason.safety", state="blocked")
        ids = missing_id_text()
        if not ids:
            raise LoraBlocked("lora.err.not_prepared")
        raise LoraBlocked("lora.err.gate_missing", ids=ids)
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
    paths = {photo_id(p): Path(p) for p in files}
    internal = bool(getattr(gate, "internal", False))
    checked: list[tuple[str, object, Decision, str | None]] = []
    for index, (pid, image) in enumerate(kept, 1):
        progress.emit(
            "prepare.safety",
            "lora.progress.safety",
            {"n": index, "total": len(kept)},
            "lora.reason.safety",
            done=index,
            total=len(kept),
        )
        decision = gate.check(image, attested=attested)
        if child_keyword(names.get(pid, ""), *_metadata_text(paths.get(pid))):
            decision = Decision(
                (*decision.reasons, "child_keyword"),
                True,
                decision.path,
                decision.trace,
                decision.audit,
                decision.shadow,
            )
        _log_stage_errors(decision)
        key = _identity(gate, image) if not internal else None
        checked.append((pid, image, decision, key))
    safe: list[tuple[str, object]] = []
    minor = False
    for pid, image, keep, is_minor in _aggregate(checked, internal):
        if keep:
            safe.append((pid, image))
        else:
            drops.append(Drop(pid, names.get(pid, pid), "lora.drop.safety"))
            minor = minor or is_minor
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
    caption_drops: list[Drop] = []
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
                long = str(text.get("long", ""))
                short = str(text.get("short", ""))
                if caption_marks_collage(long, short):
                    prepared = project.prepared / f"{pid}.png"
                    if prepared.is_file():
                        prepared.unlink()
                    caption_drops.append(Drop(pid, names.get(pid, pid), "lora.drop.collage"))
                    continue
                flag = text.get("unclothed")
                new_items.append(
                    Item(
                        pid,
                        f"prepared/{pid}.png",
                        prepend_trigger(subject.trigger, long),
                        prepend_trigger(subject.trigger, short),
                        round_no,
                        flag if isinstance(flag, bool) else None,
                    )
                )
        finally:
            progress.emit(
                "prepare.unload_captioner", "lora.progress.unload_captioner", reason_key="lora.reason.unload_captioner"
            )
            captioner.unload()
        caption_name = captioner.name
    else:
        caption_name = ""
    if caption_drops:
        drops.extend(caption_drops)
        _write_drops(project, caption_drops)
    items = [_with_trigger(subject.trigger, item) for item in existing + new_items]
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
    if mode == "simple":
        announce(progress, report_from_drops(drops, len(items), len(files)))
    if len(items) < floor:
        raise LoraUsageError("lora.err.too_few", need=floor - len(items))
    progress.emit("prepare.done", "lora.progress.done", state="done")
    return summary


def _with_trigger(trigger: str, item: Item) -> Item:
    long = prepend_trigger(trigger, item.long)
    short = prepend_trigger(trigger, item.short)
    if long == item.long and short == item.short:
        return item
    return Item(item.id, item.file, long, short, item.round, item.unclothed)


def _base(project: Project) -> dict[str, object]:
    try:
        return project.read()
    except LoraUsageError:
        return {}


def _metadata_text(path: Path | None) -> list[str]:
    out: list[str] = []
    if path is None:
        return out
    try:
        from PIL import Image

        with Image.open(path) as image:
            out += [v for v in image.info.values() if isinstance(v, str)]
            exif = image.getexif()
            for tag in (0x010E, 0x9C9E, 0x9C9F, 0x9C9C):
                value = exif.get(tag)
                if isinstance(value, bytes):
                    value = value.decode("utf-16-le", "ignore")
                if isinstance(value, str):
                    out.append(value)
    except Exception:  # noqa: BLE001
        pass
    return out


def _identity(gate: Gate, image: object) -> str | None:
    identify = getattr(gate, "identify", None)
    if not callable(identify):
        return None
    try:
        return identify(image)
    except Exception:  # noqa: BLE001
        return None


def _aggregate(checked: list[tuple[str, object, Decision, str | None]], internal: bool):
    groups: dict[str, list[float]] = {}
    for _, _, decision, key in checked:
        est = decision.trace.get("estimate")
        if key is not None and isinstance(est, (int, float)):
            groups.setdefault(key, []).append(float(est))
    for pid, image, decision, key in checked:
        if internal or key is None or key not in groups:
            yield pid, image, decision.keep, decision.minor
            continue
        mean = sum(groups[key]) / len(groups[key])
        others = [r for r in decision.reasons if r != "average_under_18"]
        if mean < AVG_MIN:
            yield pid, image, False, True
        else:
            yield pid, image, not others, decision.minor and bool(others) and others != ["single_under_16"]


def _log_stage_errors(decision: Decision) -> None:
    import sys

    for part in ("judge", "estimator"):
        info = decision.trace.get(part, {})
        if isinstance(info, dict) and "error" in info:
            print(f"safety gate {info.get('stage', part)} failed: {info['error']}", file=sys.stderr)


def _write_drops(project: Project, drops: Sequence[Drop]) -> None:
    project.root.mkdir(parents=True, exist_ok=True)
    with project.drops.open("a", encoding="utf-8") as handle:
        for drop in drops:
            handle.write(
                json.dumps({"id": drop.id, "name": drop.name, "reason": drop.reason}, ensure_ascii=False) + "\n"
            )
