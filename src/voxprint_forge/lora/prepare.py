"""``lora prepare``."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NoReturn

from voxprint_forge.lora.captioner import DEFAULT, Captioner
from voxprint_forge.lora.consent import Subject
from voxprint_forge.lora.errors import LoraBlocked, LoraStageError, LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import (
    AVG_MIN,
    INDICATORS,
    RULE_VERSION,
    UNAVAILABLE,
    Decision,
    Gate,
    JobDecision,
    child_keyword,
    decide_job,
)
from voxprint_forge.lora.job_gate import caption_texts, check_subject, check_texts, pin, refuse, unpin
from voxprint_forge.lora.limits import need_more
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

# Expert mode's line per dropped or minor-flagged photo: these gate_trace.jsonl fields.
_PRINTED = (
    "tier",
    "reasons",
    "judge",
    "reason_code",
    "markers",
    "judge_min_age",
    "judge_confidence",
    "mivolo_ages",
    "estimate",
)


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
    # Until this round passes, the project is not approved for training.
    unpin(project)
    # A child keyword in the trigger, folder or subject names refuses before any check, at any ratio.
    check_subject(project, {"kind": subject.kind, "subject": subject.to_dict()})
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
    keyword = False
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
                tier="strong",
            )
            keyword = True
        _log_stage_errors(decision)
        key = _identity(gate, image) if not internal else None
        checked.append((pid, image, decision, key))
    tiers = list(_aggregate(checked, internal))
    job = decide_job(tiers, keyword=keyword)
    safe: list[tuple[str, object]] = []
    unavailable = any(set(decision.reasons) & set(UNAVAILABLE) for _, _, decision, _ in checked)
    trace: list[dict[str, object]] = []
    for (pid, image, decision, _), tier in zip(checked, tiers, strict=True):
        keep = tier == "clear"
        row = _trace_row(pid, names.get(pid, pid), decision, keep, tier)
        trace.append(row)
        if keep:
            safe.append((pid, image))
        else:
            drops.append(Drop(pid, names.get(pid, pid), "lora.drop.safety"))
            if mode == "expert":
                _print_trace_row(row)
    if mode == "expert" and checked:
        _print_job(job)
    # Before the unload and before a refusal, so a refused run can still be measured.
    _write_gate_trace(project, trace)
    for backend in (gate.judge, gate.estimator):
        unload = getattr(backend, "unload", None)
        if callable(unload):
            unload()
    _write_drops(project, drops)
    if job.refuse:
        _refuse(project, subject, mode, progress, "photo")
    if unavailable:
        _refuse(project, subject, mode, progress, "check_unavailable", "lora.err.check_unavailable")
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
                try:
                    text = captioner.caption(crop(image, crop_mode))
                except Exception:  # noqa: BLE001
                    refuse(project, "captioner_error", "lora.err.check_unavailable")
                long = str(text.get("long", ""))
                short = str(text.get("short", ""))
                if child_keyword(long, short):
                    # A child keyword refuses the whole job at any ratio, captions included.
                    for done, _ in safe:
                        (project.prepared / f"{done}.png").unlink(missing_ok=True)
                    _write_drops(project, [Drop(pid, names.get(pid, pid), "lora.drop.safety")])
                    if mode == "expert":
                        _print_caption_refusal(names.get(pid, pid), pid)
                    _refuse(project, subject, mode, progress, "caption")
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
    check_texts(project, {"caption": caption_texts(items)})
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
    pin(project)
    summary = {
        "state": "done",
        "prepared": len(items),
        "new": len(new_items),
        "set_aside": len(drops),
        "round": round_no,
    }
    report = report_from_drops(drops, len(items), len(files))
    if mode == "simple":
        announce(progress, report)
    elif drops:
        # Expert skips the Simple-mode hints, but still says how many photos were set aside and why.
        progress.emit("prepare.refuse", report.summary_key(), report.params())
    need = need_more(len(items), mode)
    if need:
        raise LoraUsageError("lora.err.too_few", need=need)
    progress.emit("prepare.done", "lora.progress.done", state="done")
    return summary


def _refuse(
    project: Project, subject: Subject, mode: str, progress: Progress, source: str, key: str = "lora.err.blocked"
) -> NoReturn:
    """Refuse the whole job: not approved, the ``job`` drop row and the audit source (``job_gate.refuse``)."""
    project.write({**_base(project), "subject": subject.to_dict(), "mode": mode, "gate_passed": False})
    progress.emit("prepare.safety", "lora.progress.blocked", reason_key="lora.reason.safety", state="blocked")
    refuse(project, source, key)


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
    """Each photo's tier. A person whose photos average under 18 makes all of their photos strong."""
    groups: dict[str, list[float]] = {}
    for _, _, decision, key in checked:
        est = decision.trace.get("estimate")
        if key is not None and isinstance(est, (int, float)):
            groups.setdefault(key, []).append(float(est))
    for _, _, decision, key in checked:
        if not internal and key in groups and sum(groups[key]) / len(groups[key]) < AVG_MIN:
            yield "strong"
        else:
            yield decision.tier


def _log_stage_errors(decision: Decision) -> None:
    import sys

    for part in ("judge", "estimator"):
        info = decision.trace.get(part, {})
        if isinstance(info, dict) and "error" in info:
            print(f"safety gate {info.get('stage', part)} failed: {info['error']}", file=sys.stderr)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _trace_row(pid: str, name: str, decision: Decision, keep: bool, tier: str) -> dict[str, object]:
    """One ``gate_trace.jsonl`` row: codes and numbers only, no image data, paths or metadata."""
    judge = decision.trace.get("judge")
    judge = judge if isinstance(judge, dict) else {}
    verdict = judge.get("verdict")
    verdict = verdict if isinstance(verdict, dict) else {}
    indicators = verdict.get("indicators")
    indicators = indicators if isinstance(indicators, dict) else {}
    estimator = decision.trace.get("estimator")
    estimator = estimator if isinstance(estimator, dict) else {}
    ages = estimator.get("ages")
    if judge.get("ok"):
        state = "ok"
    elif "error" in judge:
        state = ":".join(str(judge[k])[:64] for k in ("error", "code") if k in judge)
    else:
        state = "missing"
    reason_code = verdict.get("reason_code")
    return {
        "file": Path(name).name,
        "id": pid,
        "judge": state,
        "reason_code": str(reason_code) if reason_code is not None else None,
        "markers": [key for key in INDICATORS if indicators.get(key) == "present"],
        "judge_min_age": _number(verdict.get("min_age")),
        "judge_confidence": _number(verdict.get("confidence")),
        "mivolo_ages": [_number(age) for age in ages] if isinstance(ages, list) else [],
        "estimate": _number(decision.trace.get("estimate")),
        "tier": tier,
        "decision": "keep" if keep else ("minor" if tier == "strong" else "drop"),
        "reasons": [str(reason) for reason in decision.reasons],
    }


def _print_trace_row(row: dict[str, object]) -> None:
    """Expert mode: one line per dropped or minor-flagged photo, with the gate numbers."""
    import sys

    def show(value: object) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:g}"
        if isinstance(value, list):
            return ",".join(show(v) for v in value) or "-"
        return str(value)

    numbers = " ".join(f"{key}={show(row[key])}" for key in _PRINTED)
    print(f"safety gate: {row['file']} ({row['id']}) {row['decision']}: {numbers}", file=sys.stderr)


def _print_job(job: JobDecision) -> None:
    """Expert mode: the whole-job verdict and the tier counts it came from."""
    import sys

    counts = " ".join(f"{tier}={count}" for tier, count in job.counts.items())
    outcome = "refused" if job.refuse else "trains"
    print(f"safety gate job: {outcome} ({job.why}) n={job.n} {counts} allowance={job.allowance}", file=sys.stderr)


def _print_caption_refusal(name: str, pid: str) -> None:
    import sys

    print(f"safety gate: {Path(name).name} ({pid}) minor: reasons=child_keyword_caption", file=sys.stderr)


def _write_gate_trace(project: Project, rows: Sequence[dict[str, object]]) -> None:
    project.root.mkdir(parents=True, exist_ok=True)
    with project.gate_trace.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_drops(project: Project, drops: Sequence[Drop]) -> None:
    """Record ``drops`` in ``drops.jsonl``, one line per photo id.

    A re-run checks photos set aside earlier again. Their old line is replaced, not repeated.
    Rows of other ids, including ``job`` rows, stay as they are and in order.
    """
    project.root.mkdir(parents=True, exist_ok=True)
    fresh: dict[str, str] = {}
    for drop in drops:
        fresh[drop.id] = json.dumps({"id": drop.id, "name": drop.name, "reason": drop.reason}, ensure_ascii=False)
    kept: list[str] = []
    if project.drops.is_file():
        for line in project.drops.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = None
            if isinstance(row, dict) and str(row.get("id")) in fresh and row.get("scope") != "job":
                continue
            kept.append(line)
    text = "".join(line + "\n" for line in (*kept, *fresh.values()))
    project.drops.write_text(text, encoding="utf-8", newline="\n")
