"""``lora prepare``."""

from __future__ import annotations

import gc
import io
import json
import os
import shutil
import sys
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import NoReturn

from voxprint_forge.lora import DEFAULT_BASE
from voxprint_forge.lora.backends import gate_unavailable_error
from voxprint_forge.lora.captioner import DEFAULT, Captioner, unclothed_flag
from voxprint_forge.lora.consent import Subject
from voxprint_forge.lora.errors import LoraStageError, LoraUsageError
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
from voxprint_forge.lora.gate_run import (
    append_drop,
    append_trace_row,
    atomic_text,
    complete,
    rewrite_trace_tail,
    start,
    trace_line_count,
)
from voxprint_forge.lora.job_gate import caption_texts, check_subject, check_texts, pin, refuse, unpin
from voxprint_forge.lora.limits import need_more
from voxprint_forge.lora.photos import (
    BLUR_THRESHOLD,
    DUP_THRESHOLD,
    KEEPABLE,
    MIN_SIDE,
    Drop,
    bytes_id,
    caption_marks_collage,
    check_count,
    crop,
    decode_photo,
    dhash,
    filter_reason,
    measure,
    near_duplicate,
    photo_id,
)
from voxprint_forge.lora.project import Item, Project
from voxprint_forge.lora.refusal import announce, outside_simple, report_from_drops, scan_files
from voxprint_forge.lora.train import MIN_PHOTOS
from voxprint_forge.lora.trigger import prepend_trigger

# Near-duplicates are kept unless ``drop_near_duplicates`` is set.
_NEAR_DUPLICATE = frozenset({"lora.drop.duplicate"})
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
# The prepared-photo hash pass and the PNG writes run on this many threads (Pillow and
# numpy release the GIL for the heavy parts). Results are taken in input order, so the
# output is the same as one thread's. Photos themselves are decoded one at a time.
WORKERS = max(1, min(4, os.cpu_count() or 1))


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
    keep: Sequence[Path] = (),
    drop_near_duplicates: bool = False,
    bases: Sequence[str] | None = None,
) -> dict[str, object]:
    """Check, crop and caption ``files``.

    A near-duplicate (a photo whose difference hash is within ``dup_threshold`` bits of an
    earlier one) is kept with a ``lora.warn.near_duplicate`` warning that names both files.
    ``drop_near_duplicates`` sets it aside as ``lora.drop.duplicate`` instead. A byte-identical
    copy of a photo already taken is always skipped.

    ``keep`` names photos (also taken as input) that stay in despite a collage or duplicate
    drop. They still go through every other check, the safety check included.

    ``bases`` is the MDL id list for this project (``moss-image-base``, ``z-image``, or
    both). It is stored on ``project.json`` only. Captions and item rows stay base-free.
    ``None`` on a brand-new project stores MOSS. ``None`` on a legacy project does not
    add the key, so its checkpoint folders stay where they are.
    """
    prior = _base(project)
    files = _with_keep(files, keep)
    if mode not in MIN_PHOTOS:
        raise LoraUsageError("lora.err.not_prepared")
    if mode == "simple" and files and outside_simple(len(files)):
        announce(
            progress,
            scan_files(
                files,
                min_side=min_side,
                blur_threshold=blur_threshold,
                dup_threshold=dup_threshold,
                drop_near_duplicates=drop_near_duplicates,
            ),
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
        raise gate_unavailable_error()
    # Before the first photo. A crash leaves this ``running``; training will not start.
    start(project)
    existing = project.items()
    known = {item.id for item in existing}
    round_no = max((item.round for item in existing), default=-1) + 1 if existing else 0
    seen: list[int] = []
    # The file name behind each hash in ``seen``, for the near-duplicate warning.
    seen_names: list[str] = []
    prepared_hash = _ordered(lambda item: _hash_of(open_photo(project.resolve(item.file))), existing)
    for item, value in zip(existing, prepared_hash, strict=True):
        if value is not None:
            seen.append(value)
            seen_names.append(item.file)
    keep_ids = {photo_id(path) for path in keep}
    drops: list[Drop] = []
    # Photos that pass the filters, as their file bytes: every phase decodes one photo at a
    # time from these bytes, so the same pixels reach every check and the full-size decoded
    # pictures are never all in memory together.
    kept: list[tuple[str, bytes]] = []
    kept_ids: set[str] = set()
    # Last file name (and path) per photo id, as ``{photo_id(p): ... for p in files}`` would give.
    names: dict[str, str] = {}
    paths: dict[str, Path] = {}
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
        blob = Path(path).read_bytes()
        pid = bytes_id(blob)
        names[pid] = Path(path).name
        paths[pid] = Path(path)
        if pid in known or pid in kept_ids:
            continue
        image = open_photo(io.BytesIO(blob))
        measures = measure(image, min_side)
        if pid in keep_ids:
            allow = KEEPABLE
        elif drop_near_duplicates:
            allow = frozenset()
        else:
            allow = _NEAR_DUPLICATE
        reason = filter_reason(
            image,
            seen,
            min_side=min_side,
            blur_threshold=blur_threshold,
            dup_threshold=dup_threshold,
            allow=allow,
            measures=measures,
        )
        if reason:
            drops.append(Drop(pid, Path(path).name, reason))
            continue
        # ``filter_reason`` added this photo's hash last; compare it with the ones before it.
        match = near_duplicate(seen[-1], seen[:-1], dup_threshold)
        if match is not None:
            progress.emit(
                "prepare.check",
                "lora.warn.near_duplicate",
                {"file": Path(path).name, "other": seen_names[match]},
            )
        seen_names.append(Path(path).name)
        kept.append((pid, blob))
        kept_ids.add(pid)
    image = None
    # Phase 1: the safety check, with only the judge and the age estimator loaded.
    try:
        checked, tiers, job = _safety_phase(project, gate, kept, names, paths, mode, attested, progress)
    except BaseException:
        _unload_gate(gate, quiet=True)
        raise
    kept.clear()
    # Phase 2: both leave memory before anything else loads.
    _unload_gate(gate)
    safe: list[tuple[str, bytes]] = []
    unavailable = any(set(decision.reasons) & set(UNAVAILABLE) for _, _, decision, _ in checked)
    for (pid, blob, _, _), tier in zip(checked, tiers, strict=True):
        if tier == "clear":
            safe.append((pid, blob))
        else:
            drops.append(Drop(pid, names.get(pid, pid), "lora.drop.safety"))
    del checked, tiers
    _write_drops(project, drops)
    if job.refuse:
        _refuse(project, subject, mode, progress, "photo", prior=prior, bases=bases)
    if unavailable:
        _refuse(
            project, subject, mode, progress, "check_unavailable", "lora.err.check_unavailable",
            prior=prior, bases=bases,
        )
    project.prepared.mkdir(parents=True, exist_ok=True)
    # The 1024 px crops, kept for the captioner so each photo is decoded and cropped once
    # here. Each photo is decoded one at a time from its kept bytes; the PNG writes then run
    # on up to WORKERS threads, in input order.
    crops: list[tuple[str, object]] = []
    for pid, blob in safe:
        image = open_photo(io.BytesIO(blob))
        crops.append((pid, crop(image, crop_mode)))
    del safe
    image = None
    written = _ordered(lambda row: row[1].save(project.prepared / f"{row[0]}.png"), crops)
    for index, _ in enumerate(written, 1):
        progress.emit(
            "prepare.crop",
            "lora.progress.crop",
            {"n": index, "total": len(crops)},
            "lora.reason.crop",
            done=index,
            total=len(crops),
        )
    new_items: list[Item] = []
    caption_drops: list[Drop] = []
    if crops:
        # Phase 3: the captioner, loaded only now that the gate models are gone.
        progress.emit("prepare.load_captioner", "lora.progress.load_captioner", reason_key="lora.reason.load_captioner")
        captioner = load_captioner(caption_model)
        if captioner is None:
            raise LoraStageError("lora.err.captioner_missing")
        try:
            for index, (pid, cropped) in enumerate(crops, 1):
                progress.emit(
                    "prepare.caption",
                    "lora.progress.caption",
                    {"n": index, "total": len(crops)},
                    "lora.reason.caption",
                    done=index,
                    total=len(crops),
                )
                try:
                    text = captioner.caption(cropped)
                except Exception:  # noqa: BLE001
                    complete(project)
                    refuse(project, "captioner_error", "lora.err.check_unavailable")
                long = str(text.get("long", ""))
                short = str(text.get("short", ""))
                if child_keyword(long, short):
                    # A child keyword refuses the whole job at any ratio, captions included.
                    for done, _ in crops:
                        (project.prepared / f"{done}.png").unlink(missing_ok=True)
                    drop = Drop(pid, names.get(pid, pid), "lora.drop.safety")
                    append_drop(project, _drop_row(drop))
                    _write_drops(project, [drop])
                    if mode == "expert":
                        _print_caption_refusal(names.get(pid, pid), pid)
                    _refuse(project, subject, mode, progress, "caption", prior=prior, bases=bases)
                if caption_marks_collage(long, short) and pid not in keep_ids:
                    prepared = project.prepared / f"{pid}.png"
                    if prepared.is_file():
                        prepared.unlink()
                    caption_drops.append(Drop(pid, names.get(pid, pid), "lora.drop.collage"))
                    continue
                # The captioner's text is stored as written; the flag follows the text.
                new_items.append(
                    Item(
                        pid,
                        f"{project.prepared.name}/{pid}.png",
                        prepend_trigger(subject.trigger, long),
                        prepend_trigger(subject.trigger, short),
                        round_no,
                        unclothed_flag(text.get("unclothed"), long, short),
                    )
                )
        finally:
            progress.emit(
                "prepare.unload_captioner", "lora.progress.unload_captioner", reason_key="lora.reason.unload_captioner"
            )
            captioner.unload()
            _free_memory()
        caption_name = captioner.name
        # Phase 4: nothing of the captioner stays referenced here.
        del captioner
    else:
        caption_name = ""
    if caption_drops:
        drops.extend(caption_drops)
        _write_drops(project, caption_drops)
    items = [_with_trigger(subject.trigger, item) for item in existing + new_items]
    # The check itself finished. A text refusal is a finished run, not an interrupted one.
    complete(project)
    check_texts(project, {"caption": caption_texts(items)})
    project.write_items(items)
    if release is not None:
        project.release.mkdir(parents=True, exist_ok=True)
        shutil.copy2(release, project.release / Path(release).name)
    data = _base(project)
    payload = {
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
    # Pin first, record its sha, then publish gate_passed. A crash cannot look approved
    # while the run is still ``running``.
    pin(project)
    complete(project)
    project.write(_with_bases(payload, prior, bases))
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


def open_photo(source: Path | io.IOBase) -> object:
    """The decoded photo at ``source`` — a path, or a byte stream read from one — or ``None``.

    ``photos.decode_photo`` does the decoding; the wrapper keeps every decode in this module
    on one name. A path that cannot be opened is an unreadable photo; decode errors are
    ``decode_photo``'s own ``None``, everything else propagates.
    """
    if isinstance(source, io.IOBase):
        return decode_photo(source)
    try:
        with Path(source).open("rb") as handle:
            return decode_photo(handle)
    except OSError:
        return None


def _ordered[T, R](work: Callable[[T], R], items: Iterable[T], workers: int | None = None) -> Iterator[R]:
    """``work(item)`` for each item, yielded in input order, run on up to ``workers`` threads.

    At most ``2 * workers`` items are in flight, which bounds memory. Stopping early cancels the rest.
    """
    workers = WORKERS if workers is None else workers
    if workers <= 1:
        yield from map(work, items)
        return
    pending: deque[Future[R]] = deque()
    with ThreadPoolExecutor(workers, thread_name_prefix="lora-prepare") as pool:
        try:
            for item in items:
                pending.append(pool.submit(work, item))
                if len(pending) >= 2 * workers:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        finally:
            for future in pending:
                future.cancel()


def _hash_of(image: object) -> int | None:
    return None if image is None else dhash(image)


def _safety_phase(
    project: Project,
    gate: Gate,
    kept: Sequence[tuple[str, bytes]],
    names: dict[str, str],
    paths: dict[str, Path],
    mode: str,
    attested: bool,
    progress: Progress,
) -> tuple[list[tuple[str, bytes, Decision, str | None]], list[str], JobDecision]:
    """Run the gate over ``kept``, one decoded photo at a time, and write ``gate_trace.jsonl``.

    Returns the checked photos (as bytes), their tiers and the job decision. Each decision
    is on disk before the next photo is checked, so a crash keeps every row written so far.
    """
    internal = bool(getattr(gate, "internal", False))
    checked: list[tuple[str, bytes, Decision, str | None]] = []
    keyword = False
    # Rows already on disk from an earlier run stay. This run appends after them.
    trace_from = trace_line_count(project)
    provisional: list[dict[str, object]] = []
    for index, (pid, blob) in enumerate(kept, 1):
        progress.emit(
            "prepare.safety",
            "lora.progress.safety",
            {"n": index, "total": len(kept)},
            "lora.reason.safety",
            done=index,
            total=len(kept),
        )
        image = open_photo(io.BytesIO(blob))
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
        del image
        checked.append((pid, blob, decision, key))
        # The decision is on disk before the next photo. A crash keeps this row.
        row = _trace_row(pid, names.get(pid, pid), decision, decision.tier == "clear", decision.tier)
        provisional.append(row)
        append_trace_row(project, row)
        if decision.tier != "clear":
            append_drop(project, _drop_row(Drop(pid, names.get(pid, pid), "lora.drop.safety")))
    tiers = list(_aggregate(checked, internal))
    job = decide_job(tiers, keyword=keyword)
    final_rows: list[dict[str, object]] = []
    for (pid, _, decision, _), tier in zip(checked, tiers, strict=True):
        keep = tier == "clear"
        row = _trace_row(pid, names.get(pid, pid), decision, keep, tier)
        final_rows.append(row)
        if not keep and mode == "expert":
            _print_trace_row(row)
        if not keep and decision.tier == "clear":
            # A cross-photo average can turn a clear photo into a drop after it was written.
            append_drop(project, _drop_row(Drop(pid, names.get(pid, pid), "lora.drop.safety")))
    if final_rows != provisional:
        rewrite_trace_tail(project, trace_from, final_rows)
    if mode == "expert" and checked:
        _print_job(job)
    return checked, tiers, job


def _unload_gate(gate: Gate, *, quiet: bool = False) -> None:
    """Unload the judge and the age estimator, then free what they left behind.

    ``quiet`` is for the error path: an unload failure there does not hide the first error.
    """
    for backend in (gate.judge, gate.estimator):
        unload = getattr(backend, "unload", None)
        if callable(unload):
            try:
                unload()
            except Exception:
                if not quiet:
                    raise
    _free_memory()


def _free_memory() -> None:
    """Collect garbage and return cached graphics memory, without importing torch when nothing loaded it."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - freeing the cache is best effort
        pass


def _with_keep(files: Sequence[Path], keep: Sequence[Path]) -> list[Path]:
    """``files`` plus any ``keep`` photo not already among them (same file, compared by resolved path)."""
    out = list(files)
    known = {Path(path).resolve() for path in out}
    for path in keep:
        if Path(path).resolve() not in known:
            known.add(Path(path).resolve())
            out.append(Path(path))
    return out


def _refuse(
    project: Project,
    subject: Subject,
    mode: str,
    progress: Progress,
    source: str,
    key: str = "lora.err.blocked",
    *,
    prior: dict[str, object] | None = None,
    bases: Sequence[str] | None = None,
) -> NoReturn:
    """Refuse the whole job: not approved, the ``job`` drop row and the audit source (``job_gate.refuse``)."""
    # The check finished. Leaving the marker ``running`` would hide this refusal behind
    # a re-run block the next time training starts.
    complete(project)
    body = {**_base(project), "subject": subject.to_dict(), "mode": mode, "gate_passed": False}
    project.write(_with_bases(body, prior or {}, bases))
    progress.emit("prepare.safety", "lora.progress.blocked", reason_key="lora.reason.safety", state="blocked")
    refuse(project, source, key)


def _bases_to_store(prior: dict[str, object], bases: Sequence[str] | None) -> list[str] | None:
    """The ``bases`` list to write, or None to leave the key as the payload already has it.

    An explicit list replaces the stored one. A brand-new project (empty record)
    stores MOSS when the caller passes no list. A legacy project is left without
    the key unless the caller passes a list.
    """
    if bases is not None:
        cleaned = [str(item).strip() for item in bases if str(item).strip()]
        return cleaned or [DEFAULT_BASE]
    if not prior:
        return [DEFAULT_BASE]
    return None


def _with_bases(payload: dict[str, object], prior: dict[str, object], bases: Sequence[str] | None) -> dict[str, object]:
    """``payload`` plus ``bases`` when this prepare should record them."""
    chosen = _bases_to_store(prior, bases)
    if chosen is None:
        return payload
    return {**payload, "bases": chosen}


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
        **_face_fields(decision.trace.get("faces")),
    }


def _face_fields(select: object) -> dict[str, object]:
    if not isinstance(select, dict):
        return {"face_rule": None, "face_why": None, "faces": [], "primary": None, "stray": []}
    faces = select.get("faces")
    return {
        "face_rule": select.get("rule"),
        "face_why": select.get("why"),
        "faces": [
            {
                "i": face.get("i"),
                "box": [_number(v) for v in face.get("box") or []],
                "area": _number(face.get("area")),
                "score": _number(face.get("score")),
                "age": _number(face.get("age")),
            }
            for face in (faces if isinstance(faces, list) else [])
            if isinstance(face, dict)
        ],
        "primary": select.get("primary"),
        "stray": list(select.get("stray") or []),
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


def _drop_row(drop: Drop) -> dict[str, str]:
    return {"id": drop.id, "name": drop.name, "reason": drop.reason}


def _write_drops(project: Project, drops: Sequence[Drop]) -> None:
    """Record ``drops`` in ``drops.jsonl``, one line per photo id.

    A re-run checks photos set aside earlier again. Their old line is replaced, not repeated.
    Rows of other ids, including ``job`` rows, stay as they are and in order.
    Safety rows may already have been appended; this compaction is what removes the copies.
    The rewrite goes to a temp file and ``os.replace``, so a crash keeps the previous file.
    """
    project.root.mkdir(parents=True, exist_ok=True)
    fresh: dict[str, str] = {}
    for drop in drops:
        fresh[drop.id] = json.dumps(_drop_row(drop), ensure_ascii=False)
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
    atomic_text(project.drops, text)
