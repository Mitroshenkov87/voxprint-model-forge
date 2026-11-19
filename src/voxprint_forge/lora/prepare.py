"""``lora prepare``."""

from __future__ import annotations

import gc
import io
import json
import os
import re
import shutil
import sys
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import NoReturn

from voxprint_forge.imageio_save import save_lossless_webp
from voxprint_forge.lora import DEFAULT_BASE
from voxprint_forge.lora.backends import gate_unavailable_error
from voxprint_forge.lora.batching import plan_batches, plan_weights, prepare_batch_size, size_class
from voxprint_forge.lora.captioner import DEFAULT, Captioner, unclothed_flag
from voxprint_forge.lora.consent import Subject
from voxprint_forge.lora.errors import LoraStageError, LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import (
    AGE_FLAG_DROPPED,
    AGE_FLAG_KEEP,
    AVG_MIN,
    INDICATORS,
    NO_FACE_KEEP_ATTESTED,
    RULE_VERSION,
    UNAVAILABLE,
    Decision,
    Gate,
    JobDecision,
    apply_attested_keeps,
    apply_no_face_majority,
    child_keyword,
    child_keyword_match,
    decide_job,
    normalize_text,
)
from voxprint_forge.lora.gate_run import (
    append_drop,
    append_trace_row,
    atomic_text,
    complete,
    note,
    rewrite_trace_tail,
    start,
    trace_line_count,
)
from voxprint_forge.lora.job_gate import caption_texts, check_subject, check_texts, pin, refuse, unpin
from voxprint_forge.lora.limits import need_more
from voxprint_forge.lora.photos import (
    BLUR_THRESHOLD,
    CROP_MODES,
    DUP_THRESHOLD,
    KEEPABLE,
    MIN_SIDE,
    Drop,
    bytes_id,
    caption_marks_collage,
    check_count,
    crop,
    crop_closeup,
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
# ``both`` writes a whole-frame and a head-and-shoulders image per photo; the captioner
# output carries one short framing phrase per prepared image.
_FULL_ID = "-full"
_CLOSE_ID = "-close"
_FULL_PHRASE = "full body"
_CLOSE_PHRASE = "close-up"
# A child word in a generated caption: re-caption tries without and with the release.txt
# attestation (``Adult (18+): yes`` and ``No minors in photos: yes``). With it, a word that
# still comes back is replaced by ADULT_WORD before the frame is dropped.
RECAPTIONS = 1
RECAPTIONS_ATTESTED = 2
ADULT_WORD = "adult"
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
# A prepared crop is lossless WebP. Projects prepared by an older version hold PNG crops;
# those still read (``prepared_path`` falls back to the legacy file).
PREPARED_SUFFIX = ".webp"
LEGACY_PREPARED_SUFFIX = ".png"
# The prepared-photo hash pass and the crop writes run on this many threads (Pillow and
# numpy release the GIL for the heavy parts). Results are taken in input order, so the
# output is the same as one thread's. Photos themselves are decoded one at a time, except
# that with a safety-check batch over 1 the next size-aware chunk (see ``plan_batches``) is
# decoded on these threads while the current one is checked (at most two chunks of decoded
# photos are held; heavy photos come in smaller chunks).
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
    crop_mode: str = "both",
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

    ``crop_mode`` is ``both`` (the default), ``subject`` or ``whole`` (Expert). ``both``
    writes the whole frame as ``{id}-full`` for every kept photo and, when the safety gate
    found a face, a head-and-shoulders close-up as ``{id}-close``; without a face the
    whole frame is the only image. The stored captions carry a short framing phrase
    (``full body`` / ``close-up``). Gate, drop and hash decisions stay tied to the source
    photo either way.

    ``bases`` is the MDL id list for this project (for example ``z-image`` or
    ``z-image-turbo``). It is stored on ``project.json`` only. Captions and item rows stay
    base-free. ``None`` on a brand-new project stores the default base (Z-Image). ``None``
    on a legacy project does not add the key, so its checkpoint folders stay where they are.
    """
    prior = _base(project)
    files = _with_keep(files, keep)
    if mode not in MIN_PHOTOS:
        raise LoraUsageError("lora.err.not_prepared")
    if crop_mode not in CROP_MODES:
        raise LoraUsageError("lora.err.crop", mode=str(crop_mode))
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
    note(project, no_minors_attested=subject.no_minors_attested)
    existing = project.items()
    known = {item.id for item in existing}
    if crop_mode == "both":
        # Each kept photo prepares as ``{id}-full`` and, with a face, ``{id}-close``:
        # a re-run must recognise a photo already prepared under either id.
        known = known | {item.id[: -len(_FULL_ID)] for item in existing if item.id.endswith(_FULL_ID)}
        known = known | {item.id[: -len(_CLOSE_ID)] for item in existing if item.id.endswith(_CLOSE_ID)}
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
    # pictures are never all in memory together. RAM cost: the encoded files themselves, about
    # 2-10 MB per camera JPEG, so up to a few GB for a 500-photo set (the decoded pictures
    # would be several times that). The clear photos' bytes stay until their crops are written.
    kept: list[tuple[str, bytes]] = []
    kept_ids: set[str] = set()
    # Last file name (and path) per photo id, as ``{photo_id(p): ... for p in files}`` would give.
    names: dict[str, str] = {}
    paths: dict[str, Path] = {}
    # Decoded (width, height) per kept photo, for the size-aware safety-check chunks.
    sizes: dict[str, tuple[int, int]] = {}
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
        sizes[pid] = (int(image.size[0]), int(image.size[1]))
    image = None
    # Phase 1: the safety check, with only the judge and the age estimator loaded.
    try:
        check_size = prepare_batch_size("check")
        print(f"prepare: batch size {check_size} (check)", file=sys.stderr)
        # The age-flag keep limit is a share of the run's input photos (distinct photos given to this
        # prepare, the ones the filters set aside included), not of the photos left after the filters.
        checked, tiers, job, age_flags = _safety_phase(
            project,
            gate,
            kept,
            names,
            paths,
            mode,
            attested,
            progress,
            batch=check_size,
            n_inputs=len(names),
            sizes=sizes,
        )
    except BaseException:
        _unload_gate(gate, quiet=True)
        raise
    kept.clear()
    # Phase 2: both leave memory before anything else loads.
    _unload_gate(gate)
    safe: list[tuple[str, bytes]] = []
    # The gate's face box per kept photo, for the ``both`` head-and-shoulders crop. The gate
    # ran once on the source photo; the close-up is geometry only and is never re-gated.
    face_boxes: dict[str, tuple[float, float, float, float]] = {}
    unavailable = any(set(decision.reasons) & set(UNAVAILABLE) for _, _, decision, _ in checked)
    for (pid, blob, decision, _), tier in zip(checked, tiers, strict=True):
        if tier == "clear":
            safe.append((pid, blob))
            box = _face_box(decision)
            if box is not None:
                face_boxes[pid] = box
        else:
            drops.append(Drop(pid, names.get(pid, pid), "lora.drop.safety"))
    # Photos the no-face majority kept: their provisional safety drop rows are purged.
    majority = {pid for pid, _, decision, _ in checked if decision.trace.get("keep_reason")}
    del checked, tiers
    _write_drops(project, drops, purge=majority)
    if job.refuse:
        _refuse(project, subject, mode, progress, "photo", prior=prior, bases=bases)
    if unavailable:
        _refuse(
            project, subject, mode, progress, "check_unavailable", "lora.err.check_unavailable",
            prior=prior, bases=bases,
        )
    project.prepared.mkdir(parents=True, exist_ok=True)
    # The 1024 px crops, as a stream: each photo is decoded from its kept bytes on this thread,
    # cropped and dropped; the crop writes run on up to WORKERS threads, in input order, with at
    # most ``2 * WORKERS`` crops in flight. ``both`` writes the whole frame as ``{id}-full`` and,
    # when the gate found a face, a head-and-shoulders close-up as ``{id}-close``; without a face
    # the whole frame is the only image. Only the prepared ids stay; the captioner reads each
    # crop back from its own lossless WebP (a legacy PNG of an old project still reads).
    prepared_ids = _prepared_ids(safe, face_boxes, crop_mode)
    written = _ordered(
        lambda row: _save_crop(project, row), _crops_of(safe, face_boxes, crop_mode)
    )
    done_ids: list[str] = []
    for index, prepared_id in enumerate(written, 1):
        done_ids.append(prepared_id)
        progress.emit(
            "prepare.crop",
            "lora.progress.crop",
            {"n": index, "total": len(prepared_ids)},
            "lora.reason.crop",
            done=index,
            total=len(prepared_ids),
        )
    if done_ids != prepared_ids:
        raise RuntimeError(f"wrote {len(done_ids)} prepared images, expected {len(prepared_ids)}")
    del safe, face_boxes, done_ids
    new_items: list[Item] = []
    caption_drops: list[Drop] = []
    # The child word behind each ``lora.drop.caption_keyword`` row, by photo id.
    drop_match: dict[str, str] = {}
    # Every child word met in a generated caption and what was done: recaptioned, replaced
    # or dropped_frame. Stored on project.json; its count goes to gate_run.json.
    caption_fixes: list[dict[str, object]] = []
    hint = subject.adult_attested
    # ``Adult content in training: no``: frames flagged unclothed are left out of training.
    clothed_only = subject.adult_content_training is False
    unclothed_ids: set[str] = set()
    if prepared_ids:
        # Phase 3: the captioner, loaded only now that the gate models are gone.
        progress.emit("prepare.load_captioner", "lora.progress.load_captioner", reason_key="lora.reason.load_captioner")
        captioner = load_captioner(caption_model)
        if captioner is None:
            raise LoraStageError("lora.err.captioner_missing")
        try:
            caption_size = prepare_batch_size("caption")
            print(f"prepare: batch size {caption_size} (caption)", file=sys.stderr)
            collage_dropped: set[str] = set()
            first_pass = _first_captions(project, captioner, prepared_ids, hint, caption_size)
            for index, (prepared_id, cropped, first) in enumerate(first_pass, 1):
                progress.emit(
                    "prepare.caption",
                    "lora.progress.caption",
                    {"n": index, "total": len(prepared_ids)},
                    "lora.reason.caption",
                    done=index,
                    total=len(prepared_ids),
                )
                source_id = _source_id(prepared_id)
                name = names.get(source_id, source_id)
                # A child word in the captioner's own text is a captioner glitch, not a user
                # text: the image is captioned again (and, attested, the word replaced); only
                # this frame is dropped if it stays. The photos themselves passed the gate.
                text, long, short, match = _clean_caption(
                    project, captioner, cropped, hint, name, prepared_id, caption_fixes, first=first
                )
                if match is not None:
                    _print_caption_keyword(name, prepared_id, match, "dropped this frame")
                    _unlink_prepared(project, prepared_id)
                    other = collage_dropped | unclothed_ids
                    if source_id not in drop_match and source_id not in other:
                        caption_drops.append(Drop(source_id, name, "lora.drop.caption_keyword"))
                    if source_id not in other:
                        drop_match.setdefault(source_id, match)
                    continue
                if caption_marks_collage(long, short) and source_id not in keep_ids:
                    # One collage caption sets the whole photo aside: every prepared image
                    # of it is removed, the close-up included. A later image of the photo is
                    # still read back and captioned first, as before, and removed when the
                    # loop reaches it (the ``collage_dropped`` branch below or a drop above).
                    for done in prepared_ids[:index]:
                        if _source_id(done) == source_id:
                            _unlink_prepared(project, done)
                    collage_dropped.add(source_id)
                    drop_match.pop(source_id, None)
                    caption_drops.append(Drop(source_id, names.get(source_id, source_id), "lora.drop.collage"))
                    continue
                if source_id in collage_dropped:
                    _unlink_prepared(project, prepared_id)
                    continue
                # The framing phrase rides on the stored caption; the flag follows the
                # captioner's own text, not the phrase.
                flag = unclothed_flag(text.get("unclothed"), long, short)
                if clothed_only and flag is True:
                    _unlink_prepared(project, prepared_id)
                    if source_id not in unclothed_ids and source_id not in drop_match:
                        caption_drops.append(Drop(source_id, name, "lora.drop.unclothed_release"))
                    unclothed_ids.add(source_id)
                    continue
                phrase = _framing_phrase(prepared_id)
                long = _append_phrase(long, phrase)
                short = _append_phrase(short, phrase)
                new_items.append(
                    Item(
                        prepared_id,
                        f"{project.prepared.name}/{prepared_id}{PREPARED_SUFFIX}",
                        prepend_trigger(subject.trigger, long),
                        prepend_trigger(subject.trigger, short),
                        round_no,
                        flag,
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
        _write_drops(
            project,
            caption_drops,
            extra={pid: {"match": word, "action": "dropped_frame"} for pid, word in drop_match.items()},
        )
    if clothed_only:
        existing = _clothed(project, existing, drops)
    note(project, caption_fixes=len(caption_fixes))
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
        "caption_fixes": caption_fixes,
    }
    # Pin first, record its sha, then publish gate_passed. A crash cannot look approved
    # while the run is still ``running``.
    pin(project, age_flags=age_flags)
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


def _open_blob(blob: bytes) -> object:
    """:func:`open_photo` on kept file bytes."""
    return open_photo(io.BytesIO(blob))


def _decoded_chunks[T, S](
    chunks: Sequence[Sequence[tuple[T, S]]],
    *,
    prefetch: bool,
    decode: Callable[[S], object] = _open_blob,  # type: ignore[assignment]
) -> Iterator[tuple[Sequence[tuple[T, S]], list[object]]]:
    """Each chunk of ``(id, source)`` rows with its pictures decoded by ``decode``, in order.

    ``decode`` defaults to :func:`open_photo` on kept file bytes. With ``prefetch`` (and more
    than one worker) the next chunk is decoded on up to ``WORKERS`` threads while the caller
    works on this one, so at most two chunks of decoded pictures are alive. Without it each
    picture is decoded on the caller's thread, one chunk at a time. A decode error propagates
    when its chunk is reached.
    """
    if not prefetch or WORKERS <= 1:
        for chunk in chunks:
            yield chunk, [decode(source) for _, source in chunk]
        return
    with ThreadPoolExecutor(WORKERS, thread_name_prefix="lora-prepare") as pool:

        def submit(chunk: Sequence[tuple[T, S]]) -> list[Future[object]]:
            return [pool.submit(decode, source) for _, source in chunk]

        ahead = submit(chunks[0]) if chunks else []
        try:
            for position, chunk in enumerate(chunks):
                current = ahead
                ahead = submit(chunks[position + 1]) if position + 1 < len(chunks) else []
                images = [future.result() for future in current]
                del current
                yield chunk, images
                del images
        finally:
            for future in ahead:
                future.cancel()


def _prepared_ids(
    safe: Sequence[tuple[str, bytes]], face_boxes: Mapping[str, object], crop_mode: str
) -> list[str]:
    """The prepared image ids for ``safe``, in write order: what :func:`_crops_of` yields, without decoding."""
    if crop_mode != "both":
        return [pid for pid, _ in safe]
    out: list[str] = []
    for pid, _ in safe:
        out.append(f"{pid}{_FULL_ID}")
        if face_boxes.get(pid) is not None:
            out.append(f"{pid}{_CLOSE_ID}")
    return out


def _crops_of(
    safe: Sequence[tuple[str, bytes]],
    face_boxes: Mapping[str, tuple[float, float, float, float]],
    crop_mode: str,
) -> Iterator[tuple[str, object]]:
    """``(prepared_id, crop)`` per prepared image, in order; each photo is decoded once and dropped after its crops."""
    for pid, blob in safe:
        image = open_photo(io.BytesIO(blob))
        if crop_mode == "both":
            full = crop(image, "whole")
            box = face_boxes.get(pid)
            close = crop_closeup(image, box) if box is not None else None
            del image
            yield f"{pid}{_FULL_ID}", full
            del full
            if close is not None:
                yield f"{pid}{_CLOSE_ID}", close
            del close
        else:
            cropped = crop(image, crop_mode)
            del image
            yield pid, cropped
            del cropped


def prepared_path(project: Project, prepared_id: str) -> Path:
    """The crop file of ``prepared_id``: its lossless WebP, or the legacy PNG when only that exists."""
    webp = project.prepared / f"{prepared_id}{PREPARED_SUFFIX}"
    if webp.is_file():
        return webp
    legacy = project.prepared / f"{prepared_id}{LEGACY_PREPARED_SUFFIX}"
    return legacy if legacy.is_file() else webp


def _unlink_prepared(project: Project, prepared_id: str) -> None:
    """Remove the WebP crop of ``prepared_id`` and a legacy PNG of the same id, if present."""
    (project.prepared / f"{prepared_id}{PREPARED_SUFFIX}").unlink(missing_ok=True)
    (project.prepared / f"{prepared_id}{LEGACY_PREPARED_SUFFIX}").unlink(missing_ok=True)


def _save_crop(project: Project, row: tuple[str, object]) -> str:
    """Write one crop as lossless ``prepared/{id}.webp`` and return its id.

    A legacy PNG crop of the same id, left by an older version, is removed.
    """
    prepared_id, image = row
    save_lossless_webp(image, project.prepared / f"{prepared_id}{PREPARED_SUFFIX}")
    (project.prepared / f"{prepared_id}{LEGACY_PREPARED_SUFFIX}").unlink(missing_ok=True)
    return prepared_id


def _read_crop(path: Path) -> object:
    """A prepared crop, fully loaded and with its file closed: the same pixels that were written.

    The crop is lossless WebP; a legacy PNG of a project prepared by an older version still reads.
    """
    from PIL import Image

    with Image.open(path) as handle:
        handle.load()
        return handle


def _hash_of(image: object) -> int | None:
    return None if image is None else dhash(image)


def _source_id(prepared_id: str) -> str:
    """The photo id a prepared image belongs to: ``both`` ids carry a ``-full``/``-close`` suffix."""
    if prepared_id.endswith(_FULL_ID):
        return prepared_id[: -len(_FULL_ID)]
    if prepared_id.endswith(_CLOSE_ID):
        return prepared_id[: -len(_CLOSE_ID)]
    return prepared_id


def _framing_phrase(prepared_id: str) -> str:
    """The short framing word the stored caption carries for a ``both`` image."""
    if prepared_id.endswith(_FULL_ID):
        return _FULL_PHRASE
    if prepared_id.endswith(_CLOSE_ID):
        return _CLOSE_PHRASE
    return ""


def _append_phrase(text: str, phrase: str) -> str:
    """``text`` with the framing phrase appended; an empty caption becomes the phrase."""
    if not phrase:
        return text
    return f"{text}, {phrase}" if text.strip() else phrase


def _face_box(decision: Decision) -> tuple[float, float, float, float] | None:
    """The primary face box (``x, y, w, h`` in photo pixels) from the gate's face record.

    The record's ``primary`` row wins; without one (a single face, or a record the gate
    kept whole) the largest box does. Nothing here runs the gate again.
    """
    record = decision.trace.get("faces")
    if not isinstance(record, dict):
        return None
    rows = record.get("faces")
    if not isinstance(rows, list):
        return None

    def box_of(row: object) -> tuple[float, float, float, float] | None:
        if not isinstance(row, dict):
            return None
        box = row.get("box")
        if not isinstance(box, list | tuple) or len(box) != 4:
            return None
        values = tuple(float(v) for v in box if isinstance(v, int | float) and not isinstance(v, bool))
        return values if len(values) == 4 and values[2] > 0 and values[3] > 0 else None

    primary = record.get("primary")
    if isinstance(primary, int) and not isinstance(primary, bool) and 0 <= primary < len(rows):
        box = box_of(rows[primary])
        if box is not None:
            return box
    best: tuple[float, float, float, float] | None = None
    best_area = 0.0
    for row in rows:
        box = box_of(row)
        if box is not None and box[2] * box[3] > best_area:
            best, best_area = box, box[2] * box[3]
    return best


def _safety_phase(
    project: Project,
    gate: Gate,
    kept: Sequence[tuple[str, bytes]],
    names: dict[str, str],
    paths: dict[str, Path],
    mode: str,
    attested: bool,
    progress: Progress,
    *,
    batch: int = 1,
    n_inputs: int | None = None,
    sizes: Mapping[str, tuple[int, int]] | None = None,
) -> tuple[list[tuple[str, bytes, Decision, str | None]], list[str], JobDecision, dict[str, object]]:
    """Run the gate over ``kept`` in size-aware chunks of decoded photos and write ``gate_trace.jsonl``.

    ``sizes`` maps a photo id to its decoded ``(width, height)``; a missing entry counts as
    ``(0, 0)`` (light). :func:`~voxprint_forge.lora.batching.plan_batches` cuts ``kept`` into
    chunks of one size class each: up to ``batch`` light photos, ``batch // 2`` medium and
    ``batch // 4`` heavy ones (never below 1), ordered by their first photo.

    ``n_inputs`` is the number of input photos of the run (default ``len(kept)``); it sets the
    age-flag keep limit of :func:`~voxprint_forge.lora.gate.apply_attested_keeps`.

    Every photo of ``kept`` is checked exactly once: a chunk goes to ``gate.check_batch`` when the
    gate has it (else ``gate.check`` per photo), and a gate that returns another number of
    decisions than photos fails the job (``RuntimeError``). With a batch over 1 the next chunk is
    decoded on worker threads while this one is checked.

    The provisional trace rows and safety drop rows are written in input order: after each chunk,
    the longest run of photos (from the first one not yet written) that all have a decision is
    appended, so a crash keeps every row of that prefix. One ``prepare.safety`` progress event is
    sent before each chunk, with the percent and ETA weighted by photo size
    (:func:`~voxprint_forge.lora.batching.plan_weights`), so the estimate is measured over whole
    chunks of real work.

    Returns the checked photos (as bytes, in input order), their tiers, the job decision and the
    attested-keeps audit record (stored in ``gate.json`` as ``age_flags`` when the gate passes).
    """
    internal = bool(getattr(gate, "internal", False))
    keyword = False
    # Rows already on disk from an earlier run stay. This run appends after them.
    trace_from = trace_line_count(project)
    size = max(1, int(batch))
    photo_sizes = [(sizes or {}).get(pid, (0, 0)) for pid, _ in kept]
    plan = plan_batches(photo_sizes, size)
    weights = plan_weights(photo_sizes)
    work_total = sum(weights)
    counts = {"light": 0, "medium": 0, "heavy": 0}
    for width, height in photo_sizes:
        counts[size_class(width, height)] += 1
    print(
        f"prepare: check chunks {len(plan)} (light {counts['light']}, medium {counts['medium']}, "
        f"heavy {counts['heavy']} photos)",
        file=sys.stderr,
    )
    # Results by input index. Rows go to disk for the longest decided prefix, in input order.
    results: list[tuple[str, bytes, Decision, str | None] | None] = [None] * len(kept)
    rows: list[dict[str, object] | None] = [None] * len(kept)
    written = 0
    # Each chunk row is (input index, bytes): the index keys the result, the bytes are decoded.
    chunks = [[(index, kept[index][1]) for index in chunk] for chunk in plan]
    check_batch = getattr(gate, "check_batch", None)
    done = 0
    work_done = 0.0
    for chunk, images in _decoded_chunks(chunks, prefetch=size > 1):
        progress.emit(
            "prepare.safety",
            "lora.progress.safety",
            {"n": done + 1, "total": len(kept)},
            "lora.reason.safety",
            done=done,
            total=len(kept),
            work_done=work_done,
            work_total=work_total,
        )
        if callable(check_batch):
            decisions = list(check_batch(images, attested=attested))
        else:
            decisions = [gate.check(image, attested=attested) for image in images]
        if len(decisions) != len(chunk):
            raise RuntimeError(f"safety gate gave {len(decisions)} decisions for {len(chunk)} photos")
        for (index, blob), image, decision in zip(chunk, images, decisions, strict=True):
            pid = kept[index][0]
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
            results[index] = (pid, blob, decision, key)
            rows[index] = _trace_row(pid, names.get(pid, pid), decision, decision.tier == "clear", decision.tier)
            done += 1
            work_done += weights[index]
        # The decided prefix is on disk before the next chunk, in input order. A crash keeps these rows.
        while written < len(kept) and (result := results[written]) is not None:
            pid, _, decision, _ = result
            append_trace_row(project, rows[written])  # type: ignore[arg-type]
            if decision.tier != "clear":
                append_drop(project, _drop_row(Drop(pid, names.get(pid, pid), "lora.drop.safety")))
            written += 1
        # No decoded photo of this chunk stays referenced while the next one is decoded.
        image = images = decisions = None
    checked = [result for result in results if result is not None]
    provisional = [row for row in rows if row is not None]
    del results, rows
    # Set-level pass: with enough photos passed with a face, a photo where no face was found is kept.
    majority = apply_no_face_majority([decision for _, _, decision, _ in checked])
    # Attested-adult person sets (product rule): no face -> keep; MiVOLO-only age flags are kept as
    # false positives up to floor(AGE_FLAG_KEEP_SHARE * input photos), first by file name; the rest drop.
    attested_keeps, age_flags = apply_attested_keeps(
        majority,
        [Path(names.get(pid, pid)).name for pid, _, _, _ in checked],
        [pid for pid, _, _, _ in checked],
        attested=attested and not internal,
        n_inputs=len(kept) if n_inputs is None else n_inputs,
    )
    checked = [
        (pid, blob, decision, key) for (pid, blob, _, key), decision in zip(checked, attested_keeps, strict=True)
    ]
    tiers = list(_aggregate(checked, internal))
    relaxed = sum(
        1
        for (_, _, decision, _), tier in zip(checked, tiers, strict=True)
        if tier == "clear" and decision.trace.get("keep_reason") in (NO_FACE_KEEP_ATTESTED, AGE_FLAG_KEEP)
    )
    exempt = sum(
        1
        for (_, _, decision, _), tier in zip(checked, tiers, strict=True)
        if tier == "ambiguous" and decision.trace.get("age_flag") == AGE_FLAG_DROPPED
    )
    job = decide_job(tiers, keyword=keyword, relaxed=relaxed, exempt=exempt)
    final_tier = {pid: tier for (pid, _, _, _), tier in zip(checked, tiers, strict=True)}
    for group in ("no_face_kept", "age_flag_kept", "age_flag_dropped"):
        for entry in age_flags[group]:
            entry["tier"] = final_tier.get(entry.get("id"), "")
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
    return checked, tiers, job, age_flags


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
    stores the default base when the caller passes no list. A legacy project is left
    without the key unless the caller passes a list.
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
        "keep_reason": decision.trace.get("keep_reason"),
        "no_face_ratio": _number(decision.trace.get("no_face_ratio")),
        "kept_reasons": [str(reason) for reason in decision.trace.get("kept_reasons") or []],
        "age_flag": decision.trace.get("age_flag"),
        "age_flag_limit": _count(decision.trace.get("age_flag_limit")),
    }


def _count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


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
    extra = f" relaxed={job.relaxed} exempt={job.exempt}" if job.relaxed or job.exempt else ""
    print(
        f"safety gate job: {outcome} ({job.why}) n={job.n} {counts} allowance={job.allowance}{extra}", file=sys.stderr
    )


def _caption(project: Project, captioner: Captioner, image: object, hint: bool = False) -> dict[str, object]:
    """One captioner call. A captioner that errors refuses the job (fail closed).

    ``hint`` (the release.txt adult attestation) is passed only when set, so the call
    without it is the plain ``caption(image)``.
    """
    try:
        return captioner.caption(image, adult_hint=True) if hint else captioner.caption(image)
    except Exception:  # noqa: BLE001
        complete(project)
        refuse(project, "captioner_error", "lora.err.check_unavailable")


def _first_captions(
    project: Project,
    captioner: Captioner,
    prepared_ids: Sequence[str],
    hint: bool,
    batch: int,
) -> Iterator[tuple[str, object, dict[str, object] | None]]:
    """``(prepared_id, crop, first caption)`` for every prepared id, in order, captioned ``batch`` crops at a time.

    Each crop is read back from its own ``prepared/{id}.webp`` (lossless WebP, so the pixels are
    the ones written; a legacy PNG of an old project still reads) only when the caller reaches
    its chunk: the current chunk and, with a batch over 1,
    the next one (read on worker threads) are the only decoded crops held here.
    Each chunk is captioned only when the caller reaches it. A captioner with ``caption_batch``
    captions a chunk in one call (``adult_hint=True`` only when ``hint`` is set, as :func:`_caption`
    does). Without it, with a chunk of one, or when the batch call raises, returns another number
    of results or a result that is not a dict, the first caption is ``None``: :func:`_clean_caption`
    then captions that crop alone through :func:`_caption`, so a captioner error still refuses the job.
    """
    size = max(1, int(batch))
    batched = getattr(captioner, "caption_batch", None)
    rows = [(prepared_id, prepared_path(project, prepared_id)) for prepared_id in prepared_ids]
    chunks = [rows[begin : begin + size] for begin in range(0, len(rows), size)]
    for chunk, images in _decoded_chunks(chunks, prefetch=size > 1, decode=_read_crop):
        texts: list[dict[str, object]] | None = None
        if callable(batched) and len(chunk) > 1:
            try:
                texts = list(batched(images, adult_hint=True) if hint else batched(images))
            except Exception as exc:  # noqa: BLE001 - each crop is captioned alone below
                _print_caption_batch(f"failed ({type(exc).__name__})")
                texts = None
            if texts is not None and (len(texts) != len(chunk) or not all(isinstance(t, dict) for t in texts)):
                _print_caption_batch(f"returned {len(texts)} results for {len(chunk)} frames")
                texts = None
        for position, (prepared_id, _) in enumerate(chunk):
            yield prepared_id, images[position], None if texts is None else texts[position]
        # No crop of this chunk stays referenced here while the next one is read.
        images = texts = None  # type: ignore[assignment]


def _print_caption_batch(what: str) -> None:
    """A batch caption call that could not be used: its frames are captioned one by one."""
    print(f"prepare: batch caption {what}: captioning those frames one by one", file=sys.stderr)


def _clean_caption(
    project: Project,
    captioner: Captioner,
    image: object,
    hint: bool,
    name: str,
    prepared_id: str,
    fixes: list[dict[str, object]],
    *,
    first: dict[str, object] | None = None,
) -> tuple[dict[str, object], str, str, str | None]:
    """The caption of one frame, its long and short text and the child word still in it, or ``None``.

    ``first`` is the frame's first caption when it was already made (a batch call); without it
    the frame is captioned here. A child word gets ``RECAPTIONS`` more tries
    (``RECAPTIONS_ATTESTED`` with ``hint``), each a single-frame call. With ``hint``, a word
    that still comes back is replaced by :data:`ADULT_WORD` and the text is checked again.
    Each step is printed to stderr and added to ``fixes``.
    """
    text = first if first is not None else _caption(project, captioner, image, hint)
    long, short = str(text.get("long", "")), str(text.get("short", ""))
    match = child_keyword_match(long, short)
    for _ in range(RECAPTIONS_ATTESTED if hint else RECAPTIONS):
        if match is None:
            break
        _print_caption_keyword(name, prepared_id, match, "re-captioning")
        fixes.append(_fix_row(prepared_id, name, match, "recaptioned"))
        text = _caption(project, captioner, image, hint)
        long, short = str(text.get("long", "")), str(text.get("short", ""))
        match = child_keyword_match(long, short)
    if match is not None and hint:
        fixed_long, fixed_short = _replace_word(long, match), _replace_word(short, match)
        if child_keyword_match(fixed_long, fixed_short) is None:
            _print_caption_keyword(name, prepared_id, match, f"replaced with '{ADULT_WORD}'")
            fixes.append(_fix_row(prepared_id, name, match, "replaced"))
            return text, fixed_long, fixed_short, None
    if match is not None:
        fixes.append(_fix_row(prepared_id, name, match, "dropped_frame"))
    return text, long, short, match


def _fix_row(prepared_id: str, name: str, match: str, action: str) -> dict[str, object]:
    return {"id": prepared_id, "name": Path(name).name, "match": match, "action": action}


def _replace_word(text: str, match: str) -> str:
    """``text`` with each occurrence of the child word ``match`` replaced by :data:`ADULT_WORD`.

    ``match`` comes from the normalized text, so a word is also found by its normalized
    form (case, ``ё``, NFKC width forms). ``a adult`` becomes ``an adult``.
    """
    exact = re.compile(rf"(?<!\w){re.escape(match)}(?!\w)", re.IGNORECASE)
    out = exact.sub(lambda found: _adult_word(found.group(0)), text)
    if out == text:
        out = re.sub(r"\w+", lambda found: _normalized_swap(found.group(0), match), text)
    return re.sub(rf"\b([Aa]) ({ADULT_WORD}|{ADULT_WORD.capitalize()})\b", r"\1n \2", out)


def _normalized_swap(word: str, match: str) -> str:
    return _adult_word(word) if normalize_text(word) == match else word


def _adult_word(original: str) -> str:
    return ADULT_WORD.capitalize() if original[:1].isupper() else ADULT_WORD


def _clothed(project: Project, items: Sequence[Item], drops: list[Drop]) -> list[Item]:
    """``items`` without the frames flagged unclothed (``Adult content in training: no``).

    Their prepared images are removed and their photo gets one ``lora.drop.unclothed_release`` row.
    """
    kept: list[Item] = []
    dropped: list[Drop] = []
    for item in items:
        if item.unclothed is not True:
            kept.append(item)
            continue
        project.resolve(item.file).unlink(missing_ok=True)
        source_id = _source_id(item.id)
        if all(drop.id != source_id for drop in dropped):
            dropped.append(Drop(source_id, source_id, "lora.drop.unclothed_release"))
    if dropped:
        drops.extend(dropped)
        _write_drops(project, dropped)
    return kept


def _print_caption_keyword(name: str, prepared_id: str, match: str, action: str) -> None:
    """Every mode: a child word in a generated caption, and what was done about it."""
    import sys

    print(f"safety gate: {Path(name).name} ({prepared_id}) caption keyword '{match}': {action}", file=sys.stderr)


def _drop_row(drop: Drop) -> dict[str, str]:
    return {"id": drop.id, "name": drop.name, "reason": drop.reason}


def _write_drops(
    project: Project,
    drops: Sequence[Drop],
    *,
    purge: Iterable[str] = (),
    extra: Mapping[str, Mapping[str, object]] | None = None,
) -> None:
    """Record ``drops`` in ``drops.jsonl``, one line per photo id.

    A re-run checks photos set aside earlier again. Their old line is replaced, not repeated.
    Photo rows of an id in ``purge`` (a provisional safety drop of a photo kept later) are removed.
    Rows of other ids, including ``job`` rows, stay as they are and in order.
    ``extra`` adds fields to the row of a photo id (the ``match`` of a caption keyword drop).
    Safety rows may already have been appended; this compaction is what removes the copies.
    The rewrite goes to a temp file and ``os.replace``, so a crash keeps the previous file.
    """
    project.root.mkdir(parents=True, exist_ok=True)
    fresh: dict[str, str] = {}
    for drop in drops:
        fresh[drop.id] = json.dumps({**_drop_row(drop), **(extra or {}).get(drop.id, {})}, ensure_ascii=False)
    replaced = fresh.keys() | set(purge)
    kept: list[str] = []
    if project.drops.is_file():
        for line in project.drops.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                row = None
            if isinstance(row, dict) and str(row.get("id")) in replaced and row.get("scope") != "job":
                continue
            kept.append(line)
    text = "".join(line + "\n" for line in (*kept, *fresh.values()))
    atomic_text(project.drops, text)
