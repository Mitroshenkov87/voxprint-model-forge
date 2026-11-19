"""``lora prepare`` batches the safety check and the captioner. Fakes only, no models."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import weakref
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.test_lora_prepare import ADULT_FACE, CLEAN, FakeCaptioner, FakeEstimator, FakeJudge, photos
from voxprint_forge.lora import batching
from voxprint_forge.lora import prepare as prepare_module
from voxprint_forge.lora.backends import JudgeParseError
from voxprint_forge.lora.consent import check_subject
from voxprint_forge.lora.errors import LoraBlocked
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import AgeReading, Face, Gate
from voxprint_forge.lora.prepare import prepare
from voxprint_forge.lora.project import Project

MINOR = replace(ADULT_FACE, min_age=12.0, reason_code="child_or_adolescent")
STATS = {"prompt_tokens": 7, "new_tokens": 3, "hit_token_limit": False, "chars": 11}


def key(image) -> str:
    """The pixels of a decoded photo or crop, as the fakes see them."""
    return hashlib.sha256(image.tobytes()).hexdigest()


def keys_of(files) -> list[str]:
    return [key(prepare_module.open_photo(path)) for path in files]


class ScriptJudge:
    """``judge`` per image; the verdict comes from the image's pixels."""

    name = "script-judge"

    def __init__(self, verdicts=None, default=ADULT_FACE):
        self.verdicts = dict(verdicts or {})
        self.default = default
        self.single: list[str] = []
        self.last_stats: dict = {}

    def verdict(self, image):
        return self.verdicts.get(key(image), self.default)

    def judge(self, image):
        self.single.append(key(image))
        self.last_stats = dict(STATS)
        return self.verdict(image)

    def unload(self):
        pass


class BatchJudge(ScriptJudge):
    """``judge_batch`` too. ``fail`` holds the batch numbers that raise; ``bad`` the image keys that fail to parse."""

    def __init__(self, verdicts=None, default=ADULT_FACE, fail=(), bad=(), short=False):
        super().__init__(verdicts, default)
        self.fail, self.bad, self.short = set(fail), set(bad), short
        self.batches: list[list[str]] = []
        self.failed: list[list[str]] = []
        self.last_batch_stats: list[dict] = []

    def judge_batch(self, images):
        number = len(self.batches) + len(self.failed)
        seen = [key(image) for image in images]
        if number in self.fail:
            self.failed.append(seen)
            raise RuntimeError("CUDA out of memory")
        self.batches.append(seen)
        out = [JudgeParseError("bad_json") if k in self.bad else self.verdict(img) for k, img in zip(seen, images)]
        if self.short:
            out = out[:-1]
        self.last_batch_stats = [dict(STATS) for _ in out]
        return out


class ScriptEstimator:
    name = "script-estimator"

    def __init__(self, ages=None, default=(34.0,), fail=()):
        self.ages, self.default, self.fail = dict(ages or {}), default, set(fail)
        self.seen: list[str] = []

    def estimate(self, image):
        k = key(image)
        self.seen.append(k)
        if k in self.fail:
            raise RuntimeError("worker gone")
        ages = self.ages.get(k, self.default)
        faces = tuple(Face((10.0, 10.0, 100.0, 100.0), 0.95) for _ in ages)
        return AgeReading(len(ages), tuple(ages), faces)

    def unload(self):
        pass


def _run(folder, files, judge, estimator, captioner=None, **kw):
    cap = captioner or FakeCaptioner()
    return prepare(
        Project(folder / "proj"),
        files,
        kw.pop("subject", None) or check_subject("person", name="Demo", consent_obtained=True, subject_adult=True),
        gate=kw.pop("gate", None) or Gate(judge, estimator),
        load_captioner=lambda choice: cap,
        progress=kw.pop("progress", None) or Progress(),
        **kw,
    )


def _trace(folder):
    return [json.loads(line) for line in (folder / "proj" / "gate_trace.jsonl").read_text().splitlines()]


def _outcome(folder, files, judge, estimator, **kw):
    try:
        _run(folder, files, judge, estimator, **kw)
        state = "trained"
    except LoraBlocked as blocked:
        state = blocked.code
    project = Project(folder / "proj")
    run_state = json.loads(project.gate_run.read_text())["state"]
    audit = folder / "proj" / "refusals.jsonl"
    sources = [json.loads(line)["source"] for line in audit.read_text().splitlines()] if audit.is_file() else []
    return state, project.read().get("gate_passed"), run_state, sources, _trace(folder), project.drops.read_text()


def test_every_photo_is_judged_once_in_batches_and_traced_in_order(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("VOXPRINT_PREPARE_BATCH", "4")
    files = photos(tmp_path / "in", 10)
    judge, estimator = BatchJudge(), ScriptEstimator()
    events: list = []
    progress = Progress()
    progress.emit = lambda step, *args, **kw: events.append((step, args, kw))  # type: ignore[method-assign]
    summary = _run(tmp_path, files, judge, estimator, progress=progress)
    expected = keys_of(files)
    assert [len(batch) for batch in judge.batches] == [4, 4, 2]
    assert [k for batch in judge.batches for k in batch] == expected
    assert judge.single == [] and estimator.seen == expected
    from voxprint_forge.lora.photos import photo_id

    rows = _trace(tmp_path)
    assert [row["id"] for row in rows] == [photo_id(path) for path in files]
    assert all(row["tier"] == "clear" for row in rows)
    safety = [args[1]["n"] for step, args, _ in events if step == "prepare.safety" and len(args) > 1]
    assert safety == [1, 5, 9]  # one event per chunk of 4, 4, 2
    assert summary["prepared"] >= 10
    err = capsys.readouterr().err
    assert "prepare: batch size 4 (check)" in err and "prepare: batch size 4 (caption)" in err


@pytest.mark.parametrize("position", [0, 3, 7])
def test_a_minor_anywhere_in_a_batch_refuses_like_the_unbatched_path(tmp_path, monkeypatch, position) -> None:
    files = photos(tmp_path / "in", 8)
    minor = keys_of(files)[position]
    results = []
    for size in ("1", "4"):
        monkeypatch.setenv("VOXPRINT_PREPARE_BATCH", size)
        folder = tmp_path / f"b{size}"
        judge = BatchJudge({minor: MINOR})
        estimator = ScriptEstimator({minor: (12.0,)})
        results.append(_outcome(folder, files, judge, estimator, mode="expert"))
        if size == "4":
            assert [len(batch) for batch in judge.batches] == [4, 4] and judge.single == []
        else:
            assert judge.batches == [] and len(judge.single) == 8
    assert results[0] == results[1]
    state, passed, run_state, sources, rows, _ = results[1]
    assert state == "lora.err.blocked" and passed is False and run_state == "complete" and sources == ["photo"]
    assert rows[position]["tier"] == "strong" and rows[position]["decision"] == "minor"
    assert all(row["tier"] == "clear" for i, row in enumerate(rows) if i != position)


def test_a_batch_that_raises_is_checked_one_by_one_and_its_minor_still_refuses(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("VOXPRINT_PREPARE_BATCH", "4")
    files = photos(tmp_path / "in", 8)
    expected = keys_of(files)
    judge = BatchJudge({expected[5]: MINOR}, fail={1})
    estimator = ScriptEstimator({expected[5]: (12.0,)})
    state, passed, *_rest = _outcome(tmp_path, files, judge, estimator, mode="expert")
    assert state == "lora.err.blocked" and passed is False
    assert judge.batches == [expected[:4]] and judge.failed == [expected[4:]]
    assert judge.single == expected[4:]
    # The background estimator read the failed batch in order (stopping early once the judge failed);
    # the one-by-one checks then read each of those photos again.
    assert estimator.seen[:4] == expected[:4] and estimator.seen[-4:] == expected[4:]
    background = estimator.seen[4:-4]
    assert background == expected[4 : 4 + len(background)]
    rows = _trace(tmp_path)
    assert len(rows) == 8 and rows[5]["tier"] == "strong"
    assert "safety gate batch judge failed (RuntimeError)" in capsys.readouterr().err


def test_an_exception_item_is_that_photos_judge_error_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_PREPARE_BATCH", "4")
    files = photos(tmp_path / "in", 10)
    expected = keys_of(files)
    judge = BatchJudge(bad={expected[2]})
    estimator = ScriptEstimator()
    state, passed, *_rest = _outcome(tmp_path, files, judge, estimator)
    # judge_error is a check that did not run: the job fails closed, as with a raising judge.
    assert state == "lora.err.check_unavailable" and passed is False
    rows = _trace(tmp_path)
    assert rows[2]["reasons"] == ["judge_error"] and rows[2]["tier"] != "clear"
    assert rows[2]["judge"] == "JudgeParseError:bad_json"
    assert all(row["tier"] == "clear" and row["reasons"] == [] for i, row in enumerate(rows) if i != 2)
    # The estimator read every photo once alongside the judge; for that photo the reading is discarded.
    assert estimator.seen == expected
    assert rows[2]["mivolo_ages"] == [] and rows[2]["estimate"] is None


def test_a_batch_with_the_wrong_count_is_checked_one_by_one(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_PREPARE_BATCH", "4")
    files = photos(tmp_path / "in", 10)
    expected = keys_of(files)
    judge = BatchJudge(short=True)
    _run(tmp_path, files, judge, ScriptEstimator())
    assert [len(batch) for batch in judge.batches] == [4, 4, 2]
    assert judge.single == expected
    assert [row["tier"] for row in _trace(tmp_path)] == ["clear"] * 10


def test_a_gate_that_miscounts_fails_the_job_closed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_PREPARE_BATCH", "4")

    class Short(Gate):
        def check_batch(self, images, *, attested=False):
            return super().check_batch(images, attested=attested)[:-1]

    judge = BatchJudge()
    with pytest.raises(RuntimeError, match="decisions for 4 photos"):
        _run(tmp_path, photos(tmp_path / "in", 10), judge, None, gate=Short(judge, ScriptEstimator()))


def _images(tmp_path, n):
    return [prepare_module.open_photo(path) for path in photos(tmp_path / "in", n)]


@pytest.mark.parametrize("internal", [False, True])
def test_check_batch_equals_check(tmp_path, internal) -> None:
    images = _images(tmp_path, 7)
    k = [key(image) for image in images]
    verdicts = {
        k[0]: ADULT_FACE,
        k[1]: MINOR,
        k[2]: CLEAN,
        k[3]: replace(ADULT_FACE, min_age=19.0, could_be_under_20=True),
        k[4]: None,
        k[5]: replace(ADULT_FACE, confidence=0.2),
        k[6]: ADULT_FACE,
    }
    ages = {k[1]: (12.0,), k[2]: (), k[3]: (18.5, 40.0), k[5]: (30.0,)}
    for attested in (False, True):
        single = Gate(BatchJudge(verdicts), ScriptEstimator(ages, fail={k[6]}), internal=internal)
        batch_judge = BatchJudge(verdicts, bad={k[5]})
        batch_estimator = ScriptEstimator(ages, fail={k[6]})
        batch = Gate(batch_judge, batch_estimator, internal=internal)
        want = [single.check(image, attested=attested) for image in images]
        got = batch.check_batch(images, attested=attested)
        assert len(batch_judge.batches) == 1 and batch_judge.single == []
        # One estimator call per image, in input order, the judge-error image included.
        assert batch_estimator.seen == k
        # The estimator raising for one image is that image's estimator error, as ``check`` gives.
        assert got[6].trace["estimator"] == {"ok": False, "stage": "estimator", "error": "RuntimeError"}
        assert "estimator_missing" in got[6].reasons and got[6].tier != "clear"
        assert len(got) == len(want)
        for i, (a, b) in enumerate(zip(want, got, strict=True)):
            if i == 5:
                # The parse failure is this image's own judge_error, nothing else changes.
                assert b.reasons == ("judge_error",) and b.tier == "unchecked" and not b.keep
                assert b.trace["estimator"] == {"ok": False, "skipped": True}
                assert b.trace["judge"] == {"ok": False, "stage": "judge.parse", "error": "JudgeParseError",
                                            "code": "bad_json", **STATS}
                continue
            assert (a.reasons, a.minor, a.path, a.tier) == (b.reasons, b.minor, b.path, b.tier), i
            assert a.trace == b.trace and a.shadow == b.shadow and a.audit == b.audit, i
            assert a.trace["judge"].get("prompt_tokens") == STATS["prompt_tokens"] or a.trace["judge"]["ok"] is False
        if internal:
            assert all(d.shadow is not None for d in got if d.reasons != ("judge_error",))


def test_check_batch_falls_back_to_check(tmp_path) -> None:
    images = _images(tmp_path, 3)
    plain = Gate(ScriptJudge(), ScriptEstimator())
    assert [d.tier for d in plain.check_batch(images, attested=True)] == ["clear"] * 3
    one = BatchJudge()
    assert [d.tier for d in Gate(one, ScriptEstimator()).check_batch(images[:1], attested=True)] == ["clear"]
    assert one.batches == [] and len(one.single) == 1

    class Custom(Gate):
        def check(self, image, *, attested=False):
            return replace(super().check(image, attested=attested), reasons=("custom",))

    judge = BatchJudge()
    assert [d.reasons for d in Custom(judge, ScriptEstimator()).check_batch(images)] == [("custom",)] * 3
    assert judge.batches == []


def test_check_batch_does_not_swallow_an_interrupt(tmp_path) -> None:
    class Stop(BatchJudge):
        def judge_batch(self, images):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        Gate(Stop(), ScriptEstimator()).check_batch(_images(tmp_path, 3))

    class Interrupted(ScriptEstimator):
        def estimate(self, image):
            raise KeyboardInterrupt

    # A BaseException on the estimator thread reaches the caller; no decision is made.
    with pytest.raises(KeyboardInterrupt):
        Gate(BatchJudge(), Interrupted()).check_batch(_images(tmp_path, 3), attested=True)


def test_estimator_errors_on_the_background_thread_match_check(tmp_path) -> None:
    images = _images(tmp_path, 4)

    class Odd(ScriptEstimator):
        """A broken reading for image 1 and a worker error with a stage and code for image 2."""

        def estimate(self, image):
            from voxprint_forge.lora.backends import WorkerError

            position = len(self.seen)
            self.seen.append(key(image))
            if position == 1:
                return "not a reading"
            if position == 2:
                raise WorkerError("worker_exited_rc1", "estimator.ipc")
            return AgeReading(1, (34.0,), (Face((10.0, 10.0, 100.0, 100.0), 0.95),))

    single = Gate(BatchJudge(), Odd(), internal=True)
    want = [single.check(image, attested=True) for image in images]
    estimator = Odd()
    got = Gate(BatchJudge(), estimator, internal=True).check_batch(images, attested=True)
    assert len(estimator.seen) == 4
    for a, b in zip(want, got, strict=True):
        assert (a.reasons, a.tier, a.trace, a.shadow) == (b.reasons, b.tier, b.trace, b.shadow)
    assert got[1].trace["estimator"]["error"] == "AttributeError" and got[1].tier != "clear"
    assert got[2].trace["estimator"] == {
        "ok": False, "stage": "estimator.ipc", "error": "WorkerError", "code": "worker_exited_rc1"
    }
    assert got[2].tier != "clear" and got[0].tier == got[3].tier == "clear"


class BatchCaptioner(FakeCaptioner):
    """``caption_batch`` too. ``child``: positions (in first-seen order) of crops whose captions name a child."""

    def __init__(self, child=(), fail=False, single_fail=False):
        super().__init__()
        self.child, self.fail, self.single_fail = set(child), fail, single_fail
        self.batches: list[list[str]] = []
        self.single: list[str] = []
        self.order: dict[str, int] = {}

    def text(self, image):
        position = self.order.setdefault(key(image), len(self.order))
        if position in self.child:
            return {"long": "a child holding a square", "short": "child with square"}
        return super().caption(image)

    def caption(self, image, adult_hint=False):
        self.single.append(key(image))
        if self.single_fail:
            raise RuntimeError("captioner gone")
        return self.text(image)

    def caption_batch(self, images, adult_hint=False):
        if self.fail:
            raise RuntimeError("CUDA out of memory")
        self.batches.append([key(image) for image in images])
        return [self.text(image) for image in images]


class PlainCaptioner(BatchCaptioner):
    caption_batch = None  # type: ignore[assignment]


def _crop_keys(folder):
    project = Project(folder / "proj")
    return sorted(key(prepare_module.open_photo(path)) for path in project.prepared.iterdir())


def test_every_crop_is_captioned_once_in_batches(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_PREPARE_BATCH", "4")
    files = photos(tmp_path / "in", 10)
    cap = BatchCaptioner()
    _run(tmp_path, files, FakeJudge(), FakeEstimator(), captioner=cap)
    assert [len(batch) for batch in cap.batches] == [4, 4, 2] and cap.single == []
    seen = [k for batch in cap.batches for k in batch]
    assert len(set(seen)) == 10 and sorted(seen) == _crop_keys(tmp_path)


def test_a_child_word_in_a_batch_caption_recaptions_alone_and_drops_the_frame(tmp_path, monkeypatch, capsys) -> None:
    files = photos(tmp_path / "in", 12)
    subject = check_subject("person", name="Demo", consent_obtained=True, subject_adult=True)
    outputs = []
    for size, captioner_type in (("1", PlainCaptioner), ("4", BatchCaptioner)):
        monkeypatch.setenv("VOXPRINT_PREPARE_BATCH", size)
        folder = tmp_path / f"b{size}"
        cap = captioner_type(child={5})
        summary = _run(folder, files, FakeJudge(), FakeEstimator(), captioner=cap, subject=subject)
        project = Project(folder / "proj")
        if size == "4":
            assert [len(batch) for batch in cap.batches] == [4, 4, 4]
            assert cap.single == [cap.batches[1][1]]  # the one recaption, single-image
        else:
            assert len(cap.single) == 13
        fixes = [(row["match"], row["action"]) for row in project.read()["caption_fixes"]]
        outputs.append((summary, fixes, project.captions.read_text(), project.drops.read_text(),
                        capsys.readouterr().err.count("caption keyword")))
    assert outputs[0] == outputs[1]
    summary, fixes, _, drops, lines = outputs[1]
    assert fixes == [("child", "recaptioned"), ("child", "dropped_frame")] and lines == 2
    assert "lora.drop.caption_keyword" in drops and summary["prepared"] == 11


def test_a_batch_caption_that_raises_falls_back_per_crop(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("VOXPRINT_PREPARE_BATCH", "4")
    files = photos(tmp_path / "in", 10)
    cap = BatchCaptioner(fail=True)
    _run(tmp_path, files, FakeJudge(), FakeEstimator(), captioner=cap)
    assert sorted(cap.single) == _crop_keys(tmp_path) and len(cap.single) == 10
    assert "batch caption failed (RuntimeError)" in capsys.readouterr().err


def test_a_per_crop_caption_error_after_a_failed_batch_still_refuses(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_PREPARE_BATCH", "4")
    files = photos(tmp_path / "in", 10)
    cap = BatchCaptioner(fail=True, single_fail=True)
    with pytest.raises(LoraBlocked) as info:
        _run(tmp_path, files, FakeJudge(), FakeEstimator(), captioner=cap)
    assert info.value.code == "lora.err.check_unavailable"
    audit = (tmp_path / "proj" / "refusals.jsonl").read_text()
    assert "captioner_error" in audit and cap.unloaded


def test_a_batch_caption_with_the_wrong_count_falls_back_per_crop(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("VOXPRINT_PREPARE_BATCH", "4")

    class Short(BatchCaptioner):
        def caption_batch(self, images, adult_hint=False):
            return super().caption_batch(images, adult_hint)[:-1]

    cap = Short()
    _run(tmp_path, photos(tmp_path / "in", 10), FakeJudge(), FakeEstimator(), captioner=cap)
    assert len(cap.single) == 10 and sorted(cap.single) == _crop_keys(tmp_path)


@pytest.mark.parametrize("size", [1, 4])
def test_captioning_reads_crops_back_chunk_by_chunk(tmp_path, monkeypatch, size) -> None:
    monkeypatch.setenv("VOXPRINT_PREPARE_BATCH", str(size))
    files = photos(tmp_path / "in", 10)
    live: list[weakref.ref] = []
    reads: list[str] = []
    lock = threading.Lock()
    most = 0
    real = prepare_module._read_crop

    def tracked(path):
        nonlocal most
        image = real(path)
        with lock:
            live.append(weakref.ref(image))
            reads.append(path.stem)
            most = max(most, sum(1 for ref in live if ref() is not None))
        return image

    monkeypatch.setattr(prepare_module, "_read_crop", tracked)
    # No crop is held in memory between writing and captioning: every one is read back.
    monkeypatch.setattr(prepare_module, "crop", _counted(prepare_module.crop, live))
    monkeypatch.setattr(prepare_module, "crop_closeup", _counted(prepare_module.crop_closeup, live))
    cap = BatchCaptioner()
    # A face on every photo: a whole frame and a close-up each, 20 prepared images.
    _run(tmp_path, files, BatchJudge(), ScriptEstimator(), captioner=cap)
    project = Project(tmp_path / "proj")
    ids = [item.id for item in project.items()]
    from voxprint_forge.lora.photos import photo_id

    assert ids == [f"{photo_id(p)}{suffix}" for p in files for suffix in ("-full", "-close")]
    assert sorted(reads) == sorted(ids) and len(reads) == 20
    # Captioned once each, in prepared order, with the pixels of its own lossless WebP crop.
    seen = [k for batch in cap.batches for k in batch] + cap.single
    assert seen == [key(prepare_module.open_photo(prepare_module.prepared_path(project, pid))) for pid in ids]
    # The current chunk, at most one prefetched chunk and the frame the loop is on.
    assert most <= 2 * size + 1, most


@pytest.mark.parametrize("size", ["1", "4"])
def test_a_collage_caption_on_the_whole_frame_still_captions_and_removes_the_close_up(
    tmp_path, monkeypatch, size
) -> None:
    monkeypatch.setenv("VOXPRINT_PREPARE_BATCH", size)

    class Collage(BatchCaptioner):
        def text(self, image):
            position = self.order.setdefault(key(image), len(self.order))
            if position == 6:  # the whole frame of photo 3
                return {"long": "a collage of squares", "short": "collage"}
            return super().text(image)

    files = photos(tmp_path / "in", 12)
    cap = Collage()
    summary = _run(tmp_path, files, BatchJudge(), ScriptEstimator(), captioner=cap)
    from voxprint_forge.lora.photos import photo_id

    pid = photo_id(files[3])
    project = Project(tmp_path / "proj")
    # Every prepared image was read back and captioned, the dropped photo's close-up included.
    assert len([k for batch in cap.batches for k in batch] + cap.single) == 24
    assert not (project.prepared / f"{pid}-full.webp").exists()
    assert not (project.prepared / f"{pid}-close.webp").exists()
    assert summary["prepared"] == 22 and all(not item.id.startswith(pid) for item in project.items())
    rows = [json.loads(line) for line in project.drops.read_text().splitlines()]
    assert [(row["id"], row["reason"]) for row in rows] == [(pid, "lora.drop.collage")]


def _counted(work, live):
    """``work`` with each crop it makes tracked in ``live``, so a crop kept in memory would show up."""

    def run(*args, **kw):
        image = work(*args, **kw)
        live.append(weakref.ref(image))
        return image

    return run


def _torch(free_gib, total_gib=16.0, cuda=True):
    gib = 1024**3
    info = (int(free_gib * gib), int(total_gib * gib))
    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda, mem_get_info=lambda: info))


def test_prepare_batch_size(monkeypatch) -> None:
    monkeypatch.delenv("VOXPRINT_PREPARE_BATCH", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    assert batching.prepare_batch_size() == 1
    assert batching.prepare_batch_size("caption") == 1
    monkeypatch.setitem(sys.modules, "torch", _torch(12.0, cuda=False))
    assert batching.prepare_batch_size("check") == batching.prepare_batch_size("caption") == 1
    assert batching.DEFAULT_BATCH == batching.MAX_BATCH == 16
    for free in (0.5, 3.0, 4.0, 10.0, 10.9, 15.5):
        monkeypatch.setitem(sys.modules, "torch", _torch(free))
        for stage in ("check", "caption"):
            assert 1 <= batching.prepare_batch_size(stage) <= batching.DEFAULT_BATCH
    # Per-stage budgets: check 0.59 GiB, caption 0.44 GiB per image, 1.0 GiB headroom.
    # A 16 GB card with the int8 Qwen loaded (about 5 GB of weights): about 10.9 GiB free -> 16 for both.
    # An 8 GB card with the same weights: about 2.5 GiB free -> a smaller batch per stage.
    cases = (
        (10.9, 16.0, 16, 16),
        (15.5, 16.0, 16, 16),
        (10.0, 16.0, 15, 16),
        (6.0, 12.0, 8, 11),
        (2.5, 8.0, 2, 3),
        (1.5, 8.0, 1, 1),
        (1.0, 8.0, 1, 1),
        (0.2, 8.0, 1, 1),
    )
    for free, total, check, caption in cases:
        monkeypatch.setitem(sys.modules, "torch", _torch(free, total))
        assert batching.prepare_batch_size() == batching.prepare_batch_size("check") == check, free
        assert batching.prepare_batch_size("caption") == caption, free
    with pytest.raises(ValueError):
        batching.prepare_batch_size("train")
    for raw, want in (("3", 3), ("100", 32), ("0", 1), ("-4", 1), (" 6 ", 6)):
        monkeypatch.setenv("VOXPRINT_PREPARE_BATCH", raw)
        assert batching.prepare_batch_size("check") == batching.prepare_batch_size("caption") == want
    monkeypatch.setenv("VOXPRINT_PREPARE_BATCH", "many")
    monkeypatch.setitem(sys.modules, "torch", _torch(10.9))
    assert (batching.prepare_batch_size("check"), batching.prepare_batch_size("caption")) == (16, 16)
    monkeypatch.setitem(sys.modules, "torch", _torch(2.5, 8.0))
    assert (batching.prepare_batch_size("check"), batching.prepare_batch_size("caption")) == (2, 3)


def test_vlm_ask_batch_pads_left_and_counts_per_item() -> None:
    torch = pytest.importorskip("torch")
    from voxprint_forge.lora.backends import VlmJudge
    from voxprint_forge.lora.captioner import QwenCaptioner

    class Inputs(dict):
        def to(self, device):
            return self

    class Tokenizer:
        padding_side = "right"
        pad_token_id = 0
        eos_token_id = 9

    class Processor:
        tokenizer = Tokenizer()

        def apply_chat_template(self, conversations, **kw):
            assert kw["padding"] is True and self.tokenizer.padding_side == "left"
            self.sides = self.tokenizer.padding_side
            assert len(conversations) == 3 and all(len(c) == 2 for c in conversations)
            ids = torch.tensor([[0, 0, 5, 5], [5, 5, 5, 5], [0, 5, 5, 5]])
            return Inputs(input_ids=ids, attention_mask=(ids != 0).long())

        def batch_decode(self, rows, skip_special_tokens=True):
            return ["".join(chr(96 + int(t)) for t in row if int(t) not in (0, 9)) for row in rows]

    class Model:
        device = "cpu"
        generation_config = SimpleNamespace(eos_token_id=[9], pad_token_id=0)

        def generate(self, input_ids, attention_mask, max_new_tokens, do_sample):
            assert do_sample is False and max_new_tokens == 3
            new = torch.tensor([[1, 9, 0], [1, 2, 3], [4, 4, 9]])
            return torch.cat([input_ids, new], dim=1)

    vlm = object.__new__(VlmJudge)
    vlm.processor, vlm.model = Processor(), Model()

    class Tiny:
        size = (64, 48)

    texts = vlm.ask_batch([Tiny(), Tiny(), Tiny()], "s", "u", 3)
    assert texts == ["a", "abc", "dd"]
    assert Processor.tokenizer.padding_side == "right"
    assert vlm.last_stats == [
        {"prompt_tokens": 2, "new_tokens": 1, "hit_token_limit": False, "chars": 1},
        {"prompt_tokens": 4, "new_tokens": 3, "hit_token_limit": True, "chars": 3},
        {"prompt_tokens": 3, "new_tokens": 2, "hit_token_limit": False, "chars": 2},
    ]
    good = json.dumps({"face_visible": "yes", "person": "yes", "min_apparent_age": 30, "indicators": {}})
    vlm.ask_batch = lambda images, system, user, n: (  # type: ignore[method-assign]
        setattr(vlm, "last_stats", [dict(STATS), dict(STATS)]) or [good, "no json here"]
    )
    verdicts = vlm.judge_batch([Tiny(), Tiny()])
    assert verdicts[0].min_age == 30.0 and isinstance(verdicts[1], JudgeParseError)
    assert vlm.last_batch_stats == [STATS, STATS]

    answers = {160: ["long one ", "a nude long"], 48: [" short", "short two"], 8: ["Yes.", "no"]}
    asks: list[tuple[str, int]] = []

    def fake_ask(images, system, user, n):
        asks.append((system, n))
        return answers[n]

    cap = object.__new__(QwenCaptioner)
    cap.vlm = SimpleNamespace(ask_batch=fake_ask)
    assert cap.caption_batch([Tiny(), Tiny()], adult_hint=True) == [
        {"long": "long one", "short": "short", "unclothed": True},
        {"long": "a nude long", "short": "short two", "unclothed": True},
    ]
    assert [n for _, n in asks] == [160, 48, 8] and all("adult" in system for system, _ in asks)
