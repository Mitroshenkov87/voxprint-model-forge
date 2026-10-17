"""LoRA data step: limits, consent, filters, the fail-closed safety gate, captions. Fakes only."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from voxprint_forge.cli import main
from voxprint_forge.lora import cli as lora_cli
from voxprint_forge.lora.backends import parse_verdict
from voxprint_forge.lora.consent import check_subject
from voxprint_forge.lora.errors import LoraBlocked, LoraStageError, LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import INDICATORS, AgeReading, Gate, Verdict, decide
from voxprint_forge.lora.photos import check_count, collect
from voxprint_forge.lora.prepare import prepare
from voxprint_forge.lora.project import Project

CLEAN = Verdict("no", "no", None, 0.9, "no_person", False, {k: "absent" for k in INDICATORS})
ADULT_FACE = Verdict("yes", "yes", 34.0, 0.9, "clearly_adult", False, {k: "absent" for k in INDICATORS})


class FakeJudge:
    name = "fake-judge"

    def __init__(self, verdict=CLEAN, fail=False):
        self.verdict, self.fail, self.unloaded = verdict, fail, False

    def judge(self, image):
        if self.fail:
            raise RuntimeError("boom")
        return self.verdict

    def unload(self):
        self.unloaded = True


class FakeEstimator:
    name = "fake-estimator"

    def __init__(self, reading=AgeReading(0, ())):
        self.reading = reading

    def estimate(self, image):
        return self.reading


class FakeCaptioner:
    name = "fake-captioner"

    def __init__(self):
        self.unloaded = False

    def caption(self, image):
        return {"long": "a textured square on a plain background", "short": "textured square"}

    def unload(self):
        self.unloaded = True


def noise(seed: int, size: int = 600) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8))


def photos(folder, count: int):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = folder / f"img{index:02d}.png"
        noise(index).save(path)
        paths.append(path)
    return paths


def obj():
    return check_subject("object", name="Blue mug", rights_confirmed=True)


def run(tmp_path, files, gate=None, captioner=None, **kw):
    cap = captioner or FakeCaptioner()
    return prepare(
        Project(tmp_path / "proj"),
        files,
        kw.pop("subject", obj()),
        gate=gate or Gate(FakeJudge(), FakeEstimator()),
        load_captioner=lambda choice: cap,
        progress=Progress(),
        **kw,
    )


def test_simple_mode_is_10_to_30_and_expert_warns_with_floor_5() -> None:
    with pytest.raises(LoraUsageError) as low:
        check_count(9, "simple")
    assert low.value.params == {"need": 1}
    with pytest.raises(LoraUsageError) as high:
        check_count(31, "simple")
    assert high.value.params == {"extra": 1}
    assert check_count(10, "simple") is None and check_count(30, "simple") is None
    assert check_count(6, "expert") == "lora.warn.count"
    assert check_count(20, "expert") is None
    with pytest.raises(LoraUsageError) as floor:
        check_count(4, "expert")
    assert floor.value.code == "lora.err.below_floor"


def test_consent_per_kind() -> None:
    with pytest.raises(LoraUsageError) as name:
        check_subject("object", rights_confirmed=True)
    assert name.value.code == "lora.err.no_name"
    with pytest.raises(LoraUsageError) as consent:
        check_subject("person", name="Ann", consent_obtained=True)
    assert consent.value.code == "lora.err.no_consent"
    for kind in ("object", "style"):
        with pytest.raises(LoraUsageError) as rights:
            check_subject(kind, name="X")
        assert rights.value.code == "lora.err.no_rights"
    person = check_subject("person", stage_name="Nova Lee", consent_obtained=True, subject_adult=True)
    assert person.trigger == "nova_lee" and person.adult_confirmed and not person.rights_confirmed


def test_consent_method_verbal_or_file(tmp_path) -> None:
    release = tmp_path / "release.txt"
    release.write_text("signed")
    base = {"name": "Demo", "consent_obtained": True, "subject_adult": True}
    verbal = check_subject("person", consent_method="verbal", **base)
    assert verbal.consent_method == "verbal" and not verbal.release_attached and verbal.release_sha256 == ""
    filed = check_subject("person", consent_method="file", release=release, **base)
    assert filed.consent_method == "file" and filed.release_attached and len(filed.release_sha256) == 64
    assert check_subject("person", **base).consent_method == "verbal"
    assert check_subject("person", release=release, **base).consent_method == "file"
    with pytest.raises(LoraUsageError) as missing:
        check_subject("person", consent_method="file", **base)
    assert missing.value.code == "lora.err.no_release"
    with pytest.raises(LoraUsageError) as both:
        check_subject("person", consent_method="verbal", release=release, **base)
    assert both.value.code == "lora.err.consent_method"
    with pytest.raises(LoraUsageError) as no_consent:
        check_subject("person", name="Demo", subject_adult=True, consent_method="verbal")
    assert no_consent.value.code == "lora.err.no_consent"
    assert check_subject("object", name="Mug", rights_confirmed=True).consent_method == ""


def test_adult_content_needs_the_18_confirmation() -> None:
    for kwargs in (
        {"kind": "person", "name": "Demo", "consent_obtained": True},
        {"kind": "object", "name": "Mug", "rights_confirmed": True},
        {"kind": "style", "name": "Ink", "rights_confirmed": True},
    ):
        with pytest.raises(LoraUsageError) as refused:
            check_subject(adult_content=True, **kwargs)
        assert refused.value.code == "lora.err.adult_content"
    person = check_subject("person", name="Demo", consent_obtained=True, subject_adult=True, adult_content=True)
    assert person.adult_content and person.adult_confirmed


class RecordingJudge(FakeJudge):
    def __init__(self, verdict):
        super().__init__(verdict)
        self.seen = 0

    def judge(self, image):
        self.seen += 1
        return super().judge(image)


@pytest.mark.parametrize(
    ("verdict", "outcome"),
    [
        (ADULT_FACE, "kept"),
        (replace(ADULT_FACE, min_age=19.0, could_be_under_20=True), "dropped"),
        (replace(ADULT_FACE, min_age=12.0, reason_code="child_or_adolescent"), "blocked"),
    ],
)
def test_age_gate_runs_the_same_with_or_without_adult_content(tmp_path, verdict, outcome) -> None:
    files = photos(tmp_path / "in", 10)
    results = []
    for flag in (False, True):
        subject = check_subject(
            "person", name="Demo", consent_obtained=True, subject_adult=True, adult_content=flag
        )
        judge = RecordingJudge(verdict)
        folder = tmp_path / f"run-{flag}"
        folder.mkdir()
        try:
            summary = run(folder, files, gate=Gate(judge, FakeEstimator(AgeReading(1, (verdict.min_age,)))),
                          subject=subject)
            state = ("kept", summary["prepared"])
        except LoraBlocked as blocked:
            state = ("blocked", blocked.code)
        except LoraUsageError as usage:
            state = ("dropped", usage.code)
        drops = [json.loads(line)["reason"] for line in (folder / "proj" / "drops.jsonl").read_text().splitlines()]
        results.append((state, judge.seen, drops))
    assert results[0] == results[1]
    assert results[0][0][0] == outcome and results[0][1] == 10


def test_release_only_keeps_a_hash(tmp_path) -> None:
    release = tmp_path / "release.txt"
    release.write_text("signed")
    subject = check_subject("object", name="Mug", rights_confirmed=True, release=release)
    assert subject.release_attached and len(subject.release_sha256) == 64
    assert str(tmp_path) not in json.dumps(subject.to_dict())


def test_collect_dedupes_and_excludes(tmp_path) -> None:
    files = photos(tmp_path / "in", 3)
    (tmp_path / "in" / "notes.txt").write_text("x")
    found = collect(tmp_path / "in", [files[0]], [files[1]])
    assert [p.name for p in found] == ["img00.png", "img02.png"]


def test_prepare_filters_crops_captions_and_unloads(tmp_path) -> None:
    files = photos(tmp_path / "in", 11)
    Image.new("RGB", (600, 600), (128, 128, 128)).save(tmp_path / "in" / "flat.png")
    noise(99, 200).save(tmp_path / "in" / "small.png")
    noise(3).save(tmp_path / "in" / "dup.jpg", quality=98)
    (tmp_path / "in" / "broken.png").write_bytes(b"not an image")
    files = collect(tmp_path / "in", [], [])
    judge, captioner = FakeJudge(), FakeCaptioner()
    result = run(tmp_path, files, gate=Gate(judge, FakeEstimator()), captioner=captioner)
    assert result["prepared"] == 11 and result["set_aside"] == 4
    reasons = sorted(
        json.loads(line)["reason"] for line in (tmp_path / "proj" / "drops.jsonl").read_text().splitlines()
    )
    assert reasons == ["lora.drop.blurry", "lora.drop.duplicate", "lora.drop.small", "lora.drop.unreadable"]
    project = Project(tmp_path / "proj")
    data = project.read()
    assert data["gate_passed"] is True and data["kind"] == "object"
    items = project.items()
    assert all(Image.open(project.root / i.file).size == (1024, 1024) for i in items)
    assert items[0].short == "textured square"
    assert judge.unloaded and captioner.unloaded


def test_missing_gate_blocks_before_any_work(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    for gate in (Gate(None, FakeEstimator()), Gate(FakeJudge(), None)):
        with pytest.raises(LoraBlocked) as info:
            run(tmp_path, files, gate=gate)
        assert info.value.code == "lora.err.gate_missing"
    assert not (tmp_path / "proj" / "prepared").exists()


def test_judge_exception_drops_every_photo(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    with pytest.raises(LoraUsageError) as info:
        run(tmp_path, files, gate=Gate(FakeJudge(fail=True), FakeEstimator()))
    assert info.value.code == "lora.err.too_few"
    reasons = {json.loads(line)["reason"] for line in (tmp_path / "proj" / "drops.jsonl").read_text().splitlines()}
    assert reasons == {"lora.drop.safety"}


def test_minor_signal_blocks_the_run(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    verdict = replace(ADULT_FACE, indicators={**ADULT_FACE.indicators, "child_face": "present"})
    with pytest.raises(LoraBlocked) as info:
        run(tmp_path, files, gate=Gate(FakeJudge(verdict), FakeEstimator(AgeReading(1, (30.0,)))))
    assert info.value.code == "lora.err.blocked"
    assert Project(tmp_path / "proj").read()["gate_passed"] is False


def test_unsure_drops_without_blocking() -> None:
    unsure = replace(CLEAN, indicators={**CLEAN.indicators, "school_context": "unsure"})
    decision = decide(unsure, AgeReading(0, ()), estimator_ok=True)
    assert not decision.keep and not decision.minor


def test_face_needs_both_judges_at_20_or_older() -> None:
    assert decide(ADULT_FACE, AgeReading(1, (31.0,)), estimator_ok=True).keep
    assert decide(ADULT_FACE, AgeReading(1, (31.0,)), estimator_ok=True).path == "full"
    assert not decide(ADULT_FACE, AgeReading(1, (19.5,)), estimator_ok=True, rule="v1").keep
    assert decide(ADULT_FACE, AgeReading(1, (16.0,)), estimator_ok=True).minor
    assert not decide(ADULT_FACE, None, estimator_ok=False).keep
    assert not decide(replace(ADULT_FACE, min_age=19.0), AgeReading(1, (30.0,)), estimator_ok=True, rule="v1").keep
    assert not decide(replace(ADULT_FACE, face_visible="unsure"), None, estimator_ok=False).keep
    youthful = replace(ADULT_FACE, reason_code="youthful_adult")
    assert not decide(youthful, AgeReading(1, (30.0,)), estimator_ok=True, rule="v1").keep
    assert not decide(replace(ADULT_FACE, could_be_under_20=None), AgeReading(1, (30.0,)), estimator_ok=True).keep
    assert not decide(replace(ADULT_FACE, confidence=0.4), AgeReading(1, (30.0,)), estimator_ok=True).keep


def test_faceless_photo_uses_the_judge_and_still_drops_on_doubt() -> None:
    body = Verdict("no", "yes", 30.0, 0.8, "clearly_adult", False, {k: "absent" for k in INDICATORS})
    decision = decide(body, AgeReading(0, ()), estimator_ok=True)
    assert decision.keep and decision.path == "body"
    assert decide(CLEAN, AgeReading(0, ()), estimator_ok=True).path == "no_person"
    assert not decide(replace(body, min_age=None), AgeReading(0, ()), estimator_ok=True).keep
    assert not decide(replace(CLEAN, indicators={}), AgeReading(0, ()), estimator_ok=True).keep
    assert not decide(None, None, estimator_ok=True).keep


def test_judge_output_parsing_fails_closed() -> None:
    good = json.dumps({**ADULT_FACE.__dict__, "min_apparent_age": 34, "any_person_could_be_under_20": False})
    assert parse_verdict("noise " + good).min_age == 34.0
    with pytest.raises(ValueError):
        parse_verdict("no json here")
    with pytest.raises(ValueError):
        parse_verdict(json.dumps({"min_apparent_age": "adult", "indicators": {}}))


def test_missing_captioner_fails(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    with pytest.raises(LoraStageError) as info:
        prepare(
            Project(tmp_path / "proj"),
            files,
            obj(),
            gate=Gate(FakeJudge(), FakeEstimator()),
            load_captioner=lambda choice: None,
            progress=Progress(),
        )
    assert info.value.code == "lora.err.captioner_missing"


def test_second_prepare_adds_a_round(tmp_path) -> None:
    run(tmp_path, photos(tmp_path / "a", 10))
    more = [tmp_path / "b" / f"n{i}.png" for i in range(5)]
    (tmp_path / "b").mkdir()
    for i, path in enumerate(more):
        noise(100 + i).save(path)
    result = run(tmp_path, more, mode="expert")
    assert result["round"] == 1 and result["prepared"] == 15
    rounds = sorted({i.round for i in Project(tmp_path / "proj").items()})
    assert rounds == [0, 1]


def test_cli_prepare_refuses_without_consent_and_runs_with_fakes(tmp_path, capsys, monkeypatch) -> None:
    photos(tmp_path / "in", 10)
    project = str(tmp_path / "proj")
    code = main(
        [
            "--json",
            "lora",
            "prepare",
            "--project",
            project,
            "--photos",
            str(tmp_path / "in"),
            "--kind",
            "person",
            "--name",
            "Ann",
        ]
    )
    assert code == 2 and json.loads(capsys.readouterr().out)["error"]["code"] == "lora.err.no_consent"
    monkeypatch.setattr(lora_cli, "_load_gate", lambda: Gate(FakeJudge(), FakeEstimator()))
    monkeypatch.setattr(lora_cli, "_load_captioner", lambda choice: FakeCaptioner())
    code = main(
        [
            "--json",
            "lora",
            "prepare",
            "--project",
            project,
            "--photos",
            str(tmp_path / "in"),
            "--kind",
            "style",
            "--name",
            "Ink wash",
            "--rights-confirmed",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert any(
        json.loads(line)["event"]["message_key"] == "lora.progress.safety"
        for line in out.splitlines()
        if line.startswith('{"event"')
    )
