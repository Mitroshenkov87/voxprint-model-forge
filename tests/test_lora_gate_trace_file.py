"""gate_trace.jsonl: one line of codes, numbers and the policy tier per checked photo.

When the job policy or a check that could not run refuses the whole job (``job_gate``),
the trace is still written, and ``drops.jsonl`` ends with a ``job`` row.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from voxprint_forge.lora.consent import check_subject
from voxprint_forge.lora.errors import LoraBlocked
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import INDICATORS, AgeReading, Gate, Verdict
from voxprint_forge.lora.photos import photo_id
from voxprint_forge.lora.prepare import prepare
from voxprint_forge.lora.project import Project

ADULT = Verdict("yes", "yes", 34.0, 0.9, "clearly_adult", False, {k: "absent" for k in INDICATORS})
CHILD_FACE = replace(ADULT, indicators={**ADULT.indicators, "child_face": "present"})
KEYS = {
    "file",
    "id",
    "judge",
    "reason_code",
    "markers",
    "judge_min_age",
    "judge_confidence",
    "mivolo_ages",
    "estimate",
    "tier",
    "decision",
    "reasons",
}


class ScriptedJudge:
    """Returns (or raises) the next scripted item per call, in photo order."""

    name = "fake-judge"

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def judge(self, image):
        item = self.script[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class ScriptedEstimator:
    """The next scripted reading per call. The gate does not call it after a judge error."""

    name = "fake-estimator"

    def __init__(self, script):
        self.script, self.calls = list(script), 0

    def estimate(self, image):
        item = self.script[self.calls]
        self.calls += 1
        return item


class FakeCaptioner:
    name = "fake-captioner"

    def caption(self, image):
        return {"long": "a person on a plain background", "short": "person"}

    def unload(self):
        pass


def photos(folder, count: int):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        path = folder / f"img{index:02d}.png"
        meta = PngInfo()
        meta.add_text("Comment", "SECRET-META Jane Roe, 1 Main St")
        rng = np.random.default_rng(index)
        Image.fromarray(rng.integers(0, 256, (600, 600, 3), dtype=np.uint8)).save(path, pnginfo=meta)
        paths.append(path)
    return paths


def person():
    return check_subject("person", name="Demo", consent_obtained=True, subject_adult=True)


def run(tmp_path, files, judge, estimator, mode="expert"):
    return prepare(
        Project(tmp_path / "proj"),
        files,
        person(),
        gate=Gate(judge, estimator),
        load_captioner=lambda choice: FakeCaptioner(),
        progress=Progress(),
        mode=mode,
    )


def rows(tmp_path):
    return [json.loads(line) for line in (tmp_path / "proj" / "gate_trace.jsonl").read_text("utf-8").splitlines()]


def drop_ids(tmp_path):
    return [json.loads(line)["id"] for line in (tmp_path / "proj" / "drops.jsonl").read_text("utf-8").splitlines()]


def mixed(tmp_path):
    """Five adults, one photo MiVOLO finds no face in, one incomplete verdict: two unchecked drops, the job runs."""
    files = photos(tmp_path / "in", 7)
    judge = ScriptedJudge([ADULT] * 6 + [replace(ADULT, face_visible="")])
    estimator = ScriptedEstimator([AgeReading(1, (31.0,))] * 5 + [AgeReading(0, ()), AgeReading(1, (31.0,))])
    return files, judge, estimator


def refused(tmp_path):
    """Five adults, one wild MiVOLO read (ambiguous, within the allowance), one judge crash.

    The allowance would let the job train; the crash is a check that could not run, so it is refused.
    """
    files = photos(tmp_path / "in", 7)
    judge = ScriptedJudge([ADULT] * 6 + [RuntimeError("/home/me/secret.png exploded")])
    estimator = ScriptedEstimator([AgeReading(1, (31.0,))] * 5 + [AgeReading(1, (15.0,))])
    return files, judge, estimator


def test_one_trace_line_per_checked_photo_with_the_numbers(tmp_path) -> None:
    files, judge, estimator = mixed(tmp_path)
    summary = run(tmp_path, files, judge, estimator)
    assert summary["prepared"] == 5
    trace = rows(tmp_path)
    assert [row["file"] for row in trace] == [path.name for path in files]
    assert [row["id"] for row in trace] == [photo_id(path) for path in files]
    assert all(set(row) == KEYS for row in trace)
    assert trace[0] == {
        "file": "img00.png",
        "id": photo_id(files[0]),
        "judge": "ok",
        "reason_code": "clearly_adult",
        "markers": [],
        "judge_min_age": 34.0,
        "judge_confidence": 0.9,
        "mivolo_ages": [31.0],
        "estimate": 32.5,
        "tier": "clear",
        "decision": "keep",
        "reasons": [],
    }
    assert trace[5]["mivolo_ages"] == [] and trace[5]["estimate"] is None
    assert trace[5]["decision"] == "drop" and trace[5]["reasons"] == ["estimator_no_face"]
    assert trace[5]["tier"] == "unchecked" and trace[6]["tier"] == "unchecked"
    assert trace[6]["judge"] == "ok" and trace[6]["estimate"] == 32.5
    assert trace[6]["decision"] == "drop" and trace[6]["reasons"] == ["judge_incomplete"]
    assert drop_ids(tmp_path) == [photo_id(files[5]), photo_id(files[6])]
    assert (tmp_path / "proj" / "gate.json").is_file()


def test_refused_trace_keeps_the_numbers_of_every_photo(tmp_path) -> None:
    files, judge, estimator = refused(tmp_path)
    with pytest.raises(LoraBlocked) as info:
        run(tmp_path, files, judge, estimator)
    assert info.value.code == "lora.err.check_unavailable"
    trace = rows(tmp_path)
    assert [row["file"] for row in trace] == [path.name for path in files]
    assert [row["decision"] for row in trace[:5]] == ["keep"] * 5
    assert trace[5]["mivolo_ages"] == [15.0] and trace[5]["estimate"] == 24.5
    assert trace[5]["decision"] == "drop" and trace[5]["reasons"] == ["single_under_16"]
    assert trace[5]["tier"] == "ambiguous" and trace[6]["tier"] == "unchecked"
    assert trace[6]["judge"] == "RuntimeError" and trace[6]["decision"] == "drop"
    assert trace[6]["reasons"] == ["judge_error"] and trace[6]["mivolo_ages"] == []
    assert drop_ids(tmp_path) == [photo_id(files[5]), photo_id(files[6]), "job"]


def test_a_check_that_could_not_run_refuses_and_still_writes_the_trace(tmp_path) -> None:
    files = photos(tmp_path / "in", 6)
    judge = ScriptedJudge([ADULT] * 5 + [RuntimeError("judge down")])
    estimator = ScriptedEstimator([AgeReading(1, (31.0,))] * 5)
    with pytest.raises(LoraBlocked) as info:
        run(tmp_path, files, judge, estimator)
    assert info.value.code == "lora.err.check_unavailable"
    trace = rows(tmp_path)
    assert len(trace) == 6 and trace[5]["reasons"] == ["judge_error"]
    assert drop_ids(tmp_path) == [photo_id(files[5]), "job"]
    assert Project(tmp_path / "proj").read()["gate_passed"] is False


@pytest.mark.parametrize("fixture", [mixed, refused])
def test_trace_holds_no_image_data_paths_or_metadata(tmp_path, fixture) -> None:
    files, judge, estimator = fixture(tmp_path)
    if fixture is refused:
        with pytest.raises(LoraBlocked):
            run(tmp_path, files, judge, estimator)
    else:
        run(tmp_path, files, judge, estimator)
    text = (tmp_path / "proj" / "gate_trace.jsonl").read_text("utf-8")
    for leak in (str(tmp_path), "secret", "SECRET-META", "Jane", "Main St", "/", "\\"):
        assert leak not in text
    for row in rows(tmp_path):
        assert set(row) == KEYS
        assert len(json.dumps(row)) < 400
        for value in row.values():
            if isinstance(value, list):
                assert all(isinstance(v, (str, int, float)) or v is None for v in value)
            else:
                assert isinstance(value, (str, int, float)) or value is None


def test_trace_is_written_when_the_run_is_refused(tmp_path) -> None:
    files = photos(tmp_path / "in", 6)
    strong = replace(CHILD_FACE, reason_code="child_or_adolescent")
    judge = ScriptedJudge([ADULT] * 3 + [strong] + [ADULT] * 2)
    adult, young = AgeReading(1, (31.0,)), AgeReading(1, (17.0,))
    estimator = ScriptedEstimator([adult] * 3 + [young] + [adult] * 2)
    with pytest.raises(LoraBlocked) as info:
        run(tmp_path, files, judge, estimator)
    assert info.value.code == "lora.err.blocked"
    trace = rows(tmp_path)
    assert len(trace) == 6
    assert [row["decision"] for row in trace] == ["keep"] * 3 + ["minor"] + ["keep"] * 2
    assert [row["tier"] for row in trace] == ["clear"] * 3 + ["strong"] + ["clear"] * 2
    assert trace[3]["markers"] == ["child_face"] and trace[3]["reasons"] == ["markers"]
    assert trace[3]["judge_min_age"] == 34.0 and trace[3]["mivolo_ages"] == [17.0]
    assert drop_ids(tmp_path) == [photo_id(files[3]), "job"]
    assert not (tmp_path / "proj" / "gate.json").exists()


def test_expert_prints_one_line_per_dropped_photo(tmp_path, capsys) -> None:
    files, judge, estimator = mixed(tmp_path)
    run(tmp_path, files, judge, estimator)
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("safety gate: ")]
    assert len(lines) == 2
    no_face, incomplete = lines
    head = f"safety gate: img05.png ({photo_id(files[5])}) drop: tier=unchecked reasons=estimator_no_face "
    assert no_face.startswith(head)
    for part in ("judge=ok", "reason_code=clearly_adult", "markers=-", "judge_min_age=34", "judge_confidence=0.9"):
        assert part in no_face
    assert "mivolo_ages=- estimate=-" in no_face
    assert incomplete.startswith("safety gate: img06.png") and "reasons=judge_incomplete judge=ok" in incomplete
    assert "mivolo_ages=31 estimate=32.5" in incomplete


def test_expert_prints_the_dropped_photos_before_the_refusal(tmp_path, capsys) -> None:
    files, judge, estimator = refused(tmp_path)
    with pytest.raises(LoraBlocked):
        run(tmp_path, files, judge, estimator)
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("safety gate: ")]
    assert len(lines) == 2
    wild, crash = lines
    head = f"safety gate: img05.png ({photo_id(files[5])}) drop: tier=ambiguous reasons=single_under_16 "
    assert wild.startswith(head)
    for part in ("judge=ok", "reason_code=clearly_adult", "markers=-", "judge_min_age=34", "judge_confidence=0.9"):
        assert part in wild
    assert "mivolo_ages=15" in wild and "estimate=24.5" in wild
    assert crash.startswith("safety gate: img06.png") and "reasons=judge_error judge=RuntimeError" in crash


def test_expert_prints_the_minor_flag_before_the_refusal(tmp_path, capsys) -> None:
    files = photos(tmp_path / "in", 6)
    judge = ScriptedJudge([ADULT] * 5 + [replace(ADULT, min_age=17.0)])
    estimator = ScriptedEstimator([AgeReading(1, (31.0,))] * 5 + [AgeReading(1, (16.5,))])
    with pytest.raises(LoraBlocked):
        run(tmp_path, files, judge, estimator)
    assert drop_ids(tmp_path) == [photo_id(files[5]), "job"]
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("safety gate: ")]
    assert lines == [
        f"safety gate: img05.png ({photo_id(files[5])}) minor: tier=strong reasons=average_under_18 judge=ok "
        "reason_code=clearly_adult markers=- judge_min_age=17 judge_confidence=0.9 mivolo_ages=16.5 estimate=16.75"
    ]


def test_simple_mode_prints_no_gate_lines_but_still_writes_the_trace(tmp_path, capsys) -> None:
    files = photos(tmp_path / "in", 12)
    judge = ScriptedJudge([ADULT] * 12)
    estimator = ScriptedEstimator([AgeReading(1, (31.0,))] * 11 + [AgeReading(0, ())])
    summary = run(tmp_path, files, judge, estimator, mode="simple")
    assert summary["prepared"] == 11
    assert not [line for line in capsys.readouterr().err.splitlines() if line.startswith("safety gate: ")]
    assert len(rows(tmp_path)) == 12


@pytest.mark.parametrize(
    ("verdict", "reading", "outcome", "tier"),
    [
        (ADULT, AgeReading(1, (31.0,)), (True, False), "clear"),
        (replace(ADULT, min_age=19.0, could_be_under_20=True), AgeReading(1, (19.0,)), (True, False), "margin"),
        (ADULT, AgeReading(1, (15.0,)), (False, False), "ambiguous"),
        (replace(ADULT, min_age=17.0), AgeReading(1, (16.5,)), (False, True), "strong"),
        (CHILD_FACE, AgeReading(1, (31.0,)), (False, True), "ambiguous"),
        (replace(ADULT, reason_code="child_or_adolescent"), AgeReading(1, (31.0,)), (False, True), "ambiguous"),
        (replace(ADULT, confidence=0.1), AgeReading(1, (31.0,)), (True, False), "ambiguous"),
    ],
)
def test_v4_reasons_are_unchanged_and_prepare_follows_the_tier(tmp_path, verdict, reading, outcome, tier) -> None:
    """v4 keep/minor per photo is what it was; prepare trains on clear photos only and refuses on the job policy.

    Five clear photos plus the one under test is below the 90 % clear share, so a strong photo refuses (and adds
    the ``job`` drop row) and one ambiguous photo is within the allowance of 1.
    """
    decision = Gate(ScriptedJudge([verdict]), ScriptedEstimator([reading])).check(object(), attested=True)
    assert (decision.keep, decision.minor) == outcome
    assert decision.tier == tier
    files = photos(tmp_path / "in", 6)
    script = [ADULT] * 5 + [verdict]
    readings = [AgeReading(1, (31.0,))] * 5 + [reading]
    job_refused = tier == "strong"
    if job_refused:
        with pytest.raises(LoraBlocked):
            run(tmp_path, files, ScriptedJudge(script), ScriptedEstimator(readings))
    else:
        summary = run(tmp_path, files, ScriptedJudge(script), ScriptedEstimator(readings))
        assert summary["prepared"] == (6 if tier == "clear" else 5)
    last = rows(tmp_path)[-1]
    assert last["tier"] == tier
    assert last["decision"] == {"clear": "keep", "strong": "minor"}.get(tier, "drop")
    assert last["reasons"] == list(decision.reasons)
    dropped = [] if tier == "clear" else [photo_id(files[5])]
    assert drop_ids(tmp_path) == dropped + (["job"] if job_refused else [])
