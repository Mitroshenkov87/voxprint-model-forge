"""gate_trace.jsonl: one line of codes and numbers per checked photo; the decisions do not change."""

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


def mixed(tmp_path):
    """Five adults, one wild MiVOLO read (drop, not minor), one judge crash (drop)."""
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
        "decision": "keep",
        "reasons": [],
    }
    assert trace[5]["mivolo_ages"] == [15.0] and trace[5]["estimate"] == 24.5
    assert trace[5]["decision"] == "drop" and trace[5]["reasons"] == ["single_under_16"]
    assert trace[6]["judge"] == "RuntimeError" and trace[6]["decision"] == "drop"
    assert trace[6]["reasons"] == ["judge_error"] and trace[6]["mivolo_ages"] == []


def test_trace_holds_no_image_data_paths_or_metadata(tmp_path) -> None:
    files, judge, estimator = mixed(tmp_path)
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
    judge = ScriptedJudge([ADULT] * 3 + [CHILD_FACE] + [ADULT] * 2)
    estimator = ScriptedEstimator([AgeReading(1, (31.0,))] * 6)
    with pytest.raises(LoraBlocked) as info:
        run(tmp_path, files, judge, estimator)
    assert info.value.code == "lora.err.blocked"
    trace = rows(tmp_path)
    assert len(trace) == 6
    assert [row["decision"] for row in trace] == ["keep"] * 3 + ["minor"] + ["keep"] * 2
    assert trace[3]["markers"] == ["child_face"] and trace[3]["reasons"] == ["markers"]
    assert trace[3]["judge_min_age"] == 34.0 and trace[3]["mivolo_ages"] == [31.0]


def test_expert_prints_one_line_per_dropped_photo(tmp_path, capsys) -> None:
    files, judge, estimator = mixed(tmp_path)
    run(tmp_path, files, judge, estimator)
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("safety gate: ")]
    assert len(lines) == 2
    wild, crash = lines
    assert wild.startswith(f"safety gate: img05.png ({photo_id(files[5])}) drop: reasons=single_under_16 ")
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
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("safety gate: ")]
    assert lines == [
        f"safety gate: img05.png ({photo_id(files[5])}) minor: reasons=average_under_18 judge=ok "
        "reason_code=clearly_adult markers=- judge_min_age=17 judge_confidence=0.9 mivolo_ages=16.5 estimate=16.75"
    ]


def test_simple_mode_prints_no_gate_lines_but_still_writes_the_trace(tmp_path, capsys) -> None:
    files = photos(tmp_path / "in", 12)
    judge = ScriptedJudge([ADULT] * 12)
    estimator = ScriptedEstimator([AgeReading(1, (31.0,))] * 11 + [AgeReading(1, (15.0,))])
    run(tmp_path, files, judge, estimator, mode="simple")
    assert not [line for line in capsys.readouterr().err.splitlines() if line.startswith("safety gate: ")]
    assert len(rows(tmp_path)) == 12


@pytest.mark.parametrize(
    ("verdict", "reading", "outcome"),
    [
        (ADULT, AgeReading(1, (31.0,)), (True, False)),
        (replace(ADULT, min_age=19.0, could_be_under_20=True), AgeReading(1, (19.0,)), (True, False)),
        (ADULT, AgeReading(1, (15.0,)), (False, False)),
        (replace(ADULT, min_age=17.0), AgeReading(1, (16.5,)), (False, True)),
        (CHILD_FACE, AgeReading(1, (31.0,)), (False, True)),
        (replace(ADULT, reason_code="child_or_adolescent"), AgeReading(1, (31.0,)), (False, True)),
        (replace(ADULT, confidence=0.1), AgeReading(1, (31.0,)), (True, False)),
    ],
)
def test_gate_decisions_are_unchanged(tmp_path, verdict, reading, outcome) -> None:
    """The v4 policy is what it was: same keep/minor per photo, and prepare follows the gate's own decision."""
    decision = Gate(ScriptedJudge([verdict]), ScriptedEstimator([reading])).check(object(), attested=True)
    assert (decision.keep, decision.minor) == outcome
    files = photos(tmp_path / "in", 6)
    script = [ADULT] * 5 + [verdict]
    readings = [AgeReading(1, (31.0,))] * 5 + [reading]
    keep, minor = outcome
    if minor:
        with pytest.raises(LoraBlocked):
            run(tmp_path, files, ScriptedJudge(script), ScriptedEstimator(readings))
    else:
        summary = run(tmp_path, files, ScriptedJudge(script), ScriptedEstimator(readings))
        assert summary["prepared"] == (6 if keep else 5)
    last = rows(tmp_path)[-1]
    assert last["decision"] == ("keep" if keep else "minor" if minor else "drop")
    assert last["reasons"] == list(decision.reasons)
    drops = (tmp_path / "proj" / "drops.jsonl").read_text("utf-8").splitlines()
    assert [json.loads(line)["id"] for line in drops] == ([] if keep else [photo_id(files[5])])
