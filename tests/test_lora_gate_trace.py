"""Every gate stage reports what happened per photo (ids and codes only); a failure still drops the photo."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from PIL import Image

from voxprint_forge.lora.backends import JudgeParseError, WorkerError, parse_verdict
from voxprint_forge.lora.gate import INDICATORS, AgeReading, Gate, Verdict

ADULT = Verdict("yes", "yes", 34.0, 0.9, "clearly_adult", False, {k: "absent" for k in INDICATORS})
IMAGE = Image.new("RGB", (16, 16))


class Judge:
    name = "judge"

    def __init__(self, verdict=ADULT, exc=None, stats=None):
        self.verdict, self.exc = verdict, exc
        self.last_stats = stats or {}

    def judge(self, image):
        if self.exc is not None:
            raise self.exc
        return self.verdict


class Estimator:
    name = "estimator"

    def __init__(self, reading=AgeReading(1, (31.0,)), exc=None):
        self.reading, self.exc = reading, exc

    def estimate(self, image):
        if self.exc is not None:
            raise self.exc
        return self.reading


def test_keep_records_both_stages() -> None:
    decision = Gate(Judge(stats={"new_tokens": 120, "hit_token_limit": False}), Estimator()).check(IMAGE, attested=True)
    assert decision.keep
    assert decision.trace["judge"]["ok"] and decision.trace["judge"]["new_tokens"] == 120
    assert decision.trace["judge"]["verdict"]["indicators"] == {k: "absent" for k in INDICATORS}
    assert decision.trace["estimator"] == {"ok": True, "n_faces": 1, "ages": [31.0]}


def test_judge_parse_error_is_a_drop_with_its_code() -> None:
    decision = Gate(Judge(exc=JudgeParseError("no_json"), stats={"hit_token_limit": True}), Estimator()).check(IMAGE)
    assert not decision.keep and decision.reasons == ("judge_error",)
    assert decision.trace["judge"] == {
        "ok": False,
        "stage": "judge.parse",
        "error": "JudgeParseError",
        "code": "no_json",
        "hit_token_limit": True,
    }


def test_judge_crash_records_type_not_message() -> None:
    decision = Gate(Judge(exc=RuntimeError("/secret/path/frame.png exploded")), Estimator()).check(IMAGE)
    assert not decision.keep
    assert decision.trace["judge"] == {"ok": False, "stage": "judge", "error": "RuntimeError"}
    assert "secret" not in json.dumps(decision.trace)


def test_estimator_failure_is_recorded_and_still_drops_a_face_photo() -> None:
    decision = Gate(Judge(), Estimator(exc=WorkerError("OutOfMemoryError"))).check(IMAGE)
    assert not decision.keep and "estimator_missing" in decision.reasons
    assert decision.trace["estimator"] == {
        "ok": False,
        "stage": "estimator.worker",
        "error": "WorkerError",
        "code": "OutOfMemoryError",
    }


def test_trace_cuts_free_text_from_the_verdict() -> None:
    chatty = replace(ADULT, reason_code="a long sentence describing the whole picture in detail")
    decision = Gate(Judge(chatty), Estimator()).check(IMAGE)
    assert len(decision.trace["judge"]["verdict"]["reason_code"]) <= 24


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("no braces here", "no_json"),
        ('{"a": 1,}', "bad_json"),
        ('{"min_apparent_age": "x"}', "bad_age"),
        ('{"min_apparent_age": 30}', "no_indicators"),
        ('{"face_visible": "yes", "indicators": {"child_face"', "no_json"),
    ],
)
def test_parse_codes(raw, code) -> None:
    with pytest.raises(JudgeParseError) as err:
        parse_verdict(raw)
    assert err.value.code == code and isinstance(err.value, ValueError)
