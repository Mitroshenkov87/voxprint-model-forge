"""Strict gate decision table (v1 decides) and shadow v3 (logs only, decision 01:24)."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from voxprint_forge.lora.gate import DEFAULT_RULE, INDICATORS, AgeReading, Gate, Verdict, decide

MARKERS = ["child_body", "child_face", "undeveloped_figure", "school_context", "lacks_adult_features"]
ABSENT = {k: "absent" for k in INDICATORS}
ADULT = Verdict("yes", "yes", 30.0, 0.9, "clearly_adult", False, dict(ABSENT))
BODY = Verdict("no", "yes", 30.0, 0.8, "clearly_adult", False, dict(ABSENT))


def face(age: float) -> AgeReading:
    return AgeReading(1, (age,))


class J:
    name = "j"

    def __init__(self, v):
        self.v = v

    def judge(self, image):
        return self.v


class E:
    name = "e"

    def __init__(self, r):
        self.r = r

    def estimate(self, image):
        return self.r


def test_strict_is_default() -> None:
    assert DEFAULT_RULE == "v1"


STRICT = [
    (ADULT, face(31.0), True),
    (ADULT, face(16.0), False),
    (ADULT, face(19.0), False),
    (replace(ADULT, min_age=19.0), face(30.0), False),
    (replace(ADULT, reason_code="youthful_adult"), face(30.0), False),
    (replace(ADULT, reason_code="could_be_under_20"), face(30.0), False),
    (replace(ADULT, could_be_under_20=None), face(30.0), False),
    (replace(ADULT, confidence=0.4), face(30.0), False),
    (replace(ADULT, indicators={**ABSENT, "infant_or_putto": "present"}), face(30.0), False),
    (replace(ADULT, indicators={**ABSENT, "child_face": "unsure"}), face(30.0), False),
    (BODY, AgeReading(0, ()), True),
    (replace(BODY, indicators={**ABSENT, "childlike_body": "present"}), AgeReading(0, ()), False),
] + [(replace(ADULT, indicators={**ABSENT, m: "present"}), face(25.0), False) for m in MARKERS]


@pytest.mark.parametrize(("verdict", "reading", "keep"), STRICT)
def test_strict_table(verdict, reading, keep) -> None:
    assert decide(verdict, reading, estimator_ok=True).keep is keep


def test_strict_minor_blocks() -> None:
    assert decide(ADULT, face(16.0), estimator_ok=True).minor


@pytest.mark.parametrize(("verdict", "reading", "keep"), STRICT)
def test_shadow_never_changes_the_outcome(verdict, reading, keep) -> None:
    d = Gate(J(verdict), E(reading)).check(object())
    assert d.keep is keep
    assert d.reasons == decide(verdict, reading, estimator_ok=True).reasons
    assert d.shadow is not None and d.shadow["rule"] == "v3"


def test_shadow_records_would_be_keep_for_youthful_adult() -> None:
    d = Gate(J(ADULT), E(face(19.0))).check(object())
    assert not d.keep
    assert d.shadow["keep"] is True and d.shadow["audit"]["min_estimate"] == 19.0
    assert "prompt" not in json.dumps(d.shadow)


def test_shadow_keeps_floor_and_markers() -> None:
    assert Gate(J(ADULT), E(face(16.0))).check(object()).shadow["keep"] is False
    bad = replace(ADULT, indicators={**ABSENT, "child_body": "present"})
    assert Gate(J(bad), E(face(25.0))).check(object()).shadow["keep"] is False


def test_unknown_rule_rejected() -> None:
    with pytest.raises(ValueError):
        decide(ADULT, face(30.0), estimator_ok=True, rule="v9")
