"""Gate rule v3 decision table (Aleksandr 00:49 + the under-18 floor)."""

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


def test_default_is_v3() -> None:
    assert DEFAULT_RULE == "v3"


def test_est_16_with_adult_verdict_drops_and_blocks() -> None:
    d = decide(ADULT, face(16.0), estimator_ok=True)
    assert not d.keep and d.minor and d.audit is None


def test_judge_age_16_drops() -> None:
    assert not decide(replace(ADULT, min_age=16.0), face(30.0), estimator_ok=True).keep


def test_est_19_adult_no_markers_passes_and_is_audited() -> None:
    d = decide(ADULT, face(19.0), estimator_ok=True)
    assert d.keep and d.audit is not None
    assert d.audit["min_estimate"] == 19.0 and d.audit["verdict"] == "clearly_adult"
    assert d.audit["markers"] == ABSENT
    assert not decide(ADULT, face(19.0), estimator_ok=True, rule="v1").keep


def test_youthful_adult_verdict_passes_and_is_audited() -> None:
    d = decide(replace(ADULT, reason_code="youthful_adult", min_age=19.0), face(22.0), estimator_ok=True)
    assert d.keep and d.audit is not None


def test_youthful_adult_at_25_passes() -> None:
    assert decide(replace(ADULT, reason_code="youthful_adult", min_age=25.0), face(25.0), estimator_ok=True).keep


@pytest.mark.parametrize("marker", MARKERS)
def test_est_25_with_youth_marker_drops(marker: str) -> None:
    d = decide(replace(ADULT, indicators={**ABSENT, marker: "present"}), face(25.0), estimator_ok=True)
    assert not d.keep and d.audit is None


def test_vlm_unsure_drops() -> None:
    assert not decide(replace(ADULT, reason_code="could_be_under_20"), face(30.0), estimator_ok=True).keep
    assert not decide(replace(ADULT, could_be_under_20=None), face(19.0), estimator_ok=True).keep
    assert not decide(replace(ADULT, could_be_under_20=True), face(19.0), estimator_ok=True).keep
    assert not decide(replace(ADULT, indicators={**ABSENT, "child_face": "unsure"}), face(19.0), estimator_ok=True).keep
    assert not decide(replace(ADULT, confidence=0.4), face(19.0), estimator_ok=True).keep


def test_putto_drops_and_blocks() -> None:
    d = decide(replace(ADULT, indicators={**ABSENT, "infant_or_putto": "present"}), face(30.0), estimator_ok=True)
    assert not d.keep and d.minor


def test_faceless_unchanged_between_v1_and_v3() -> None:
    cases = [
        BODY,
        replace(BODY, min_age=19.0),
        replace(BODY, reason_code="youthful_adult"),
        replace(BODY, indicators={**ABSENT, "childlike_body": "present"}),
        replace(BODY, min_age=None),
    ]
    for v in cases:
        a = decide(v, AgeReading(0, ()), estimator_ok=True, rule="v1")
        b = decide(v, AgeReading(0, ()), estimator_ok=True, rule="v3")
        assert (a.reasons, a.minor, a.path) == (b.reasons, b.minor, b.path)
        assert b.audit is None


def test_unknown_rule_rejected() -> None:
    with pytest.raises(ValueError):
        decide(ADULT, face(30.0), estimator_ok=True, rule="v9")
    with pytest.raises(ValueError):
        Gate(None, None, rule="v9")


def test_audit_holds_no_free_text() -> None:
    d = decide(ADULT, face(19.0), estimator_ok=True)
    assert "prompt" not in json.dumps(d.audit) and "path" not in d.audit
