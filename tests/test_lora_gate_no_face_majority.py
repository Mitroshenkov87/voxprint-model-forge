"""No-face majority: a photo where no face was found is kept when most of the set passed with a face.

The pass runs over the whole set after every photo is decided. Only a photo whose one reason is
``estimator_no_face`` and that nothing reads as under age is eligible; under-age and margin faces stay dropped.
In ``prepare`` on an attested set, the attested no-face keep (``apply_attested_keeps``) then keeps a
no-face photo whatever the ratio; see tests/test_lora_gate_attested_keeps.py.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from PIL import Image

from tests.test_lora_gate_trace_file import ADULT, ScriptedEstimator, ScriptedJudge, photos, rows, run
from voxprint_forge.lora.gate import (
    NO_FACE_KEEP,
    NO_FACE_MAJORITY,
    RULE_VERSION,
    AgeReading,
    Gate,
    apply_no_face_majority,
    face_pass_ratio,
)
from voxprint_forge.lora.photos import photo_id

FACE = AgeReading(1, (31.0,))
NO_FACE = AgeReading(0, ())
BODY = replace(ADULT, face_visible="no")
MARGIN_JUDGE = replace(ADULT, min_age=19.0)
MINOR_JUDGE = replace(ADULT, min_age=17.0)
UNDER_AGE_FACE = AgeReading(1, (15.0,))
MARKED_JUDGE = replace(ADULT, indicators={**ADULT.indicators, "child_face": "present"})
# 18-20 on both models with nothing else found: clear since rule 6.
MARGIN_FACE = (replace(ADULT, min_age=19.0, could_be_under_20=True), AgeReading(1, (19.0,)))


def check(verdict, reading):
    return Gate(ScriptedJudge([verdict]), ScriptedEstimator([reading])).check(object(), attested=True)


def clear(n):
    return [check(ADULT, FACE) for _ in range(n)]


def no_face(verdict=ADULT):
    return check(verdict, NO_FACE)


def test_threshold_default_and_rule_version() -> None:
    assert NO_FACE_MAJORITY == 0.80
    assert NO_FACE_KEEP == "kept_no_face_majority"
    assert RULE_VERSION == 6


def test_no_face_photo_is_unchecked_per_photo() -> None:
    decision = no_face()
    assert decision.reasons == ("estimator_no_face",) and decision.tier == "unchecked" and decision.path == "full"


def test_eighty_percent_keeps_the_no_face_photos() -> None:
    before = [*clear(8), no_face(), no_face()]
    after = apply_no_face_majority(before)
    assert face_pass_ratio(before) == pytest.approx(0.80)
    for decision in after[8:]:
        assert decision.keep and decision.tier == "clear" and not decision.minor
        assert decision.path == "full"
        assert decision.trace["keep_reason"] == "kept_no_face_majority"
        assert decision.trace["no_face_ratio"] == pytest.approx(0.80)
        # The per-photo trace is kept.
        assert {"judge", "estimator", "faces", "estimate"} <= set(decision.trace)
    assert after[:8] == before[:8] and all("keep_reason" not in d.trace for d in after[:8])
    # The input is not changed.
    assert before[8].reasons == ("estimator_no_face",) and "keep_reason" not in before[8].trace


def test_below_eighty_percent_drops_them() -> None:
    before = [*clear(7), no_face(), no_face()]
    after = apply_no_face_majority(before)
    assert face_pass_ratio(before) == pytest.approx(7 / 9)
    for decision in after[7:]:
        assert decision.reasons == ("estimator_no_face",) and decision.tier == "unchecked"
        assert "keep_reason" not in decision.trace


def test_threshold_argument() -> None:
    before = [*clear(7), no_face(), no_face()]
    assert apply_no_face_majority(before, threshold=0.75)[7].tier == "clear"
    assert apply_no_face_majority([*clear(8), no_face(), no_face()], threshold=0.9)[8].tier == "unchecked"


def test_empty_set() -> None:
    assert face_pass_ratio([]) == 0.0
    assert apply_no_face_majority([]) == []


def test_body_only_photos_do_not_count_toward_the_ratio() -> None:
    body = check(BODY, NO_FACE)
    assert body.keep and body.tier == "clear" and body.path == "body"
    before = [*clear(7), body, no_face(), no_face()]
    assert face_pass_ratio(before) == pytest.approx(0.70)
    assert [d.tier for d in apply_no_face_majority(before)[8:]] == ["unchecked", "unchecked"]


def test_an_18_plus_margin_face_is_clear_and_counts_toward_the_ratio() -> None:
    margin = check(*MARGIN_FACE)
    assert margin.keep and margin.tier == "clear"
    before = [*clear(7), margin, no_face(), no_face()]
    assert face_pass_ratio(before) == pytest.approx(0.80)
    after = apply_no_face_majority(before)
    assert after[7].tier == "clear"
    assert [d.tier for d in after[8:]] == ["clear", "clear"]


def test_under_age_face_stays_dropped_when_the_majority_is_met() -> None:
    under = check(ADULT, UNDER_AGE_FACE)
    strong = check(MINOR_JUDGE, AgeReading(1, (16.5,)))
    assert under.tier == "ambiguous" and strong.tier == "strong"
    after = apply_no_face_majority([*clear(8), under, no_face()])
    assert after[8].reasons == under.reasons and after[8].tier == "ambiguous" and not after[8].keep
    assert after[9].tier == "clear" and after[9].trace["keep_reason"] == NO_FACE_KEEP
    after = apply_no_face_majority([*clear(8), strong, no_face()])
    assert after[8].tier == "strong" and after[8].minor and not after[8].keep
    assert after[9].tier == "clear"


def test_an_18_plus_margin_face_is_kept_as_it_is_when_the_majority_is_met() -> None:
    margin = check(*MARGIN_FACE)
    after = apply_no_face_majority([*clear(8), margin, no_face()])
    assert after[8].tier == "clear" and "keep_reason" not in after[8].trace
    assert after[9].tier == "clear"


def test_margin_judge_age_with_no_face_is_eligible() -> None:
    decision = no_face(MARGIN_JUDGE)
    assert decision.reasons == ("estimator_no_face",) and decision.tier == "margin"
    kept = apply_no_face_majority([*clear(8), decision])[8]
    assert kept.keep and kept.tier == "clear" and kept.trace["no_face_ratio"] == pytest.approx(8 / 9)


def test_judge_under_age_with_no_face_stays_dropped() -> None:
    decision = no_face(MINOR_JUDGE)
    assert decision.reasons == ("estimator_no_face",) and decision.tier == "strong"
    after = apply_no_face_majority([*clear(9), decision])
    assert after[9].tier == "strong" and not after[9].keep


def test_no_face_with_another_reason_stays_dropped() -> None:
    incomplete = no_face(replace(ADULT, indicators={}))
    assert set(incomplete.reasons) == {"judge_incomplete", "estimator_no_face"}
    marked = no_face(replace(ADULT, indicators={**ADULT.indicators, "child_face": "present"}))
    assert "markers" in marked.reasons
    after = apply_no_face_majority([*clear(18), incomplete, marked])
    assert after[18].reasons == incomplete.reasons and after[18].tier == incomplete.tier
    assert after[19].reasons == marked.reasons and after[19].tier == marked.tier


def drops(tmp_path):
    path = tmp_path / "proj" / "drops.jsonl"
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()] if path.is_file() else []


def test_prepare_keeps_no_face_photos_at_eighty_percent(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    judge = ScriptedJudge([ADULT] * 10)
    estimator = ScriptedEstimator([FACE] * 8 + [NO_FACE] * 2)
    summary = run(tmp_path, files, judge, estimator)
    assert summary["prepared"] == 10
    trace = rows(tmp_path)
    assert len(trace) == 10
    for row in trace[8:]:
        assert row["tier"] == "clear" and row["decision"] == "keep" and row["reasons"] == []
        assert row["keep_reason"] == "kept_no_face_majority"
        assert row["no_face_ratio"] == pytest.approx(0.80)
        assert row["mivolo_ages"] == [] and row["estimate"] is None
    assert all(row["keep_reason"] is None and row["no_face_ratio"] is None for row in trace[:8])
    # The provisional safety drops of the kept photos are gone.
    assert drops(tmp_path) == []
    prepared = {
        p.stem.removesuffix("-full").removesuffix("-close")
        for p in (tmp_path / "proj" / "prepared").glob("*.webp")
    }
    assert {photo_id(path) for path in files} <= prepared


def test_prepare_keeps_no_face_photos_below_eighty_percent_on_an_attested_set(tmp_path) -> None:
    """Below the majority the no-face photos are still kept: on an attested set no face -> keep (2026-11-18)."""
    files = photos(tmp_path / "in", 9)
    judge = ScriptedJudge([ADULT] * 9)
    estimator = ScriptedEstimator([FACE] * 7 + [NO_FACE] * 2)
    summary = run(tmp_path, files, judge, estimator)
    assert summary["prepared"] == 9
    trace = rows(tmp_path)
    for row in trace[7:]:
        assert row["tier"] == "clear" and row["reasons"] == [] and row["decision"] == "keep"
        assert row["keep_reason"] == "kept_no_face" and row["no_face_ratio"] is None
        assert row["kept_reasons"] == ["estimator_no_face"]
    assert drops(tmp_path) == []


def test_prepare_keeps_the_no_face_photo_and_drops_the_under_age_face(tmp_path) -> None:
    # The under-age face is judge-side (a child-face marker): the attested age-flag keep does not cover it.
    files = photos(tmp_path / "in", 10)
    judge = ScriptedJudge([ADULT] * 8 + [MARKED_JUDGE, ADULT])
    estimator = ScriptedEstimator([FACE] * 9 + [NO_FACE])
    summary = run(tmp_path, files, judge, estimator)
    assert summary["prepared"] == 9
    trace = rows(tmp_path)
    assert trace[8]["tier"] == "ambiguous" and trace[8]["decision"] == "drop" and trace[8]["keep_reason"] is None
    assert trace[9]["tier"] == "clear" and trace[9]["keep_reason"] == "kept_no_face_majority"
    assert drops(tmp_path) == [{"id": photo_id(files[8]), "name": files[8].name, "reason": "lora.drop.safety"}]


def test_prepare_purges_only_the_kept_photos_drop_rows(tmp_path) -> None:
    """A filter drop from the same run and an under-age face stay in drops.jsonl; the kept no-face photo does not."""
    files = photos(tmp_path / "in", 10)
    tiny = tmp_path / "in" / "tiny.png"
    Image.new("RGB", (64, 64), "white").save(tiny)
    judge = ScriptedJudge([ADULT] * 8 + [MARKED_JUDGE, ADULT])
    estimator = ScriptedEstimator([FACE] * 9 + [NO_FACE])
    summary = run(tmp_path, [*files, tiny], judge, estimator)
    assert summary["prepared"] == 9
    reasons = {row["id"]: row["reason"] for row in drops(tmp_path)}
    assert photo_id(files[9]) not in reasons
    assert reasons[photo_id(files[8])] == "lora.drop.safety"
    assert photo_id(tiny) in reasons and reasons[photo_id(tiny)] != "lora.drop.safety"
