"""Attested-adult person sets (final rule, 2026-11-18): no face -> keep; MiVOLO-only age flags kept up to a share.

Rule 1: a photo where MiVOLO ran and found no face is never dropped by the age gate, at any no-face ratio.
Rule 2: MiVOLO under-age flags the judge does not share are false positives. The first
``floor(AGE_FLAG_KEEP_SHARE * input photos)`` of them in file-name order are kept, the rest are dropped with
their normal drop reason. No refusal. Everything else in the minors rule is unchanged.
Synthetic data only: scripted judge and estimator fakes, random-noise PNGs.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace

import pytest

from tests.test_lora_gate_trace_file import (
    ADULT,
    CHILD_FACE,
    FakeCaptioner,
    ScriptedEstimator,
    ScriptedJudge,
    person,
    photos,
    rows,
)
from voxprint_forge.lora.errors import LoraBlocked
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import (
    AGE_FLAG_DROPPED,
    AGE_FLAG_KEEP,
    AGE_FLAG_KEEP_SHARE,
    AGE_FLAG_KEPT,
    NO_FACE_KEEP_ATTESTED,
    AgeReading,
    Decision,
    Face,
    Gate,
    age_flag_keep_limit,
    apply_attested_keeps,
    decide_job,
)
from voxprint_forge.lora.photos import photo_id
from voxprint_forge.lora.prepare import prepare
from voxprint_forge.lora.project import Project

FACE = AgeReading(1, (31.0,))
NO_FACE = AgeReading(0, ())
# MiVOLO says 8.19 while the judge says 20: single_under_16 (and the average is under 18).
JUDGE_20 = replace(ADULT, min_age=20.0)
FLAG_READING = AgeReading(1, (8.19,))
# An extra spurious low-score face aged 5.41; the main face has a low score too, so no primary is chosen.
SPURIOUS = AgeReading(2, (30.0, 5.41), (Face((100.0, 100.0, 300.0, 300.0), 0.8), Face((10.0, 10.0, 40.0, 40.0), 0.3)))
# The real no-face example: no face found, judge ok, judge_min_age 18.0, confidence 0.95.
NO_FACE_JUDGE = replace(ADULT, min_age=18.0, confidence=0.95)


def check(verdict, reading, *, estimator_raises: bool = False) -> Decision:
    class Broken:
        name = "broken"

        def estimate(self, image):
            raise RuntimeError("estimator down")

    estimator = Broken() if estimator_raises else ScriptedEstimator([reading])
    return Gate(ScriptedJudge([verdict]), estimator).check(object(), attested=True)


def clear():
    return check(ADULT, FACE)


def flagged():
    return check(JUDGE_20, FLAG_READING)


def names(n: int) -> list[str]:
    return [f"photo_{i:03d}.png" for i in range(n)]


def keep_all(decisions, file_names, *, attested=True, n_inputs=None):
    return apply_attested_keeps(
        decisions,
        file_names,
        [f"id{i:03d}" for i in range(len(decisions))],
        attested=attested,
        n_inputs=len(decisions) if n_inputs is None else n_inputs,
    )


# --- the limit ------------------------------------------------------------------


def test_share_constant_and_the_limit_rounds_down() -> None:
    assert AGE_FLAG_KEEP_SHARE == 0.10
    assert age_flag_keep_limit(180) == 18
    assert age_flag_keep_limit(179) == 17
    assert age_flag_keep_limit(10) == 1
    assert age_flag_keep_limit(9) == 0  # floor(0.9) = 0: small sets round down
    assert age_flag_keep_limit(0) == 0
    assert all(age_flag_keep_limit(n) == n // 10 for n in range(0, 1001))


def test_flag_examples_are_age_flags_before_the_pass() -> None:
    single = flagged()
    assert set(single.reasons) == {"single_under_16", "average_under_18"} and single.tier == "ambiguous"
    spurious = check(ADULT, SPURIOUS)
    assert spurious.trace["faces"]["why"] == "low_score"
    assert spurious.reasons == ("single_under_16",) and spurious.tier == "ambiguous"


# --- rule 2: age flags ------------------------------------------------------------


def test_180_inputs_with_18_flagged_keeps_all_18() -> None:
    decisions = [clear() for _ in range(162)] + [flagged() for _ in range(18)]
    out, audit = keep_all(decisions, names(180))
    assert all(d.tier == "clear" and d.keep for d in out)
    for d in out[162:]:
        assert d.trace["keep_reason"] == AGE_FLAG_KEEP and d.trace["age_flag"] == AGE_FLAG_KEPT
        assert d.trace["age_flag_limit"] == 18 and "single_under_16" in d.trace["kept_reasons"]
    assert audit["age_flag_keep_limit"] == 18 and audit["age_flagged"] == 18
    assert len(audit["age_flag_kept"]) == 18 and audit["age_flag_dropped"] == []


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_180_inputs_with_19_flagged_keeps_18_and_drops_the_last_by_file_name_in_any_order(seed) -> None:
    rng = random.Random(seed)
    file_names = names(180)
    flagged_at = set(rng.sample(range(180), 19))
    last = max(file_names[i] for i in flagged_at)
    rows_in = [(file_names[i], flagged() if i in flagged_at else clear()) for i in range(180)]
    rng.shuffle(rows_in)  # processing order is not file-name order
    out, audit = keep_all([d for _, d in rows_in], [n for n, _ in rows_in])
    by_name = {name: d for (name, _), d in zip(rows_in, out, strict=True)}
    dropped = [name for name, d in by_name.items() if not d.keep]
    assert dropped == [last]
    lost = by_name[last]
    assert lost.tier == "ambiguous" and "single_under_16" in lost.reasons
    assert lost.trace["age_flag"] == AGE_FLAG_DROPPED and lost.trace["age_flag_limit"] == 18
    assert "keep_reason" not in lost.trace
    assert [e["file"] for e in audit["age_flag_dropped"]] == [last]
    kept_files = [e["file"] for e in audit["age_flag_kept"]]
    assert kept_files == sorted(file_names[i] for i in flagged_at)[:18]


def test_9_inputs_with_1_flagged_drops_it() -> None:
    decisions = [clear() for _ in range(8)] + [flagged()]
    out, audit = keep_all(decisions, names(9))
    assert not out[8].keep and out[8].tier == "ambiguous" and out[8].trace["age_flag"] == AGE_FLAG_DROPPED
    assert audit["age_flag_keep_limit"] == 0 and audit["age_flag_kept"] == []
    assert [e["file"] for e in audit["age_flag_dropped"]] == ["photo_008.png"]


def test_the_limit_is_a_share_of_the_input_photos_not_the_checked_ones() -> None:
    decisions = [clear() for _ in range(8)] + [flagged()]
    out, audit = keep_all(decisions, names(9), n_inputs=10)
    assert out[8].keep and audit["age_flag_keep_limit"] == 1 and audit["inputs"] == 10


@pytest.mark.parametrize(
    ("verdict", "reading"),
    [
        (replace(ADULT, min_age=19.0), AgeReading(1, (8.0,))),  # the judge reads under 20 itself
        (replace(ADULT, min_age=17.0), AgeReading(1, (8.0,))),  # the judge reads under 18
        (CHILD_FACE, AgeReading(1, (8.0,))),  # a judge marker
        (replace(ADULT, reason_code="child_or_adolescent"), AgeReading(1, (8.0,))),
        (replace(ADULT, confidence=0.3), AgeReading(1, (8.0,))),  # the judge reading is not confident
        (replace(ADULT, face_visible=""), AgeReading(1, (8.0,))),  # incomplete verdict
    ],
)
def test_judge_side_findings_are_not_age_flags(verdict, reading) -> None:
    before = check(verdict, reading)
    assert not before.keep or before.tier != "clear"
    out, audit = keep_all([*[clear() for _ in range(99)], before], names(100))
    assert out[99].reasons == before.reasons and out[99].tier == before.tier
    assert "age_flag" not in out[99].trace and audit["age_flagged"] == 0


def test_a_child_keyword_is_not_an_age_flag() -> None:
    d = flagged()
    keyword = Decision((*d.reasons, "child_keyword"), True, d.path, d.trace, tier="strong")
    out, _ = keep_all([*[clear() for _ in range(99)], keyword], names(100))
    assert out[99].tier == "strong" and not out[99].keep


def test_a_set_that_is_not_attested_is_unchanged() -> None:
    decisions = [clear() for _ in range(99)] + [flagged(), check(NO_FACE_JUDGE, NO_FACE)]
    out, audit = keep_all(decisions, names(101), attested=False)
    assert out == decisions and [d.trace for d in out] == [d.trace for d in decisions]
    assert audit["attested"] is False and audit["age_flag_keep_limit"] == 0


# --- rule 1: no face --------------------------------------------------------------


@pytest.mark.parametrize("face_visible", ["yes", "no", "unsure"])
def test_the_real_no_face_example_is_kept_whatever_the_ratio(face_visible) -> None:
    before = check(replace(NO_FACE_JUDGE, face_visible=face_visible), NO_FACE)
    assert before.tier == "margin" and not (before.tier == "clear")
    # No photo passed with a face: the no-face majority would never keep it.
    out, audit = keep_all([before, flagged(), flagged()], names(3))
    kept = out[0]
    assert kept.keep and kept.tier == "clear" and not kept.minor
    assert kept.trace["keep_reason"] == NO_FACE_KEEP_ATTESTED
    assert kept.trace["kept_reasons"] == list(before.reasons)
    assert [e["file"] for e in audit["no_face_kept"]] == ["photo_000.png"]


def test_a_no_face_photo_the_judge_reads_as_adult_is_kept() -> None:
    before = check(ADULT, NO_FACE)
    assert before.reasons == ("estimator_no_face",) and before.tier == "unchecked"
    out, _ = keep_all([before], names(1))
    assert out[0].keep and out[0].tier == "clear" and out[0].trace["keep_reason"] == NO_FACE_KEEP_ATTESTED


@pytest.mark.parametrize(
    "verdict",
    [
        replace(NO_FACE_JUDGE, min_age=17.0),  # the judge says under 18
        replace(NO_FACE_JUDGE, indicators={**ADULT.indicators, "child_body": "present"}),
        replace(NO_FACE_JUDGE, reason_code="child_or_adolescent"),
        replace(NO_FACE_JUDGE, indicators={}),  # incomplete verdict
        replace(NO_FACE_JUDGE, confidence=0.2),  # not a confident reading
    ],
)
def test_no_face_with_a_judge_finding_stays_as_it_was(verdict) -> None:
    before = check(verdict, NO_FACE)
    assert before.tier != "clear"
    out, audit = keep_all([before], names(1))
    assert out[0].tier == before.tier and out[0].reasons == before.reasons and audit["no_face_kept"] == []


def test_an_estimator_that_did_not_run_is_not_a_no_face_photo() -> None:
    before = check(NO_FACE_JUDGE, None, estimator_raises=True)
    assert "estimator_missing" in before.reasons
    out, _ = keep_all([before], names(1))
    assert out[0].tier == before.tier and not out[0].keep


# --- the job ----------------------------------------------------------------------


def test_job_dropped_age_flags_never_refuse_and_kept_ones_do_not_lift_a_strong_refusal() -> None:
    # 30 flagged photos of 180 dropped past the limit would be over the allowance of 10: no refusal.
    tiers = ["clear"] * 150 + ["clear"] * 18 + ["ambiguous"] * 12
    assert decide_job(tiers, keyword=False, relaxed=18, exempt=12).refuse is False
    assert decide_job(tiers, keyword=False, relaxed=18).why == "over_allowance"
    # 26 clear + 2 kept as false positives + 1 strong of 30: the clear share is still 26/30, so it refuses.
    tiers = ["clear"] * 28 + ["strong"] + ["margin"]
    assert decide_job(tiers, keyword=False).why == "clear_share"
    assert decide_job(tiers, keyword=False, relaxed=2).why == "strong"


# --- prepare ------------------------------------------------------------------------


def _prepare(tmp_path, files, judge, estimator):
    return prepare(
        Project(tmp_path / "proj"),
        files,
        person(),
        gate=Gate(judge, estimator),
        load_captioner=lambda choice: FakeCaptioner(),
        progress=Progress(),
        mode="expert",
        crop_mode="whole",
    )


def _drops(tmp_path):
    path = tmp_path / "proj" / "drops.jsonl"
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()] if path.is_file() else []


@pytest.fixture(scope="module")
def photos_180(tmp_path_factory):
    return photos(tmp_path_factory.mktemp("in180"), 180)


@pytest.mark.parametrize("n_flagged", [18, 19])
def test_prepare_180_inputs_keeps_18_age_flags_first_by_file_name(tmp_path, photos_180, n_flagged) -> None:
    files = list(photos_180)
    flagged_at = set(range(180 - n_flagged, 180))  # the last names: the 19th is img179.png
    order = list(reversed(files))  # processed in reverse file-name order
    judge = ScriptedJudge([JUDGE_20 if files.index(p) in flagged_at else ADULT for p in order])
    estimator = ScriptedEstimator([FLAG_READING if files.index(p) in flagged_at else FACE for p in order])
    summary = _prepare(tmp_path, order, judge, estimator)
    assert summary["prepared"] == 180 - (n_flagged - 18)
    trace = {row["file"]: row for row in rows(tmp_path)}
    kept = sorted(name for name, row in trace.items() if row["age_flag"] == AGE_FLAG_KEPT)
    dropped = [name for name, row in trace.items() if row["age_flag"] == AGE_FLAG_DROPPED]
    assert kept == sorted(files[i].name for i in flagged_at)[:18]
    assert all(trace[name]["keep_reason"] == AGE_FLAG_KEEP and trace[name]["decision"] == "keep" for name in kept)
    assert all(trace[name]["age_flag_limit"] == 18 for name in kept + dropped)
    gate = json.loads((tmp_path / "proj" / "gate.json").read_text("utf-8"))["age_flags"]
    assert gate["age_flag_keep_limit"] == 18 and gate["inputs"] == 180 and gate["age_flagged"] == n_flagged
    assert sorted(e["file"] for e in gate["age_flag_kept"]) == kept
    if n_flagged == 19:
        assert dropped == ["img179.png"]
        row = trace["img179.png"]
        assert row["decision"] == "drop" and row["tier"] == "ambiguous" and "single_under_16" in row["reasons"]
        assert _drops(tmp_path) == [{"id": photo_id(files[179]), "name": "img179.png", "reason": "lora.drop.safety"}]
        assert [e["file"] for e in gate["age_flag_dropped"]] == ["img179.png"]
    else:
        assert dropped == [] and _drops(tmp_path) == [] and gate["age_flag_dropped"] == []
    assert not (tmp_path / "proj" / "refusals.jsonl").exists()


def test_prepare_9_inputs_with_1_flagged_drops_it_and_trains(tmp_path) -> None:
    files = photos(tmp_path / "in", 9)
    judge = ScriptedJudge([ADULT] * 8 + [JUDGE_20])
    estimator = ScriptedEstimator([FACE] * 8 + [FLAG_READING])
    summary = _prepare(tmp_path, files, judge, estimator)
    assert summary["prepared"] == 8
    row = rows(tmp_path)[8]
    assert row["decision"] == "drop" and row["age_flag"] == AGE_FLAG_DROPPED and row["age_flag_limit"] == 0
    assert _drops(tmp_path) == [{"id": photo_id(files[8]), "name": files[8].name, "reason": "lora.drop.safety"}]
    gate = json.loads((tmp_path / "proj" / "gate.json").read_text("utf-8"))["age_flags"]
    assert gate["age_flag_keep_limit"] == 0 and [e["file"] for e in gate["age_flag_dropped"]] == [files[8].name]


def test_prepare_keeps_no_face_photos_with_no_majority(tmp_path) -> None:
    """Half the set has no face (ratio 0.5, under NO_FACE_MAJORITY): every no-face photo is still kept."""
    files = photos(tmp_path / "in", 10)
    judge = ScriptedJudge([ADULT] * 5 + [NO_FACE_JUDGE] * 5)
    estimator = ScriptedEstimator([FACE] * 5 + [NO_FACE] * 5)
    summary = _prepare(tmp_path, files, judge, estimator)
    assert summary["prepared"] == 10
    trace = rows(tmp_path)
    for row in trace[5:]:
        assert row["mivolo_ages"] == [] and row["judge_min_age"] == 18.0 and row["judge_confidence"] == 0.95
        assert row["tier"] == "clear" and row["decision"] == "keep" and row["keep_reason"] == NO_FACE_KEEP_ATTESTED
    assert _drops(tmp_path) == []
    gate = json.loads((tmp_path / "proj" / "gate.json").read_text("utf-8"))["age_flags"]
    assert sorted(e["file"] for e in gate["no_face_kept"]) == [p.name for p in files[5:]]


def test_prepare_keeps_judge_side_minor_refusal(tmp_path) -> None:
    """A judge-side strong photo still refuses below the clear share; age-flag keeps do not count toward it."""
    files = photos(tmp_path / "in", 10)
    judge = ScriptedJudge([ADULT] * 8 + [JUDGE_20, replace(ADULT, min_age=17.0)])
    estimator = ScriptedEstimator([FACE] * 8 + [FLAG_READING, AgeReading(1, (15.0,))])
    with pytest.raises(LoraBlocked):
        _prepare(tmp_path, files, judge, estimator)
    trace = rows(tmp_path)
    assert trace[8]["age_flag"] == AGE_FLAG_KEPT and trace[9]["tier"] == "strong"
