"""Minors-gate job policy (rc.12): per-photo tiers, the clear share, the ambiguous allowance, keywords. Fakes only."""

from __future__ import annotations

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
    run,
)
from voxprint_forge.lora.consent import check_subject
from voxprint_forge.lora.errors import LoraBlocked
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import AgeReading, Gate, allowance, decide, decide_job
from voxprint_forge.lora.prepare import prepare
from voxprint_forge.lora.project import Project


def face(age: float) -> AgeReading:
    return AgeReading(1, (age,))


# One fake photo per tier: (judge verdict, MiVOLO reading).
KIND = {
    "clear": (ADULT, face(31.0)),
    "ambiguous": (ADULT, face(15.0)),  # one wild MiVOLO read, the judge says 34
    "margin": (replace(ADULT, min_age=19.0), face(19.0)),  # 18-20 adult margin
    "strong": (replace(ADULT, min_age=17.0), face(15.0)),  # MiVOLO < 16 and judge < 18
}


def gate_for(kinds: list[str]) -> Gate:
    return Gate(ScriptedJudge([KIND[k][0] for k in kinds]), ScriptedEstimator([KIND[k][1] for k in kinds]))


def prepare_set(tmp_path, **counts: int):
    """Prepare (Expert) one photo per counted kind, in the order given."""
    kinds = [kind for kind, count in counts.items() for _ in range(count)]
    gate = gate_for(kinds)
    return run(tmp_path, photos(tmp_path / "in", len(kinds)), gate.judge, gate.estimator)


def tier(verdict, age: float | None) -> str:
    return decide(verdict, None if age is None else face(age), estimator_ok=age is not None, rule="v4").tier


@pytest.mark.parametrize(
    ("verdict", "age", "expected"),
    [
        (ADULT, 31.0, "clear"),
        (replace(ADULT, min_age=20.0), 20.0, "clear"),
        # strong: judge child_or_adolescent and MiVOLO < 18; MiVOLO < 16 and judge < 18
        (replace(ADULT, reason_code="child_or_adolescent"), 17.0, "strong"),
        (replace(ADULT, min_age=17.0), 15.0, "strong"),
        # not covered by the policy but both models under 18: fail closed, as v4 did
        (replace(ADULT, min_age=17.0), 17.0, "strong"),
        # ambiguous: one model says minor, the other says 20 or older
        (CHILD_FACE, 31.0, "ambiguous"),
        (replace(ADULT, reason_code="child_or_adolescent"), 20.0, "ambiguous"),
        (replace(ADULT, min_age=20.0), 17.0, "ambiguous"),
        (ADULT, 15.0, "ambiguous"),
        (replace(ADULT, min_age=15.0), 25.0, "ambiguous"),
        # ambiguous: a minor reading whose average sits in [18, 20)
        (replace(ADULT, min_age=19.0), 17.0, "ambiguous"),
        # ambiguous: low judge confidence, v4 reads it again
        (replace(ADULT, confidence=0.4), 31.0, "ambiguous"),
        (replace(ADULT, confidence=None), 31.0, "ambiguous"),
        # margin: no minor reading, but not both models at 20 or older
        (replace(ADULT, min_age=19.0), 19.0, "margin"),
        (replace(ADULT, min_age=18.0), 30.0, "margin"),
        (ADULT, 19.5, "margin"),
        # unchecked: MiVOLO did not run; dropped, never refuses
        (ADULT, None, "unchecked"),
        # the judge alone says minor while MiVOLO did not run: fail closed
        (CHILD_FACE, None, "strong"),
    ],
)
def test_photo_tiers(verdict, age, expected) -> None:
    assert tier(verdict, age) == expected


def test_a_judge_crash_is_unchecked() -> None:
    gate = Gate(ScriptedJudge([RuntimeError("boom")]), ScriptedEstimator([]))
    assert gate.check(object(), attested=True).tier == "unchecked"


@pytest.mark.parametrize(
    ("n", "expected"), [(1, 1), (10, 1), (30, 1), (31, 1), (45, 6), (59, 10), (60, 10), (100, 10)]
)
def test_allowance(n, expected) -> None:
    assert allowance(n) == expected


def test_job_decision_order() -> None:
    assert decide_job(["clear"] * 29 + ["strong"], keyword=True).why == "child_keyword"
    assert decide_job(["clear"] * 27 + ["strong"] * 3, keyword=False).why == "clear_share"
    assert decide_job(["clear"] * 26 + ["strong"] + ["margin"] * 3, keyword=False).why == "strong"
    assert decide_job(["clear"] * 26 + ["ambiguous"] * 2 + ["margin"] * 2, keyword=False).why == "over_allowance"
    within = decide_job(["clear"] * 26 + ["ambiguous"] + ["unchecked"] * 3, keyword=False)
    assert not within.refuse and within.why == "within_allowance" and within.allowance == 1


def test_30_with_1_ambiguous_trains_on_29(tmp_path) -> None:
    assert prepare_set(tmp_path, clear=29, ambiguous=1)["prepared"] == 29
    assert [row["tier"] for row in rows(tmp_path)].count("ambiguous") == 1


def test_30_below_the_clear_share_with_1_ambiguous_and_3_margin_trains_on_26(tmp_path) -> None:
    assert prepare_set(tmp_path, clear=26, ambiguous=1, margin=3)["prepared"] == 26


def test_30_below_the_clear_share_with_2_ambiguous_refuses(tmp_path) -> None:
    with pytest.raises(LoraBlocked) as info:
        prepare_set(tmp_path, clear=26, ambiguous=2, margin=2)
    assert info.value.code == "lora.err.blocked"
    assert Project(tmp_path / "proj").read()["gate_passed"] is False


def test_60_with_10_ambiguous_trains_and_11_refuses(tmp_path) -> None:
    assert prepare_set(tmp_path / "ten", clear=50, ambiguous=10)["prepared"] == 50
    with pytest.raises(LoraBlocked):
        prepare_set(tmp_path / "eleven", clear=49, ambiguous=11)


@pytest.mark.parametrize("n", [6, 30, 45, 60])
def test_one_strong_photo_refuses_at_any_size_below_the_clear_share(tmp_path, n) -> None:
    clear = n * 8 // 10
    with pytest.raises(LoraBlocked):
        prepare_set(tmp_path, clear=clear, strong=1, margin=n - clear - 1)


def test_30_with_27_clear_and_3_flagged_trains_on_27_even_past_the_allowance_and_with_a_strong_photo(
    tmp_path,
) -> None:
    assert prepare_set(tmp_path, clear=27, ambiguous=2, strong=1)["prepared"] == 27
    trace = rows(tmp_path)
    assert sorted(row["tier"] for row in trace if row["decision"] != "keep") == ["ambiguous", "ambiguous", "strong"]


def test_30_with_26_clear_and_one_strong_refuses(tmp_path) -> None:
    with pytest.raises(LoraBlocked):
        prepare_set(tmp_path, clear=26, strong=1, margin=3)


def test_a_filename_keyword_refuses_at_any_ratio(tmp_path) -> None:
    files = photos(tmp_path / "in", 30)
    files[7] = files[7].rename(files[7].with_name("kid_portrait.png"))
    gate = gate_for(["clear"] * 30)
    with pytest.raises(LoraBlocked):
        run(tmp_path, files, gate.judge, gate.estimator)
    trace = rows(tmp_path)
    assert trace[7]["tier"] == "strong" and "child_keyword" in trace[7]["reasons"]


def test_a_caption_keyword_refuses_at_any_ratio(tmp_path) -> None:
    class Teen(FakeCaptioner):
        calls = 0

        def caption(self, image):
            self.calls += 1
            if self.calls == 3:
                return {"long": "a teen on a plain background", "short": "person"}
            return super().caption(image)

    project = Project(tmp_path / "proj")
    with pytest.raises(LoraBlocked):
        prepare(
            project,
            photos(tmp_path / "in", 30),
            person(),
            gate=gate_for(["clear"] * 30),
            load_captioner=lambda choice: Teen(),
            progress=Progress(),
            mode="expert",
        )
    assert project.read()["gate_passed"] is False
    assert not list(project.prepared.glob("*.png")) and not project.items()


@pytest.mark.parametrize(
    "fields",
    [{"name": "Kid Demo"}, {"stage_name": "Schoolgirl"}, {"trigger": "teen_vxp"}, {"provenance_note": "my children"}],
)
def test_a_subject_keyword_refuses_before_any_check(tmp_path, fields) -> None:
    subject = check_subject("person", **{"name": "Demo", **fields}, consent_obtained=True, subject_adult=True)
    gate = gate_for([])
    with pytest.raises(LoraBlocked):
        prepare(
            Project(tmp_path / "proj"),
            photos(tmp_path / "in", 10),
            subject,
            gate=gate,
            load_captioner=lambda choice: FakeCaptioner(),
            progress=Progress(),
        )
    assert gate.judge.calls == 0


def test_expert_prints_the_job_verdict(tmp_path, capsys) -> None:
    prepare_set(tmp_path, clear=26, ambiguous=1, margin=3)
    lines = [line for line in capsys.readouterr().err.splitlines() if line.startswith("safety gate job: ")]
    assert lines == [
        "safety gate job: trains (within_allowance) n=30 clear=26 margin=3 unchecked=0 ambiguous=1 strong=0 allowance=1"
    ]
