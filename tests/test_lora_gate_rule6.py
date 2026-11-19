"""Minors gate rule 6: caption glitches are re-captioned, adult idioms pass, 18+ margin photos are clear. Fakes only."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tests.test_lora_job_gate import adult_gate, assert_refused, person
from tests.test_lora_prepare import FakeCaptioner, photos
from voxprint_forge.lora.errors import LoraBlocked
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import (
    INDICATORS,
    RULE_VERSION,
    AgeReading,
    Verdict,
    child_keyword,
    child_keyword_match,
    decide,
    decide_job,
)
from voxprint_forge.lora.prepare import prepare
from voxprint_forge.lora.project import Project

ABSENT = {k: "absent" for k in INDICATORS}
YOUTHFUL = Verdict("yes", "yes", 18.0, 0.95, "youthful_adult", False, dict(ABSENT))


class ScriptedCaptioner(FakeCaptioner):
    """Returns the scripted long caption for each call number (1-based), else the clean default."""

    def __init__(self, script: dict[int, str]) -> None:
        super().__init__()
        self.script, self.calls = script, 0

    def caption(self, image, adult_hint=False):
        self.calls += 1
        if self.calls in self.script:
            item = self.script[self.calls]
            if isinstance(item, Exception):
                raise item
            return {"long": item, "short": "a woman"}
        return super().caption(image)


def run(tmp_path, captioner, *, count=10, subject=None):
    project = Project(tmp_path / "proj")
    summary = prepare(
        project,
        photos(tmp_path / "in", count),
        subject or person(),
        gate=adult_gate(),
        load_captioner=lambda choice: captioner,
        progress=Progress(),
        mode="expert",
    )
    return project, summary


def drops(project: Project) -> list[dict]:
    if not project.drops.is_file():
        return []
    return [json.loads(line) for line in project.drops.read_text("utf-8").splitlines()]


def test_rule_version() -> None:
    assert RULE_VERSION == 6


# --- 1. a caption glitch is re-captioned, then only its frame is dropped ------------------


def test_a_glitched_caption_is_re_captioned_once_and_the_clean_one_is_kept(tmp_path, capsys) -> None:
    captioner = ScriptedCaptioner({3: "a teen standing by a wall"})
    project, summary = run(tmp_path, captioner)
    assert summary["prepared"] == 10 and captioner.calls == 11
    assert project.read()["gate_passed"] is True
    assert not drops(project)
    assert not any(child_keyword(item.long, item.short) for item in project.items())
    err = capsys.readouterr().err
    assert "caption keyword 'teen': re-captioning" in err and "dropped this frame" not in err


def test_a_caption_that_glitches_twice_drops_only_that_frame_and_the_job_trains(tmp_path, capsys) -> None:
    captioner = ScriptedCaptioner({3: "kids playing", 4: "a kid on a bench"})
    project, summary = run(tmp_path, captioner)
    assert summary["prepared"] == 9 and captioner.calls == 11
    assert project.read()["gate_passed"] is True and (project.root / "gate.json").is_file()
    assert len(list(project.prepared.glob("*.webp"))) == 9
    rows = drops(project)
    assert len(rows) == 1
    assert rows[0]["reason"] == "lora.drop.caption_keyword" and rows[0]["match"] == "kid"
    dropped = rows[0]["id"]
    assert dropped not in {item.id.removesuffix("-full") for item in project.items()}
    assert not (project.root / "refusals.jsonl").exists()
    err = capsys.readouterr().err
    assert "caption keyword 'kids': re-captioning" in err
    assert f"({dropped}-full) caption keyword 'kid': dropped this frame" in err


def test_the_caption_keyword_lines_print_in_simple_mode_too(tmp_path, capsys) -> None:
    captioner = ScriptedCaptioner({1: "a schoolgirl", 2: "a schoolgirl"})
    project = Project(tmp_path / "proj")
    prepare(
        project,
        photos(tmp_path / "in", 11),
        person(),
        gate=adult_gate(),
        load_captioner=lambda choice: captioner,
        progress=Progress(),
    )
    err = capsys.readouterr().err
    assert "caption keyword 'schoolgirl': re-captioning" in err and "dropped this frame" in err
    assert len(project.items()) == 10


def test_a_captioner_error_on_the_re_caption_still_refuses(tmp_path) -> None:
    captioner = ScriptedCaptioner({1: "a teen", 2: RuntimeError("captioner crashed")})
    with pytest.raises(LoraBlocked) as info:
        run(tmp_path, captioner)
    assert_refused(info, Project(tmp_path / "proj"), key="lora.err.check_unavailable", source="captioner_error")


@pytest.mark.parametrize(
    ("field", "value", "source"),
    [("trigger", "teen_anna", "trigger"), ("stage_name", "kid demo", "project_name")],
)
def test_user_written_texts_still_refuse(tmp_path, field, value, source) -> None:
    captioner = ScriptedCaptioner({})
    with pytest.raises(LoraBlocked) as info:
        run(tmp_path, captioner, subject=person(**{field: value}))
    assert_refused(info, Project(tmp_path / "proj"), source=source)
    assert captioner.calls == 0


def test_child_keyword_match_names_the_word() -> None:
    assert child_keyword_match("a woman", "two kids playing") == "kids"
    assert child_keyword_match("16yo girl") == "16yo"
    assert child_keyword_match("under 18") == "under 18"
    assert child_keyword_match("a woman in kid gloves") is None


# --- 2. adult idioms ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["kid gloves", "wearing kid gloves", "kid-glove treatment", "kidding", "kidded", "minor details",
     "minor scratches on the wall", "in a minor key", "a minor detail", "minor changes", "minor flaws",
     "minor imperfections", "minor blemishes", "minor wrinkles", "minor issues", "minor edits", "a minor role"],
)
def test_adult_idioms_do_not_match(text) -> None:
    assert not child_keyword(text)


@pytest.mark.parametrize(
    "text",
    ["a minor", "minors", "a teen", "schoolgirl outfit", "kid", "kids playing", "child", "minor girl",
     "minor child", "kid gloves on a kid", "minor details of a minor", "t33n", "сhild", "kіd gloves",
     "школьница", "16 лет"],
)
def test_child_words_still_match(text) -> None:
    assert child_keyword(text)


# --- 3. an 18+ margin photo is clear --------------------------------------------------------


def tier(verdict, ages):
    return decide(verdict, AgeReading(len(ages), tuple(ages)), estimator_ok=True, rule="v4")


def test_the_reported_youthful_adult_photo_is_clear() -> None:
    decision = tier(YOUTHFUL, [24.75])
    assert decision.keep and decision.tier == "clear"
    assert decision.trace["estimate"] == 21.375


@pytest.mark.parametrize(
    ("verdict", "ages", "expected"),
    [
        (replace(YOUTHFUL, min_age=19.0), [19.0], "clear"),
        (replace(YOUTHFUL, min_age=17.9), [30.0], "strong"),
        (YOUTHFUL, [17.9], "strong"),
        (replace(YOUTHFUL, indicators={**ABSENT, "child_face": "present"}), [24.0], "ambiguous"),
        (replace(YOUTHFUL, reason_code="child_or_adolescent"), [24.0], "ambiguous"),
        (replace(YOUTHFUL, confidence=0.3), [24.0], "ambiguous"),
    ],
)
def test_only_an_all_18_plus_reading_becomes_clear(verdict, ages, expected) -> None:
    assert tier(verdict, ages).tier == expected


def test_the_judge_only_margin_stays_margin() -> None:
    body = replace(YOUTHFUL, face_visible="no", min_age=19.0)
    assert decide(body, None, estimator_ok=False, rule="v4").tier == "margin"


def test_18_plus_photos_count_toward_the_clear_share(tmp_path) -> None:
    tiers = [tier(YOUTHFUL, [24.75]).tier] * 27 + ["ambiguous"] * 3
    assert decide_job(tiers, keyword=False).why == "clear_share"
    gate = adult_gate(18.5, verdict=YOUTHFUL)
    project = Project(tmp_path / "proj")
    summary = prepare(
        project,
        photos(tmp_path / "in", 10),
        person(),
        gate=gate,
        load_captioner=lambda choice: FakeCaptioner(),
        progress=Progress(),
    )
    assert summary["prepared"] == 10 and not drops(project)
