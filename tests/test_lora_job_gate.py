"""Job-level minors gate: one refusal before training. Fakes only."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tests.lora_fakes import TinyAdapter, gradient, make_project
from tests.test_lora_prepare import ADULT_FACE, FakeCaptioner, FakeEstimator, FakeJudge, photos
from voxprint_forge.lora.consent import check_subject
from voxprint_forge.lora.errors import LoraBlocked, LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import AgeReading, Gate, child_keyword
from voxprint_forge.lora.job_gate import preflight_refuse, repin_captions
from voxprint_forge.lora.prepare import prepare
from voxprint_forge.lora.presets import build_config
from voxprint_forge.lora.project import Item, Project
from voxprint_forge.lora.train import train


def tiny():
    return build_config("quick", "balanced", 24, steps=2, resolution=64, save_every=2, batch=2, previews=False)


def person(**kw):
    fields = {"name": "Demo", "consent_obtained": True, "subject_adult": True, **kw}
    return check_subject("person", **fields)


def adult_gate(age: float = 34.0, verdict=ADULT_FACE) -> Gate:
    return Gate(FakeJudge(replace(verdict, min_age=age)), FakeEstimator(AgeReading(1, (age,))))


def run_prepare(root, files, *, gate=None, subject=None, captioner=None, name="proj"):
    cap = captioner or FakeCaptioner()
    project = Project(root / name)
    prepare(
        project,
        files,
        subject or person(),
        gate=gate or adult_gate(),
        load_captioner=lambda choice: cap,
        progress=Progress(),
    )
    return project


def assert_refused(info, project: Project, key: str = "lora.err.blocked", source: str | None = None) -> None:
    assert info.value.code == key
    assert project.read()["gate_passed"] is False
    assert not (project.root / "gate.json").exists()
    if source is not None:
        audit = [json.loads(line) for line in (project.root / "refusals.jsonl").read_text().splitlines()]
        assert audit[-1]["source"] == source


class WordCaptioner(FakeCaptioner):
    def __init__(self, long: str) -> None:
        super().__init__()
        self.long = long

    def caption(self, image):
        return {"long": self.long, "short": "a woman"}


class BrokenCaptioner(FakeCaptioner):
    def caption(self, image):
        raise RuntimeError("captioner crashed")


# --- photo signal -------------------------------------------------------------


def one_child(age: float, margin: int = 0):
    """A gate whose 4th photo is ``child_or_adolescent`` with MiVOLO reading ``age``.

    The next ``margin`` photos read 19 on MiVOLO (18-20 adult margin); the rest are adults.
    """

    class OneChild(FakeJudge):
        seen = 0

        def judge(self, image):
            OneChild.seen += 1
            if OneChild.seen == 4:
                return replace(ADULT_FACE, reason_code="child_or_adolescent")
            return ADULT_FACE

    class OneYoung(FakeEstimator):
        seen = 0

        def estimate(self, image):
            OneYoung.seen += 1
            if OneYoung.seen == 4:
                return AgeReading(1, (age,))
            return AgeReading(1, (19.0 if 4 < OneYoung.seen <= 4 + margin else 34.0,))

    return Gate(OneChild(), OneYoung())


def test_one_strong_photo_below_the_clear_share_refuses_the_whole_job(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    with pytest.raises(LoraBlocked) as info:
        run_prepare(tmp_path, files, gate=one_child(15.0, margin=1))
    project = Project(tmp_path / "proj")
    assert_refused(info, project, source="photo")
    assert not list(project.prepared.glob("*.png")) if project.prepared.exists() else True


def test_one_ambiguous_photo_at_the_clear_share_is_dropped_and_the_job_trains(tmp_path) -> None:
    files = photos(tmp_path / "in", 11)
    project = run_prepare(tmp_path, files, gate=one_child(34.0))
    assert len(project.items()) == 10
    assert project.read()["gate_passed"] is True and (project.root / "gate.json").is_file()


def test_a_single_reading_under_16_refuses_the_job(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    gate = Gate(FakeJudge(replace(ADULT_FACE, min_age=30.0)), FakeEstimator(AgeReading(1, (15.0,))))
    with pytest.raises(LoraBlocked) as info:
        run_prepare(tmp_path, files, gate=gate)
    assert_refused(info, Project(tmp_path / "proj"), source="photo")


# --- text sources -------------------------------------------------------------


def test_caption_with_a_child_word_refuses_the_job(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    with pytest.raises(LoraBlocked) as info:
        run_prepare(tmp_path, files, captioner=WordCaptioner("a teenage girl standing by a wall"))
    assert_refused(info, Project(tmp_path / "proj"), source="caption")


def test_edited_caption_with_a_child_word_refuses_training(tmp_path) -> None:
    project = make_project(tmp_path / "p", 10)
    items = project.items()
    items[3] = replace(items[3], long="tcube, a schoolgirl in a classroom")
    project.write_items(items)
    with pytest.raises(LoraBlocked) as info:
        repin_captions(project)
    assert_refused(info, project, source="caption")


def test_trigger_refuses_the_job(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    with pytest.raises(LoraBlocked) as info:
        run_prepare(tmp_path, files, subject=person(trigger="teen_anna"))
    assert_refused(info, Project(tmp_path / "proj"), source="trigger")


def test_project_name_refuses_the_job(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    with pytest.raises(LoraBlocked) as info:
        run_prepare(tmp_path, files, name="loli_set")
    assert_refused(info, Project(tmp_path / "loli_set"), source="project_name")
    with pytest.raises(LoraBlocked):
        run_prepare(tmp_path, files, subject=person(stage_name="Школьница Аня"), name="other")


def test_sample_prompt_refuses_training(tmp_path) -> None:
    project = make_project(tmp_path / "p", 10)
    adapter = TinyAdapter()
    with pytest.raises(LoraBlocked) as info:
        train(project, tiny(), adapter, Progress(), preview_prompts=["tcube, 16yo girl on a beach"])
    assert_refused(info, project, source="sample_prompt")
    assert adapter.calls == 0 and not project.checkpoints.exists()


def test_setting_refuses_training(tmp_path) -> None:
    project = make_project(tmp_path / "p", 10)
    project.write({**project.read(), "style_hint": "school uniform, 25 years old"})
    with pytest.raises(LoraBlocked) as info:
        train(project, tiny(), TinyAdapter(), Progress())
    assert_refused(info, project, source="setting")
    young = make_project(tmp_path / "q", 10)
    data = young.read()
    subject = {**data["subject"], "kind": "person", "adult_confirmed": True, "attr": 16}
    young.write({**data, "kind": "person", "subject": subject})
    with pytest.raises(LoraBlocked):
        preflight_refuse(young)


@pytest.mark.parametrize(
    "word",
    ["ребёнок", "ребенка", "девочке", "мальчишки", "школьница", "подростку", "несовершеннолетняя",
     "малолетка", "лоли", "детьми", "rebenok", "devochka", "shkolnitsa", "16 лет", "шестнадцатилетняя", "сhild"],
)
def test_ru_keyword_refuses_the_job(tmp_path, word) -> None:
    project = make_project(tmp_path / "p", 10)
    items = project.items()
    items[0] = replace(items[0], short=f"tcube, {word}")
    project.write_items(items)
    with pytest.raises(LoraBlocked) as info:
        repin_captions(project)
    assert_refused(info, project, source="caption")


@pytest.mark.parametrize(
    "text",
    ["девушка", "молодая женщина 19 лет", "студентка, 18 лет", "мальчишник", "детектив", "малышка", "ребята",
     "одноклассники", "фото 5 лет назад", "10 лет опыта", "амурская область", "школа танцев", "young woman",
     "kidney", "19yo", "eighteen-year-old", "фото 2016 года", "school uniform"],
)
def test_adult_words_do_not_trip_the_keyword_list(text) -> None:
    assert not child_keyword(text)


# --- fail closed --------------------------------------------------------------


@pytest.mark.parametrize("broken", ["judge", "estimator", "captioner"])
def test_a_check_that_errors_refuses_the_job(tmp_path, broken) -> None:
    files = photos(tmp_path / "in", 10)

    class Crash:
        name = "crash"

        def judge(self, image):
            raise RuntimeError("down")

        estimate = judge

    gate = adult_gate()
    if broken == "judge":
        gate = Gate(Crash(), gate.estimator)
    elif broken == "estimator":
        gate = Gate(gate.judge, Crash())
    captioner = BrokenCaptioner() if broken == "captioner" else None
    with pytest.raises(LoraBlocked) as info:
        run_prepare(tmp_path, files, gate=gate, captioner=captioner)
    assert_refused(info, Project(tmp_path / "proj"), key="lora.err.check_unavailable")


def test_a_check_that_errors_at_preflight_refuses_training(tmp_path, monkeypatch) -> None:
    project = make_project(tmp_path / "p", 10)

    def down(*args):
        raise RuntimeError("keyword check down")

    monkeypatch.setattr("voxprint_forge.lora.job_gate.child_keyword", down)
    with pytest.raises(LoraBlocked) as info:
        train(project, tiny(), TinyAdapter(), Progress())
    assert_refused(info, project, key="lora.err.check_unavailable", source="check_error")


def test_extend_prepare_with_the_gate_missing_leaves_the_old_set_unapproved(tmp_path) -> None:
    project = run_prepare(tmp_path, photos(tmp_path / "in", 10))
    assert project.read()["gate_passed"] is True
    with pytest.raises(LoraBlocked):
        run_prepare(tmp_path, photos(tmp_path / "more", 10), gate=Gate(None, FakeEstimator()))
    with pytest.raises(LoraBlocked):
        preflight_refuse(project)


# --- integrity ----------------------------------------------------------------


@pytest.mark.parametrize("change", ["changed", "added", "removed", "captions"])
def test_a_file_changed_after_the_gate_refuses_training(tmp_path, change) -> None:
    project = make_project(tmp_path / "p", 10)
    preflight_refuse(project)
    if change == "changed":
        gradient(99).save(project.prepared / "p03.png")
    elif change == "added":
        gradient(99).save(project.prepared / "extra.png")
    elif change == "removed":
        (project.prepared / "p03.png").unlink()
    else:
        project.write_items([*project.items()[:-1], Item("p09", "prepared/p09.png", "tcube, other", "tcube", 0)])
    adapter = TinyAdapter()
    with pytest.raises(LoraBlocked) as info:
        train(project, tiny(), adapter, Progress())
    assert_refused(info, project, key="lora.err.dataset_changed", source="hash_mismatch")
    assert adapter.calls == 0


def test_caption_edit_repins_only_when_images_still_match(tmp_path) -> None:
    project = make_project(tmp_path / "p", 10)
    gradient(99).save(project.prepared / "p03.png")
    with pytest.raises(LoraBlocked) as info:
        repin_captions(project)
    assert info.value.code == "lora.err.dataset_changed"


# --- adults who look 18-20 ------------------------------------------------------


@pytest.mark.parametrize("age", [18.5, 19.0])
def test_youthful_adult_18_to_20_is_dropped_as_margin_and_never_refuses(tmp_path, age) -> None:
    """Job policy: the 18-20 margin is not trained on, but it never refuses the job either."""
    verdict = replace(ADULT_FACE, reason_code="youthful_adult", could_be_under_20=True)
    with pytest.raises(LoraUsageError) as info:
        run_prepare(tmp_path, photos(tmp_path / "in", 10), gate=adult_gate(age, verdict))
    assert info.value.code == "lora.err.too_few"
    project = Project(tmp_path / "proj")
    assert not (project.root / "refusals.jsonl").exists()
    assert {row["tier"] for row in map(json.loads, project.gate_trace.read_text().splitlines())} == {"margin"}


@pytest.mark.parametrize("age", [20.0])
def test_youthful_adult_at_20_passes_prepare_and_preflight(tmp_path, age) -> None:
    verdict = replace(ADULT_FACE, reason_code="youthful_adult", could_be_under_20=True)
    project = run_prepare(tmp_path, photos(tmp_path / "in", 10), gate=adult_gate(age, verdict))
    assert project.read()["gate_passed"] is True
    clearance = preflight_refuse(project, prompts=["vxp, a young woman, 19 years old, close-up portrait"])
    assert clearance.rule == 4
    assert not (project.root / "refusals.jsonl").exists()
