"""``lora prepare`` runs in phases: gate models, then the captioner, never both. Fakes only."""

from __future__ import annotations

import json
import weakref

import pytest

from tests.test_lora_prepare import ADULT_FACE, FakeCaptioner, FakeEstimator, FakeJudge, obj, photos
from voxprint_forge.lora import prepare as prepare_module
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.gate import AgeReading, Gate
from voxprint_forge.lora.prepare import prepare
from voxprint_forge.lora.project import Project


class Residency:
    """Which fake models are loaded right now, and every overlap seen."""

    def __init__(self) -> None:
        self.loaded: set[str] = set()
        self.log: list[tuple[str, str]] = []

    def load(self, name: str) -> None:
        self.loaded.add(name)
        self.log.append(("load", name))

    def unload(self, name: str) -> None:
        self.loaded.discard(name)
        self.log.append(("unload", name))


class TrackedJudge(FakeJudge):
    def __init__(self, live: Residency, verdict=ADULT_FACE) -> None:
        super().__init__(verdict)
        self.live = live
        live.load("judge")

    def judge(self, image):
        assert "captioner" not in self.live.loaded
        return super().judge(image)

    def unload(self):
        super().unload()
        self.live.unload("judge")


class TrackedEstimator(FakeEstimator):
    def __init__(self, live: Residency) -> None:
        super().__init__(AgeReading(1, (34.0,)))
        self.live = live
        live.load("estimator")

    def estimate(self, image):
        assert "captioner" not in self.live.loaded
        return super().estimate(image)

    def unload(self):
        self.live.unload("estimator")


class TrackedCaptioner(FakeCaptioner):
    def __init__(self, live: Residency) -> None:
        super().__init__()
        self.live = live
        live.load("captioner")

    def caption(self, image):
        assert not self.live.loaded & {"judge", "estimator"}
        return super().caption(image)

    def unload(self):
        super().unload()
        self.live.unload("captioner")


def _person():
    from voxprint_forge.lora.consent import check_subject

    return check_subject("person", name="Demo", consent_obtained=True, subject_adult=True)


def _load(live: Residency):
    def load(choice):
        # The gate models must be gone before the captioner loads.
        assert not live.loaded, live.loaded
        return TrackedCaptioner(live)

    return load


def test_judge_and_captioner_are_never_loaded_together(tmp_path) -> None:
    live = Residency()
    gate = Gate(TrackedJudge(live), TrackedEstimator(live))
    summary = prepare(
        Project(tmp_path / "proj"),
        photos(tmp_path / "in", 10),
        _person(),
        gate=gate,
        load_captioner=_load(live),
        progress=Progress(),
    )
    assert summary["prepared"] == 10
    assert live.loaded == set()
    order = [entry for entry in live.log if entry[0] == "load" or entry[1] == "captioner"]
    assert order == [("load", "judge"), ("load", "estimator"), ("load", "captioner"), ("unload", "captioner")]
    unload_gate = max(live.log.index(("unload", "judge")), live.log.index(("unload", "estimator")))
    assert unload_gate < live.log.index(("load", "captioner"))


def test_gate_models_unload_when_the_safety_check_is_interrupted(tmp_path) -> None:
    live = Residency()

    class Interrupted(TrackedJudge):
        def judge(self, image):
            raise KeyboardInterrupt

    loaded = []
    with pytest.raises(KeyboardInterrupt):
        prepare(
            Project(tmp_path / "proj"),
            photos(tmp_path / "in", 10),
            _person(),
            gate=Gate(Interrupted(live), TrackedEstimator(live)),
            load_captioner=lambda choice: loaded.append(choice),
            progress=Progress(),
        )
    assert live.loaded == set() and loaded == []


def test_an_unload_error_after_an_interruption_does_not_hide_it(tmp_path) -> None:
    live = Residency()

    class Stuck(TrackedEstimator):
        def unload(self):
            raise RuntimeError("stuck")

    class Interrupted(TrackedJudge):
        def judge(self, image):
            raise KeyboardInterrupt

    judge = Interrupted(live)
    with pytest.raises(KeyboardInterrupt):
        prepare(
            Project(tmp_path / "proj"),
            photos(tmp_path / "in", 10),
            _person(),
            gate=Gate(judge, Stuck(live)),
            load_captioner=lambda choice: None,
            progress=Progress(),
        )
    assert judge.unloaded


def test_decoded_photos_are_not_all_held_at_once(tmp_path, monkeypatch) -> None:
    live: list[weakref.ref] = []
    most = 0
    real = prepare_module.open_photo

    def tracked(path):
        nonlocal most
        most = max(most, sum(1 for ref in live if ref() is not None))
        image = real(path)
        if image is not None:
            live.append(weakref.ref(image))
        return image

    monkeypatch.setattr(prepare_module, "open_photo", tracked)
    summary = prepare(
        Project(tmp_path / "proj"),
        photos(tmp_path / "in", 12),
        obj(),
        gate=Gate(FakeJudge(), FakeEstimator()),
        load_captioner=lambda choice: FakeCaptioner(),
        progress=Progress(),
    )
    assert summary["prepared"] == 12
    # Filter, gate and crop each decode every photo (the captioner gets the crop): 3 x 12, one alive at a time.
    assert len(live) == 36
    assert most <= 1


def test_ids_names_and_trace_match_the_files(tmp_path) -> None:
    files = photos(tmp_path / "in", 10)
    # A byte-identical copy is skipped; its name is the one the trace and drops use, as before.
    copy = tmp_path / "in" / "zz-copy.png"
    copy.write_bytes(files[0].read_bytes())
    prepare(
        Project(tmp_path / "proj"),
        [*files, copy],
        obj(),
        gate=Gate(FakeJudge(), FakeEstimator()),
        load_captioner=lambda choice: FakeCaptioner(),
        progress=Progress(),
    )
    rows = [json.loads(line) for line in (tmp_path / "proj" / "gate_trace.jsonl").read_text().splitlines()]
    from voxprint_forge.lora.photos import photo_id

    assert [row["id"] for row in rows] == [photo_id(path) for path in files]
    assert [row["file"] for row in rows] == ["zz-copy.png", *(path.name for path in files[1:])]
