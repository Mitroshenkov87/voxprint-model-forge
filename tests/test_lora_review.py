"""Review rounds: state transitions, include/exclude sets, and a tiny CPU stage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from tests.lora_fakes import TinyAdapter, gradient, make_project
from voxprint_forge.lora.bases import BaseInfo
from voxprint_forge.lora.errors import LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.presets import TrainConfig, build_config
from voxprint_forge.lora.project import Item, Project
from voxprint_forge.lora.review import (
    apply_file,
    build_sets,
    can_continue,
    check_round_options,
    counts,
    load_state,
    plan_samples,
    sample_steps_of,
    samples_for_round,
    save_state,
    set_verdicts,
    state_path,
    step_plan,
    steps_for_stage,
    transition,
)
from voxprint_forge.lora.train import train


def _idle() -> dict[str, object]:
    return transition(
        {},
        "start",
        rounds=2,
        samples_per_round=3,
        prompts=["a cube", "a cube on a table"],
        seed=7,
        base="draw-test",
        config={"steps": 1},
    )


def _paused(state: dict[str, object] | None = None) -> dict[str, object]:
    current = _idle() if state is None else state
    samples = plan_samples(current)
    return transition(current, "pause", checkpoint="checkpoints/step-000001.safetensors", samples=samples)


def test_rounds_move_idle_training_review_and_finished() -> None:
    training = _idle()
    assert training["phase"] == "training"
    assert training["index"] == 0
    review = _paused(training)
    assert review["phase"] == "review"
    assert review["index"] == 1
    assert can_continue(review)
    assert [sample["id"] for sample in review["samples"]] == ["r1-s00", "r1-s01", "r1-s02"]
    assert [sample["seed"] for sample in review["samples"]] == [7, 8, 9]
    marked = transition(review, "mark", id="r1-s00", verdict="like")
    marked = transition(marked, "mark", id="r1-s01", verdict="dislike")
    assert counts(marked) == {"liked": 1, "disliked": 1, "unmarked": 1, "round": 1, "rounds": 2}
    again = transition(marked, "continue")
    assert again["phase"] == "training"
    assert again["samples"] == []
    assert again["history"][0]["samples"][0]["verdict"] == "like"
    second = transition(again, "pause", checkpoint="checkpoints/step-000001.safetensors", samples=plan_samples(again))
    assert second["index"] == 2
    assert not can_continue(second)
    with pytest.raises(LoraUsageError) as blocked:
        transition(second, "continue")
    assert blocked.value.code == "lora.err.review_finished"
    finished = transition(second, "finish")
    assert finished["phase"] == "finished"
    with pytest.raises(LoraUsageError) as pending:
        transition(review, "start", rounds=1, samples_per_round=1)
    assert pending.value.code == "lora.err.review_pending"


def test_marks_reject_conflicts_unknown_ids_and_the_wrong_phase() -> None:
    review = _paused()
    with pytest.raises(LoraUsageError) as conflict:
        set_verdicts(review, like=["r1-s00"], dislike=["r1-s00"])
    assert conflict.value.code == "lora.err.review_conflict"
    with pytest.raises(LoraUsageError) as missing:
        transition(review, "mark", id="r9-s00", verdict="like")
    assert missing.value.code == "lora.err.review_sample"
    with pytest.raises(LoraUsageError) as phase:
        transition(_idle(), "finish")
    assert phase.value.code == "lora.err.review_phase"
    with pytest.raises(LoraUsageError) as rounds:
        transition({}, "start", rounds=0, samples_per_round=1)
    assert rounds.value.code == "lora.err.review_rounds"


def test_a_paused_review_reloads_from_the_project_folder(tmp_path: Path) -> None:
    project = Project(tmp_path)
    review = _paused()
    review = transition(review, "mark", id="r1-s02", verdict="dislike")
    save_state(project, review)
    loaded = load_state(project)
    assert loaded["phase"] == "review"
    assert counts(loaded)["disliked"] == 1
    assert state_path(project).is_file()


def test_include_is_originals_plus_likes_and_dislikes_are_negatives_only_when_supported() -> None:
    originals = (
        Item("p00", "prepared/p00.png", "long", "short", 0),
        Item("p01", "prepared/p01.png", "long", "short", 0),
    )
    review = transition(_paused(), "mark", id="r1-s00", verdict="like")
    review = transition(review, "mark", id="r1-s01", verdict="dislike")
    plain = build_sets(originals, review, supports_negatives=False)
    assert [item.id for item in plain.include] == ["p00", "p01", "r1-s00"]
    assert plain.negatives == ()
    assert plain.excluded == ("r1-s01",)
    backed = build_sets(originals, review, supports_negatives=True)
    assert [item.id for item in backed.negatives] == ["r1-s01"]
    assert "r1-s02" not in {item.id for item in backed.include}
    assert "r1-s02" not in {item.id for item in backed.negatives}
    kept = transition(review, "continue")
    later = build_sets(originals, kept, supports_negatives=False)
    assert [item.id for item in later.include] == ["p00", "p01", "r1-s00"]


def test_a_verdict_file_marks_the_open_round(tmp_path: Path) -> None:
    review = _paused()
    path = tmp_path / "marks.txt"
    path.write_text("like r1-s00\n# skip\ndislike r1-s01\n", encoding="utf-8")
    marked = apply_file(review, path)
    assert counts(marked)["liked"] == 1
    blob = tmp_path / "marks.json"
    blob.write_text(json.dumps({"like": ["r1-s01"], "unmarked": ["r1-s00"]}), encoding="utf-8")
    flipped = apply_file(marked, blob)
    assert counts(flipped) == {"liked": 1, "disliked": 0, "unmarked": 2, "round": 1, "rounds": 2}


class DrawingAdapter(TinyAdapter):
    """Tiny adapter that returns a gradient for each review sample."""

    info = BaseInfo("draw-test", "example/draw", "0" * 40, "Apache-2.0", "Draw")
    encoded: list[int] = []
    preview_steps: list[int | None] = []

    def encode(self, images, captions, config):
        type(self).encoded.append(len(images))
        return super().encode(images, captions, config)

    def preview(self, model, prompt, seed, config, steps=None):
        type(self).preview_steps.append(steps)
        if seed % 5 == 4:
            return None
        return gradient(seed, config.resolution)


class NegativeAdapter(DrawingAdapter):
    """Same drawings, and a negative loss the review can call."""

    supports_negatives = True
    negative_calls = 0

    def negative_loss(self, model, batch, caption_keys, rng):
        type(self).negative_calls += 1
        latents = torch.stack([sample.latents for sample in batch])
        return model(latents).sum() * 0


def _config(**overrides: object) -> TrainConfig:
    values = {"vram": 24, "steps": 1, "resolution": 64, "save_every": 1, "batch": 1, "previews": False}
    values.update(overrides)
    vram = int(values.pop("vram"))
    return build_config("quick", "balanced", vram, **values)


def test_a_stage_pauses_for_review_and_the_next_pass_uses_likes(tmp_path: Path) -> None:
    DrawingAdapter.encoded = []
    DrawingAdapter.preview_steps = []
    NegativeAdapter.negative_calls = 0
    project = make_project(tmp_path, count=10)
    first = train(
        project, _config(steps=101), DrawingAdapter(), Progress(), review_rounds=2, samples_per_round=3,
        first_round_steps=100, preview_prompts=["a cube"],
    )
    assert first["state"] == "review"
    assert first["steps"] == 100
    state = load_state(project)
    assert state["first_round_steps"] == 100
    assert state["first_round_samples"] == 6
    assert state["sample_steps"] == 28
    assert state["total_steps"] == 101
    assert DrawingAdapter.preview_steps
    assert set(DrawingAdapter.preview_steps) == {28}
    assert state["phase"] == "review"
    assert state["index"] == 1
    kept = [sample for sample in state["samples"] if isinstance(sample, dict)]
    assert kept
    assert all((project.root / str(sample["file"])).is_file() for sample in kept)
    with Image.open(project.root / str(kept[0]["file"])) as image:
        assert image.getexif() == {}
        assert not image.info.get("parameters")
    assert DrawingAdapter.encoded == [10]
    liked = str(kept[0]["id"])
    disliked = str(kept[1]["id"]) if len(kept) > 1 else liked
    state = transition(state, "mark", id=liked, verdict="like")
    if disliked != liked:
        state = transition(state, "mark", id=disliked, verdict="dislike")
    save_state(project, state)
    second = train(project, _config(steps=101), NegativeAdapter(), Progress(), review_continue=True)
    assert second["state"] == "review"
    assert DrawingAdapter.encoded[1] == 10 + (2 if disliked != liked else 1)
    if disliked != liked:
        assert NegativeAdapter.negative_calls > 0
    plain = make_project(tmp_path / "plain", count=10)
    done = train(plain, _config(), DrawingAdapter(), Progress())
    assert done["state"] == "done"
    assert not state_path(plain).is_file()


def _envelope(text: str) -> dict[str, object]:
    start = text.rindex("\n{") if "\n{" in text else text.index("{")
    return json.loads(text[start:])


def test_cli_review_continue_flags_and_finish(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from voxprint_forge.cli import build_parser, main

    parsed = build_parser().parse_args(["lora", "train", "--project", str(tmp_path), "--review-rounds", "2"])
    assert parsed.review_rounds == 2
    assert parsed.samples_per_round == 8
    assert parsed.first_round_steps == 200
    assert parsed.first_round_samples == 6
    assert parsed.sample_steps == 28
    continued = build_parser().parse_args(["lora", "continue", str(tmp_path), "--no-more-reviews"])
    assert continued.no_more_reviews is True
    straight = build_parser().parse_args(["lora", "train", "--project", str(tmp_path)])
    assert straight.review_rounds == 0
    project = make_project(tmp_path / "proj", count=10)
    review = _paused()
    save_state(project, review)
    capsys.readouterr()
    assert main(["--json", "lora", "review", str(project.root)]) == 0
    listed = _envelope(capsys.readouterr().out)["result"]
    assert listed["phase"] == "review"
    assert len(listed["samples"]) == 3
    assert len(listed["originals"]) == 10
    marks = tmp_path / "marks.txt"
    marks.write_text("like r1-s00\ndislike r1-s01\n", encoding="utf-8")
    assert main(["--json", "lora", "review", str(project.root), "--from-file", str(marks), "--like", "r1-s02"]) == 0
    marked = _envelope(capsys.readouterr().out)["result"]
    assert marked["liked"] == 2 and marked["disliked"] == 1
    progress = project.root / "train.json"
    progress.write_text(
        json.dumps({"step": 1, "total": 1, "loss": 0.2, "elapsed_seconds": 3, "eta_seconds": 0, "stage": "review"}),
        encoding="utf-8",
    )
    assert main(["--json", "lora", "status", str(project.root)]) == 0
    status = _envelope(capsys.readouterr().out)["result"]
    assert status["step"] == 1
    assert status["review"]["phase"] == "review"
    assert main(["--json", "lora", "finish", str(project.root)]) == 0
    finished = _envelope(capsys.readouterr().out)["result"]
    assert finished["state"] == "done"
    assert load_state(project)["phase"] == "finished"


def test_step_plan_splits_the_remaining_steps_and_train_full_takes_the_rest() -> None:
    def opened(rounds: int, total: int = 1500, first: int = 200) -> dict[str, object]:
        return transition(
            {}, "start", rounds=rounds, samples_per_round=8, first_round_steps=first,
            first_round_samples=6, sample_steps=28, total_steps=total, prompts=["a cube"],
            seed=1, base="draw-test", config={"steps": total},
        )

    three = opened(3)
    assert step_plan(three) == (200, 650, 650)
    assert sum(step_plan(three)) == 1500
    four = opened(4)
    assert step_plan(four) == (200, 433, 433, 434)
    assert sum(step_plan(four)) == 1500
    assert step_plan(opened(1)) == (200,)
    assert len(plan_samples(three)) == 6
    assert samples_for_round(three, 1) == 6
    paused = transition(three, "pause", checkpoint="checkpoints/step.safetensors", samples=plan_samples(three))
    again = transition(paused, "continue")
    assert samples_for_round(again, 2) == 8
    assert len(plan_samples(again)) == 8
    assert sample_steps_of(three) == 28
    full = transition(paused, "train_full")
    assert full["phase"] == "training"
    assert full["no_more_reviews"] is True
    assert full["samples"] == []
    assert full["history"]
    assert steps_for_stage(full) == 1300
    finished = transition(full, "finish")
    assert finished["phase"] == "finished"
    with pytest.raises(LoraUsageError) as blocked:
        transition(three, "finish")
    assert blocked.value.code == "lora.err.review_phase"
    last = transition(again, "pause", checkpoint="c", samples=plan_samples(again))
    last = transition(last, "continue")
    last = transition(last, "pause", checkpoint="c", samples=[])
    assert not can_continue(last)
    with pytest.raises(LoraUsageError):
        transition(last, "continue")
    done = transition(last, "train_full")
    assert done["no_more_reviews"] is True
    assert steps_for_stage(done) == 0
    check_round_options(first_round_steps=100, first_round_samples=6, samples_per_round=8, sample_steps=28)
    for bad in (50, 600):
        with pytest.raises(LoraUsageError) as limited:
            check_round_options(first_round_steps=bad, first_round_samples=6, samples_per_round=8, sample_steps=28)
        assert limited.value.code == "lora.err.first_round_steps"


def test_train_full_with_nothing_left_finishes(tmp_path: Path) -> None:
    DrawingAdapter.encoded = []
    project = make_project(tmp_path, count=10)
    first = train(
        project, _config(steps=2), DrawingAdapter(), Progress(), review_rounds=2, samples_per_round=1,
        preview_prompts=["a cube"],
    )
    assert first["state"] == "review"
    state = load_state(project)
    liked = next(sample["id"] for sample in state["samples"] if isinstance(sample, dict))
    save_state(project, transition(state, "mark", id=str(liked), verdict="like"))
    done = train(project, _config(steps=2), DrawingAdapter(), Progress(), review_continue=True, no_more_reviews=True)
    assert done["state"] == "done"
    assert load_state(project)["phase"] == "finished"


def test_status_prints_the_first_pictures_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from voxprint_forge.cli import main

    project = make_project(tmp_path, count=10)
    (project.root / "train.json").write_text(
        json.dumps({
            "step": 40, "total": 200, "loss": 0.2, "elapsed_seconds": 90, "eta_seconds": 400,
            "stage": "step", "round": 1, "first_pictures_seconds": 480, "seconds_per_sample": 18,
            "sample_seconds_estimated": False,
        }),
        encoding="utf-8",
    )
    capsys.readouterr()
    assert main(["--json", "lora", "status", str(project.root)]) == 0
    status = _envelope(capsys.readouterr().out)["result"]
    assert status["first_pictures"] == "First pictures in about 8 min (18 s per picture)"


def test_train_refuses_a_first_round_outside_100_to_500(tmp_path: Path) -> None:
    project = make_project(tmp_path, count=10)
    for bad in (50, 600):
        with pytest.raises(LoraUsageError) as limited:
            train(
                project, _config(), DrawingAdapter(), Progress(), review_rounds=2, first_round_steps=bad,
            )
        assert limited.value.code == "lora.err.first_round_steps"


def test_train_refuses_to_start_while_a_review_is_waiting(tmp_path: Path) -> None:
    project = make_project(tmp_path, count=10)
    train(project, _config(), DrawingAdapter(), Progress(), review_rounds=1, samples_per_round=1)
    with pytest.raises(LoraUsageError) as pending:
        train(project, _config(), DrawingAdapter(), Progress(), review_rounds=1, samples_per_round=1)
    assert pending.value.code == "lora.err.review_pending"
