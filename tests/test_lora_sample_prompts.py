"""Review-sample prompts follow the prepared captions' attire."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.lora_fakes import make_project
from voxprint_forge.lora.captioner import text_marks_unclothed
from voxprint_forge.lora.project import Item
from voxprint_forge.lora.review import load_state, passes_output_check
from voxprint_forge.lora.sample_prompts import (
    ATTIRE_MODES,
    FRAMINGS,
    build_sample_prompts,
    marks_unclothed,
    read_attire,
    resolve_attire,
    store_attire,
)


def _caps(*flags: bool | None) -> list[dict[str, object]]:
    rows = []
    for flag in flags:
        row: dict[str, object] = {"long": "a person standing", "short": "standing"}
        if isinstance(flag, bool):
            row["unclothed"] = flag
        rows.append(row)
    return rows


def test_auto_follows_any_unclothed_caption() -> None:
    assert resolve_attire("auto", _caps(False, False)) == "as_photos"
    assert resolve_attire("auto", _caps(False, True)) == "unclothed"
    assert resolve_attire("auto", []) == "as_photos"
    assert resolve_attire("as_photos", _caps(True)) == "as_photos"
    assert resolve_attire("clothed", _caps(True)) == "clothed"
    assert resolve_attire("unclothed", _caps(False)) == "unclothed"
    with pytest.raises(ValueError):
        resolve_attire("fancy", _caps(False))


def test_a_missing_flag_can_come_from_the_caption_text() -> None:
    assert marks_unclothed({"long": "a nude figure", "short": "figure"}) is True
    assert marks_unclothed({"long": "a coated figure", "short": "figure"}) is False
    assert marks_unclothed({"unclothed": False, "long": "a nude figure"}) is False
    assert marks_unclothed({"unclothed": True, "long": "a coated figure"}) is True
    assert text_marks_unclothed("fully nude.") is True
    assert text_marks_unclothed("denude") is False


def test_prompts_rotate_framings_from_the_seed_and_round() -> None:
    captions = _caps(False)
    first = build_sample_prompts("tcube", count=3, seed=0, round_no=1, mode="auto", captions=captions)
    again = build_sample_prompts("tcube", count=3, seed=0, round_no=1, mode="auto", captions=captions)
    assert first == again
    assert first == [
        "tcube, waist-up, wearing near-skin-tone underwear",
        "tcube, full body, wearing near-skin-tone underwear",
        "tcube, close-up portrait, wearing near-skin-tone underwear",
    ]
    shifted = build_sample_prompts("tcube", count=3, seed=2, round_no=1, mode="clothed", captions=_caps(True))
    assert shifted[0] == "tcube, close-up portrait, wearing near-skin-tone underwear"
    nxt = build_sample_prompts("tcube", count=4, seed=0, round_no=2, mode="auto", captions=captions)
    assert [part.split(", ")[1] for part in nxt] == [
        "full body", "close-up portrait", "waist-up", "full body",
    ]
    bare = build_sample_prompts("", count=1, seed=2, round_no=1, mode="unclothed", captions=captions)
    assert bare == ["close-up portrait, unclothed"]
    assert set(FRAMINGS) == {"close-up portrait", "waist-up", "full body"}


def test_as_photos_does_not_add_clothing_and_clothed_does() -> None:
    mixed = _caps(False, True)
    as_photos = build_sample_prompts("mira", count=2, seed=2, round_no=1, mode="as_photos", captions=mixed)
    forced = build_sample_prompts("mira", count=2, seed=2, round_no=1, mode="clothed", captions=mixed)
    assert all(prompt.endswith(", unclothed") for prompt in as_photos)
    assert all("underwear" not in prompt for prompt in as_photos)
    assert all(prompt.endswith(", wearing near-skin-tone underwear") for prompt in forced)
    auto = build_sample_prompts("mira", count=1, seed=2, round_no=1, mode="auto", captions=mixed)
    assert auto == ["mira, close-up portrait, unclothed"]


def test_sample_attire_is_stored_on_the_project(tmp_path: Path) -> None:
    project = make_project(tmp_path, count=10)
    assert read_attire(project.read()) == "auto"
    assert store_attire(project, "unclothed") == "unclothed"
    assert project.read()["sample_attire"] == "unclothed"
    assert read_attire(project.read()) == "unclothed"
    store_attire(project, "unclothed")
    assert ATTIRE_MODES == ("auto", "as_photos", "clothed", "unclothed")


def test_the_output_check_still_keeps_only_a_picture() -> None:
    assert passes_output_check(None) is False
    assert passes_output_check(object()) is True
    source = Path(passes_output_check.__code__.co_filename).read_text(encoding="utf-8")
    body = source.split("def passes_output_check", 1)[1].split("def ", 1)[0]
    assert "return image is not None" in body


def test_a_review_without_a_prompt_file_stores_sample_prompts(tmp_path: Path) -> None:
    from tests.test_lora_review import DrawingAdapter, _config
    from voxprint_forge.lora.events import Progress
    from voxprint_forge.lora.train import train

    project = make_project(tmp_path, count=10)
    items = project.items()
    project.write_items([
        Item(item.id, item.file, item.long, item.short, item.round, index == 0)
        for index, item in enumerate(items)
    ])
    DrawingAdapter.preview_steps = []
    train(
        project, _config(steps=2), DrawingAdapter(), Progress(),
        review_rounds=1, samples_per_round=3, first_round_samples=3, sample_attire="auto",
    )
    state = load_state(project)
    prompts = [str(prompt) for prompt in state["prompts"]]
    assert state["custom_prompts"] is False
    assert prompts
    assert all(prompt.startswith("tcube, ") and prompt.endswith(", unclothed") for prompt in prompts)
    assert project.read()["sample_attire"] == "auto"
    kept = [sample for sample in state["samples"] if isinstance(sample, dict)]
    assert kept
    assert all(str(sample["prompt"]).endswith(", unclothed") for sample in kept)


def test_an_explicit_prompt_list_is_kept(tmp_path: Path) -> None:
    from tests.test_lora_review import DrawingAdapter, _config
    from voxprint_forge.lora.events import Progress
    from voxprint_forge.lora.train import train

    project = make_project(tmp_path, count=10)
    train(
        project, _config(steps=2), DrawingAdapter(), Progress(),
        review_rounds=1, samples_per_round=1, preview_prompts=["a cube"],
    )
    state = load_state(project)
    assert state["prompts"] == ["a cube"]
    assert state["custom_prompts"] is True


def test_the_cli_flag_parses_and_the_next_round_uses_the_saved_mode(tmp_path: Path) -> None:
    from tests.test_lora_review import DrawingAdapter, _config
    from voxprint_forge.cli import build_parser
    from voxprint_forge.lora.events import Progress
    from voxprint_forge.lora.train import train

    parsed = build_parser().parse_args(
        ["lora", "train", "--project", str(tmp_path), "--sample-attire", "clothed"],
    )
    assert parsed.sample_attire == "clothed"
    project = make_project(tmp_path / "proj", count=10)
    project.write_items([
        Item(item.id, item.file, item.long, "nude", item.round, None) for item in project.items()
    ])
    train(
        project, _config(steps=3), DrawingAdapter(), Progress(),
        review_rounds=2, samples_per_round=2, first_round_samples=2,
    )
    first = load_state(project)
    assert all(str(prompt).endswith(", unclothed") for prompt in first["prompts"])
    store_attire(project, "clothed")
    before = sorted(path.name for path in (project.root / "review" / "round-0001").glob("*.png"))
    second = train(project, _config(steps=3), DrawingAdapter(), Progress(), review_continue=True)
    assert second["state"] == "review"
    state = load_state(project)
    assert all(str(prompt).endswith(", wearing near-skin-tone underwear") for prompt in state["prompts"])
    assert sorted(path.name for path in (project.root / "review" / "round-0001").glob("*.png")) == before
    assert json.loads((project.root / "project.json").read_text(encoding="utf-8"))["sample_attire"] == "clothed"
