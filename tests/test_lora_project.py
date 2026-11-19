"""Project folder layout: fixed, documented paths and reading projects stored by older versions."""

from __future__ import annotations

import json
from pathlib import Path

from tests.lora_fakes import gradient, make_project
from voxprint_forge.lora import refset
from voxprint_forge.lora.project import Item, Project
from voxprint_forge.lora.review import load_state, save_state


def test_layout_folder_paths(tmp_path: Path) -> None:
    """Every folder property is the documented name directly under the root."""
    project = Project(tmp_path / "demo")
    assert project.prepared == project.root / "prepared"
    assert project.checkpoints == project.root / "checkpoints"
    assert project.previews == project.root / "previews"
    assert project.review == project.root / "review"
    assert project.eval == project.root / "eval"
    assert project.export == project.root / "export"
    assert project.release == project.root / "release"
    assert project.refset == project.root / "refset"


def test_layout_file_paths(tmp_path: Path) -> None:
    """Every file property is the documented name directly under the root."""
    project = Project(tmp_path / "demo")
    assert project.file == project.root / "project.json"
    assert project.captions == project.root / "captions.jsonl"
    assert project.drops == project.root / "drops.jsonl"
    assert project.gate_trace == project.root / "gate_trace.jsonl"
    assert project.pause == project.root / "PAUSE"
    assert project.train_state == project.root / "train.json"
    assert project.train_log == project.root / "train.log"
    assert project.vram_log == project.root / "vram.csv"


def test_resolve_joins_stored_relative_paths(tmp_path: Path) -> None:
    """Paths stored in project files are relative to the root (docs/LORA.md, "Project folder")."""
    project = Project(tmp_path)
    assert project.resolve("prepared/p00.png") == project.root / "prepared" / "p00.png"
    assert project.resolve(Path("review") / "round-0000" / "s00.png") == project.review / "round-0000" / "s00.png"
    assert project.resolve("checkpoints/step-000001.safetensors") == project.checkpoints / "step-000001.safetensors"


def test_resolve_passes_absolute_paths_through(tmp_path: Path) -> None:
    """A stored absolute path is used as is, so moved projects still resolve."""
    project = Project(tmp_path)
    outside = tmp_path.parent / "elsewhere" / "photo.png"
    assert project.resolve(outside) == outside
    assert project.resolve(str(outside)) == outside


def test_items_reads_project_from_older_version(tmp_path: Path) -> None:
    """An old captions.jsonl (same on-disk names) still opens: items resolve against the root."""
    project = make_project(tmp_path, 3, rounds=[0, 0, 1])
    for item in project.items():
        assert item.file.startswith("prepared/")
        assert project.resolve(item.file).is_file()
    assert [item.round for item in project.items()] == [0, 0, 1]


def test_items_accepts_stored_absolute_file(tmp_path: Path) -> None:
    """A captions.jsonl row with an absolute file path still lists the item."""
    project = Project(tmp_path)
    project.prepared.mkdir(parents=True)
    photo = tmp_path.parent / "outside.png"
    gradient(1).save(photo)
    project.write_items([Item("abs1", str(photo), "long", "short", 0)])
    found = project.items()
    assert [item.id for item in found] == ["abs1"]
    assert found[0].file == str(photo)


def test_items_skips_missing_prepared_photo(tmp_path: Path) -> None:
    """A row whose photo is gone is not listed; the rest still are."""
    project = make_project(tmp_path, 2)
    (project.prepared / "p01.png").unlink()
    assert [item.id for item in project.items()] == ["p00"]


def test_review_state_lives_in_review_dir(tmp_path: Path) -> None:
    """save_state/load_state round-trip through the fixed review folder."""
    project = Project(tmp_path)
    state = {"phase": "review", "index": 1, "rounds": 3}
    save_state(project, state)
    assert (project.review / "state.json").is_file()
    loaded = load_state(project)
    assert loaded["phase"] == "review" and loaded["index"] == 1


def test_refset_folder_uses_project_layout(tmp_path: Path) -> None:
    """refset paths come from the same central layout."""
    project = Project(tmp_path)
    assert refset.folder(project) == project.refset
    assert refset.folder(tmp_path) == project.refset
    assert refset.state_file(project) == project.refset / "state.json"
    assert refset.prompts_file(project) == project.refset / "prompts.json"


def test_root_files_stay_beside_folders(tmp_path: Path) -> None:
    """Training progress files live at the root next to the folders, as documented."""
    project = make_project(tmp_path, 1)
    for name in ("train.json", "train.log", "vram.csv"):
        (project.root / name).write_text("", encoding="utf-8")
    assert project.train_state.is_file() and project.train_log.is_file() and project.vram_log.is_file()
    json.loads(project.file.read_text(encoding="utf-8"))
