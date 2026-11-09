"""tools/sign_model_index.py refuses to sign while any TODO remains."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "sign_model_index.py"


def _load():
    spec = importlib.util.spec_from_file_location("sign_model_index", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _handover(tmp_path: Path, **turbo_fields: object) -> Path:
    row = {
        "id": "z-image-turbo",
        "repo": "Mitroshenkov87/voxprint-mirror-z-image-turbo",
        "revision": "TODO-pending-upload",
        "sha256": "ab" * 32,
        "default": True,
        "optional": False,
        "note": "TODO-pending-upload: waiting for the descriptor.",
    }
    row.update(turbo_fields)
    path = tmp_path / "handover.json"
    path.write_text(
        json.dumps(
            {
                "head": "15edeab0f7a874c27d2b38ef6ecb69106ff94df8",
                "serial": 9,
                "entries": [
                    {
                        "id": "z-image",
                        "repo": "datasets/example/index",
                        "revision": "04cc4abb7c5069926f75c9bfde9ef43d49423021",
                        "sha256": "cd" * 32,
                        "default": False,
                        "optional": True,
                    },
                    row,
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_refuses_when_turbo_revision_is_not_a_commit(tmp_path: Path) -> None:
    module = _load()
    handover = _handover(tmp_path)
    assert module.main(["--handover", str(handover), "--turbo-revision", "TODO-pending-upload", "--dry-run"]) == 2


def test_refuses_when_a_todo_remains_after_substitution(tmp_path: Path) -> None:
    module = _load()
    handover = _handover(tmp_path, note="TODO still waiting on another field")
    # Apply a real-looking revision; the note still contains TODO.
    # First put TODO on a *different* entry so note cleanup on turbo cannot clear it.
    data = json.loads(handover.read_text(encoding="utf-8"))
    data["entries"][0]["note"] = "TODO-remove-this"
    handover.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    assert (
        module.main(
            [
                "--handover",
                str(handover),
                "--turbo-revision",
                "a" * 40,
                "--dry-run",
            ]
        )
        == 2
    )


def test_todo_paths_finds_nested_markers() -> None:
    module = _load()
    assert module._todo_paths({"entries": [{"revision": "TODO-pending-upload"}]}) == ["$.entries[0].revision"]
    assert module._todo_paths({"revision": "f332072aa78be7aecdf3ee76d5c247082da564a6"}) == []

def test_retired_ids_include_handover_and_compressed() -> None:
    module = _load()
    retired = module._retired_ids({"retired_rows": [{"id": "mirror-z-image-turbo"}]})
    assert "mirror-z-image-turbo" in retired
    assert "moss-image-compressed" in retired
    assert "mirror-z-image-turbo-controlnet-union-2.1" not in retired


def test_build_index_drops_retired_rows() -> None:
    module = _load()
    handover = {
        "head": "15edeab0f7a874c27d2b38ef6ecb69106ff94df8",
        "serial": 9,
        "entries": [],
        "retired_rows": [{"id": "mirror-z-image-turbo"}],
    }
    document = module.build_index(
        handover, serial=10, head=handover["head"], generated_at="2026-11-10T00:00:00Z"
    )
    ids = [row["id"] for row in document["models"]]
    assert "mirror-z-image-turbo" not in ids
    assert "moss-image-compressed" not in ids
    assert "z-image-turbo" in ids
    assert "mirror-z-image-turbo-controlnet-union-2.1" in ids
    assert document["serial"] == 10

