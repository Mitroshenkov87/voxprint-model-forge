"""The age-gate judge also loads from the pre-#47 folder name; with neither folder the gate stays closed."""

from __future__ import annotations

from pathlib import Path

import pytest

from voxprint_forge.lora import backends


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setattr(backends, "model_dir", lambda mdl_id: tmp_path / "model-forge" / mdl_id)
    loaded: list[Path] = []

    class FakeJudge:
        def __init__(self, folder: Path, device: str = "cuda") -> None:
            loaded.append(folder)

    monkeypatch.setattr(backends, "VlmJudge", FakeJudge)
    return tmp_path / "model-forge", loaded


def test_legacy_folder_is_found(models) -> None:
    root, loaded = models
    (root / "qwen3-vl-4b").mkdir(parents=True)
    assert backends.installed_dir(backends.JUDGE_ID) == root / "qwen3-vl-4b"
    gate = backends.load_gate("cpu")
    assert loaded == [root / "qwen3-vl-4b"]
    assert gate.judge is not None


def test_new_name_wins(models) -> None:
    root, loaded = models
    (root / "qwen3-vl-4b").mkdir(parents=True)
    (root / "qwen3-vl-4b-instruct").mkdir()
    backends.load_gate("cpu")
    assert loaded == [root / "qwen3-vl-4b-instruct"]


def test_neither_folder_blocks(models) -> None:
    root, loaded = models
    assert backends.installed_dir(backends.JUDGE_ID) is None
    gate = backends.load_gate("cpu")
    assert loaded == [] and not gate.available


def test_broken_legacy_folder_blocks(models, monkeypatch) -> None:
    root, _ = models
    (root / "qwen3-vl-4b").mkdir(parents=True)

    def boom(folder: Path, device: str = "cuda") -> None:
        raise OSError("bad weights")

    monkeypatch.setattr(backends, "VlmJudge", boom)
    assert not backends.load_gate("cpu").available
