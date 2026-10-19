"""The judge loads only its own folder. A stock folder does not open the gate."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from transformers import AutoProcessor

from voxprint_forge.app.settings import CAPTION_MODELS
from voxprint_forge.lora import backends, captioner
from voxprint_forge.lora.consent import check_subject
from voxprint_forge.lora.errors import LoraBlocked
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.prepare import prepare
from voxprint_forge.lora.project import Project

JUDGE = "huihui-qwen3-vl-4b-instruct-abliterated"
STOCK = ("qwen3-vl-4b-instruct", "qwen3-vl-4b")


@pytest.fixture
def models(tmp_path, monkeypatch):
    monkeypatch.setattr(backends, "model_dir", lambda mdl_id: tmp_path / "model-forge" / mdl_id)
    loaded: list[Path] = []

    class FakeJudge:
        def __init__(self, folder: Path, device: str = "cuda") -> None:
            loaded.append(folder)

    monkeypatch.setattr(backends, "VlmJudge", FakeJudge)
    return tmp_path / "model-forge", loaded


def test_installed_folder_loads(models) -> None:
    root, loaded = models
    (root / JUDGE).mkdir(parents=True)
    assert backends.installed_dir(backends.JUDGE_ID) == root / JUDGE
    gate = backends.load_gate("cpu")
    assert loaded == [root / JUDGE]
    assert gate.judge is not None


def test_stock_folder_does_not_satisfy_the_judge(models) -> None:
    root, loaded = models
    for name in STOCK:
        (root / name).mkdir(parents=True)
    assert backends.installed_dir(backends.JUDGE_ID) is None
    gate = backends.load_gate("cpu")
    assert loaded == [] and gate.judge is None and not gate.available


def test_missing_folder_blocks(models) -> None:
    _root, loaded = models
    assert backends.installed_dir(backends.JUDGE_ID) is None
    gate = backends.load_gate("cpu")
    assert loaded == [] and not gate.available


def test_broken_folder_blocks(models, monkeypatch) -> None:
    root, _loaded = models
    (root / JUDGE).mkdir(parents=True)

    def boom(folder: Path, device: str = "cuda") -> None:
        raise OSError("bad weights")

    monkeypatch.setattr(backends, "VlmJudge", boom)
    assert not backends.load_gate("cpu").available


def test_missing_judge_folder_refuses_prepare(models, tmp_path) -> None:
    root, _loaded = models
    for name in STOCK:
        (root / name).mkdir(parents=True)
    gate = backends.load_gate("cpu")
    assert gate.judge is None and not gate.available
    photos = tmp_path / "in"
    photos.mkdir()
    files = []
    for index in range(10):
        path = photos / f"{index}.png"
        Image.new("RGB", (64, 64), (index, index, index)).save(path)
        files.append(path)
    with pytest.raises(LoraBlocked) as info:
        prepare(
            Project(tmp_path / "proj"),
            files,
            check_subject("object", name="Mug", rights_confirmed=True),
            gate=gate,
            load_captioner=lambda choice: None,
            progress=Progress(),
        )
    assert info.value.code == "lora.err.gate_missing"
    assert not (tmp_path / "proj" / "prepared").exists()


def test_florence_fallback_is_selectable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(backends, "model_dir", lambda mdl_id: tmp_path / mdl_id)
    (tmp_path / captioner.FALLBACK).mkdir()
    for name in STOCK:
        (tmp_path / name).mkdir()
    loaded: list[Path] = []

    class FakeFlorence:
        name = captioner.FALLBACK

        def __init__(self, folder: Path, device: str = "cuda") -> None:
            loaded.append(folder)

    monkeypatch.setitem(captioner._LOADERS, captioner.FALLBACK, FakeFlorence)
    assert CAPTION_MODELS == (captioner.DEFAULT, captioner.FALLBACK)
    chosen = captioner.load_captioner(captioner.FALLBACK)
    assert chosen is not None and chosen.name == captioner.FALLBACK
    assert loaded == [tmp_path / captioner.FALLBACK]
    fallback = captioner.load_captioner(captioner.DEFAULT)
    assert fallback is not None and fallback.name == captioner.FALLBACK


def test_processor_loads_chat_template_jinja(tmp_path) -> None:
    folder = tmp_path / "processor"
    folder.mkdir()
    (folder / "vocab.json").write_text('{"<|endoftext|>": 0, "hello": 1}', encoding="utf-8")
    (folder / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (folder / "tokenizer_config.json").write_text(
        '{"tokenizer_class": "Qwen2Tokenizer", "model_max_length": 32, '
        '"unk_token": "<|endoftext|>", "eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>"}',
        encoding="utf-8",
    )
    template = "template {{ messages }}"
    (folder / "chat_template.jinja").write_text(template, encoding="utf-8")
    processor = AutoProcessor.from_pretrained(str(folder))
    assert processor.chat_template == template
    assert backends.load_processor(folder).chat_template == template


def test_stock_folder_is_not_a_captioner(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(backends, "model_dir", lambda mdl_id: tmp_path / mdl_id)
    for name in STOCK:
        (tmp_path / name).mkdir(parents=True)
    assert captioner.load_captioner(captioner.DEFAULT) is None
    assert captioner.FALLBACK == "florence2-large"
    assert captioner.LICENCES[captioner.DEFAULT] == "Apache-2.0"
