"""LoRA engine: strings, presets, events, train/extend/pause/resume on a tiny CPU model."""

from __future__ import annotations

import json
import random
import re
import threading

import pytest
import torch
from safetensors.torch import load_file, save_file

from tests.lora_fakes import TinyAdapter, make_project
from voxprint_forge.lora import bases
from voxprint_forge.lora.errors import LoraBlocked, LoraStageError, LoraUsageError
from voxprint_forge.lora.events import Progress, sentence
from voxprint_forge.lora.i18n import catalog, text
from voxprint_forge.lora.presets import build_config
from voxprint_forge.lora.project import Item
from voxprint_forge.lora.train import estimate, select_items, train

PLACEHOLDER = re.compile(r"\{(\w+)\}")


def tiny(**overrides):
    values = {"vram": 24, "steps": 6, "resolution": 64, "save_every": 3, "batch": 2, "previews": False}
    values.update(overrides)
    vram = values.pop("vram")
    return build_config("quick", "balanced", vram, **values)


def test_catalogs_have_the_same_keys_and_placeholders() -> None:
    en, ru = catalog("en"), catalog("ru")
    assert set(en) == set(ru)
    for key in en:
        assert set(PLACEHOLDER.findall(en[key])) == set(PLACEHOLDER.findall(ru[key])), key
        assert en[key] and ru[key], key


def test_catalogs_cover_the_plan_tables() -> None:
    en = catalog("en")
    for step in ("photos", "subject", "prepare", "train", "result"):
        for part in ("title", "desc", "help"):
            assert f"lora.step.{step}.{part}" in en
    for key in (
        "lora.err.blocked",
        "lora.err.gate_missing",
        "lora.drop.safety",
        "lora.progress.safety",
        "lora.warn.count",
    ):
        assert key in en
    assert text("lora.err.too_few", "ru", need=3) == "Добавьте ещё хотя бы 3 фото, чтобы продолжить."


def test_presets_follow_memory_tier_and_likeness() -> None:
    big = build_config("balanced", "balanced", 24)
    small = build_config("balanced", "close", 16)
    assert (big.batch, big.grad_accum, big.grad_checkpointing, big.base_fp8) == (2, 1, False, False)
    assert (small.batch, small.grad_accum, small.grad_checkpointing, small.base_fp8) == (1, 2, True, True)
    assert small.steps == 1950 and small.caption_dropout == 0.05
    assert build_config("best", "flexible", 24).steps == 2100
    with pytest.raises(ValueError):
        build_config("balanced", "balanced", 24, resolution=100)


def test_events_give_sentences_and_eta_after_warmup() -> None:
    now = [0.0]
    progress = Progress(clock=lambda: now[0])
    events = []
    for step in range(1, 6):
        now[0] += 2.0
        events.append(
            progress.emit(
                "train.step",
                "lora.progress.train",
                {"step": step, "steps": 10},
                "lora.reason.train",
                done=step,
                total=10,
            )
        )
    assert [e.eta_seconds for e in events[:2]] == [None, None]
    assert events[-1].eta_seconds == 10
    assert events[-1].percent == 50.0
    assert "About 1 min left" in sentence(events[-1], "en")
    assert sentence(events[0], "ru").startswith("Обучение: шаг 1 из 10 — модель изучает ваши фото — Оцениваю")


def test_train_writes_checkpoints_and_events(tmp_path) -> None:
    project = make_project(tmp_path)
    adapter = TinyAdapter()
    progress = Progress()
    result = train(project, tiny(), adapter, progress)
    assert result["state"] == "done" and result["step"] == 6
    assert (project.checkpoints / "step-000003.safetensors").is_file()
    assert (project.checkpoints / "step-000006.safetensors").is_file()
    assert adapter.released
    states = {e.state for e in progress.events}
    assert states == {"running", "done"}
    record = json.loads(project.file.read_text())["train"]
    assert record["base"] == "tiny-test" and record["checkpoint"] == "step-000006.safetensors"
    tensors = load_file(str(project.checkpoints / "step-000006.safetensors"))
    assert set(tensors) == {"to_q.lora_A.weight", "to_q.lora_B.weight", "to_v.lora_A.weight", "to_v.lora_B.weight"}
    assert tensors["to_q.lora_B.weight"].abs().sum() > 0


def test_pause_then_resume_matches_one_run(tmp_path) -> None:
    straight = make_project(tmp_path / "a")
    train(straight, tiny(), TinyAdapter(), Progress())
    paused = make_project(tmp_path / "b")
    stop = threading.Event()

    def sink(event):
        if event.params.get("step") == 2:
            stop.set()

    first = train(paused, tiny(), TinyAdapter(), Progress(sink), stop=stop)
    assert first["state"] == "paused" and first["step"] == 2
    second = train(paused, tiny(steps=99), TinyAdapter(), Progress(), resume=True)
    assert second["state"] == "done" and second["step"] == 6
    a = load_file(str(straight.checkpoints / "step-000006.safetensors"))
    b = load_file(str(paused.checkpoints / "step-000006.safetensors"))
    for key in a:
        assert torch.allclose(a[key], b[key], atol=1e-6), key


def test_resume_without_checkpoint_is_refused(tmp_path) -> None:
    with pytest.raises(LoraUsageError) as info:
        train(make_project(tmp_path), tiny(), TinyAdapter(), Progress(), resume=True)
    assert info.value.code == "lora.err.no_checkpoint"


def test_gate_not_passed_blocks_and_few_photos_refuse(tmp_path) -> None:
    with pytest.raises(LoraBlocked):
        train(make_project(tmp_path / "g", gate_passed=False), tiny(), TinyAdapter(), Progress())
    with pytest.raises(LoraUsageError) as info:
        train(make_project(tmp_path / "f", count=8), tiny(), TinyAdapter(), Progress())
    assert info.value.key == "lora.err.too_few" and info.value.params == {"need": 2}
    result = train(make_project(tmp_path / "e", count=6, mode="expert"), tiny(steps=1), TinyAdapter(), Progress())
    assert result["state"] == "done"


def test_extend_refuses_a_different_base(tmp_path) -> None:
    project = make_project(tmp_path / "p")
    other = tmp_path / "other.safetensors"
    save_file({"x": torch.zeros(1)}, str(other), metadata={"voxprint.lora.base": "z-image"})
    with pytest.raises(LoraUsageError) as info:
        train(project, tiny(), TinyAdapter(), Progress(), from_lora=other)
    assert info.value.code == "lora.err.base_mismatch"


def test_extend_continues_from_a_lora(tmp_path) -> None:
    first = make_project(tmp_path / "one")
    train(first, tiny(), TinyAdapter(), Progress())
    source = first.checkpoints / "step-000006.safetensors"
    tensors = load_file(str(source))
    lora = tmp_path / "model.safetensors"
    save_file(
        tensors,
        str(lora),
        metadata={"voxprint.lora.base": "tiny-test", "voxprint.lora.rank": "16", "voxprint.lora.alpha": "16"},
    )
    second = make_project(tmp_path / "two", count=12, rounds=[0] * 8 + [1] * 4)
    result = train(second, tiny(steps=2, save_every=1), TinyAdapter(), Progress(), from_lora=lora)
    assert result["state"] == "done"
    assert json.loads(second.file.read_text())["train"]["extended"] is True


def test_replay_keeps_a_share_of_old_photos() -> None:
    items = [Item(str(i), f"{i}.png", "", "", 0 if i < 10 else 1) for i in range(14)]
    chosen = select_items(items, 0.25, random.Random(0))
    assert sum(1 for i in chosen if i.round == 1) == 4
    assert sum(1 for i in chosen if i.round == 0) == 3


def test_divergence_fails_with_plain_error(tmp_path) -> None:
    adapter = TinyAdapter()
    adapter.nan_after = 4
    with pytest.raises(LoraStageError) as info:
        train(make_project(tmp_path), tiny(), adapter, Progress())
    assert info.value.code == "lora.err.diverged"


def test_no_gpu_is_reported(tmp_path, monkeypatch) -> None:
    adapter = TinyAdapter()
    adapter.requires_gpu = True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(LoraStageError) as info:
        train(make_project(tmp_path), tiny(), adapter, Progress())
    assert info.value.code == "lora.err.no_gpu"


def test_dry_run_estimates_time(tmp_path) -> None:
    plan = estimate(make_project(tmp_path), tiny(), TinyAdapter())
    assert plan["eta_seconds"] == 3 and plan["steps"] == 6 and plan["photos"] == 10


def test_builtin_bases_are_permissive_and_zimage_is_default() -> None:
    ids = {info.mdl_id: info for info in bases.available()}
    assert bases.DEFAULT_BASE == "z-image"
    assert ids["z-image"].permissive and ids["moss-image-base"].permissive
    with pytest.raises(LoraUsageError):
        bases.get_adapter("qwen-image-edit")
    with pytest.raises(LoraStageError):
        bases.get_adapter("moss-image-base").load(None, tiny(), "cpu")
    with pytest.raises(LoraUsageError):
        bases.get_adapter("z-image").load(None, tiny(train_text_encoder=True), "cpu")
