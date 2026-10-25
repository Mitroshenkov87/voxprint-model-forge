"""MOSS text encoder quantization: option parsing, staged placement order, missing-library error."""

from __future__ import annotations

import argparse
import importlib.util
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests import test_lora_moss as tiny
from tests.lora_fakes import make_project
from voxprint_forge.diffusion import moss
from voxprint_forge.lora import bases, te_quant
from voxprint_forge.lora.cli import add_command
from voxprint_forge.lora.errors import LoraStageError, LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.i18n import text
from voxprint_forge.lora.presets import build_config
from voxprint_forge.lora.train import train


def test_resolve_default_env_and_bad_value(monkeypatch) -> None:
    monkeypatch.delenv(te_quant.ENV, raising=False)
    assert te_quant.resolve() == "auto"
    monkeypatch.setenv(te_quant.ENV, "INT8")
    assert te_quant.resolve() == "int8"
    assert te_quant.resolve("nf4") == "nf4"
    with pytest.raises(LoraUsageError) as caught:
        te_quant.resolve("int3")
    assert caught.value.key == "lora.err.te_quant_mode"


def test_config_and_cli_flag(monkeypatch) -> None:
    monkeypatch.delenv(te_quant.ENV, raising=False)
    monkeypatch.delenv(te_quant.ENV_DEVICE, raising=False)
    assert build_config("quick", "balanced", 16).te_quant == "auto"
    assert build_config("quick", "balanced", 16).te_device == "gpu"
    monkeypatch.setenv(te_quant.ENV, "nf4")
    assert build_config("quick", "balanced", 16).te_quant == "nf4"
    assert build_config("quick", "balanced", 16, te_quant="int8").te_quant == "int8"
    with pytest.raises(ValueError):
        build_config("quick", "balanced", 16, te_quant="fp2")
    parser = argparse.ArgumentParser()
    add_command(parser.add_subparsers(dest="command"))
    args = parser.parse_args(["lora", "train", "--project", "p", "--te-quant", "int8"])
    assert args.te_quant == "int8"
    assert (
        moss.generation_parser()
        .parse_args(["--snapshot", "s", "--prompt", "x", "--out", "o.png", "--te-quant", "nf4"])
        .te_quant
        == "nf4"
    )


def test_missing_library_is_a_localized_stage_error(monkeypatch) -> None:
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "bitsandbytes" else real(name, *a)
    )
    te_quant.check_available("none")
    with pytest.raises(LoraStageError) as caught:
        te_quant.check_available("int8")
    assert caught.value.key == "lora.err.te_quant_missing"
    assert "bitsandbytes" in text("lora.err.te_quant_missing", "en", mode="int8")
    assert "bitsandbytes" in text("lora.err.te_quant_missing", "ru", mode="int8")
    monkeypatch.setattr(te_quant, "cuda_available", lambda: True)
    with pytest.raises(LoraStageError):
        te_quant.plan("nf4", "gpu", 16)
    with pytest.raises(LoraStageError) as tight:
        te_quant.plan("auto", "gpu", 16)
    assert tight.value.key == "lora.err.te_quant_missing"


def test_plan_by_tier_and_overrides(monkeypatch) -> None:
    monkeypatch.delenv(te_quant.ENV, raising=False)
    monkeypatch.delenv(te_quant.ENV_DEVICE, raising=False)
    monkeypatch.setattr(te_quant, "cuda_available", lambda: True)
    monkeypatch.setattr(te_quant, "available", lambda: True)
    assert te_quant.plan("auto", None, 24) == te_quant.TePlan("none", "cuda")
    assert te_quant.plan(None, None, 16) == te_quant.TePlan("int8", "cuda")
    assert te_quant.plan("nf4", "gpu", 24) == te_quant.TePlan("nf4", "cuda")
    with pytest.raises(LoraStageError) as tight:
        te_quant.plan("int8", "cpu", 16)
    assert tight.value.key == "lora.err.cpu_fallback"
    assert te_quant.plan("none", "cpu", 24).device == "cpu"
    monkeypatch.setattr(te_quant, "cuda_available", lambda: False)
    assert te_quant.plan("auto", "gpu", 24).device == "cpu"
    with pytest.raises(LoraStageError) as no_card:
        te_quant.plan("auto", "gpu", 16)
    assert no_card.value.key == "lora.err.cpu_fallback"


def test_staged_order_frees_encoder_before_dit(monkeypatch, tmp_path) -> None:
    order: list[str] = []
    seen = tiny._install_pipe(monkeypatch)
    loader = moss.load_pipeline
    parts = moss.load_encoder_parts

    def spy_parts(root, **kwargs):
        order.append(f"encoders:{kwargs['te_quant']}@{kwargs['te_device']}")
        return parts(root, **kwargs)

    def spy_pipeline(root, **kwargs):
        order.append(f"dit:te={kwargs['with_text_encoder']}")
        return loader(root, **kwargs)

    monkeypatch.setattr(moss, "load_encoder_parts", spy_parts)
    monkeypatch.setattr(moss, "load_pipeline", spy_pipeline)
    adapter = bases.get_adapter("moss-image-base")
    monkeypatch.setattr(type(adapter), "_check", lambda self, w, c: te_quant.TePlan(c.te_quant, "cuda"))
    adapter.requires_gpu = False
    encode, free = adapter.encode, adapter.free_text_encoder

    def spy_encode(*a):
        order.append("encode")
        return encode(*a)

    def spy_free():
        order.append("free")
        free()
        assert adapter.pipe.text_encoder is None

    adapter.encode, adapter.free_text_encoder = spy_encode, spy_free
    weights = tmp_path / "w"
    weights.mkdir()
    config = tiny._run_config()
    config = type(config)(**{**config.to_dict(), "te_quant": "int8"})
    done = train(make_project(tmp_path / "p"), config, adapter, Progress(), base_dir=weights)
    assert done["state"] == "done"
    assert order == ["encoders:int8@cuda", "encode", "free", "dit:te=False"]
    assert seen["pipe"].text_encoder is None
    assert adapter.pipe.vae is not None


def test_encode_staged_frees_the_encoder_and_returns_cpu() -> None:
    class Tok:
        def apply_chat_template(self, msgs, **kw):
            return msgs[0]["content"]

        def __call__(self, text, **kw):
            ids = torch.tensor([[len(text) % 7 + 1, 2, 3]])
            return SimpleNamespace(to=lambda device: {"input_ids": ids})

    class Enc(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(10, 4)

        def forward(self, input_ids, **kw):
            return SimpleNamespace(last_hidden_state=self.emb(input_ids))

    calls = []

    def loader(root, te_quant, device):
        calls.append(te_quant)
        return Tok(), Enc()

    out = moss.encode_staged(__import__("pathlib").Path("."), ["a cube"], te_quant="int8", loader=loader)
    assert calls == ["int8"] and set(out) == {"a cube", ""}
    assert all(e.device.type == "cpu" and m.dtype == torch.bool for e, m in out.values())


def test_generate_encodes_before_reading_the_dit(monkeypatch, tmp_path) -> None:
    order: list[str] = []
    monkeypatch.setattr(te_quant, "cuda_available", lambda: True)
    monkeypatch.setattr(te_quant, "available", lambda: True)

    def staged(root, prompts, **kw):
        order.append(f"encode:{kw['te_quant']}@{kw['te_device']}")
        return {p: (torch.zeros(1, 4, 8), torch.ones(1, 4, dtype=torch.bool)) for p in [*prompts, ""]}

    class P:
        transformer = nn.Linear(1, 1)
        vae = None

        def __call__(self, prompt=None, **kw):
            assert prompt is None and kw["prompt_embeds"].shape == (1, 4, 8)
            return SimpleNamespace(images=["img"])

    def pipeline(root, **kw):
        order.append(f"dit:te={kw['with_text_encoder']}")
        return P()

    monkeypatch.setattr(moss, "encode_staged", staged)
    monkeypatch.setattr(moss, "load_pipeline", pipeline)
    assert moss.generate(tmp_path, "a cube", te_quant="int8") == "img"
    assert order == ["encode:int8@cuda", "dit:te=False"]


def test_quant_config_kinds(monkeypatch) -> None:
    assert te_quant.quant_config("none") is None
    monkeypatch.setattr(te_quant, "check_available", lambda mode: None)
    try:
        int8 = te_quant.quant_config("int8")
        nf4 = te_quant.quant_config("nf4")
    except ImportError:
        pytest.skip("transformers without BitsAndBytesConfig")
    assert int8.load_in_8bit and not int8.load_in_4bit
    assert nf4.load_in_4bit and nf4.bnb_4bit_quant_type == "nf4"
