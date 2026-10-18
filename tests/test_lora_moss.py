"""MOSS-Image LoRA on a tiny CPU stand-in: no weights download, no GPU.

The stand-in has the same projection names as the vendored transformer, so inject, the
flow-matching loss, the checkpoint and ``apply_lora`` run the real adapter code.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn

from tests.lora_fakes import make_project
from voxprint_forge.diffusion import moss
from voxprint_forge.errors import StageError
from voxprint_forge.lora import bases
from voxprint_forge.lora.bases import MossImageAdapter
from voxprint_forge.lora.errors import LoraStageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.export import export
from voxprint_forge.lora.layers import LoRALinear
from voxprint_forge.lora.presets import build_config
from voxprint_forge.lora.train import train

_PROJECTIONS = ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.ModuleDict({name: nn.Linear(128, 128, bias=False) for name in _PROJECTIONS})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.proj["q"].weight.dtype)
        return self.proj["o"](self.proj["q"](x))


class _TinyMoss(nn.Module):
    """One block, latent channels 128, so a 64 px picture packs to a legal input."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block()])

    def forward(self, hidden, timestep, encoder_hidden_states, encoder_attention_mask=None, return_dict=True):
        batch, channels, height, width = hidden.shape
        x = hidden.flatten(2).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        sample = x.transpose(1, 2).reshape(batch, channels, height, width)
        return SimpleNamespace(sample=sample)


class _Vae(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.bn = SimpleNamespace(running_mean=torch.zeros(128), running_var=torch.ones(128))
        self.config = SimpleNamespace(batch_norm_eps=1e-5)

    def encode(self, image: torch.Tensor) -> object:
        batch, _, height, width = image.shape
        sample = torch.randn(batch, 32, height // 8, width // 8, device=image.device, dtype=torch.float32)
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: sample))


class _Pipe:
    def __init__(self) -> None:
        self.transformer = _TinyMoss()
        self.vae = _Vae()
        self.text_encoder = nn.Linear(1, 1)
        self.tokenizer = None
        self.calls: list[dict] = []

    def encode_prompt(self, prompt):
        text = prompt if isinstance(prompt, str) else str(prompt)
        embeds = torch.zeros(1, 4, 128)
        embeds[0, :, 0] = float(len(text))
        return embeds, torch.ones(1, 4, dtype=torch.bool)

    def __call__(self, prompt=None, **kwargs):
        self.calls.append({"prompt": prompt, "generator": getattr(kwargs.get("generator"), "device", None)})
        return SimpleNamespace(images=[Image.new("RGB", (8, 8), (40, 80, 120))])


def _install_pipe(monkeypatch) -> dict:
    seen: dict = {}

    def load_pipeline(root, **kwargs):
        torch.manual_seed(0)
        pipe = _Pipe()
        pipe.transformer.to(dtype=kwargs.get("dtype") or torch.bfloat16)
        path = kwargs.get("lora_path")
        if path is not None:
            moss.apply_lora(pipe.transformer, path, float(kwargs.get("lora_scale", 1.0)))
        seen["pipe"] = pipe
        seen["lora"] = path
        return pipe

    monkeypatch.setattr(moss, "load_pipeline", load_pipeline)
    return seen


def _run_config():
    return build_config(
        "quick",
        "balanced",
        24,
        steps=3,
        resolution=64,
        save_every=1,
        batch=1,
        grad_accum=1,
        previews=False,
        rank=4,
        alpha=4,
        caption_dropout=0.0,
        caption_mode="long",
        seed=0,
        grad_checkpointing=True,
    )


def _forward(model: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    embeds = torch.zeros(1, 4, 128)
    mask = torch.ones(1, 4, dtype=torch.bool)
    with torch.no_grad():
        return model(hidden.to(torch.bfloat16), torch.tensor([250.0]), embeds.to(torch.bfloat16), mask).sample.float()


def test_cpu_stand_in_trains_resumes_exports_and_loads(monkeypatch, tmp_path) -> None:
    seen = _install_pipe(monkeypatch)
    weights = tmp_path / "weights"
    weights.mkdir()
    config = _run_config()
    straight_adapter = bases.get_adapter("moss-image-base")
    straight_adapter.requires_gpu = False
    held: dict = {}
    shapes: list[tuple[int, ...]] = []
    encode = straight_adapter.encode

    def spy_encode(images, captions, cfg):
        samples = encode(images, captions, cfg)
        shapes.append(tuple(samples[0].latents.shape))
        pair = samples[0].captions["long"]
        assert pair[0].ndim == 2 and pair[1].dtype == torch.bool
        return samples

    def spy_load(folder, cfg, device):
        model = MossImageAdapter.load(straight_adapter, folder, cfg, device)
        assert model.layers[0].forward.__name__ == "wrapped"
        held["model"] = model
        return model

    straight_adapter.encode = spy_encode
    straight_adapter.load = spy_load
    straight = make_project(tmp_path / "straight")
    done = train(straight, config, straight_adapter, Progress(), base_dir=weights)
    assert done["state"] == "done" and done["step"] == 3
    assert shapes == [(128, 4, 4)]
    assert (straight.checkpoints / "step-000001.safetensors").is_file()
    assert (straight.checkpoints / "step-000003.safetensors").is_file()
    loras = [m for m in held["model"].modules() if isinstance(m, LoRALinear)]
    assert loras and all(m.lora_A.weight.dtype == torch.float32 for m in loras)
    assert all(m.base.weight.dtype == torch.bfloat16 and not m.base.weight.requires_grad for m in loras)
    assert any(m.lora_B.weight.detach().abs().sum() > 0 for m in loras)

    paused = make_project(tmp_path / "paused")
    paused_adapter = bases.get_adapter("moss-image-base")
    paused_adapter.requires_gpu = False

    def sink(event):
        if event.params.get("step") == 1:
            paused.pause.write_text("", encoding="utf-8")

    first = train(paused, config, paused_adapter, Progress(sink), base_dir=weights)
    assert first["state"] == "paused" and first["step"] == 1
    paused.pause.unlink()
    second = train(paused, config, paused_adapter, Progress(), resume=True, base_dir=weights)
    assert second["state"] == "done" and second["step"] == 3
    left = load_file(str(straight.checkpoints / "step-000003.safetensors"))
    right = load_file(str(paused.checkpoints / "step-000003.safetensors"))
    assert set(left) == set(right)
    for key in left:
        assert torch.allclose(left[key], right[key], atol=1e-5), key

    exported = export(straight, Progress(), out=tmp_path / "cube.safetensors")
    assert exported["state"] == "done"
    image_path = tmp_path / "out.png"
    lora_file = tmp_path / "cube.safetensors"
    image = moss.generate(weights, "a red cube", steps=1, height=64, width=64, lora_path=lora_file, out=image_path)
    assert image_path.is_file() and seen["lora"] == lora_file
    assert image.size == (8, 8)
    torch.manual_seed(1)
    hidden = torch.randn(1, 128, 4, 4)
    trained = _forward(held["model"], hidden)
    merged = _forward(seen["pipe"].transformer, hidden)
    assert torch.allclose(trained, merged, atol=1e-2, rtol=1e-2)
    torch.manual_seed(0)
    base = _TinyMoss().to(dtype=torch.bfloat16)
    assert not torch.allclose(merged, _forward(base, hidden), atol=1e-3)


def test_dry_run_prints_a_plan(monkeypatch, tmp_path, capsys) -> None:
    from voxprint_forge.cli import main

    monkeypatch.setattr(bases, "_moss_weights_dir", lambda: None)
    make_project(tmp_path)
    code = main(["lora", "train", "--project", str(tmp_path), "--base", "moss-image-base", "--vram", "24", "--dry-run"])
    plan = json.loads(capsys.readouterr().out)
    assert code == 0
    assert plan["state"] == "dry-run" and plan["base"] == "moss-image-base"
    assert plan["memory_plan"]["peak_gb"] == pytest.approx(11.8, abs=0.05)
    assert plan["vram_need_gb"] < 16


def test_memory_plan_matches_the_measured_4090_run(monkeypatch) -> None:
    monkeypatch.setattr(bases, "_moss_weights_dir", lambda: None)
    adapter = MossImageAdapter()
    anchor = build_config("balanced", "balanced", 24, batch=1, grad_accum=1)
    assert adapter.seconds_per_step(anchor) == pytest.approx(0.7)
    peak = adapter.memory_plan(build_config("balanced", "balanced", 24)).peak_gb
    assert peak == pytest.approx(11.8, abs=0.05)
    assert adapter.memory_plan(build_config("balanced", "balanced", 16)).peak_gb == pytest.approx(peak, abs=0.01)
    assert adapter.vram_need_gb(build_config("quick", "balanced", 16)) < 16
    assert adapter.vram_need_gb(build_config("best", "balanced", 24)) < 24
    assert adapter.memory_plan(build_config("balanced", "balanced", 24, grad_checkpointing=False)).peak_gb > 24


def test_generation_parser_accepts_a_lora_file() -> None:
    args = moss.generation_parser().parse_args(
        [
            "--snapshot",
            "snap",
            "--prompt",
            "a red cube",
            "--out",
            "out.png",
            "--lora",
            "cube.safetensors",
            "--lora-scale",
            "0.5",
        ]
    )
    assert args.lora.name == "cube.safetensors" and args.lora_scale == 0.5


def test_apply_lora_merges_a_prefixed_key_and_rejects_an_empty_file(tmp_path) -> None:
    torch.manual_seed(0)
    module = _TinyMoss()
    before = module.layers[0].proj["q"].weight.detach().clone()
    path = tmp_path / "peft.safetensors"
    rank = 2
    save_file(
        {
            "base_model.model.transformer.layers.0.proj.q.lora_A.weight": torch.ones(rank, 128),
            "base_model.model.transformer.layers.0.proj.q.lora_B.weight": torch.ones(128, rank),
        },
        str(path),
        metadata={"voxprint.lora.rank": "2", "voxprint.lora.alpha": "2"},
    )
    assert moss.apply_lora(module, path, 1.0) == 1
    assert not torch.equal(module.layers[0].proj["q"].weight, before)
    empty = tmp_path / "empty.safetensors"
    save_file({"other.weight": torch.zeros(2, 2)}, str(empty))
    with pytest.raises(StageError) as caught:
        moss.apply_lora(module, empty, 1.0)
    assert caught.value.code == "moss_lora"


class _Parked:
    """diffusers' device rule: the first module in sorted order, which is the text encoder."""

    def __init__(self, encoder_device: str) -> None:
        self.text_encoder = nn.Linear(1, 1)
        self.text_encoder.to(encoder_device)
        self.transformer = nn.Linear(1, 1)
        self.vae = nn.Linear(1, 1)
        self.during: list[object] = []

    @property
    def _execution_device(self) -> torch.device:
        for name in sorted(("text_encoder", "transformer", "vae")):
            module = getattr(self, name, None)
            if isinstance(module, nn.Module):
                return next(module.parameters()).device
        return torch.device("cpu")

    def __call__(self, **kwargs):
        self.during.append(self.text_encoder)
        generator = kwargs["generator"]
        if generator.device.type != "cpu":
            raise RuntimeError(generator.device.type)
        return SimpleNamespace(images=[Image.new("RGB", (4, 4))])


def test_preview_parks_the_text_encoder_and_seeds_a_cpu_generator() -> None:
    adapter = MossImageAdapter()
    adapter.device = "cpu"
    adapter.pipe = _Parked("meta")
    row = (torch.ones(3, 8), torch.ones(3, dtype=torch.bool))
    adapter._prompts = {"a red cube": row, "": (torch.zeros(3, 8), torch.ones(3, dtype=torch.bool))}
    config = build_config("balanced", "balanced", 24, previews=True)
    image = adapter.preview(nn.Identity(), "a red cube", 7, config)
    assert image is not None and adapter.pipe.during == [None]
    assert isinstance(adapter.pipe.text_encoder, nn.Module)


def test_preview_device_mismatch_is_named() -> None:
    class _CudaDevice(_Parked):
        @property
        def _execution_device(self) -> torch.device:
            return torch.device("cuda")

    adapter = MossImageAdapter()
    adapter.device = "cpu"
    adapter.pipe = _CudaDevice("meta")
    row = (torch.ones(3, 8), torch.ones(3, dtype=torch.bool))
    adapter._prompts = {"a red cube": row, "": row}
    with pytest.raises(LoraStageError) as caught:
        adapter.preview(nn.Identity(), "a red cube", 1, build_config("balanced", "balanced", 24))
    assert caught.value.key == "lora.err.preview_device"
