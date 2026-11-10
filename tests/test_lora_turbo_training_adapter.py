"""Z-Image-Turbo training adapter: flags, freeze, exclusion, metadata. Offline, no GPU."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from voxprint_forge.cli import build_parser
from voxprint_forge.lora import bases
from voxprint_forge.lora.errors import LoraUsageError
from voxprint_forge.lora.export import (
    base_short,
    compatible_bases_json,
    compatible_bases_list,
    header_metadata,
    trained_base_tag,
)
from voxprint_forge.lora.layers import inject, lora_parameters, lora_state
from voxprint_forge.lora.project import bases_for_choice
from voxprint_forge.lora.training_adapter import (
    DEFAULT_FILE,
    DEFAULT_REPO,
    DEFAULT_REVISION,
    DEFAULT_SHA256,
    TURBO_REVISION,
    attach_frozen,
    bind,
    remap_key,
    set_enabled,
    sha256_file,
)
from voxprint_forge.mdl import check_lora_base


class _TinyTurbo(nn.Module):
    """A handful of Linear layers named like the Z-Image attention targets."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.attention = nn.Module()
        self.attention.to_q = nn.Linear(width, width, bias=False)
        self.attention.to_k = nn.Linear(width, width, bias=False)
        self.attention.to_v = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            for linear in (self.attention.to_q, self.attention.to_k, self.attention.to_v):
                linear.weight.copy_(torch.eye(width))


def _adapter_file(path: Path, width: int = 8, rank: int = 2, keys: dict[str, torch.Tensor] | None = None) -> Path:
    if keys is None:
        keys = {}
        for name in ("attention.to_q", "attention.to_k", "attention.to_v"):
            keys[f"{name}.lora_A.weight"] = torch.ones(rank, width)
            keys[f"{name}.lora_B.weight"] = torch.ones(width, rank)
    save_file(keys, str(path))
    return path


def test_cli_parses_training_adapter_flags(tmp_path) -> None:
    parsed = build_parser().parse_args(
        [
            "lora",
            "train",
            "--project",
            str(tmp_path),
            "--base",
            "z-image-turbo",
            "--training-adapter",
            DEFAULT_REPO,
            "--training-adapter-revision",
            DEFAULT_REVISION,
            "--training-adapter-file",
            DEFAULT_FILE,
        ]
    )
    assert parsed.base == "z-image-turbo"
    assert parsed.training_adapter == DEFAULT_REPO
    assert parsed.training_adapter_revision == DEFAULT_REVISION
    assert parsed.training_adapter_file == DEFAULT_FILE
    omitted = build_parser().parse_args(["lora", "train", "--project", str(tmp_path), "--base", "z-image-turbo"])
    assert omitted.training_adapter is None
    assert omitted.training_adapter_revision is None
    assert omitted.training_adapter_file is None
    prepare = build_parser().parse_args(["lora", "prepare", "--project", str(tmp_path), "--base", "z-image-turbo"])
    assert prepare.base == "z-image-turbo" and prepare.base_explicit is True
    assert bases_for_choice("z-image-turbo") == ["z-image-turbo"]
    retrain = build_parser().parse_args(["lora", "retrain", str(tmp_path), "--base", "z-image-turbo"])
    assert retrain.base == "z-image-turbo"


def test_bind_defaults_on_turbo_and_rejects_other_bases(tmp_path) -> None:
    turbo = bases.get_adapter("z-image-turbo", training=True)
    bind(turbo, None, None, None)
    assert turbo.training_adapter.source == DEFAULT_REPO
    assert turbo.training_adapter.revision == DEFAULT_REVISION
    assert turbo.training_adapter.filename == DEFAULT_FILE
    assert turbo.training_adapter.sha256 == DEFAULT_SHA256
    local = tmp_path / "custom.safetensors"
    local.write_bytes(b"not-used")
    bind(turbo, str(local), None, None)
    assert turbo.training_adapter.path == local
    other = bases.get_adapter("z-image", training=True)
    bind(other, None, None, None)
    assert getattr(other, "training_adapter", None) is None
    with pytest.raises(LoraUsageError) as caught:
        bind(other, DEFAULT_REPO, None, None)
    assert caught.value.key == "lora.err.training_adapter_base"


def test_turbo_adapter_is_registered_and_hidden_with_z_image(monkeypatch) -> None:
    info = bases.get_adapter("z-image-turbo").info
    assert info.mdl_id == "z-image-turbo"
    assert info.repo == "Tongyi-MAI/Z-Image-Turbo"
    assert info.revision == TURBO_REVISION
    assert info.licence == "Apache-2.0"
    assert "z-image-turbo" in {row.mdl_id for row in bases.trainable()}
    monkeypatch.setenv("VOXPRINT_FORGE_TURBO_LORA", "0")
    monkeypatch.delenv("VOXPRINT_FORGE_ZIMAGE_LORA", raising=False)
    assert "z-image-turbo" in bases.hidden()
    assert "z-image" not in bases.hidden()


def test_frozen_adapter_is_excluded_from_saved_state(tmp_path) -> None:
    torch.manual_seed(0)
    model = _TinyTurbo()
    targets = ("attention.to_q", "attention.to_k", "attention.to_v")
    layers = inject(model, targets, rank=2, alpha=2)
    path = _adapter_file(tmp_path / "adapter.safetensors")
    attach_frozen(layers, path, scale=1.0)
    for layer in layers.values():
        assert layer.has_training_adapter
        assert layer.training_adapter_scale == 1.0
        assert layer.training_A is not None and layer.training_B is not None
        assert not layer.training_A.weight.requires_grad
        assert not layer.training_B.weight.requires_grad
        assert layer.lora_A.weight.requires_grad and layer.lora_B.weight.requires_grad
    trainable = set(lora_parameters(layers))
    assert all(param.requires_grad for param in trainable)
    assert all(id(layer.training_A.weight) not in {id(p) for p in trainable} for layer in layers.values())
    state = lora_state(layers)
    assert all("training_" not in key for key in state)
    assert set(state) == {
        "attention.to_q.lora_A.weight",
        "attention.to_q.lora_B.weight",
        "attention.to_k.lora_A.weight",
        "attention.to_k.lora_B.weight",
        "attention.to_v.lora_A.weight",
        "attention.to_v.lora_B.weight",
    }
    saved = tmp_path / "character.safetensors"
    save_file(state, str(saved))
    loaded = load_file(str(saved))
    assert set(loaded) == set(state)


def test_disabling_the_adapter_changes_the_forward_and_does_not_touch_saved_weights(tmp_path) -> None:
    torch.manual_seed(1)
    model = _TinyTurbo()
    layers = inject(model, ("attention.to_q", "attention.to_k", "attention.to_v"), rank=2, alpha=2)
    # Keep the character LoRA at zero so only the frozen adapter can move the output.
    with torch.no_grad():
        for layer in layers.values():
            layer.lora_B.weight.zero_()
    attach_frozen(layers, _adapter_file(tmp_path / "adapter.safetensors"), scale=1.0)
    x = torch.ones(1, 8)
    trained = model.attention.to_q(x)
    set_enabled(model, False)
    preview = model.attention.to_q(x)
    set_enabled(model, True)
    restored = model.attention.to_q(x)
    assert not torch.allclose(trained, preview)
    assert torch.allclose(trained, restored)
    assert torch.allclose(preview, torch.ones(1, 8))  # identity base + zero character LoRA


def test_comfy_prefix_is_stripped() -> None:
    assert remap_key("diffusion_model.attention.to_q.lora_A.weight") == "attention.to_q.lora_A.weight"
    assert remap_key("transformer.attention.to_q.lora_down.weight") == "attention.to_q.lora_A.weight"


def test_metadata_and_canvas_are_turbo_only() -> None:
    assert trained_base_tag("z-image-turbo", TURBO_REVISION) == "z-image-turbo@f332072a"
    assert compatible_bases_list("z-image-turbo") == ["z-image-turbo"]
    assert compatible_bases_json("z-image-turbo") == '["z-image-turbo"]'
    assert "z-image" not in compatible_bases_list("z-image-turbo")
    assert base_short("z-image-turbo") == "zimgturbo"
    # Full Z-Image LoRAs stay accepted on Turbo; the reverse is not true.
    assert compatible_bases_list("z-image") == ["z-image", "z-image-turbo"]
    sidecar = {
        "base": {"mdl_id": "z-image-turbo", "revision": TURBO_REVISION},
        "compatible_bases": ["z-image-turbo"],
    }
    check_lora_base(sidecar, "z-image-turbo")
    with pytest.raises(LoraUsageError) as caught:
        check_lora_base(sidecar, "z-image")
    assert caught.value.key == "lora.err.base_mismatch"
    z_image = {"base": {"mdl_id": "z-image"}, "compatible_bases": ["z-image", "z-image-turbo"]}
    check_lora_base(z_image, "z-image-turbo")
    check_lora_base(z_image, "z-image")


def test_export_headers_use_turbo_only_compatible_bases() -> None:
    meta = header_metadata(
        {
            "subject": {"kind": "person", "name": "Garnis_AI", "consent_obtained": True, "adult_confirmed": True},
            "train": {
                "base": "z-image-turbo",
                "base_revision": TURBO_REVISION,
                "step": 1200,
                "config": {"rank": 64, "alpha": 64},
            },
        }
    )
    assert meta["voxprint.lora.trained_base"] == "z-image-turbo@f332072a"
    assert meta["voxprint.lora.compatible_bases"] == '["z-image-turbo"]'
    assert meta["voxprint.lora.base"] == "z-image-turbo"


def test_default_v2_pin_and_sha256_helper(tmp_path) -> None:
    payload = b"voxprint-fake-adapter"
    path = tmp_path / DEFAULT_FILE
    path.write_bytes(payload)
    import hashlib

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()
    assert DEFAULT_SHA256 == "20a541d3e016ab8de0da076321b48b6cd9b3ffd072d9df830a068220ab2265f6"
    assert len(DEFAULT_REVISION) == 40
