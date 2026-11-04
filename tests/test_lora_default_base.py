"""New projects default to MOSS-Image. A record with no base is still Z-Image."""

from __future__ import annotations

import pytest
from safetensors.torch import load_file, save_file

from tests.lora_fakes import TinyAdapter, make_project
from voxprint_forge.cli import build_parser
from voxprint_forge.lora import DEFAULT_BASE, OPTIONAL_BASES, bases, legacy_base, resolve_base
from voxprint_forge.lora.bases import BaseInfo
from voxprint_forge.lora.errors import LoraUsageError
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.export import lora_base_of
from voxprint_forge.lora.presets import build_config
from voxprint_forge.lora.train import lora_base, train


class _ZImageStub(TinyAdapter):
    """CPU stand-in registered nowhere: its MDL id is the legacy Z-Image base."""

    info = BaseInfo("z-image", "example/z-image", "0" * 40, "Apache-2.0", "Z-Image stub")


def _quick(**overrides):
    values = {"steps": 2, "resolution": 64, "save_every": 1, "batch": 2, "previews": False}
    values.update(overrides)
    return build_config("quick", "balanced", 24, **values)


def test_new_commands_default_to_moss(tmp_path) -> None:
    assert DEFAULT_BASE == "moss-image-base"
    assert OPTIONAL_BASES == ("z-image",)
    assert legacy_base() == "z-image"
    info = bases.get_adapter(DEFAULT_BASE).info
    assert info.display_name == "MOSS-Image SFT"
    assert info.repo == "OpenMOSS-Team/MOSS-Image-SFT"
    assert info.revision == "fc5de0d56fc910cbca6f6625a70f5f582b26deb1"
    assert info.mirror == "Mitroshenkov87/voxprint-mirror-moss-image-sft"
    assert info.variant == "sft"
    assert bases.get_adapter("z-image").info.variant == "z-image"
    train_args = build_parser().parse_args(["lora", "train", "--project", str(tmp_path)])
    finish_args = build_parser().parse_args(["lora", "refset", "finish", "--project", str(tmp_path)])
    assert train_args.base == "moss-image-base"
    assert finish_args.base == "moss-image-base"


def test_a_record_without_base_information_resolves_to_z_image(tmp_path) -> None:
    project = make_project(tmp_path)
    data = project.read()
    assert "base" not in data
    train_record = data.get("train")
    assert not isinstance(train_record, dict)
    assert resolve_base(None) == "z-image"
    assert resolve_base("") == "z-image"
    assert resolve_base("  ") == "z-image"
    assert resolve_base({"revision": "abc"}) == "z-image"
    assert resolve_base({"mdl_id": ""}) == "z-image"
    assert resolve_base("moss-image-base") == "moss-image-base"
    assert resolve_base({"mdl_id": " moss-image-base "}) == "moss-image-base"
    assert lora_base_of({}) == "z-image"
    assert lora_base_of({"voxprint.lora.base": "  ", "voxprint.lora.base_family": ""}) == "z-image"
    assert lora_base_of({"voxprint.lora.base": "z-image"}) == "z-image"
    assert lora_base_of({"voxprint.lora.base_family": "moss-image-base", "voxprint.lora.base": "z-image"}) == (
        "moss-image-base"
    )


def test_hiding_moss_does_not_relabel_legacy_loras(monkeypatch) -> None:
    monkeypatch.delenv("VOXPRINT_FORGE_MOSS_LORA", raising=False)
    assert DEFAULT_BASE == "moss-image-base"
    assert "moss-image-base" in {info.mdl_id for info in bases.trainable()}
    monkeypatch.setenv("VOXPRINT_FORGE_MOSS_LORA", "0")
    assert bases.hidden() == frozenset({"moss-image-base"})
    assert [info.mdl_id for info in bases.trainable()] == ["z-image"]
    assert "moss-image-base" in {info.mdl_id for info in bases.available()}
    with pytest.raises(LoraUsageError) as caught:
        bases.get_adapter(DEFAULT_BASE, training=True)
    assert caught.value.key == "lora.err.base_missing"
    # Export and other reads of an existing run still see the hidden base.
    assert bases.get_adapter(DEFAULT_BASE).info.mdl_id == "moss-image-base"
    assert resolve_base(None) == legacy_base() == "z-image"


def test_a_lora_file_without_a_base_tag_extends_as_z_image(tmp_path) -> None:
    first = make_project(tmp_path / "one")
    train(first, _quick(), _ZImageStub(), Progress())
    source = first.checkpoints / "step-000002.safetensors"
    blank = tmp_path / "legacy.safetensors"
    save_file(load_file(str(source)), str(blank))
    assert "zimg6b" not in blank.name and "moss5b" not in blank.name
    assert lora_base_of(lora_base(blank)) == "z-image"
    with pytest.raises(LoraUsageError) as caught:
        train(make_project(tmp_path / "other"), _quick(steps=1), TinyAdapter(), Progress(), from_lora=blank)
    assert caught.value.code == "lora.err.base_mismatch"
    second = make_project(tmp_path / "two")
    result = train(second, _quick(steps=1), _ZImageStub(), Progress(), from_lora=blank)
    assert result["state"] == "done"
    assert second.read()["train"]["base"] == "z-image"
    # The stand-in is not registered, so the real Z-Image adapter stays in place.
    assert bases.get_adapter("z-image").info.repo == "Tongyi-MAI/Z-Image"
