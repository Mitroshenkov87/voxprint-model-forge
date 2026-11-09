"""A voxprint-model/1 LoRA sidecar names its base, and the loader refuses a mismatch."""

from __future__ import annotations

import pytest

from voxprint_forge.lora.bases import BaseInfo
from voxprint_forge.lora.errors import LoraUsageError
from voxprint_forge.lora.export import sidecar_document
from voxprint_forge.mdl import check_lora_base

MOSS = BaseInfo(
    "moss-image-base",
    "example/moss",
    "1" * 40,
    "Apache-2.0",
    "MOSS stub",
    sha256="ab" * 32,
    mirror="example/mirror-moss",
)
ZIMAGE = BaseInfo(
    "z-image",
    "example/z-image",
    "2" * 40,
    "Apache-2.0",
    "Z-Image stub",
    sha256="",
    mirror="",
)


def _sidecar(tmp_path, info: BaseInfo) -> dict[str, object]:
    target = tmp_path / f"lora.{info.mdl_id}.safetensors"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"stub-weights")
    data = {
        "kind": "object",
        "subject": {"kind": "object", "adult_content": False},
        "train": {"base": info.mdl_id, "base_revision": info.revision, "step": 4, "steps_planned": 4},
    }
    return sidecar_document("tcube-deadbeef", data, target, info, None, ("cube",))


def test_sidecar_base_block_is_required_on_a_new_document(tmp_path) -> None:
    document = _sidecar(tmp_path, MOSS)
    assert document["schema"] == "voxprint-model/1"
    assert document["type"] == "lora"
    assert set(document["base"]) == {"mdl_id", "revision", "sha256", "mirror"}
    assert document["base"] == {
        "mdl_id": "moss-image-base",
        "revision": "1" * 40,
        "sha256": "ab" * 32,
        "mirror": "example/mirror-moss",
    }
    assert document["trained_base"] == "moss-image-base@" + "1" * 8
    assert document["compatible_bases"] == ["moss-image-base"]
    zimage = _sidecar(tmp_path / "z", ZIMAGE)
    assert zimage["base"] == {"mdl_id": "z-image", "revision": "2" * 40, "sha256": "", "mirror": ""}
    assert zimage["trained_base"] == "z-image@" + "2" * 8
    assert zimage["compatible_bases"] == ["z-image", "z-image-turbo"]


def test_loader_rejects_a_moss_lora_on_z_image(tmp_path) -> None:
    moss = _sidecar(tmp_path, MOSS)
    check_lora_base(moss, "moss-image-base")
    with pytest.raises(LoraUsageError) as caught:
        check_lora_base(moss, "z-image")
    assert caught.value.code == "lora.err.base_mismatch"
    assert caught.value.key == "lora.err.base_mismatch"
    assert caught.value.params["base"] == "moss-image-base"
    assert caught.value.params["target"] == "z-image"
    assert caught.value.params["cli"] == "z-image"
    assert "forge lora retrain --base z-image PROJECT" in str(caught.value)

    zimage = _sidecar(tmp_path / "z", ZIMAGE)
    check_lora_base(zimage, "z-image")
    check_lora_base(zimage, "z-image-turbo")
    with pytest.raises(LoraUsageError) as other:
        check_lora_base(zimage, "moss-image-base")
    assert other.value.code == "lora.err.base_mismatch"
    with pytest.raises(LoraUsageError):
        check_lora_base(moss, "z-image-turbo")


def test_a_descriptor_without_a_base_is_z_image() -> None:
    check_lora_base({}, "z-image")
    check_lora_base({"base": {}}, "z-image")
    check_lora_base({"base": ""}, "z-image")
    check_lora_base({"base": "z-image", "compatible_bases": ["z-image"]}, "z-image")
    check_lora_base({"base": "z-image", "compatible_bases": ["z-image"]}, "z-image-turbo")
    check_lora_base({}, "z-image-turbo")
    with pytest.raises(LoraUsageError) as caught:
        check_lora_base({}, "moss-image-base")
    assert caught.value.code == "lora.err.base_mismatch"
    assert caught.value.params["base"] == "z-image"
    assert caught.value.params["cli"] == "moss"
    assert "forge lora retrain --base moss PROJECT" in str(caught.value)
