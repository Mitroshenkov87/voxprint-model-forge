"""Export names, safetensors headers and the sidecar record the LoRA base."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import torch
from safetensors.torch import save_file

from tests.lora_fakes import TinyAdapter
from tests.test_lora_export import header, trained
from voxprint_forge.lora import bases
from voxprint_forge.lora.bases import BaseInfo
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.export import (
    base_short,
    compatible_bases_json,
    compatible_bases_list,
    dataset_digest,
    export,
    export_package,
    lora_base_of,
    lora_variant_of,
    trained_base_tag,
)
from voxprint_forge.lora.project import Item, package_stem
from voxprint_forge.lora.train import lora_base
from voxprint_forge.mdl import check_lora_base

_DATASET = "c59e4e3a142e3a1f69a497d9e83ff813bf285e972e6b78f460bba6fa5d17d781"


def _set_base(project, base: str, revision: str) -> None:
    data = project.read()
    train = dict(data["train"])
    train["base"] = base
    train["base_revision"] = revision
    # Drop a variant left by the stand-in trainer so the adapter's variant applies.
    train.pop("base_variant", None)
    project.write({**data, "train": train})


def test_base_short_tags_and_dataset_digest_are_stable() -> None:
    assert base_short("moss-image-base") == "moss5b"
    assert base_short("z-image") == "zimg6b"
    assert base_short("tiny-test") == "tinytest"
    left = [
        Item("b", "prepared/b.png", "long-b", "short-b", 1),
        Item("a", "prepared/a.png", "long-a", "short-a", 0),
    ]
    right = [
        Item("a", "other/a.png", "long-a", "short-a", 9, True),
        Item("b", "other/b.png", "long-b", "short-b", 3, False),
    ]
    assert dataset_digest(left) == dataset_digest(right) == _DATASET
    changed = [Item("a", "prepared/a.png", "long-a", "edited"), Item("b", "prepared/b.png", "long-b", "short-b")]
    assert dataset_digest(changed) != _DATASET


def test_safetensors_and_voxlora_names_carry_the_base(tmp_path) -> None:
    project = trained(tmp_path)
    digest = dataset_digest(project.items())
    _set_base(project, "z-image", "a" * 40)
    saved = export(project, Progress())
    z_path = Path(str(saved["file"]))
    assert z_path.name == f"{saved['id']}.zimg6b.safetensors"
    z_meta = header(z_path)
    assert z_meta["voxprint.lora.base"] == "z-image"
    assert z_meta["voxprint.lora.base_family"] == "z-image"
    assert z_meta["voxprint.lora.base_variant"] == "z-image"
    assert z_meta["voxprint.lora.base_sha256"] == ""
    assert z_meta["voxprint.lora.base_revision"] == "a" * 40
    assert z_meta["voxprint.lora.trained_base"] == "z-image@" + "a" * 8
    assert z_meta["voxprint.lora.compatible_bases"] == '["z-image","z-image-turbo"]'
    assert z_meta["voxprint.lora.dataset_id"] == digest
    assert lora_base_of(z_meta) == "z-image"
    sidecar = json.loads((z_path.parent / "voxprint-model.json").read_text(encoding="utf-8"))
    assert sidecar["base"] == {"mdl_id": "z-image", "revision": "a" * 40, "sha256": "", "mirror": ""}
    assert sidecar["trained_base"] == "z-image@" + "a" * 8
    assert sidecar["compatible_bases"] == ["z-image", "z-image-turbo"]
    assert sidecar["file"] == z_path.name
    check_lora_base(sidecar, "z-image")
    check_lora_base(sidecar, "z-image-turbo")

    _set_base(project, "moss-image-base", "b" * 40)
    moss = export(project, Progress(), out=tmp_path / "picked.safetensors")
    moss_meta = header(moss["file"])
    assert Path(str(moss["file"])).name == "picked.safetensors"
    assert moss_meta["voxprint.lora.base"] == "moss-image-base"
    assert moss_meta["voxprint.lora.base_family"] == "moss-image-base"
    assert moss_meta["voxprint.lora.base_variant"] == "sft"
    assert moss_meta["voxprint.lora.dataset_id"] == digest
    moss_side = json.loads((tmp_path / "voxprint-model.json").read_text(encoding="utf-8"))
    info = bases.get_adapter("moss-image-base").info
    assert moss_side["base"] == {
        "mdl_id": "moss-image-base",
        "revision": "b" * 40,
        "sha256": info.sha256,
        "mirror": info.mirror,
    }
    assert moss_meta["voxprint.lora.trained_base"] == "moss-image-base@" + "b" * 8
    assert moss_meta["voxprint.lora.compatible_bases"] == '["moss-image-base"]'
    assert moss_side["trained_base"] == "moss-image-base@" + "b" * 8
    assert moss_side["compatible_bases"] == ["moss-image-base"]
    assert info.mirror == "Mitroshenkov87/voxprint-mirror-moss-image-sft"
    assert info.repo == "OpenMOSS-Team/MOSS-Image-SFT"
    assert info.variant == "sft"
    data = project.read()
    recorded = dict(data["train"])
    recorded["base_variant"] = "base"
    project.write({**data, "train": recorded})
    stored = export(project, Progress(), out=tmp_path / "stored.safetensors")
    assert header(stored["file"])["voxprint.lora.base_variant"] == "base"

    packed = export_package(project, Progress())
    package = Path(str(packed["file"]))
    assert package.name == f"{package_stem(project.read())}.moss5b.voxlora"
    with zipfile.ZipFile(package) as archive:
        doc = json.loads(archive.read("voxprint-model.json"))
        weights = archive.read("model.safetensors")
    assert doc["base"] == "moss-image-base"
    unpacked = tmp_path / "model.safetensors"
    unpacked.write_bytes(weights)
    packed_meta = header(unpacked)
    assert packed_meta["voxprint.lora.base_family"] == "moss-image-base"
    assert packed_meta["voxprint.lora.base_variant"] == "base"
    assert packed_meta["voxprint.lora.dataset_id"] == digest
    assert lora_base_of(packed_meta) == "moss-image-base"


def test_adapter_sha_is_copied_and_a_missing_project_base_exports_as_z_image(tmp_path) -> None:
    class ShaAdapter(TinyAdapter):
        info = BaseInfo(
            "sha-base",
            "example/sha-base",
            "c" * 40,
            "Apache-2.0",
            "Sha",
            sha256="ab" * 32,
            mirror="example/mirror-sha",
        )

    bases.register(ShaAdapter)
    try:
        project = trained(tmp_path / "sha")
        data = project.read()
        train = dict(data["train"])
        train["base"] = "sha-base"
        train["base_revision"] = "c" * 40
        train["base_sha256"] = "ff" * 32  # the adapter pin wins when it is set
        project.write({**data, "train": train})
        saved = export(project, Progress())
        meta = header(saved["file"])
        assert Path(str(saved["file"])).name.endswith(".shabase.safetensors")
        assert meta["voxprint.lora.base_sha256"] == "ab" * 32
        assert meta["voxprint.lora.base_variant"] == "base"
        assert meta["voxprint.lora.base_family"] == "sha-base"
        sidecar = json.loads((project.root / "export" / "voxprint-model.json").read_text(encoding="utf-8"))
        assert sidecar["base"]["sha256"] == "ab" * 32
        assert sidecar["base"]["mirror"] == "example/mirror-sha"
        assert sidecar["compatible_bases"] == ["sha-base"]
        assert sidecar["trained_base"] == "sha-base@" + "c" * 8
        assert meta["voxprint.lora.trained_base"] == "sha-base@" + "c" * 8
        assert meta["voxprint.lora.compatible_bases"] == '["sha-base"]'
    finally:
        bases.unregister("sha-base")

    project = trained(tmp_path / "legacy")
    data = project.read()
    train = {
        key: value
        for key, value in dict(data["train"]).items()
        if key not in {"base", "base_revision", "base_variant", "base_sha256"}
    }
    project.write({**data, "train": train})
    saved = export(project, Progress())
    assert Path(str(saved["file"])).name.endswith(".zimg6b.safetensors")
    meta = header(saved["file"])
    assert meta["voxprint.lora.base"] == "z-image"
    assert meta["voxprint.lora.base_family"] == "z-image"
    assert meta["voxprint.lora.base_variant"] == "z-image"
    assert meta["voxprint.lora.base_sha256"] == ""
    assert lora_base_of(meta) == "z-image"
    assert lora_variant_of(meta) == "z-image"
    side = json.loads((project.root / "export" / "voxprint-model.json").read_text(encoding="utf-8"))
    assert side["base"]["mdl_id"] == "z-image"
    assert side["trained_base"] == "z-image@04cc4abb"
    assert side["compatible_bases"] == ["z-image", "z-image-turbo"]
    assert meta["voxprint.lora.trained_base"] == "z-image@04cc4abb"
    assert meta["voxprint.lora.compatible_bases"] == '["z-image","z-image-turbo"]'
    # Legacy voxprint.lora.base and the short tag stay unchanged.
    assert meta["voxprint.lora.base"] == "z-image"
    check_lora_base(side, "z-image")
    check_lora_base(side, "z-image-turbo")


def test_a_file_with_no_base_metadata_reads_as_z_image(tmp_path) -> None:
    path = tmp_path / "old-name.safetensors"
    save_file({"x": torch.zeros(1)}, str(path))
    assert lora_base_of(lora_base(path)) == "z-image"
    assert lora_variant_of(lora_base(path)) == "z-image"
    family = tmp_path / "family.safetensors"
    save_file({"x": torch.zeros(1)}, str(family), metadata={"voxprint.lora.base_family": "moss-image-base"})
    assert lora_base_of(lora_base(family)) == "moss-image-base"
    assert lora_variant_of(lora_base(family)) == "sft"
    assert b"voxprint.lora.base" not in path.read_bytes()


def test_canvas_metadata_contract_strings() -> None:
    """Verbatim Canvas keys for a full Z-Image@04cc4abb run and a MOSS run."""
    assert trained_base_tag("z-image", "04cc4abb7c5069926f75c9bfde9ef43d49423021") == "z-image@04cc4abb"
    assert compatible_bases_json("z-image") == '["z-image","z-image-turbo"]'
    assert compatible_bases_list("z-image") == ["z-image", "z-image-turbo"]
    assert trained_base_tag("moss-image-base", "fc5de0d56fc910cbca6f6625a70f5f582b26deb1") == "moss-image-base@fc5de0d5"
    assert compatible_bases_json("moss-image-base") == '["moss-image-base"]'
    assert base_short("z-image") == "zimg6b"
    assert base_short("moss-image-base") == "moss5b"
