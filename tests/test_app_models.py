"""manifests/model-forge.models.json: schema, pins, packaged copy, base revisions."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from voxprint_forge import app_models
from voxprint_forge.errors import StageError
from voxprint_forge.lora import bases

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "model-forge.models.json"
SCHEMA = ROOT / "schemas" / "voxprint-app-models.schema.json"

REQUIRED = {
    "z-image": "04cc4abb",
    "huihui-qwen3-vl-4b-instruct-abliterated": "ce72a7c2",
    "mivolo-v2": "53393526",
    "yunet-2023mar": "47534e27",
}
OPTIONAL = {
    "florence2-large": "21a599d4",
    "moss-image-base": "1ceede80",
    "z-image-turbo": "f332072a",
    "dinov2-base": "f9e44c81",
}


def _doc() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_manifest_matches_the_schema() -> None:
    jsonschema.validate(_doc(), json.loads(SCHEMA.read_text(encoding="utf-8")))


def test_packaged_copy_is_identical() -> None:
    packaged = ROOT / "src" / "voxprint_forge" / "data" / "app-models.json"
    assert packaged.read_bytes() == MANIFEST.read_bytes()


def test_required_and_optional_pins() -> None:
    models = app_models.load()
    assert {m.id: m.revision[:8] for m in models if m.required} == REQUIRED
    assert {m.id: m.revision[:8] for m in models if not m.required} == OPTIONAL
    assert [m.id for m in models][:4] == list(REQUIRED)


def test_every_model_is_permissive_and_carries_no_weights_in_git() -> None:
    for model in app_models.load(MANIFEST):
        assert model.license in app_models.PERMISSIVE, model.id
        assert all(len(str(f["sha256"])) == 64 for f in model.files)
    assert "qwen-image-edit" not in MANIFEST.read_text(encoding="utf-8").lower()


def test_bases_use_the_manifest_revision() -> None:
    pins = {m.id: m for m in app_models.load()}
    for mdl_id in ("z-image", "moss-image-base"):
        info = bases._REGISTRY[mdl_id].info
        assert info.revision == pins[mdl_id].revision
        assert info.repo == pins[mdl_id].source


def test_missing_required_reads_the_manifest() -> None:
    assert app_models.missing_required(lambda mdl_id: mdl_id != "mivolo-v2") == ["mivolo-v2"]
    assert app_models.missing_required(lambda mdl_id: mdl_id not in OPTIONAL) == []


VL_ID = "huihui-qwen3-vl-4b-instruct-abliterated"
VL_REVISION = "ce72a7c22aacb493fb94478de3bfbe834c61844a"
VL_REPO = "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated"
VL_SIZE = 8_891_670_863
# Loadable files at the pinned revision. The card and .gitattributes are not in the app manifests.
VL_FILES = {
    "added_tokens.json": (707, "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680"),
    "chat_template.jinja": (5292, "3636d0f0bd6bef02654cdffdc447b79cb2cef8ab02cc75267345946291a489e4"),
    "config.json": (1548, "3d3fd7a91d2399f1ee98d64b142d18d698237fa73f84b97be3f75553e900885c"),
    "generation_config.json": (213, "4d9818c3d27895c0058828a5f68bc7a4de80c3ae1bcdac90936ce24178063f59"),
    "merges.txt": (1_671_853, "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5"),
    "model-00001-of-00002.safetensors": (
        4_990_497_880,
        "bbe234f80ac7e829b56b26028a893d9d266520a24132e8cde018986b778bc9f2",
    ),
    "model-00002-of-00002.safetensors": (
        3_885_221_448,
        "b1dc38b900dba6dd2841882efb9112c078ad1c3efa186a85f557a7ee0a716cbd",
    ),
    "model.safetensors.index.json": (64_778, "c321e968083cc64dbdc77b9cc17842396cf376399f4ff4bca6b0a001fd22afef"),
    "preprocessor_config.json": (782, "93585062a80db5e8ca038efc7726a3e6411d9db948472d81d63c6303993be8c5"),
    "special_tokens_map.json": (613, "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd"),
    "tokenizer.json": (11_422_654, "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    "tokenizer_config.json": (5_445, "81ec7bb9530159b326c0bef1d0b6c33d392090524014ea3f0123a3c1eb9c2af5"),
    "video_preprocessor_config.json": (817, "59c5c9eb52182eb14c06ffb10ca9effd29adce5f238a95de23ca14a38dbd2cb1"),
    "vocab.json": (2_776_833, "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910"),
}


def test_vl_manifest_rows_match_the_pin() -> None:
    packaged = ROOT / "src" / "voxprint_forge" / "data" / "app-models.json"
    for path in (MANIFEST, packaged):
        model = next(item for item in json.loads(path.read_text(encoding="utf-8"))["models"] if item["id"] == VL_ID)
        assert model["revision"] == VL_REVISION
        assert model["repo"] == VL_REPO
        assert model["license"] == "apache-2.0"
        got = {item["path"]: (item["size"], item["sha256"]) for item in model["files"]}
        assert got == VL_FILES
        assert model["size"] == VL_SIZE == sum(size for size, _digest in VL_FILES.values())


def test_parse_rejects_a_branch_or_a_wrong_size() -> None:
    doc = _doc()
    doc["models"][0]["revision"] = "main"
    with pytest.raises(StageError, match="full commit"):
        app_models.parse(doc)
    doc = _doc()
    doc["models"][0]["size"] += 1
    with pytest.raises(StageError, match="size"):
        app_models.parse(doc)
    doc = _doc()
    doc["optional"].append(doc["required"][0])
    with pytest.raises(StageError, match="both"):
        app_models.parse(doc)
