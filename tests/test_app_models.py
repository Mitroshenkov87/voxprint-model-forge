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
    "qwen3-vl-4b-instruct": "ebb281ec",
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
