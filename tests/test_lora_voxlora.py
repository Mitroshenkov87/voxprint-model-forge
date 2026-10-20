"""The ``.voxlora`` package: round trip, digest, no source photos, stripped metadata."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import jsonschema
import pytest
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from safetensors.torch import load_file

from tests.lora_fakes import TinyAdapter, make_project
from tests.test_lora_export import header, trained
from voxprint_forge.cli import main
from voxprint_forge.lora import bases
from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.export import SCALE_DEFAULT, export_package


@pytest.fixture(autouse=True)
def tiny_base():
    bases.register(TinyAdapter)
    yield
    bases.unregister("tiny-test")


def _envelope(text: str) -> dict[str, object]:
    start = text.rindex("\n{") if "\n{" in text else text.index("{")
    return json.loads(text[start:])


def test_package_round_trip_digest_and_schema(tmp_path: Path) -> None:
    project = trained(tmp_path, preview=True)
    data = project.read()
    subject = dict(data["subject"])  # type: ignore[arg-type]
    subject.update({
        "kind": "person",
        "stage_name": "Mira",
        "consent_obtained": True,
        "adult_confirmed": True,
        "consent_method": "verbal",
    })
    project.write({**data, "subject": subject, "kind": "person"})
    marker = b"SOURCE-PHOTO-MARKER"
    photos = project.root / "photos"
    photos.mkdir()
    (photos / "source.png").write_bytes(marker)
    (project.prepared / "p00.png").write_bytes((project.prepared / "p00.png").read_bytes() + marker)
    info = PngInfo()
    info.add_text("comment", "UserPhotoMetaSecret")
    wide = Image.new("RGB", (1400, 900), (20, 40, 60))
    wide.save(project.previews / "wide.png", pnginfo=info)
    result = export_package(project, Progress())
    target = Path(str(result["file"]))
    assert target.name == "Mira.voxlora"
    blob = target.read_bytes()
    assert marker not in blob
    assert b"UserPhotoMetaSecret" not in blob
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        assert all(item.compress_type == zipfile.ZIP_STORED for item in archive.infolist())
        names = archive.namelist()
        assert names[0] == "model.safetensors"
        assert "voxprint-model.json" in names
        assert "preview.webp" in names
        extras = [name for name in names if name.startswith("previews/")]
        assert extras and len(extras) <= 6
        weights = archive.read("model.safetensors")
        doc = json.loads(archive.read("voxprint-model.json"))
        preview = archive.read("preview.webp")
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "docs" / "schemas" / "voxprint-model-1.json").read_text()
    )
    jsonschema.validate(doc, schema)
    assert doc["weights_sha256"] == hashlib.sha256(weights).hexdigest()
    assert doc["base"] == "tiny-test"
    assert doc["kind"] == "subject"
    assert doc["scale_default"] == SCALE_DEFAULT
    assert doc["scale_min"] == 0.0 and doc["scale_max"] == 1.2
    assert doc["name"] == "Mira"
    assert doc["steps"] == 4
    assert set(doc["attest"]) == {"consent_obtained", "subject_adult"}
    assert doc["attest"]["consent_obtained"] is True and doc["attest"]["subject_adult"] is True
    assert "consent_method" not in json.dumps(doc)
    with Image.open(io.BytesIO(preview)) as image:
        assert image.format == "WEBP"
        assert max(image.size) <= 1024
        assert "UserPhotoMetaSecret" not in image.info
    unpacked = tmp_path / "model.safetensors"
    unpacked.write_bytes(weights)
    tensors = load_file(str(unpacked))
    assert tensors and all(key.endswith((".lora_A.weight", ".lora_B.weight")) for key in tensors)
    meta = header(unpacked)
    assert any(key.startswith("voxprint.subject.") for key in meta)
    assert any(key.startswith("voxprint.lora.") for key in meta)
    assert all(key.startswith(("voxprint.subject.", "voxprint.lora.")) for key in meta)


def test_thing_maps_to_object_and_cli_package_is_separate_from_safetensors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    project = trained(tmp_path, preview=False)
    data = project.read()
    subject = dict(data["subject"])  # type: ignore[arg-type]
    subject["kind"] = "thing"
    project.write({**data, "subject": subject, "kind": "thing"})
    out = tmp_path / "pack.voxlora"
    capsys.readouterr()
    assert main(["--json", "lora", "export", str(project.root), "-o", str(out)]) == 0
    packed = _envelope(capsys.readouterr().out)["result"]
    assert packed["state"] == "done"
    with zipfile.ZipFile(out) as archive:
        doc = json.loads(archive.read("voxprint-model.json"))
    assert doc["kind"] == "object"
    assert main(["--json", "lora", "export", str(project.root), "--project", str(project.root)]) == 2
    both = _envelope(capsys.readouterr().out)
    assert both["error"]["code"] == "lora.err.export_project"
    assert main(["--json", "lora", "export", "--project", str(project.root)]) == 0
    saved = project.root / "export"
    assert any(path.suffix == ".safetensors" for path in saved.iterdir())
    assert main(["--json", "lora", "export"]) == 2


def test_package_refuses_a_project_without_a_checkpoint(tmp_path: Path) -> None:
    from voxprint_forge.lora.errors import LoraUsageError

    project = make_project(tmp_path / "empty")
    with pytest.raises(LoraUsageError) as missing:
        export_package(project, Progress())
    assert missing.value.code == "lora.err.no_checkpoint"
