"""voxprint-model.json, the index commands, and the Plotweaver export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from tests.complete_card import complete_card
from voxprint_forge.cli import main
from voxprint_forge.diffusion.int4.publish import publish_verified
from voxprint_forge.diffusion.publish import _publish_hub
from voxprint_forge.errors import EXIT_USAGE, StageError
from voxprint_forge.hashing import sha256_file
from voxprint_forge.layout import Layout
from voxprint_forge.manifest import write_run_manifest
from voxprint_forge.mdl import (
    INDEX_REPO,
    MANIFEST_NAME,
    SCHEMA_ID,
    Listing,
    RemoteFile,
    RemoteRepo,
    backfill,
    build_document,
    check_document,
    gb_to_gib,
    schema_path,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads(schema_path().read_text(encoding="utf-8"))


def _schema(document: dict[str, object]) -> None:
    jsonschema.Draft202012Validator.check_schema(SCHEMA)
    jsonschema.validate(document, SCHEMA)


def _sample(*, engine: str = "llama.cpp", min_gb: float = 16, rec_gb: float = 24) -> dict[str, object]:
    version = {"llama.cpp": "b11476", "diffusers": ">=0.37.1,<0.38", "nunchaku": ">=1.0"}[engine]
    task = {"llama.cpp": "text-generation", "diffusers": "text-to-image", "nunchaku": "image-to-image"}[engine]
    apps = ("plotweaver", "audiobook-builder") if task == "text-generation" else ("canvas",)
    flags = ("least-censored",) if engine == "llama.cpp" else ()
    return build_document(
        model_id="t-lite-it-2.1",
        display_name="VOXPrint t-lite-it-2.1",
        app_ids=apps,
        task=task,
        license="Apache-2.0",
        files=[{"path": "model.gguf", "size": 4, "sha256": "ab" * 32}],
        mirrors=[{"kind": "hf", "ref": "Mitroshenkov87/voxprint-t-lite-it-2.1", "revision": "main"}],
        min_vram_gb=min_gb,
        recommended_vram_gb=rec_gb,
        gpu_min_cc=6.0 if engine == "llama.cpp" else 8.0,
        engine=engine,
        version_req=version,
        base_model="org/base",
        card_url="https://huggingface.co/Mitroshenkov87/voxprint-t-lite-it-2.1",
        flags=flags,
        on_demand=True,
        date="2026-10-11",
    )


def test_vram_uses_mdl_gib() -> None:
    assert gb_to_gib(24) == 23.5
    assert gb_to_gib(16) == 15.5
    document = _sample()
    assert document["tier"] == {
        "min_vram_gib": 15.5,
        "recommended_vram_gib": 23.5,
        "gpu_min_cc": 6.0,
    }


def test_built_documents_match_the_schema() -> None:
    for engine in ("llama.cpp", "diffusers", "nunchaku"):
        document = _sample(engine=engine)
        _schema(document)
        check_document(document)
        assert document["schema"] == SCHEMA_ID
        assert document["license"] == "apache-2.0"
        assert "licence" not in document
        assert document["files"][0]["path"] == "model.gguf"
        assert MANIFEST_NAME not in {item["path"] for item in document["files"]}


def test_schema_and_checker_reject_a_missing_license() -> None:
    document = _sample()
    del document["license"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(document, SCHEMA)
    with pytest.raises(StageError):
        check_document(document)


def test_plotweaver_export_still_writes_the_run_manifest(tmp_path: Path) -> None:
    result = write_run_manifest(Layout(tmp_path), {"models": []}, date="2026-10-11")
    assert (tmp_path / "manifests" / "2026-10-11-a100.json").is_file()
    assert (tmp_path / "manifests" / "2026-10-11-a100.MODELS.md").is_file()
    assert "voxprint-model.json" not in result["instructions"]


def test_index_build_writes_local_index_and_does_not_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr("voxprint_forge.mdl._upload_index", lambda repo, path: called.append(repo))
    folder = tmp_path / "models" / "tiny"
    folder.mkdir(parents=True)
    document = _sample()
    path = folder / MANIFEST_NAME
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    dest = tmp_path / "index.json"
    assert main(["index", "build", "--from", str(tmp_path / "models"), "--out", str(dest)]) == 0
    rows = json.loads(dest.read_text(encoding="utf-8"))
    assert rows == [
        {
            "id": "t-lite-it-2.1",
            "repo": "Mitroshenkov87/voxprint-t-lite-it-2.1",
            "revision": "main",
            "sha256": sha256_file(path),
        }
    ]
    assert called == []


def test_index_push_requires_the_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr("voxprint_forge.mdl._upload_index", lambda repo, path: called.append(repo))
    index = tmp_path / "index.json"
    index.write_text("[]\n", encoding="utf-8")
    assert main(["index", "push", "--index", str(index)]) == EXIT_USAGE
    assert called == []
    assert main(["index", "push", "--push", "--index", str(index)]) == 0
    assert called == [INDEX_REPO]


def test_backfill_writes_local_files_and_does_not_upload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr("voxprint_forge.mdl._upload_index", lambda repo, path: called.append(repo))
    digest = "cd" * 32
    repo = RemoteRepo(
        repo_id="Mitroshenkov87/voxprint-example",
        revision="abc123",
        license="apache-2.0",
        files=(RemoteFile("weights/model.gguf", 3, digest),),
        base_model="org/example",
        task="text-generation",
    )
    skipped = RemoteRepo(
        repo_id="Mitroshenkov87/voxprint-model-index",
        revision="main",
        license="apache-2.0",
        files=(RemoteFile("index.json", 2, digest),),
    )

    def fake_list(user: str = "Mitroshenkov87") -> Listing:
        assert user == "Mitroshenkov87"
        return Listing((repo, skipped), (("Mitroshenkov87/voxprint-bare", "no licence tag"),))

    monkeypatch.setattr("voxprint_forge.mdl.list_published_repos", fake_list)
    out = tmp_path / "backfill"
    assert main(["index", "backfill", "--out", str(out)]) == 0
    written = out / "Mitroshenkov87" / "voxprint-example" / MANIFEST_NAME
    document = json.loads(written.read_text(encoding="utf-8"))
    _schema(document)
    assert document["id"] == "example"
    assert document["files"] == [{"path": "weights/model.gguf", "size": 3, "sha256": digest}]
    assert document["mirrors"] == [{"kind": "hf", "ref": repo.repo_id, "revision": "abc123"}]
    assert not (out / "Mitroshenkov87" / "voxprint-model-index").exists()
    assert called == []
    assert backfill((skipped,), tmp_path / "direct") == []


class _Hub:
    def __init__(self) -> None:
        self.hf_bytes: dict[str, bytes] = {}
        self.kaggle_bytes: dict[str, bytes] = {}

    def hf_private_repo(self, repo_id: str) -> str:
        return f"https://huggingface.co/{repo_id}"

    def hf_upload(self, repo_id: str, path: Path, name: str) -> None:
        del repo_id
        self.hf_bytes[name] = path.read_bytes()

    def hf_stat(self, repo_id: str, name: str) -> tuple[str, int]:
        del repo_id
        payload = self.hf_bytes[name]
        return hashlib.sha256(payload).hexdigest(), len(payload)

    def kaggle_private_model(self, owner: str, slug: str, folder: Path) -> str:
        del owner, slug, folder
        return "https://www.kaggle.com/models/example"

    def kaggle_upload_variation(self, owner: str, slug: str, variation: str, folder: Path) -> str:
        for path in folder.iterdir():
            if path.is_file() and path.name not in {"model-instance-metadata.json", "model-metadata.json"}:
                self.kaggle_bytes[path.name] = path.read_bytes()
        return f"{owner}/{slug}/gguf/{variation}"

    def kaggle_stat(self, handle: str, name: str) -> tuple[str, int]:
        del handle
        payload = self.kaggle_bytes[name]
        return hashlib.sha256(payload).hexdigest(), len(payload)


def test_canvas_publish_uploads_the_manifest_next_to_the_card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_USER", "mitroshenkov87")
    monkeypatch.setattr("voxprint_forge.diffusion.publish.place_weight", lambda path, digest=None: None)
    output = tmp_path / "out"
    output.mkdir()
    (output / "README.md").write_text(complete_card(), encoding="utf-8")
    (output / "small.safetensors").write_bytes(b"small")
    (output / "large.safetensors").write_bytes(b"large-weights")
    remote = _Hub()
    _publish_hub(
        output=output,
        hf_repo="Mitroshenkov87/voxprint-mirror-z-image-turbo",
        kaggle_slug="voxprint-mirror-z-image-turbo",
        title="VOXPrint Z-Image Turbo tiers",
        description="Apache-2.0.",
        tiers={
            "small.safetensors": ("16gb", 16, "Tongyi-MAI/Z-Image-Turbo", "rev"),
            "large.safetensors": ("24gb", 24, "Tongyi-MAI/Z-Image-Turbo", "rev"),
        },
        variations={"16gb": ["small.safetensors"], "24gb": ["large.safetensors"]},
        repo="Tongyi-MAI/Z-Image-Turbo",
        revision="rev",
        licence="apache-2.0",
        remote=remote,  # type: ignore[arg-type]
        sleep=lambda _seconds: None,
        attempts=1,
        card_text=complete_card(),
        day="2026-10-11",
    )
    assert MANIFEST_NAME in remote.hf_bytes
    assert MANIFEST_NAME in remote.kaggle_bytes
    document = json.loads(remote.hf_bytes[MANIFEST_NAME])
    _schema(document)
    assert document["tier"]["min_vram_gib"] == 15.5
    assert document["tier"]["recommended_vram_gib"] == 23.5
    assert document["runtime"]["engine"] == "diffusers"
    assert document["app_ids"] == ["canvas"]
    paths = {item["path"] for item in document["files"]}
    assert MANIFEST_NAME not in paths
    assert "small.safetensors" in paths
    assert "README.md" in paths


def test_int4_drive_fallback_drops_the_hub_mirror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_USER", "Mitroshenkov87")
    (tmp_path / "weight.bin").write_bytes(b"weights")
    seen: dict[str, object] = {}

    def hub(_repo: str, files: list[object]) -> list[dict[str, object]]:
        seen["hub"] = [getattr(item, "name") for item in files]
        raise StageError("hub down", code="upload_mismatch")

    def drive(_folder: str, files: list[object]) -> list[dict[str, object]]:
        manifest = next(item for item in files if getattr(item, "name") == MANIFEST_NAME)
        seen["drive"] = json.loads(Path(getattr(manifest, "path")).read_text(encoding="utf-8"))
        return [{"name": MANIFEST_NAME, "target": "drive"}]

    published = publish_verified(
        tmp_path,
        slug="voxprint-example",
        card_text=complete_card(),
        licence="apache-2.0",
        hub=hub,
        drive=drive,
        date="2026-10-11",
    )
    assert published.verified is True
    assert published.hf == "failed"
    assert published.drive == "uploaded"
    assert MANIFEST_NAME in seen["hub"]
    document = seen["drive"]
    assert isinstance(document, dict)
    _schema(document)
    assert [item["kind"] for item in document["mirrors"]] == ["gdrive"]
    assert document["mirrors"][0]["ref"] == "Voxprint/model-forge/2026-10-11/voxprint-example"
    assert MANIFEST_NAME not in {item["path"] for item in document["files"]}
    assert (ROOT / "schemas" / "voxprint-model.schema.json").is_file()
