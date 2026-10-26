"""voxprint-model.json, the index commands, and the Plotweaver export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from tests.complete_card import complete_card
from voxprint_forge.cli import main
from voxprint_forge.diffusion.int4.publish import publish_verified
from voxprint_forge.diffusion.publish import _publish_hub
from voxprint_forge.errors import EXIT_STAGE, EXIT_USAGE, StageError
from voxprint_forge.hashing import sha256_file
from voxprint_forge.layout import Layout
from voxprint_forge.manifest import write_run_manifest
from voxprint_forge.mdl import (
    HOSTED_REPO,
    INDEX_REPO,
    MANIFEST_NAME,
    SCHEMA_ID,
    Listing,
    RemoteFile,
    RemoteRepo,
    backfill,
    build_document,
    check_document,
    descriptor_commit,
    gb_to_gib,
    host_descriptors,
    hosted_path,
    index_document,
    is_hosted,
    next_serial,
    per_file_sources,
    push_index,
    retarget_mirrors,
    schema_path,
    write_index,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads(schema_path().read_text(encoding="utf-8"))


def _schema(document: dict[str, object]) -> None:
    jsonschema.Draft202012Validator.check_schema(SCHEMA)
    jsonschema.validate(document, SCHEMA)


@pytest.fixture(autouse=True)
def _published_index(monkeypatch: pytest.MonkeyPatch) -> None:
    head = "15edeab0f7a874c27d2b38ef6ecb69106ff94df8"
    monkeypatch.setattr("voxprint_forge.mdl.published_index_state", lambda: (9, head))


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
    asked: list[str] = []

    def lookup(repo: str) -> str:
        asked.append(repo)
        return "ab" * 20

    monkeypatch.setattr("voxprint_forge.mdl.descriptor_commit", lookup)
    folder = tmp_path / "models" / "tiny"
    folder.mkdir(parents=True)
    document = _sample()
    path = folder / MANIFEST_NAME
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    dest = tmp_path / "index.json"
    assert main(["index", "build", "--from", str(tmp_path / "models"), "--out", str(dest)]) == 0
    rows = json.loads(dest.read_text(encoding="utf-8"))["models"]
    assert rows == [
        {
            "id": "t-lite-it-2.1",
            "repo": "Mitroshenkov87/voxprint-t-lite-it-2.1",
            "revision": "ab" * 20,
            "sha256": sha256_file(path),
        }
    ]
    assert asked == ["Mitroshenkov87/voxprint-t-lite-it-2.1"]
    assert called == []


def test_index_build_accepts_an_explicit_descriptor_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbid(repo: str) -> str:
        raise AssertionError(repo)

    monkeypatch.setattr("voxprint_forge.mdl.descriptor_commit", forbid)
    folder = tmp_path / "models" / "tiny"
    folder.mkdir(parents=True)
    path = folder / MANIFEST_NAME
    path.write_text(json.dumps(_sample()) + "\n", encoding="utf-8")
    dest = tmp_path / "index.json"
    pin = "Mitroshenkov87/voxprint-t-lite-it-2.1=" + ("cd" * 20)
    assert main(["index", "build", "--from", str(folder.parent), "--out", str(dest), "--revision", pin]) == 0
    rows = json.loads(dest.read_text(encoding="utf-8"))["models"]
    assert rows[0]["revision"] == "cd" * 20


def test_index_revision_must_be_a_full_commit(tmp_path: Path) -> None:
    dest = tmp_path / "index.json"
    code = main(
        [
            "index",
            "build",
            "--from",
            str(tmp_path),
            "--out",
            str(dest),
            "--revision",
            "Mitroshenkov87/voxprint-t-lite-it-2.1=main",
        ]
    )
    assert code == EXIT_USAGE
    assert not dest.exists()


def test_a_branch_name_is_not_a_descriptor_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("voxprint_forge.mdl.descriptor_commit", lambda repo: "main")
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / MANIFEST_NAME).write_text(json.dumps(_sample()) + "\n", encoding="utf-8")
    dest = tmp_path / "index.json"
    assert main(["index", "build", "--from", str(folder), "--out", str(dest)]) == EXIT_STAGE
    assert not dest.exists()


def test_unresolved_descriptor_commit_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(repo: str) -> str:
        raise StageError(f"{repo} descriptor commit could not be resolved", code="mdl")

    monkeypatch.setattr("voxprint_forge.mdl.descriptor_commit", missing)
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / MANIFEST_NAME).write_text(json.dumps(_sample()) + "\n", encoding="utf-8")
    dest = tmp_path / "index.json"
    assert main(["index", "build", "--from", str(folder), "--out", str(dest)]) == EXIT_STAGE
    assert not dest.exists()


def _signed_fixture(tmp_path: Path, serial: int = 10) -> tuple[Path, Path, bytes]:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat

    key = Ed25519PrivateKey.generate()
    key_file = tmp_path / "key.pem"
    key_file.write_bytes(key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))
    row = {"id": "tiny", "repo": "Mitroshenkov87/voxprint-tiny", "revision": "ab" * 20, "sha256": "cd" * 32}
    index = tmp_path / "index.json"
    document = index_document([row], serial=serial, head="15edeab", generated_at="2026-10-18T05:00:00Z")
    index.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    from cryptography.hazmat.primitives.serialization import PublicFormat

    public = key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return index, key_file, public


def test_index_push_requires_the_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import base64

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    called: list[tuple[str, str]] = []
    monkeypatch.setattr("voxprint_forge.mdl._upload_index", lambda repo, path, sig: called.append((repo, sig.name)))
    index, key_file, public = _signed_fixture(tmp_path)
    monkeypatch.setenv("VOXPRINT_MDL_INDEX_KEY_FILE", str(key_file))
    assert main(["index", "push", "--index", str(index)]) == EXIT_USAGE
    assert called == []
    assert main(["index", "push", "--push", "--index", str(index)]) == 0
    assert called == [(INDEX_REPO, "index.json.sig")]
    signature = base64.b64decode((tmp_path / "index.json.sig").read_bytes())
    Ed25519PublicKey.from_public_bytes(public).verify(signature, index.read_bytes())


def test_index_push_refuses_a_stale_serial_or_a_bare_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []
    monkeypatch.setattr("voxprint_forge.mdl._upload_index", lambda repo, path, sig: called.append(repo))
    index, key_file, _ = _signed_fixture(tmp_path, serial=11)
    with pytest.raises(StageError):
        push_index(index, key_file=key_file, published_serial=11)
    index.write_text("[]\n", encoding="utf-8")
    with pytest.raises(StageError):
        push_index(index, key_file=key_file, published_serial=None)
    assert called == []
    assert not (tmp_path / "index.json.sig").exists()


def test_index_serial_follows_the_published_one() -> None:
    # voxprint-common 0.5.0 bundles serial 9, so nothing below 10 is published.
    assert next_serial(None) == 10
    assert next_serial(1) == 10
    assert next_serial(9) == 10
    assert next_serial(12) == 13


def test_strict_priority_sources_and_030_fields_are_accepted() -> None:
    document = _sample()
    ours = {"kind": "hf", "ref": "Mitroshenkov87/voxprint-t-lite-it-2.1", "revision": "ab" * 20}
    digest = "ab" * 32
    document["mirror_policy"] = "strict-priority"
    document["files"][0]["sources"] = [{"kind": "hf", "ref": "author/repo", "revision": "cd" * 20}, ours]
    document["file_groups"] = [{"paths": ["a.gguf"], "mirror_policy": "race", "sources": [ours]}]
    document["files"][0]["onnx_prebuilt"] = {
        "path": "a.onnx",
        "sha256": digest,
        "size": 12,
        "url": "https://example.com/a.onnx",
        "built_for": {"onnxruntime": "1.22", "opset": 17},
    }
    document["prebuilt"] = [
        {
            "id": "fallback-a",
            "role": "fallback",
            "replaces": "a.gguf",
            "files": [{"path": "fallback/a.gguf", "size": 3, "sha256": digest}],
        }
    ]
    document["post_install"] = {
        "cli": "voxprint-forge-cli",
        "args": ["--prepare-model", "{id}"],
        "timeout_s": 60,
        "outputs": [{"path": "ready.flag"}],
    }
    document["runtime"]["engine"] = "onnxruntime"
    check_document(document)
    _schema(document)
    document["mirror_policy"] = "fastest"
    with pytest.raises(StageError):
        check_document(document)


def test_build_document_emits_onnx_prebuilt() -> None:
    digest = "ab" * 32
    artifact = {
        "path": "weights.onnx",
        "size": 8,
        "sha256": digest,
        "onnx_prebuilt": {
            "path": "weights.prebuilt.onnx",
            "sha256": digest,
            "url": "https://example.com/weights.prebuilt.onnx",
            "built_for": {"onnxruntime": ">=1.22", "opset": 17},
        },
    }
    document = build_document(
        model_id="onnx-demo",
        display_name="ONNX demo",
        app_ids=["model-forge"],
        task="image-to-image",
        license="apache-2.0",
        files=[artifact],
        mirrors=[{"kind": "hf", "ref": "Mitroshenkov87/voxprint-onnx-demo", "revision": "ab" * 20}],
        min_vram_gb=4,
        recommended_vram_gb=8,
        gpu_min_cc=8.0,
        engine="onnxruntime",
        version_req=">=1.22",
        base_model="org/example",
        card_url="",
        onnx_prebuilt={
            "path": "model.onnx",
            "sha256": digest,
            "url": "https://example.com/model.onnx",
            "built_for": {"onnxruntime": "1.22.0", "opset": 18},
        },
        post_install={"cli": "voxprint-forge-cli", "args": ["mdl", "prepare", "{id}", "{model_dir}"]},
    )
    assert "onnx_prebuilt" in document["files"][0]
    assert document["files"][0]["onnx_prebuilt"]["path"] == "weights.prebuilt.onnx"
    assert document["onnx_prebuilt"]["path"] == "model.onnx"
    assert document["post_install"]["cli"] == "voxprint-forge-cli"
    _schema(document)


def test_bundled_serial_nine_index_still_parses() -> None:
    import hashlib
    from importlib import resources

    from voxprint_common.mdl.index import load_index

    pin = json.loads(
        resources.files("voxprint_common.mdl.data").joinpath("model-index.pin.json").read_text(encoding="utf-8")
    )
    raw = resources.files("voxprint_common.mdl.data").joinpath("model-index.json").read_bytes()
    document = json.loads(raw.decode("utf-8"))
    assert pin["serial"] == 9
    assert document["serial"] == 9
    assert document["head"] == pin["head"]
    assert document["generated_at"] == pin["generated_at"]
    assert isinstance(document["models"], list) and document["models"]
    assert hashlib.sha256(raw).hexdigest() == pin["sha256"]
    bundled = load_index(remote=False)
    assert bundled.source == "bundled"
    assert bundled.serial == 9
    assert len(bundled.entries) == len(document["models"])


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


# Full rows from dataset commit d2325ed. The vendored descriptors store a later
# weights commit on our mirror. The index revision is the descriptor commit.
DERIVATIVE_ROWS = [
    {
        "id": "gemma4-31b",
        "repo": "Mitroshenkov87/voxprint-gemma4-31b",
        "revision": "674f6bf503aaf4684a52ec6c53ce00158f25c544",
        "sha256": "4c53e3fe900639860827cd93ce0e8e31cf163dfbf9d1877e504f5b436ebf0b3c",
    },
    {
        "id": "huihui-qwen36-27b",
        "repo": "Mitroshenkov87/voxprint-huihui-qwen36-27b",
        "revision": "e18da4c2633c2b9206779bff7d223a14921aa60a",
        "sha256": "2050fd29586f19a9e61b0cef12c7c4f051c5013f2276c752f3571f2a7743c33d",
    },
]
_WEIGHTS_COMMITS = (
    "26495e1a45d8bde085a0681c3c43044eff5c8ac5",
    "cc596c1a74791b7fca57d037ef168dd57c76ed34",
)


def test_index_build_reproduces_the_derivative_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = ROOT / "tests" / "fixtures" / "descriptors"
    asked: list[str] = []

    def lookup(repo: str) -> str:
        asked.append(repo)
        pinned = {row["repo"]: row["revision"] for row in DERIVATIVE_ROWS}
        if repo not in pinned:
            raise StageError(f"{repo} descriptor commit could not be resolved", code="mdl")
        return pinned[repo]

    monkeypatch.setattr("voxprint_forge.mdl.descriptor_commit", lookup)
    for row in DERIVATIVE_ROWS:
        path = source / f"voxprint-{row['id']}" / MANIFEST_NAME
        document = json.loads(path.read_text(encoding="utf-8"))
        _schema(document)
        assert sha256_file(path) == row["sha256"]
        assert document["mirrors"][0]["ref"] != row["repo"]
        assert document["mirrors"][1]["ref"] == row["repo"]
        assert document["mirrors"][1]["revision"] in _WEIGHTS_COMMITS
    dest = tmp_path / "index.json"
    assert main(["index", "build", "--from", str(source), "--out", str(dest)]) == 0
    text = dest.read_text(encoding="utf-8")
    assert json.loads(text)["models"] == DERIVATIVE_ROWS
    assert asked == [row["repo"] for row in DERIVATIVE_ROWS]
    for weights in _WEIGHTS_COMMITS:
        assert weights not in text


def test_explicit_revisions_skip_the_descriptor_lookup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbid(repo: str) -> str:
        raise AssertionError(repo)

    monkeypatch.setattr("voxprint_forge.mdl.descriptor_commit", forbid)
    source = ROOT / "tests" / "fixtures" / "descriptors"
    dest = tmp_path / "index.json"
    argv = ["index", "build", "--from", str(source), "--out", str(dest)]
    for row in DERIVATIVE_ROWS:
        argv.extend(["--revision", f"{row['repo']}={row['revision']}"])
    assert main(argv) == 0
    assert json.loads(dest.read_text(encoding="utf-8"))["models"] == DERIVATIVE_ROWS


def test_descriptor_commit_reads_the_published_file(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakeApi:
        def get_paths_info(self, repo_id: str, paths: list[str], **kwargs: object) -> list[SimpleNamespace]:
            seen["repo"] = repo_id
            seen["paths"] = paths
            seen["kwargs"] = kwargs
            return [SimpleNamespace(last_commit=SimpleNamespace(oid=DERIVATIVE_ROWS[0]["revision"]))]

    monkeypatch.setattr("voxprint_forge.mdl.HfApi", FakeApi)
    assert descriptor_commit("Mitroshenkov87/voxprint-gemma4-31b") == DERIVATIVE_ROWS[0]["revision"]
    assert seen["paths"] == [MANIFEST_NAME]
    assert seen["kwargs"] == {"repo_type": "model", "expand": True, "token": False}


def test_descriptor_commit_fails_without_a_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeApi:
        def get_paths_info(self, repo_id: str, paths: list[str], **kwargs: object) -> list[SimpleNamespace]:
            del repo_id, paths, kwargs
            return [SimpleNamespace(last_commit=None)]

    monkeypatch.setattr("voxprint_forge.mdl.HfApi", FakeApi)
    with pytest.raises(StageError) as caught:
        descriptor_commit("Mitroshenkov87/voxprint-gemma4-31b")
    assert caught.value.code == "mdl"
    assert "could not be resolved" in str(caught.value)


def test_descriptor_commit_scrubs_a_token(monkeypatch: pytest.MonkeyPatch) -> None:
    token = "hf_" + "ab" * 8
    monkeypatch.setenv("HF_TOKEN", token)

    class FakeApi:
        def get_paths_info(self, repo_id: str, paths: list[str], **kwargs: object) -> list[SimpleNamespace]:
            del repo_id, paths, kwargs
            raise RuntimeError(f"denied for {token}")

    monkeypatch.setattr("voxprint_forge.mdl.HfApi", FakeApi)
    with pytest.raises(StageError) as caught:
        descriptor_commit("Mitroshenkov87/voxprint-gemma4-31b")
    assert caught.value.code == "mdl"
    assert token not in str(caught.value)


def _author_only(tmp_path: Path, model_id: str = "t-lite-it-2.1") -> Path:
    document = _sample()
    document["id"] = model_id
    document["mirrors"] = [
        {
            "kind": "hf",
            "ref": "paperscarecrow/Gemma-4-31B-it-abliterated",
            "revision": "2491ee58472b82a2372d288f982943b9d5e6a82c",
        }
    ]
    document["card_url"] = "https://huggingface.co/paperscarecrow/Gemma-4-31B-it-abliterated"
    folder = tmp_path / model_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / MANIFEST_NAME
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    return path


def test_author_only_descriptor_is_hosted_in_the_index_dataset(tmp_path: Path) -> None:
    path = _author_only(tmp_path)
    asked: list[str] = []

    def lookup(repo: str) -> str:
        asked.append(repo)
        return "cd" * 20

    rows = write_index([path], tmp_path / "index.json", commits=lookup, serial=11, head="ab" * 20)
    assert rows == [
        {
            "id": "t-lite-it-2.1",
            "repo": "datasets/Mitroshenkov87/voxprint-model-index",
            "revision": "cd" * 20,
            "sha256": sha256_file(path),
        }
    ]
    assert asked == ["datasets/Mitroshenkov87/voxprint-model-index/descriptors/t-lite-it-2.1/voxprint-model.json"]


def test_hosted_rows_take_one_dataset_pin(tmp_path: Path) -> None:
    paths = [_author_only(tmp_path, "a-model"), _author_only(tmp_path, "b-model")]

    def forbid(repo: str) -> str:
        raise AssertionError(repo)

    rows = write_index(
        paths,
        tmp_path / "index.json",
        revisions={HOSTED_REPO: "ef" * 20},
        commits=forbid,
        serial=11,
        head="ab" * 20,
    )
    assert [(row["id"], row["repo"], row["revision"]) for row in rows] == [
        ("a-model", HOSTED_REPO, "ef" * 20),
        ("b-model", HOSTED_REPO, "ef" * 20),
    ]


def test_unresolved_hosted_descriptor_writes_nothing(tmp_path: Path) -> None:
    path = _author_only(tmp_path)

    def missing(repo: str) -> str:
        raise StageError(f"{repo} descriptor commit could not be resolved", code="mdl")

    with pytest.raises(StageError) as caught:
        write_index([path], tmp_path / "index.json", commits=missing, serial=11, head="ab" * 20)
    assert caught.value.code == "mdl"
    assert not (tmp_path / "index.json").exists()


def test_a_drive_only_descriptor_is_not_hosted(tmp_path: Path) -> None:
    document = _sample()
    document["mirrors"] = [{"kind": "gdrive", "ref": "folder-id", "revision": ""}]
    assert not is_hosted(document)
    assert is_hosted(json.loads(_author_only(tmp_path).read_text(encoding="utf-8")))
    assert not is_hosted(_sample())


def test_descriptor_commit_reads_a_hosted_file_from_the_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakeApi:
        def get_paths_info(self, repo_id: str, paths: list[str], **kwargs: object) -> list[SimpleNamespace]:
            seen.update(repo=repo_id, paths=paths, kwargs=kwargs)
            return [SimpleNamespace(last_commit=SimpleNamespace(oid="cd" * 20))]

    monkeypatch.setattr("voxprint_forge.mdl.HfApi", FakeApi)
    assert descriptor_commit(hosted_path("z-image")) == "cd" * 20
    assert seen == {
        "repo": "Mitroshenkov87/voxprint-model-index",
        "paths": ["descriptors/z-image/voxprint-model.json"],
        "kwargs": {"repo_type": "dataset", "expand": True, "token": False},
    }


def test_host_uploads_only_official_source_descriptors(tmp_path: Path) -> None:
    hosted = _author_only(tmp_path / "src", "z-image")
    ours = tmp_path / "src" / "ours" / MANIFEST_NAME
    ours.parent.mkdir(parents=True)
    ours.write_text(json.dumps(_sample()) + "\n", encoding="utf-8")
    sent: list[tuple[list[tuple[str, Path]], str]] = []

    def upload(items: list[tuple[str, Path]], message: str) -> str:
        sent.append((items, message))
        return "12" * 20

    result = host_descriptors(tmp_path / "src", upload=upload)
    assert result == {
        "repo": HOSTED_REPO,
        "commit": "12" * 20,
        "paths": ["descriptors/z-image/voxprint-model.json"],
        "pushed": True,
    }
    assert sent[0][0] == [("descriptors/z-image/voxprint-model.json", hosted)]


def test_host_needs_push(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[object] = []
    monkeypatch.setattr("voxprint_forge.mdl._upload_descriptors", lambda *args: called.append(args))
    _author_only(tmp_path)
    assert main(["index", "host", "--from", str(tmp_path)]) == 2
    assert called == []


def test_hosted_path_rejects_a_bad_id() -> None:
    with pytest.raises(StageError):
        hosted_path("../escape")


def test_explicit_ours_is_the_index_repo(tmp_path: Path) -> None:
    document = _sample()
    document["mirrors"] = [
        {"kind": "hf", "ref": "Mitroshenkov87/voxprint-t-lite-it-2.1", "revision": "main"},
        {"kind": "hf", "ref": "example/held-copy", "revision": "abc123", "ours": True},
    ]
    document["card_url"] = "https://huggingface.co/example/held-copy"
    _schema(document)
    path = tmp_path / MANIFEST_NAME
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    pin = "e" * 40

    def forbid(repo: str) -> str:
        raise AssertionError(repo)

    rows = write_index([path], tmp_path / "index.json", revisions={"example/held-copy": pin}, commits=forbid)
    assert rows[0] == {
        "id": "t-lite-it-2.1",
        "repo": "example/held-copy",
        "revision": pin,
        "sha256": sha256_file(path),
    }


def test_two_owned_mirrors_fail(tmp_path: Path) -> None:
    document = _sample()
    document["mirrors"] = [
        {"kind": "hf", "ref": "Mitroshenkov87/voxprint-a", "revision": "main"},
        {"kind": "hf", "ref": "Mitroshenkov87/voxprint-b", "revision": "main"},
    ]
    path = tmp_path / MANIFEST_NAME
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    with pytest.raises(StageError) as caught:
        write_index([path], tmp_path / "index.json")
    assert caught.value.code == "mdl"
    assert "more than one forge-owned" in str(caught.value)


def test_retarget_card_url_uses_our_repo(tmp_path: Path) -> None:
    path = tmp_path / MANIFEST_NAME
    path.write_text(json.dumps(_sample()) + "\n", encoding="utf-8")
    retarget_mirrors(
        path,
        [
            {
                "kind": "hf",
                "ref": "huihui-ai/Huihui-Qwen3.6-27B-abliterated-MTP-GGUF",
                "revision": "2da8c0432f0a42c25c14cbc99f90028a90d3dcd3",
            },
            {
                "kind": "hf",
                "ref": "Mitroshenkov87/voxprint-huihui-qwen36-27b",
                "revision": "cc596c1a74791b7fca57d037ef168dd57c76ed34",
            },
        ],
    )
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["card_url"] == "https://huggingface.co/Mitroshenkov87/voxprint-huihui-qwen36-27b"
    retarget_mirrors(
        path,
        [{"kind": "gdrive", "ref": "Voxprint/model-forge/2026-10-11/tiny", "revision": ""}],
    )
    cleared = json.loads(path.read_text(encoding="utf-8"))
    assert cleared["card_url"] == ""


def test_per_file_sources_put_the_author_first_only_where_the_bytes_match() -> None:
    document = _sample()
    author = {"kind": "hf", "ref": "author/repo", "revision": "cd" * 20}
    document["mirrors"] = [author, *document["mirrors"]]
    first = document["files"][0]["path"]
    result = per_file_sources(document, [first])
    assert result["mirror_policy"] == "strict-priority"
    assert result["files"][0]["sources"][0] == author
    assert result["files"][0]["sources"][1]["ref"].startswith("Mitroshenkov87/")
    assert result["mirrors"] == document["mirrors"]
    for before, after in zip(document["files"], result["files"], strict=True):
        assert {key: after[key] for key in ("path", "size", "sha256")} == before
    with pytest.raises(StageError):
        per_file_sources(document, ["missing.gguf"])


def test_an_abliterated_name_is_not_a_least_censored_claim() -> None:
    import dataclasses

    from voxprint_forge.mdl import document_from_repo

    repo = RemoteRepo(
        repo_id="Mitroshenkov87/voxprint-example-abliterated",
        revision="abc123",
        license="apache-2.0",
        files=(RemoteFile("model.gguf", 3, "cd" * 32),),
        base_model="org/example-abliterated",
        task="text-generation",
    )
    assert document_from_repo(repo)["flags"] == []
    verified = dataclasses.replace(repo, flags=("least-censored",))
    assert document_from_repo(verified)["flags"] == ["least-censored"]

