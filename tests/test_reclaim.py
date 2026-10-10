"""Eager disk freeing: one verified mirror, then leftovers, then a re-check."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from tests.complete_card import complete_card
from voxprint_forge.diffusion.common import Ctx, on_mirror_verified
from voxprint_forge.diffusion.planload import load_canvas_plan
from voxprint_forge.diffusion.retain import consumers_for, release_consumer, source_store
from voxprint_forge.disk import ensure_free
from voxprint_forge.errors import DiskShortError
from voxprint_forge.layout import Layout
from voxprint_forge.reclaim import free_hf_revisions, free_orphaned_blobs, reclaim_for_disk, reclaim_model_work
from voxprint_forge.upload import Artifact, publish_private

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
BLOB = "b" * 40


def _cache(root: Path, payload: bytes) -> tuple[Path, Path]:
    cache = root / "hub"
    repo = cache / "models--org--toy"
    blob = repo / "blobs" / BLOB
    blob.parent.mkdir(parents=True)
    blob.write_bytes(payload)
    snap = repo / "snapshots" / COMMIT
    snap.mkdir(parents=True)
    (snap / "weights.safetensors").symlink_to(Path("..") / ".." / "blobs" / BLOB)
    refs = repo / "refs"
    refs.mkdir()
    (refs / "main").write_text(COMMIT, encoding="utf-8")
    return cache, blob


def test_deleting_a_snapshot_leaves_the_blob(tmp_path: Path) -> None:
    payload = b"x" * 4096
    cache, blob = _cache(tmp_path, payload)
    shutil.rmtree(cache / "models--org--toy" / "snapshots")
    assert blob.is_file()
    assert blob.stat().st_size == len(payload)
    freed = free_orphaned_blobs(cache)
    assert freed == len(payload)
    assert not blob.exists()


def test_delete_revisions_frees_the_blob_bytes(tmp_path: Path) -> None:
    payload = b"y" * 2048
    cache, blob = _cache(tmp_path, payload)
    freed = free_hf_revisions(cache, (COMMIT,))
    assert not blob.exists()
    assert freed == len(payload)


def test_disk_wait_reclaims_then_rechecks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    forge = tmp_path / "forge-out"
    work = forge / "work" / "huihui"
    source = work / "source"
    source.mkdir(parents=True)
    (source / "weights.safetensors").write_bytes(b"s" * 1000)
    staging = work / "upload-staging"
    staging.mkdir()
    (staging / "pack.bin").write_bytes(b"p" * 200)
    partial = work / "quants" / "24gb" / "Q4.gguf"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"q" * 300)
    kept = work / "quants" / "24gb" / "Q5.gguf"
    kept.write_bytes(b"k" * 400)
    Path(str(kept) + ".sha256").write_text("abc\n", encoding="utf-8")
    marker = forge / "stages" / "huihui" / "upload.json"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps({"state": "UPLOADED", "uploads": {"hf": "uploaded"}}) + "\n", encoding="utf-8")
    cache, blob = _cache(tmp_path, b"z" * 128)
    shutil.rmtree(cache / "models--org--toy" / "snapshots")
    calls = {"n": 0}

    def free(_path: Path) -> int:
        calls["n"] += 1
        if calls["n"] == 1:
            return 0
        return 200 * 1024**3

    monkeypatch.setattr("voxprint_forge.disk.free_bytes", free)
    got = ensure_free(forge, 1024, "huihui", forge_out=forge, cache_dir=cache)
    assert got == 200 * 1024**3
    assert calls["n"] == 2
    assert not source.exists()
    assert not staging.exists()
    assert not partial.exists()
    assert kept.is_file()
    assert not blob.exists()
    err = capsys.readouterr().err
    assert f"reclaim {source} " in err
    assert "bytes" in err


def test_disk_wait_still_raises_when_nothing_is_deletable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("voxprint_forge.disk.free_bytes", lambda _path: 0)
    with pytest.raises(DiskShortError):
        ensure_free(tmp_path, 1, "empty", forge_out=tmp_path, cache_dir=tmp_path / "no-cache")


def test_running_model_keeps_its_partial(tmp_path: Path) -> None:
    forge = tmp_path / "forge-out"
    partial = forge / "work" / "huihui" / "quants" / "Q4.gguf"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"q" * 10)
    status = {"updated": "2026-10-10T00:00:00Z", "items": {"huihui": {"state": "RUNNING", "pid": os.getpid()}}}
    (forge / "status.json").write_text(json.dumps(status), encoding="utf-8")
    reclaim_for_disk(forge)
    assert partial.is_file()


def test_f16_stays_until_quantize_finished(tmp_path: Path) -> None:
    forge = tmp_path / "forge-out"
    f16 = forge / "work" / "tiny" / "f16.gguf"
    f16.parent.mkdir(parents=True)
    f16.write_bytes(b"f" * 20)
    reclaim_for_disk(forge)
    assert f16.is_file()
    stages = forge / "stages" / "tiny"
    stages.mkdir(parents=True)
    (stages / "imatrix.json").write_text("{}\n", encoding="utf-8")
    (stages / "quantize.json").write_text("{}\n", encoding="utf-8")
    reclaim_for_disk(forge)
    assert not f16.exists()


def test_later_consumer_keeps_the_shared_source(tmp_path: Path) -> None:
    plan = load_canvas_plan(ROOT / "plans" / "canvas-zimage-tiers-v1.toml")
    names = consumers_for(plan, "turbo")
    tree = source_store(tmp_path, plan.id) / "turbo"
    tree.mkdir(parents=True)
    (tree / "weights.safetensors").write_bytes(b"w" * 32)
    for name in names:
        if name == "bake-off":
            continue
        assert release_consumer(tmp_path, plan.id, "turbo", name, names) is False
    reclaim_for_disk(tmp_path, plan=plan)
    assert tree.is_dir()
    folder = tmp_path / "work" / plan.id / "source-releases" / "turbo"
    (folder / "bake-off").write_text("released\n", encoding="utf-8")
    reclaim_for_disk(tmp_path, plan=plan)
    assert not tree.exists()


def test_hf_verify_drops_sources_and_keeps_the_quant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_USER", "tester")
    monkeypatch.setenv("KAGGLE_USER", "tester")
    model = tmp_path / "model"
    source = model / "source"
    source.mkdir(parents=True)
    (source / "model.safetensors").write_bytes(b"safetensors-bytes")
    (model / "f16.gguf").write_bytes(b"f16-weights")
    quant = model / "quants" / "16gb" / "Q4.gguf"
    quant.parent.mkdir(parents=True)
    payload = b"gguf-bytes"
    quant.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    seen: list[str] = []

    class Remote:
        def hf_private_repo(self, repo_id: str) -> str:
            del repo_id
            return "https://huggingface.co/tester/voxprint-tiny"

        def hf_upload(self, repo_id: str, file_path: Path, name: str) -> None:
            del repo_id, file_path, name

        def hf_stat(self, repo_id: str, name: str) -> tuple[str, int]:
            del repo_id, name
            return digest, len(payload)

        def kaggle_private_model(self, owner: str, slug: str, folder: Path) -> str:
            del owner, slug, folder
            return "https://www.kaggle.com/models/example"

        def kaggle_upload_variation(self, owner: str, slug: str, variation: str, folder: Path) -> str:
            del owner, folder
            return f"tester/{slug}/gguf/{variation}"

        def kaggle_stat(self, handle: str, name: str) -> tuple[str, int]:
            del handle, name
            return digest, len(payload)

    def on_verified(target: str) -> None:
        seen.append(target)
        reclaim_model_work(model, drop_staging=target != "hf")
        if target == "hf":
            assert not source.exists()
            assert not (model / "f16.gguf").exists()
            assert quant.is_file()

    publish_private(
        Remote(),
        [Artifact(quant, "Q4.gguf", digest, len(payload), "16gb", "Q4_K_M", "gguf")],
        model_id="tiny",
        title="T",
        description=complete_card(),
        licence="apache-2.0",
        attempts=1,
        sleep=lambda _seconds: None,
        work=model / "upload-staging",
        on_verified=on_verified,
    )
    assert seen == ["hf", "kaggle"]
    assert quant.is_file()
    assert not (model / "upload-staging").exists()


def test_canvas_mirror_keeps_a_later_consumer(tmp_path: Path) -> None:
    plan = load_canvas_plan(ROOT / "plans" / "canvas-zimage-tiers-v1.toml")
    ctx = Ctx(
        repo=ROOT,
        forge_out=tmp_path,
        plan=plan,
        part=plan.part("base"),
        token=None,
        layout=Layout(tmp_path),
    )
    tree = source_store(tmp_path, plan.id) / "turbo"
    tree.mkdir(parents=True)
    (tree / "weights.safetensors").write_bytes(b"w" * 16)
    scratch = ctx.upstream / "notes.txt"
    scratch.parent.mkdir(parents=True)
    scratch.write_bytes(b"n")
    staging = ctx.upstream / "kaggle-staging"
    staging.mkdir()
    (staging / "pack.bin").write_bytes(b"s")
    on_mirror_verified(ctx, "hf")
    assert tree.is_dir()
    assert not scratch.exists()
    assert staging.is_dir()
    on_mirror_verified(ctx, "kaggle")
    assert tree.is_dir()
    assert not ctx.upstream.exists()
