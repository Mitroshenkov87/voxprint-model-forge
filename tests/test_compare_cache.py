"""Shared compare downloads and GPU-neutral notes. No GPU and no network."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from voxprint_forge.compare.cache import cache_name, reclaim_all
from voxprint_forge.compare.cli import execute
from voxprint_forge.compare.engine import fetch_arm
from voxprint_forge.compare.plan import load_compare_plan
from voxprint_forge.compare.reclaim import component_bytes
from voxprint_forge.compare.results import fp8_note
from voxprint_forge.compare.run import release_cache
from voxprint_forge.kl.run import speed_facts

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "plans" / "compare-synth4.toml"
TURBO = "Tongyi-MAI/Z-Image-Turbo"
TURBO_REV = "f332072aa78be7aecdf3ee76d5c247082da564a6"


def _fill(local_dir: str) -> None:
    dest = Path(local_dir)
    (dest / "transformer").mkdir(parents=True, exist_ok=True)
    (dest / "transformer" / "model.safetensors").write_bytes(b"T" * 8)
    (dest / "transformer" / "config.json").write_text("{}", encoding="utf-8")
    (dest / "vae").mkdir(exist_ok=True)
    (dest / "vae" / "model.safetensors").write_bytes(b"V" * 4)


def test_one_base_snapshot_is_hardlinked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def download(**kwargs: object) -> None:
        calls.append(str(kwargs["repo_id"]))
        _fill(str(kwargs["local_dir"]))

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    folder = tmp_path / "fp8"
    folder.mkdir()
    (folder / "zimage-turbo-transformer-fp8.safetensors").write_bytes(b"F")
    monkeypatch.setenv("FORGE_FP8_DIR", str(folder))
    job = load_compare_plan(PLAN).job("D1")
    cache = tmp_path / "compare-cache"
    fetch_arm(job.arm("a"), tmp_path / "a", cache=cache)
    fetch_arm(job.arm("c"), tmp_path / "c", cache=cache)
    assert calls == [TURBO]
    reference = tmp_path / "a" / "weights" / "vae" / "model.safetensors"
    shared = tmp_path / "c" / "base" / "vae" / "model.safetensors"
    assert os.path.samefile(reference, shared)
    assert not (tmp_path / "c" / "base" / "transformer" / "model.safetensors").exists()
    assert (tmp_path / "c" / "base" / "transformer" / "config.json").is_file()
    sizes = component_bytes(tmp_path / "c")
    assert "base/transformer/model.safetensors" not in sizes
    assert sizes["base/vae/model.safetensors"] == 4
    assert sizes["weights/zimage-turbo-transformer-fp8.safetensors"] == 1


def test_queue_drops_the_turbo_base_when_d2_holds_nothing(tmp_path: Path) -> None:
    plan = load_compare_plan(PLAN)
    assert plan.queue == ("D1", "D3", "D2", "D4")
    cache = tmp_path / "compare-cache"
    name = cache_name(TURBO, TURBO_REV, None)
    entry = cache / name
    entry.mkdir(parents=True)
    (entry / "source.txt").write_text(f"{TURBO}\n{TURBO_REV}\n*\n", encoding="utf-8")
    assert release_cache(plan, "D1", cache, (9, 0)) == []
    assert entry.is_dir()
    assert release_cache(plan, "D3", cache, (9, 0)) == [name]
    assert not entry.exists()
    entry.mkdir(parents=True)
    (entry / "source.txt").write_text(f"{TURBO}\n{TURBO_REV}\n*\n", encoding="utf-8")
    assert release_cache(plan, "D3", cache, (8, 0)) == []
    assert release_cache(plan, "D2", cache, (8, 0)) == [name]


def test_reclaim_cache_deletes_every_snapshot(tmp_path: Path) -> None:
    cache = tmp_path / "compare-cache" / "one"
    cache.mkdir(parents=True)
    (cache / "source.txt").write_text("example/weights\nabc\n*\n", encoding="utf-8")
    report = execute(ROOT, tmp_path, None, PLAN, reclaim_cache=True)
    assert report["exit_code"] == 0
    assert report["uploaded"] is False
    assert reclaim_all(tmp_path / "compare-cache") == []
    assert not cache.exists()


def test_fp8_notes_use_the_detected_gpu() -> None:
    job = load_compare_plan(PLAN).job("D1")
    text = fp8_note(job, "NVIDIA H100 NVL", "9.0")
    assert "A100" not in text
    assert "c-dq: FP8 weights expanded to BF16; s/img and VRAM are BF16 numbers." in text
    assert "sm_90" in text
    assert "NVIDIA H100 NVL" in text
    assert "c: FP8 E4M3" in text
    assert "d: FP8 E4M3" in text


def test_fp8_note_skips_disabled_arms() -> None:
    job = load_compare_plan(PLAN).job("D1")
    disabled = replace(
        job,
        arms=tuple(replace(arm, enabled=False) if arm.id == "d" else arm for arm in job.arms),
    )
    text = fp8_note(disabled, "NVIDIA H100 NVL", "9.0")
    assert "d: FP8 E4M3" not in text
    assert "c: FP8 E4M3" in text
    assert "c-dq: FP8 weights expanded to BF16" in text


def test_local_fp8_loads_without_base_transformer_weights(tmp_path: Path) -> None:
    """A child process loads a tiny on-disk pipeline that has no transformer weights."""
    script = ROOT / "tests" / "compare_fp8_shell.py"
    done = subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "layers_swapped" in done.stdout
    assert "diffusers" not in sys.modules


def test_tokens_per_second_is_a100_only_on_sm80() -> None:
    hopper = speed_facts("NVIDIA H100 NVL", (9, 0))
    assert hopper["tokens_per_second_is_a100"] is False
    assert hopper["tokens_per_second_gpu"] == "NVIDIA H100 NVL"
    assert "A100" not in str(hopper["gpu_note"])
    ampere = speed_facts("NVIDIA A100-SXM4-80GB", (8, 0))
    assert ampere["tokens_per_second_is_a100"] is True
    cpu = speed_facts("", None)
    assert cpu["tokens_per_second_is_a100"] is False
    assert "this GPU" in str(cpu["gpu_note"])


def test_script_headers_say_synth4() -> None:
    for name in ("bootstrap.sh", "vm_run_compare.sh", "vm_run_kl.sh"):
        header = (ROOT / "scripts" / name).read_text(encoding="utf-8").splitlines()[1]
        assert "synth4" in header
        assert "A100" not in header

