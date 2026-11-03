"""UI VRAM readings go through gpu_probe, not torch."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from voxprint_common.gpu_probe import GpuDevice

from voxprint_forge.gui.gauges import vram_status
from voxprint_forge.lora.presets import build_config, detect_vram_gb
from voxprint_forge.preflight import _probe_cuda, memory_status
from voxprint_forge.vram_probe import status, total_bytes, used_bytes

ROOT = Path(__file__).resolve().parents[1]
_MIB = 1024 * 1024


def _gpu(mib: int | None) -> GpuDevice:
    return GpuDevice(index=0, name="test", driver="610.73.01", compute=(8, 9), memory_mib=mib)


def _patch_gpus(monkeypatch: pytest.MonkeyPatch, devices: list[GpuDevice]) -> None:
    monkeypatch.setattr("voxprint_common.gpu_probe.query_gpus", lambda runner=None: devices)


def test_tier_keeps_the_20_gib_cut(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_gpus(monkeypatch, [_gpu(16384)])
    assert detect_vram_gb() == 16
    assert total_bytes() == 16384 * _MIB
    assert _probe_cuda() == 16384 * _MIB
    small = build_config("balanced", "balanced", "auto")
    assert small.vram == 16 and small.base_fp8 is True

    _patch_gpus(monkeypatch, [_gpu(20480)])
    assert detect_vram_gb() == 24
    assert build_config("balanced", "balanced", "auto").vram == 24

    _patch_gpus(monkeypatch, [_gpu(20479)])
    assert detect_vram_gb() == 16

    _patch_gpus(monkeypatch, [_gpu(24576)])
    assert detect_vram_gb() == 24
    assert _probe_cuda() == 24576 * _MIB


def test_missing_gpu_matches_the_old_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_gpus(monkeypatch, [])
    assert detect_vram_gb() == 16
    assert _probe_cuda() is None
    assert vram_status() is None

    _patch_gpus(monkeypatch, [_gpu(None)])
    assert detect_vram_gb() == 16
    assert _probe_cuda() is None
    assert status() is None

    def boom(runner: object = None) -> list[GpuDevice]:
        raise RuntimeError("smi down")

    monkeypatch.setattr("voxprint_common.gpu_probe.query_gpus", boom)
    assert detect_vram_gb() == 16
    assert _probe_cuda() is None
    assert vram_status() is None


def test_vram_status_reports_total_and_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_gpus(monkeypatch, [_gpu(24576)])
    used = 4 * 1024**3
    monkeypatch.setattr("voxprint_forge.vram_probe.used_bytes", lambda index=0: used)
    assert vram_status() == (24576 * _MIB, used)

    monkeypatch.setattr("voxprint_forge.vram_probe.used_bytes", lambda index=0: None)
    assert vram_status() is None


def test_used_bytes_parses_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        seen.append(command)
        return SimpleNamespace(returncode=0, stdout="1536\n4096\n")

    monkeypatch.setattr("voxprint_forge.vram_probe.subprocess.run", run)
    assert used_bytes() == 1536 * _MIB
    assert used_bytes(1) == 4096 * _MIB
    assert "memory.used" in seen[0][1]


def test_used_bytes_rejects_a_bad_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    scripted = iter(
        (
            SimpleNamespace(returncode=0, stdout="N/A\n"),
            SimpleNamespace(returncode=1, stdout=""),
            SimpleNamespace(returncode=0, stdout="-4\n"),
        )
    )

    def run(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        return next(scripted)

    monkeypatch.setattr("voxprint_forge.vram_probe.subprocess.run", run)
    assert used_bytes() is None
    assert used_bytes() is None
    assert used_bytes() is None

    def explode(_command: list[str], **_kwargs: object) -> SimpleNamespace:
        raise OSError("no nvidia-smi")

    monkeypatch.setattr("voxprint_forge.vram_probe.subprocess.run", explode)
    assert used_bytes() is None


def test_memory_status_is_still_ram() -> None:
    total, available = memory_status()
    assert isinstance(total, int) and isinstance(available, int)
    assert total > 0
    assert 0 <= available <= total


def test_ui_vram_path_does_not_import_torch() -> None:
    """Importing and calling the UI probes must not load torch or the park."""
    script = r"""
import sys
import voxprint_common.gpu_probe as gpu_probe
from voxprint_common.gpu_probe import GpuDevice

def query(runner=None):
    return [GpuDevice(index=0, name="Test", driver="610.73.01", compute=(8, 9), memory_mib=24576)]

gpu_probe.query_gpus = query

import voxprint_forge.vram_probe as vram_probe
vram_probe.used_bytes = lambda index=0: 5 * 1024 ** 3

from voxprint_forge.preflight import memory_status, _probe_cuda
from voxprint_forge.lora.presets import detect_vram_gb
from voxprint_forge.gui.gauges import vram_status
from voxprint_forge.lora import backends, refset
from voxprint_forge.app import settings

total, free = memory_status()
assert isinstance(total, int) and isinstance(free, int) and total >= free >= 0
assert _probe_cuda() == 24576 * 1024 * 1024
assert detect_vram_gb() == 24
assert vram_status() == (24576 * 1024 * 1024, 5 * 1024 ** 3)
assert refset.empty_state()["schema"] == "voxprint-refset/1"
assert backends.LEGACY_IDS
assert settings.WIZARD == "lora"

leaked = [
    name for name in sys.modules
    if name == "torch" or name.startswith("torch.") or name == "voxprint_forge.model_park"
]
if leaked:
    raise SystemExit("imported " + ", ".join(sorted(leaked)))

import voxprint_forge.model_park as model_park
assert model_park._vram_bytes() == 5 * 1024 ** 3
leaked = [name for name in sys.modules if name == "torch" or name.startswith("torch.")]
if leaked:
    raise SystemExit("park vram imported " + ", ".join(sorted(leaked)))
print("ok")
"""
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ok" in proc.stdout
