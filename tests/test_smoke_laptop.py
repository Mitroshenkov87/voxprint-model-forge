"""CPU contract for the Windows laptop smoke. No CUDA and no diffusers import."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from voxprint_forge.diffusion.nunchaku_zimage import WINDOWS_WHEEL
from voxprint_forge.diffusion.smoke_laptop import (
    SCHEMA,
    TIME_CAP_SKIP,
    WHEEL_URL,
    exit_code,
    main,
    max_abs_rel,
    parse_gpu_csv,
    parse_steps,
    result_document,
    run_plan,
    suitable_gpu,
    write_result,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke_laptop.ps1"
BASH = ROOT / "scripts" / "vm_run_diffusion_int4.sh"


def test_import_does_not_load_diffusers() -> None:
    """Importing the smoke module must not import diffusers or voxprint_common."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "import builtins\n"
            "real = builtins.__import__\n"
            "def guarded(name, *args, **kwargs):\n"
            "    if name == 'voxprint_common' or name.startswith('voxprint_common.'):\n"
            "        raise ImportError(name)\n"
            "    return real(name, *args, **kwargs)\n"
            "builtins.__import__ = guarded\n"
            "import voxprint_forge.diffusion.smoke_laptop\n"
            "import voxprint_forge.diffusion.int4\n"
            "assert 'diffusers' not in sys.modules\n"
            "assert 'voxprint_forge.diffusion.int4.stage' not in sys.modules\n"
            "assert 'voxprint_common' not in sys.modules\n",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_steps_errors_and_gpu_gate() -> None:
    """Step order, kernel error, and the 16 GB sm_89 gate are pure functions."""
    assert parse_steps("") == ("zimage",)
    assert parse_steps("zimage") == ("zimage",)
    with pytest.raises(Exception, match="unknown smoke step"):
        parse_steps("wan")
    left = torch.tensor([1.0, 2.0, 4.0])
    right = torch.tensor([1.0, 2.5, 4.0])
    max_abs, max_rel = max_abs_rel(left, right)
    assert max_abs == pytest.approx(0.5)
    assert max_rel == pytest.approx(0.25)
    gpu = parse_gpu_csv("NVIDIA GeForce RTX 4090 Laptop GPU, 16376, 617.42, 8.9")
    assert gpu["driver"] == "617.42"
    assert suitable_gpu(int(gpu["memory_mib"]), 8, 9) is True
    assert suitable_gpu(81920, 8, 0) is False
    assert suitable_gpu(8192, 8, 9) is False


def test_plan_isolates_failures_and_the_time_cap() -> None:
    """One failed step is recorded and the next step still runs."""
    clock = {"now": 0.0}

    def fail() -> dict[str, object]:
        raise RuntimeError("missing weight")

    def ok() -> dict[str, object]:
        clock["now"] = 10.0
        return {"id": "zimage", "status": "pass", "notes": [], "images": ["z1.webp"]}

    records = run_plan(
        ("zimage",),
        {"zimage": fail},
        deadline=100.0,
        now=lambda: clock["now"],
    )
    assert records[0]["status"] == "fail"
    assert "missing weight" in str(records[0]["error"])
    assert exit_code(records) == 1

    passed = run_plan(
        ("zimage",),
        {"zimage": ok},
        deadline=100.0,
        now=lambda: clock["now"],
    )
    assert passed[0]["status"] == "pass"

    def skipped_later() -> dict[str, object]:
        return {"id": "zimage", "status": "pass"}

    capped = run_plan(
        ("zimage",),
        {"zimage": skipped_later},
        deadline=0.0,
        now=lambda: 1.0,
    )
    assert capped[0]["error"] == TIME_CAP_SKIP
    assert exit_code([{"status": "pass"}, {"status": "pass"}]) == 0

    from voxprint_forge.diffusion.smoke_prefetch import DownloadError

    def broken_download() -> dict[str, object]:
        raise DownloadError("setup: download failed: transformer")

    with pytest.raises(DownloadError, match="setup: download failed"):
        run_plan(("zimage",), {"zimage": broken_download}, deadline=100.0, now=lambda: 0.0)


def test_result_schema_and_setup_exit() -> None:
    """The document name and a missing-argument exit do not touch the network."""
    payload = result_document(
        records=[{"id": "zimage", "status": "pass"}],
        code=0,
        results_dir=Path("results"),
        minutes_cap=90,
        seconds_cap=5400,
        selected=("zimage",),
        gpu={"name": "RTX 4090 Laptop", "compute_capability": "8.9"},
        versions={"torch": "2.11.0"},
        started="2026-10-11T00:00:00Z",
    )
    assert payload["schema"] == SCHEMA
    assert payload["ok"] is True
    assert payload["exit_code"] == 0
    assert main(["run"]) == 2


def test_script_is_unattended_and_pins_the_windows_wheel() -> None:
    """The PowerShell entry has no prompt and pins the cp312 cu130 wheel."""
    text = SCRIPT.read_text(encoding="utf-8")
    assert "Read-Host" not in text
    assert "HF_TOKEN=" not in text
    assert "Remove-Item Env:HF_TOKEN" in text
    assert WINDOWS_WHEEL.sha256 in text
    assert WINDOWS_WHEEL.filename in text
    assert WHEEL_URL in text
    for needle in (
        "LOCALAPPDATA",
        "VoxprintForgeSmoke",
        "KeepVenv",
        "HF_HOME",
        "result.json",
        "voxprint-forge-smoke-laptop/1",
        "cu130",
        "diffusers==0.37.1",
        "transformers==4.57.6",
        "3.12",
        "--no-deps",
        "$Steps = 'zimage'",
        "Minutes = 90",
        "hf_xet",
        "hf_transfer",
        "Force",
        "WindowStyle Hidden",
        "-NoProfile",
        "latest.txt",
    ):
        assert needle in text
    bash = BASH.read_text(encoding="utf-8")
    assert "FORGE_WAN_REPO" in bash
    assert "FORGE_WAN_PYTHON" in bash
    assert "lcpp.patch" in bash
    assert 'CMAKE_CUDA_ARCHITECTURES="${FORGE_CUDA_ARCHS:-80;89;90;120}"' in bash
    assert "deepcompressor" not in bash
    assert "--own-edit" not in bash
    assert "smoke_laptop" not in bash
    assert "smoke-tiny" not in text
    assert "--blocks" not in text


def test_powershell_parses_when_present() -> None:
    """Windows CI parses the script. Ubuntu skips when pwsh is absent."""
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("pwsh is not installed")
    proc = subprocess.run(
        [
            shell,
            "-NoProfile",
            "-Command",
            (
                "$tokens=$null; $errors=$null; "
                "[void][System.Management.Automation.Language.Parser]::ParseFile("
                f"'{SCRIPT}', [ref]$tokens, [ref]$errors); "
                "if ($errors -and $errors.Count -gt 0) { $errors | Out-String; exit 1 }"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_written_result_roundtrip(tmp_path: Path) -> None:
    """result.json is the schema document and nothing else is required on disk."""
    path = tmp_path / "result.json"
    payload = result_document(
        records=[],
        code=3,
        results_dir=tmp_path,
        minutes_cap=90,
        seconds_cap=5400,
        selected=("zimage",),
        gpu=None,
        versions={},
        started="2026-10-11T00:00:00Z",
        error="no suitable GPU",
    )
    write_result(path, payload)
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == SCHEMA


def test_download_retry() -> None:
    """A dropped connection is retryable. A hash mismatch is not."""
    from voxprint_forge.diffusion.smoke_laptop import _reuse_step
    from voxprint_forge.diffusion.smoke_prefetch import is_retryable_download

    class ChunkedEncodingError(Exception):
        pass

    assert is_retryable_download(ChunkedEncodingError("Connection broken: IncompleteRead"))
    assert not is_retryable_download(ValueError("sha256"))
    previous = {"steps": [{"id": "zimage", "status": "pass", "notes": []}]}
    reused = _reuse_step(previous, "zimage", force=False)
    assert reused is not None
    assert reused["status"] == "pass"
    assert "reused" in reused["notes"]
    assert _reuse_step(previous, "zimage", force=True) is None
    assert _reuse_step(previous, "other", force=False) is None


def test_verify_download_rejects_a_short_file(tmp_path: Path) -> None:
    """A truncated weight is a setup error and is not left on disk."""
    from voxprint_forge.diffusion.smoke_prefetch import DownloadError, call_with_retry, verify_download

    path = tmp_path / "weight.bin"
    path.write_bytes(b"abc")
    with pytest.raises(DownloadError, match="setup: download failed"):
        verify_download(path, sha256="0" * 64, size_bytes=1_000_000)
    assert not path.exists()

    calls = {"n": 0}

    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 2:
            raise ConnectionResetError("connection reset")
        return "ok"

    assert call_with_retry(flaky, "weight", attempts=3, sleep=lambda _delay: None) == "ok"

    checks = {"n": 0}

    def short_then_ok() -> str:
        checks["n"] += 1
        if checks["n"] < 2:
            raise DownloadError("setup: download failed: short")
        return "ok"

    assert call_with_retry(short_then_ok, "weight", attempts=3, sleep=lambda _delay: None) == "ok"

    def always_short() -> None:
        raise DownloadError("setup: download failed: short")

    with pytest.raises(DownloadError, match="^setup: download failed: short$"):
        call_with_retry(always_short, "weight", attempts=2, sleep=lambda _delay: None)
