"""CPU contract for the Windows laptop smoke. No CUDA and no diffusers import."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from voxprint_forge.diffusion.nunchaku_edit import WINDOWS_WHEEL
from voxprint_forge.diffusion.smoke_laptop import (
    DEEPCOMPRESSOR_WINDOWS_SKIP,
    SCHEMA,
    TIME_CAP_SKIP,
    WHEEL_URL,
    deepcompressor_skip,
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
    assert parse_steps("") == ("smoke-tiny", "quantfunc", "zimage", "calib-quant-timing")
    assert parse_steps("zimage,smoke-tiny") == ("smoke-tiny", "zimage")
    with pytest.raises(Exception, match="unknown smoke step"):
        parse_steps("wan")
    assert deepcompressor_skip("win32") == DEEPCOMPRESSOR_WINDOWS_SKIP
    assert deepcompressor_skip("linux") is None
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
        raise RuntimeError("missing smooth_factor")

    def ok() -> dict[str, object]:
        clock["now"] = 10.0
        return {"id": "zimage", "status": "pass", "notes": [], "images": ["z1.webp"]}

    records = run_plan(
        ("quantfunc", "zimage"),
        {"quantfunc": fail, "zimage": ok},
        deadline=100.0,
        now=lambda: clock["now"],
    )
    assert records[0]["status"] == "fail"
    assert "smooth_factor" in str(records[0]["error"])
    assert records[1]["status"] == "pass"
    assert exit_code(records) == 1

    def skipped_later() -> dict[str, object]:
        return {"id": "smoke-tiny", "status": "pass"}

    capped = run_plan(
        ("smoke-tiny", "quantfunc"),
        {"smoke-tiny": skipped_later, "quantfunc": ok},
        deadline=0.0,
        now=lambda: 1.0,
    )
    assert capped[0]["error"] == TIME_CAP_SKIP
    assert capped[1]["status"] == "skip"
    assert exit_code([{"status": "pass"}, {"status": "pass"}]) == 0

    from voxprint_forge.diffusion.smoke_prefetch import DownloadError

    def broken_download() -> dict[str, object]:
        raise DownloadError("setup: download failed: transformer")

    with pytest.raises(DownloadError, match="setup: download failed"):
        run_plan(("quantfunc",), {"quantfunc": broken_download}, deadline=100.0, now=lambda: 0.0)


def test_result_schema_and_setup_exit() -> None:
    """The document name and a missing-argument exit do not touch the network."""
    payload = result_document(
        records=[{"id": "smoke-tiny", "status": "pass"}],
        code=0,
        results_dir=Path("results"),
        minutes_cap=90,
        seconds_cap=5400,
        selected=("smoke-tiny",),
        gpu={"name": "RTX 4090 Laptop", "compute_capability": "8.9"},
        versions={"torch": "2.11.0"},
        started="2026-10-11T00:00:00Z",
    )
    assert payload["schema"] == SCHEMA
    assert payload["ok"] is True
    assert payload["exit_code"] == 0
    assert main(["run"]) == 2
    assert main(["preflight"]) == 2


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
        "smoke-tiny,quantfunc,zimage",
        "Minutes = 90",
        "diffusion-int4-v4-smoke2",
        "hf_xet",
        "hf_transfer",
        "Force",
        "Blocks",
        "WindowStyle Hidden",
        "-NoProfile",
        "latest.txt",
    ):
        assert needle in text
    bash = BASH.read_text(encoding="utf-8")
    assert bash.index("PREFLIGHT-OK") < bash.index('OUT" run --svd-rank')
    assert "The long quant does not start" in bash
    assert "set +e" in bash


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
        selected=("smoke-tiny",),
        gpu=None,
        versions={},
        started="2026-10-11T00:00:00Z",
        error="no suitable GPU",
    )
    write_result(path, payload)
    assert json.loads(path.read_text(encoding="utf-8"))["schema"] == SCHEMA


def test_tiny_config_reaches_the_nunchaku_constructor() -> None:
    """A star-args init hides num_layers. Binding the parent signature keeps one block."""
    import inspect

    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

    from voxprint_forge.diffusion.int4.qwen_tiny import tiny_config
    from voxprint_forge.diffusion.nunchaku_config import bind_qwen_config_signature, layers_agree

    class Hidden(QwenImageTransformer2DModel):
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs.pop("offload", None)
            super().__init__(*args, **kwargs)

    hidden_keys = set(inspect.signature(Hidden.__init__).parameters)
    assert "num_layers" not in hidden_keys
    bind_qwen_config_signature(Hidden, QwenImageTransformer2DModel)
    assert "num_layers" in inspect.signature(Hidden.__init__).parameters
    model = Hidden.from_config(tiny_config())
    assert len(model.transformer_blocks) == 1
    assert layers_agree(int(model.config.num_layers), len(model.transformer_blocks))
    assert not layers_agree(1, 60)


def test_download_retry_and_extrapolation() -> None:
    """A dropped connection is retryable, and the H100 scale uses both TFLOPs and bandwidth."""
    from voxprint_forge.diffusion.smoke_prefetch import is_retryable_download
    from voxprint_forge.diffusion.smoke_timing import SMOKE_SAMPLES, extrapolate_quant, reuse_step

    class ChunkedEncodingError(Exception):
        pass

    assert is_retryable_download(ChunkedEncodingError("Connection broken: IncompleteRead"))
    assert not is_retryable_download(ValueError("sha256"))
    report = extrapolate_quant(sample_seconds=[1.0, 3.0], block_seconds=[10.0, 30.0])
    assert report["seconds_per_calibration_sample"] == pytest.approx(2.0)
    assert report["seconds_per_block"] == pytest.approx(20.0)
    assert report["full_blocks"] == 60
    assert report["laptop_seconds"] == pytest.approx(1.0 + 3.0 + 20.0 * 60)
    assert float(report["h100_seconds"]) < float(report["laptop_seconds"])
    assert "576" in str(report["assumption"])
    assert "3350" in str(report["assumption"])
    previous = {"steps": [{"id": "quantfunc", "status": "pass", "notes": []}]}
    assert reuse_step(previous, "quantfunc", force=False)["status"] == "pass"
    assert reuse_step(previous, "quantfunc", force=True) is None
    assert reuse_step(previous, "zimage", force=False) is None
    assert "reused" in reuse_step(previous, "quantfunc", force=False)["notes"]
    assert SMOKE_SAMPLES == 2


def test_verify_download_rejects_a_short_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A truncated weight is a setup error and is not left on disk."""
    monkeypatch.delenv("FORGE_DEEPCOMPRESSOR", raising=False)
    from voxprint_forge.diffusion.int4.config import write_configs
    from voxprint_forge.diffusion.smoke_prefetch import DownloadError, call_with_retry, verify_download
    from voxprint_forge.diffusion.smoke_timing import SMOKE_SAMPLES

    configs = write_configs(tmp_path, "qwen-image-edit-2511", rank=256, num_samples=SMOKE_SAMPLES)
    collect = configs["collect"].read_text(encoding="utf-8")
    overlay = configs["overlay"].read_text(encoding="utf-8")
    assert f"num_samples: {SMOKE_SAMPLES}" in collect
    assert f"num_samples: {SMOKE_SAMPLES}" in overlay
    assert "num_samples: 64" not in collect
    assert "num_samples: 64" not in overlay

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
