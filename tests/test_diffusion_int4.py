"""CPU tests for the Wan 2.2 diffusion-int4 stage. Remotes and GPUs stay mocked."""

from __future__ import annotations

import json
import os
import site
import subprocess
import sys
from pathlib import Path

import pytest

from voxprint_forge.diffusion.int4.cli import main
from voxprint_forge.diffusion.int4.context import StageContext
from voxprint_forge.diffusion.int4.disk import DiskFloorError, guard_disk, reclaim_sources
from voxprint_forge.diffusion.int4.fetch import fetch_weights
from voxprint_forge.diffusion.int4.pins import (
    DRIVE_FOLDER_ID,
    WAN_EXPERTS,
    WAN_QUANTS,
    expected_minutes,
    planned_steps,
    skip_wan,
)
from voxprint_forge.diffusion.int4.publish import live_drive, render_int4_card
from voxprint_forge.diffusion.int4.stage import run
from voxprint_forge.diffusion.int4.wan import expert_shards, gguf_name
from voxprint_forge.errors import LicenceError, StageError
from voxprint_forge.statusfile import read_status

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "vm_run_diffusion_int4.sh"
POISON = '<script>alert("xss")</script>'


def _ctx(tmp_path: Path) -> tuple[StageContext, Path, Path]:
    forge_out = tmp_path / "forge-out"
    done = tmp_path / "done"
    watch = tmp_path / "watch.json"
    forge_out.mkdir()
    ctx = StageContext(repo=ROOT, forge_out=forge_out, dry_run=True, svd_rank=256)
    return ctx, done, watch


def test_dry_run_is_resumable_and_writes_watch_and_status(tmp_path: Path) -> None:
    ctx, done, watch = _ctx(tmp_path)
    first = run(ctx, done=done, watch=watch)
    assert first["expected_min"] == 190
    assert expected_minutes() == 190
    ids = [step.step_id for step in planned_steps()]
    assert ids == [
        "wan-2.2-gguf",
        "wan-2.2-fp8",
        "wan-2.2-smoke",
        "wan-2.2-publish",
        "wan-2.2-reclaim",
    ]
    gguf = next(row for row in first["steps"] if row["id"] == "wan-2.2-gguf")
    names = {Path(str(path)).name for path in gguf["files"]}
    for expert in WAN_EXPERTS:
        for quant_name in WAN_QUANTS:
            assert gguf_name(expert, quant_name) in names
    fp8 = next(row for row in first["steps"] if row["id"] == "wan-2.2-fp8")
    assert fp8["format"] == "voxprint-fp8-v1"
    document = json.loads(watch.read_text(encoding="utf-8"))
    assert document["schema"] == "voxprint-forge-diffusion-int4/1"
    assert document["expected_min"] == 190
    assert {job["id"] for job in document["jobs"]} == set(ids)
    status = read_status(ctx.forge_out)
    items = status["items"]
    assert isinstance(items, dict)
    assert items["wan-2.2-publish"]["state"] == "UPLOADED"
    assert items["wan-2.2-publish"]["last_heartbeat"]
    second = run(ctx, done=done, watch=watch)
    assert all(row["skipped"] is True for row in second["steps"])


def test_skip_wan_drops_steps_and_the_weight_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_SKIP_WAN", "1")
    assert skip_wan() is True
    assert expected_minutes() == 0
    assert planned_steps() == ()
    seen: list[str] = []

    def licence(repo: str, _revision: str, _token: str | None) -> str:
        seen.append(repo)
        return "apache-2.0"

    def snap(**kwargs: object) -> None:
        raise AssertionError(f"skip must not download {kwargs}")

    monkeypatch.setattr("voxprint_forge.diffusion.int4.fetch.snapshot_download", snap)
    paths = fetch_weights(tmp_path / "skipped", None, licence=licence)
    assert paths["FORGE_WAN_CKPT"] == ""
    assert seen == []
    assert json.loads((tmp_path / "skipped" / "paths.json").read_text(encoding="utf-8"))["FORGE_WAN_CKPT"] == ""


def test_fetch_downloads_wan_experts_and_refuses_a_bad_licence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def snap(**kwargs: object) -> None:
        folder = Path(str(kwargs["local_dir"]))
        assert str(kwargs["repo_id"]).startswith("Wan")
        for name in ("high_noise_model", "low_noise_model"):
            (folder / name).mkdir(parents=True, exist_ok=True)
            (folder / name / "model.safetensors").write_bytes(b"wan")

    monkeypatch.setattr("voxprint_forge.diffusion.int4.fetch.snapshot_download", snap)

    def refuse(_repo: str, _revision: str, _token: str | None) -> str:
        raise LicenceError("refused", code="non_permissive")

    with pytest.raises(LicenceError):
        fetch_weights(tmp_path / "refused", None, licence=refuse)
    assert not (tmp_path / "refused").exists()
    paths = fetch_weights(tmp_path / "ok", None, licence=lambda *_args: "apache-2.0")
    assert (Path(paths["FORGE_WAN_CKPT"]) / "high_noise_model" / "model.safetensors").is_file()
    assert (Path(paths["FORGE_WAN_CKPT"]) / "low_noise_model" / "model.safetensors").is_file()


def test_numbered_expert_shards_win(tmp_path: Path) -> None:
    expert = tmp_path / "high_noise_model"
    expert.mkdir()
    (expert / "diffusion_pytorch_model-00001-of-00002.safetensors").write_bytes(b"a")
    (expert / "diffusion_pytorch_model.safetensors").write_bytes(b"full")
    found = expert_shards(expert)
    assert len(found) == 1
    assert "00001" in found[0].name


def test_disk_floor_aborts_and_recovers(tmp_path: Path) -> None:
    with pytest.raises(DiskFloorError) as exc:
        guard_disk(tmp_path, ratio=lambda _path: 0.01, reclaim=lambda: 0)
    assert exc.value.exit_code == 2
    assert exc.value.code == "disk_floor"
    state = {"seen": 0}

    def ratio(_path: Path) -> float:
        state["seen"] += 1
        return 0.01 if state["seen"] == 1 else 0.5

    assert guard_disk(tmp_path, ratio=ratio, reclaim=lambda: 1) == 0.5


def test_reclaim_keeps_sources_until_the_mirror_verifies(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"keep")
    with pytest.raises(StageError):
        reclaim_sources([str(source)], verified=False)
    assert (source / "weights.bin").is_file()
    assert reclaim_sources([str(source)], verified=True) == [str(source)]
    assert not source.exists()


def test_drive_folder_id_is_the_voxprint_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def upload(_files: list[object], *, model_id: str, date: str) -> list[dict[str, object]]:
        del model_id, date
        seen["folder"] = os.environ["FORGE_DRIVE_ROOT_FOLDER_ID"]
        return []

    monkeypatch.setattr("voxprint_forge.diffusion.int4.publish.upload_drive", upload)
    live_drive(DRIVE_FOLDER_ID, [])
    assert seen["folder"] == "13qzgTnnkqvZrf_DK9fmymJ8ICdOuwDJT"


def test_card_escapes_untrusted_text() -> None:
    text = render_int4_card(
        ROOT,
        title=POISON,
        base_models=(POISON,),
        pipeline_tag="text-to-video",
        quant_lines=(POISON,),
        eval_lines=("ok",),
        credits=(POISON,),
        kind="wan",
    )
    assert "<script>" not in text
    assert "&lt;script&gt;" in text


def test_estimates_print_both_gpu_tables(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_GPU", raising=False)
    monkeypatch.delenv("FORGE_COMPUTE_CAP", raising=False)
    assert main(["--root", str(ROOT), "estimates"]) == 0
    text = capsys.readouterr().out
    assert "planning gpu: a100" in text
    assert "A100 80GB" in text
    assert "H100 80GB" in text
    assert text.count("total  190 min") == 2
    assert "wan-2.2-gguf  100 min" in text
    assert "qwen-edit" not in text
    monkeypatch.setenv("FORGE_GPU", "h100")
    assert main(["--json", "--root", str(ROOT), "estimates"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["gpu"] == "h100"
    assert payload["result"]["expected_min"] == 190
    assert payload["result"]["gpus"]["a100"]["expected_min"] == 190
    assert payload["result"]["gpus"]["h100"]["expected_min"] == 190


def test_cli_dry_run_json_and_disk_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code = main(
        [
            "--json",
            "--root",
            str(ROOT),
            "--forge-out",
            str(tmp_path / "out"),
            "run",
            "--dry-run",
            "--done-dir",
            str(tmp_path / "done"),
            "--watch",
            str(tmp_path / "watch.json"),
        ]
    )
    assert code == 0
    monkeypatch.setattr("voxprint_forge.diffusion.int4.disk.free_ratio", lambda _path: 0.01)
    floor = main(
        [
            "--json",
            "--root",
            str(ROOT),
            "--forge-out",
            str(tmp_path / "out"),
            "disk-floor",
            "--path",
            str(tmp_path),
            "--done-dir",
            str(tmp_path / "empty-done"),
        ]
    )
    assert floor == 2


@pytest.mark.skipif(sys.platform == "win32", reason="the rental script is bash")
def test_script_dry_run(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(tmp_path / "home"),
            "FORGE_OUT": str(tmp_path / "forge-out"),
            "FORGE_DONE_DIR": str(tmp_path / "done"),
            "FORGE_WATCH": str(tmp_path / "watch.json"),
            "FORGE_HOST_KIND": "local",
            "FORGE_PYTHON": sys.executable,
            "PYTHONPATH": site.getusersitepackages(),
        }
    )
    (tmp_path / "home").mkdir()
    done = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "ALL-DONE" in done.stdout
    assert (tmp_path / "watch.json").is_file()
    text = SCRIPT.read_text(encoding="utf-8")
    assert "1ea34ff48f87168174e12956e200b1d908b1c5ff" in text
    assert "6ea2651e7df66d7585f6ffee804b20e92fb38b8a" in text
    assert "c8c07d658a6cefc5a50cfdf6be7d726503612303" in text
    assert "lcpp.patch" in text
    assert 'CMAKE_CUDA_ARCHITECTURES="80;90"' in text
    assert "venv-wan" in text
    for name in ("FORGE_WAN_REPO", "FORGE_WAN_PYTHON", "FORGE_COMFY_GGUF", "FORGE_LLAMA_SRC", "FORGE_LLAMA_BIN"):
        assert name in text
    assert "deepcompressor" not in text
    assert "--own-edit" not in text
    assert "--qwen-image" not in text
    assert "smoke_laptop" not in text


def test_forge_help_lists_diffusion_int4() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "voxprint_forge", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0
    assert "diffusion-int4" in done.stdout
    assert "compare" in done.stdout
    assert "kl" in done.stdout
