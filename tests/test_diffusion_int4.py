"""CPU tests for the diffusion-int4 stage. Remotes and GPUs stay mocked."""

from __future__ import annotations

import json
import os
import site
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open

from voxprint_forge.diffusion.int4 import calib, measure
from voxprint_forge.diffusion.int4.calib import calibration_prompts, image_root, load_prompts
from voxprint_forge.diffusion.int4.cli import main
from voxprint_forge.diffusion.int4.config import fast_int4_yaml, quant_argv, write_configs
from voxprint_forge.diffusion.int4.context import StageContext
from voxprint_forge.diffusion.int4.disk import DiskFloorError, guard_disk
from voxprint_forge.diffusion.int4.evalrule import Arm, decide_winner, dry_run_arms
from voxprint_forge.diffusion.int4.fetch import fetch_quantfunc, fetch_weights
from voxprint_forge.diffusion.int4.gate import decide_own_edit, load_phase0_baseline, quantfunc_r128_gate
from voxprint_forge.diffusion.int4.gpu import (
    EMULATED_NOTE,
    NATIVE_NOTE,
    device_fp8_report,
    fp8_mode,
    parse_compute_cap,
    planning_gpu,
    step_minutes,
    total_minutes,
)
from voxprint_forge.diffusion.int4.measure import _require_native_bindings, bind_fp8_linears, fp8_layout
from voxprint_forge.diffusion.int4.patch import (
    _BUILD_ANCHOR,
    _COLLECT_ANCHOR,
    _FEATURES_ANCHOR,
    _TASK_ANCHOR,
    _YAML_ANCHOR,
    apply_checkout,
    assert_apache,
)
from voxprint_forge.diffusion.int4.pins import (
    DEFAULT_SVD_RANK,
    DRIVE_FOLDER_ID,
    QUANTFUNC_R32,
    QUANTFUNC_R32_SHA256,
    QUANTFUNC_R128,
    QUANTFUNC_R128_SHA256,
    QUANTFUNC_R256,
    QUANTFUNC_R256_SHA256,
    expected_minutes,
    planned_steps,
    skip_wan,
)
from voxprint_forge.diffusion.int4.publish import live_drive, publish_verified, render_int4_card
from voxprint_forge.diffusion.int4.qwen import _eval_lines, _winner_files, eval_step, publish_step, reclaim_sources
from voxprint_forge.diffusion.int4.stage import run
from voxprint_forge.diffusion.int4.wan import WAN_EXPERTS, WAN_QUANTS, expert_shards, gguf_name
from voxprint_forge.diffusion.int4.watch import build_watch
from voxprint_forge.errors import ChecksumError, LicenceError, StageError
from voxprint_forge.statusfile import read_status

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "vm_run_diffusion_int4.sh"
POISON = '<script>alert("xss")</script>'


def _ctx(tmp_path: Path, *, rank: int = DEFAULT_SVD_RANK, qwen_image: bool = False) -> tuple[StageContext, Path, Path]:
    forge_out = tmp_path / "forge-out"
    done = tmp_path / "done"
    watch = tmp_path / "watch.json"
    forge_out.mkdir()
    ctx = StageContext(repo=ROOT, forge_out=forge_out, dry_run=True, svd_rank=rank)
    return ctx, done, watch


def test_calibration_covers_languages_and_image_refs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = load_prompts(ROOT)
    kinds = {kind for item in prompts for kind in item["kinds"]}
    langs = {item["lang"] for item in prompts}
    assert {"edit", "people", "text", "hands"} <= kinds
    assert langs == {"ru", "en"}
    for item in prompts:
        assert (ROOT / "data" / "diffusion_calib" / str(item["image"])).is_file()
    override = tmp_path / "swatches"
    (override / "images").mkdir(parents=True)
    for name in ("c1-evening.png", "c2-hands.png", "c3-sign.png", "people-platform.png"):
        (override / "images" / name).write_bytes(b"png")
    monkeypatch.setenv("FORGE_CALIB_IMAGE_DIR", str(override))
    assert image_root(ROOT) == override
    assert load_prompts(ROOT)[0]["lang"] in {"ru", "en"}


def test_fast_int4_rank_and_quant_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_DEEPCOMPRESSOR", raising=False)
    assert "rank: 256" in fast_int4_yaml()
    assert "rank: 32" in fast_int4_yaml(rank=32)
    configs = write_configs(tmp_path, "qwen-image-edit-2511", rank=256)
    assert configs["model"].parent == tmp_path / "configs"
    argv = quant_argv(tmp_path / "deepcompressor", configs, tmp_path / "quant")
    text = " ".join(argv)
    assert "int4.yaml" in text
    assert "fast.yaml" in text
    assert "--skip-eval" in argv
    assert "--save-model" in argv
    assert "guidance_scale: 4.0" in configs["model"].read_text(encoding="utf-8")


def test_winner_rule_quality_speed_and_empty() -> None:
    decision = decide_winner(dry_run_arms())
    assert decision.winner == "own-r256"
    assert decision.rule == "quality"
    assert decision.passed["quantfunc-r128"] is False
    missing = [
        Arm(arm.arm_id, arm.peak_vram_smi_mib, arm.seconds_per_image, None, None, None, None, arm.rank)
        for arm in dry_run_arms()
    ]
    speed = decide_winner(missing)
    assert speed.rule == "speed"
    assert speed.winner == "quantfunc-r128"
    huge = [Arm("fp8", 30 * 1024, 1.0, None, None, None, None, None)]
    assert decide_winner(huge).winner is None


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


def test_licence_refusal_skips_drive_and_keeps_sources(tmp_path: Path) -> None:
    called: list[str] = []

    def drive(_folder: str, _files: list[object]) -> list[dict[str, object]]:
        called.append("drive")
        return []

    with pytest.raises(LicenceError):
        publish_verified(
            tmp_path,
            slug="voxprint-example",
            card_text="no",
            licence="cc-by-nc-4.0",
            hub=lambda _repo, _files: [],
            drive=drive,
        )
    assert called == []
    source = tmp_path / "source"
    source.mkdir()
    (source / "weights.bin").write_bytes(b"keep")
    with pytest.raises(StageError):
        reclaim_sources([str(source)], verified=False)
    assert (source / "weights.bin").is_file()
    assert reclaim_sources([str(source)], verified=True) == [str(source)]
    assert not source.exists()


def test_hub_failure_falls_through_to_drive(tmp_path: Path) -> None:
    (tmp_path / "weight.bin").write_bytes(b"weights")
    card = render_int4_card(
        ROOT,
        title="Voxprint example",
        base_models=("Qwen/Qwen-Image-Edit-2511@rev",),
        pipeline_tag="image-to-image",
        quant_lines=("int4",),
        eval_lines=("dry-run",),
        credits=("Qwen, for the base model.",),
        kind="int4",
    )
    seen: list[str] = []

    def hub(_repo: str, _files: list[object]) -> list[dict[str, object]]:
        raise StageError("hub down", code="upload_mismatch")

    def drive(_folder: str, _files: list[object]) -> list[dict[str, object]]:
        seen.append("drive")
        return [{"name": "weight.bin", "sha256": "abc", "size": 7, "target": "drive", "url": "file://x"}]

    published = publish_verified(
        tmp_path,
        slug="voxprint-example",
        card_text=card,
        licence="apache-2.0",
        hub=hub,
        drive=drive,
    )
    assert seen == ["drive"]
    assert published.verified is True
    assert published.hf == "failed"
    assert published.drive == "uploaded"


def test_card_escapes_untrusted_text() -> None:
    text = render_int4_card(
        ROOT,
        title=POISON,
        base_models=(POISON,),
        pipeline_tag="image-to-image",
        quant_lines=(POISON,),
        eval_lines=("ok",),
        credits=(POISON,),
        kind="int4",
    )
    assert "<script>" not in text
    assert "&lt;script&gt;" in text


def test_quantfunc_winner_file_is_published(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    quant = tmp_path / "quantfunc"
    quant.mkdir()
    weight = quant / QUANTFUNC_R128
    weight.write_bytes(b"int4")
    monkeypatch.setenv("FORGE_QUANTFUNC", str(quant))
    ctx = StageContext(repo=ROOT, forge_out=tmp_path / "out", dry_run=True, svd_rank=256)
    comparison = ctx.job("qwen-image-edit-2511") / "comparison.json"
    comparison.parent.mkdir(parents=True)
    comparison.write_text(json.dumps({"winner": "quantfunc-r128", "rule": "speed", "dry_run": True}), encoding="utf-8")
    assert _winner_files(ctx, "qwen-image-edit-2511", "quantfunc-r128") == [weight]
    names: list[str] = []

    def hub(_repo: str, files: list[object]) -> list[dict[str, object]]:
        names.extend(getattr(item, "name") for item in files)
        return [{"name": "ok", "sha256": "abc", "size": 1, "target": "hf", "url": "file://hf"}]

    ctx.hub = hub
    ctx.drive = lambda _folder, _files: [{"name": "ok", "sha256": "abc", "size": 1, "target": "drive", "url": "d"}]
    result = publish_step(ctx, "qwen-image-edit-2511")
    assert result["verified"] is True
    assert QUANTFUNC_R128 in names


def test_dry_run_is_resumable_and_writes_watch_and_status(tmp_path: Path) -> None:
    ctx, done, watch = _ctx(tmp_path)
    first = run(ctx, done=done, watch=watch, qwen_image=False)
    assert first["winner"] == "own-r256"
    assert first["expected_min"] == 1260
    own_edit = first["own_edit"]
    assert isinstance(own_edit, dict)
    assert own_edit["mode"] == "on"
    assert own_edit["action"] == "quantize"
    assert first["phase0_baseline"] is None
    assert expected_minutes(qwen_image=False) == 1260
    ids = [step.step_id for step in planned_steps(qwen_image=False)]
    assert "qwen-image-quant" not in ids
    export = next(row for row in first["steps"] if row["id"] == "qwen-edit-2511-export")
    with safe_open(str(export["file"]), framework="pt") as handle:
        meta = handle.metadata() or {}
    assert json.loads(meta["config"])["zero_cond_t"] is True
    quant = next(row for row in first["steps"] if row["id"] == "qwen-edit-2511-quant")
    argv = " ".join(str(part) for part in quant["argv"])
    assert "int4.yaml" in argv and "fast.yaml" in argv and "--skip-eval" in argv
    gguf = next(row for row in first["steps"] if row["id"] == "wan-2.2-gguf")
    names = {Path(str(path)).name for path in gguf["files"]}
    for expert in WAN_EXPERTS:
        for quant_name in WAN_QUANTS:
            assert gguf_name(expert, quant_name) in names
    fp8 = next(row for row in first["steps"] if row["id"] == "wan-2.2-fp8")
    assert fp8["format"] == "voxprint-fp8-v1"
    document = json.loads(watch.read_text(encoding="utf-8"))
    assert document["schema"] == "voxprint-forge-diffusion-int4/1"
    assert document["expected_min"] == 1260
    assert {job["id"] for job in document["jobs"]} == set(ids)
    status = read_status(ctx.forge_out)
    items = status["items"]
    assert isinstance(items, dict)
    assert items["qwen-edit-2511-publish"]["state"] == "UPLOADED"
    assert items["qwen-edit-2511-publish"]["last_heartbeat"]
    assert status["own_edit"]["action"] == "quantize"
    assert status["phase0_baseline"] is None
    second = run(ctx, done=done, watch=watch, qwen_image=False)
    assert all(row["skipped"] is True for row in second["steps"])


def test_qwen_image_export_clears_zero_cond_t(tmp_path: Path) -> None:
    ctx, done, watch = _ctx(tmp_path)
    result = run(ctx, done=done, watch=watch, qwen_image=True)
    export = next(row for row in result["steps"] if row["id"] == "qwen-image-export")
    with safe_open(str(export["file"]), framework="pt") as handle:
        meta = handle.metadata() or {}
    assert json.loads(meta["config"])["zero_cond_t"] is False
    assert "qwen-image-quant" in {row["id"] for row in result["steps"]}


def test_patch_is_idempotent_and_licence_is_apache(tmp_path: Path) -> None:
    root = tmp_path / "deepcompressor"
    files = {
        root / "deepcompressor" / "app" / "diffusion" / "pipeline" / "config.py": _TASK_ANCHOR + _BUILD_ANCHOR,
        root / "deepcompressor" / "app" / "diffusion" / "dataset" / "data" / "__init__.py": _FEATURES_ANCHOR
        + _YAML_ANCHOR,
        root / "deepcompressor" / "app" / "diffusion" / "dataset" / "collect" / "calib.py": _COLLECT_ANCHOR,
    }
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    licence = root / "LICENSE"
    licence.write_text("Apache License\nVersion 2.0\n", encoding="utf-8")
    assert_apache(licence, what="deepcompressor")
    assert apply_checkout(root)["pipeline"] == "applied"
    assert apply_checkout(root)["pipeline"] == "present"
    broken = tmp_path / "other" / "LICENSE"
    broken.parent.mkdir()
    broken.write_text("MIT\n", encoding="utf-8")
    with pytest.raises(LicenceError):
        assert_apache(broken, what="other")


def test_fp8_layout_and_wan_expert_dirs(tmp_path: Path) -> None:
    diffusers = tmp_path / "pipe"
    diffusers.mkdir()
    (diffusers / "model_index.json").write_text("{}", encoding="utf-8")
    assert fp8_layout(diffusers) == "diffusers"
    shards = tmp_path / "shards"
    shards.mkdir()
    (shards / "model.safetensors").write_bytes(b"fp8")
    assert fp8_layout(shards) == "shards"
    expert = tmp_path / "high_noise_model"
    expert.mkdir()
    (expert / "diffusion_pytorch_model-00001-of-00002.safetensors").write_bytes(b"a")
    (expert / "diffusion_pytorch_model.safetensors").write_bytes(b"full")
    found = expert_shards(expert)
    assert len(found) == 1
    assert "00001" in found[0].name


def test_fetch_checks_sha_and_refuses_a_bad_licence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def download(**kwargs: object) -> None:
        path = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"bytes")

    monkeypatch.setattr("voxprint_forge.diffusion.int4.fetch.hf_hub_download", download)
    monkeypatch.setattr(
        "voxprint_forge.diffusion.int4.fetch.sha256_file",
        lambda path: "deadbeef",
    )
    with pytest.raises(ChecksumError):
        fetch_quantfunc(tmp_path, None)

    def good_hash(path: Path) -> str:
        if path.name == QUANTFUNC_R128:
            return QUANTFUNC_R128_SHA256
        if path.name == QUANTFUNC_R32:
            return QUANTFUNC_R32_SHA256
        return QUANTFUNC_R256_SHA256

    monkeypatch.setattr("voxprint_forge.diffusion.int4.fetch.sha256_file", good_hash)

    def snap(**kwargs: object) -> None:
        folder = Path(str(kwargs["local_dir"]))
        repo = str(kwargs["repo_id"])
        folder.mkdir(parents=True, exist_ok=True)
        if repo.startswith("Wan"):
            for name in ("high_noise_model", "low_noise_model"):
                (folder / name).mkdir(parents=True, exist_ok=True)
                (folder / name / "model.safetensors").write_bytes(b"wan")
        elif repo.startswith("Qwen"):
            (folder / "model_index.json").write_text("{}", encoding="utf-8")
        else:
            (folder / "transformer.safetensors").write_bytes(b"fp8")

    monkeypatch.setattr("voxprint_forge.diffusion.int4.fetch.snapshot_download", snap)
    calls: list[str] = []

    def refuse(_repo: str, _revision: str, _token: str | None) -> str:
        calls.append("licence")
        raise LicenceError("refused", code="non_permissive")

    with pytest.raises(LicenceError):
        fetch_weights(tmp_path / "refused", None, licence=refuse)
    assert calls == ["licence"]
    assert not (tmp_path / "refused" / "quantfunc").exists()
    paths = fetch_weights(tmp_path / "ok", None, licence=lambda *_args: "apache-2.0")
    assert Path(paths["FORGE_QUANTFUNC"], QUANTFUNC_R128).is_file()
    assert Path(paths["FORGE_QUANTFUNC"], QUANTFUNC_R32).is_file()
    assert (Path(paths["FORGE_WAN_CKPT"]) / "high_noise_model" / "model.safetensors").is_file()
    assert json.loads((tmp_path / "ok" / "paths.json").read_text(encoding="utf-8"))["FORGE_QWEN_EDIT"]


def test_drive_folder_id_is_the_voxprint_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    def upload(_files: list[object], *, model_id: str, date: str) -> list[dict[str, object]]:
        del model_id, date
        seen["folder"] = os.environ["FORGE_DRIVE_ROOT_FOLDER_ID"]
        return []

    monkeypatch.setattr("voxprint_forge.diffusion.int4.publish.upload_drive", upload)
    live_drive(DRIVE_FOLDER_ID, [])
    assert seen["folder"] == "13qzgTnnkqvZrf_DK9fmymJ8ICdOuwDJT"


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


def test_script_documents_the_rental_contract() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    for needle in (
        "FORGE_HOST_KIND",
        "--dry-run",
        "--smoke-tiny",
        "--preflight",
        "PREFLIGHT-OK",
        "--qwen-image",
        "watch.json",
        "status.json",
        "0.15",
        "exports.sh",
        "ALL-DONE",
        "sm_80",
        "sm_90",
        "1.8",
        "cu128",
        "80;90",
        "A100 or H100 80GB",
        "--own-edit",
        "--phase0-result",
        "FORGE_OWN_EDIT:-on",
        "FORGE_SKIP_WAN",
        "FORGE_WAIT_FOR",
        "FORGE_PRIVATE_CALIB",
        "FORGE_COMMON_SRC",
        "--no-deps",
        "datasets>=3.0,<4",
        "pinned %s already at %s",
    ):
        assert needle in text


def test_h100_quant_minutes_and_fp8_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    assert parse_compute_cap("NVIDIA H100 80GB HBM3, 9.0") == (9, 0)
    assert fp8_mode(8, 0) == "emulated"
    assert fp8_mode(8, 9) == "native"
    assert fp8_mode(9, 0) == "native"
    steps = planned_steps(qwen_image=False)
    quant = next(step for step in steps if step.step_id == "qwen-edit-2511-quant")
    assert step_minutes(quant, "a100") == 900
    assert step_minutes(quant, "h100") == 500
    assert total_minutes(steps, "a100") == 1260
    assert total_minutes(steps, "h100") == 860
    optional = planned_steps(qwen_image=True)
    assert total_minutes(optional, "a100") == 2270
    assert total_minutes(optional, "h100") == 1470
    monkeypatch.delenv("FORGE_GPU", raising=False)
    monkeypatch.delenv("FORGE_COMPUTE_CAP", raising=False)
    assert planning_gpu() == "a100"
    monkeypatch.setenv("FORGE_COMPUTE_CAP", "9.0")
    assert planning_gpu() == "h100"
    native = device_fp8_report(probe=False)
    assert native["mode"] == "native"
    assert native["sm"] == "sm_90"
    assert native["note"] == NATIVE_NOTE
    assert native["speed_vs_fp8_representative"] is True
    monkeypatch.setenv("FORGE_GPU", "a100")
    monkeypatch.setenv("FORGE_COMPUTE_CAP", "8.0")
    emulated = device_fp8_report(probe=False)
    assert emulated["gpu"] == "a100"
    assert emulated["sm"] == "sm_80"
    assert emulated["mode"] == "emulated"
    assert emulated["note"] == EMULATED_NOTE
    assert emulated["speed_vs_fp8_representative"] is False
    monkeypatch.setenv("FORGE_GPU", "h100")
    watched = build_watch(qwen_image=False, done=set(), gpu=planning_gpu())
    assert watched["gpu"] == "h100"
    assert watched["expected_min"] == 860
    quant_job = next(job for job in watched["jobs"] if job["id"] == "qwen-edit-2511-quant")
    assert quant_job["expected_min"] == 500


def test_eval_records_emulated_fp8_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_COMPUTE_CAP", "8.0")
    monkeypatch.delenv("FORGE_GPU", raising=False)
    ctx = StageContext(repo=ROOT, forge_out=tmp_path / "out", dry_run=True, svd_rank=256)
    payload = eval_step(ctx, "qwen-image-edit-2511")
    baseline = payload["fp8_baseline"]
    assert isinstance(baseline, dict)
    assert baseline["mode"] == "emulated"
    assert "not representative" in str(baseline["note"])
    assert any("not representative" in line for line in _eval_lines(payload))


def test_native_fp8_binds_a_linear_and_refuses_zero() -> None:
    parent = torch.nn.Sequential(torch.nn.Linear(16, 16, bias=False))
    weight = torch.ones(16, 16, dtype=torch.float8_e4m3fn)
    scale = torch.ones((), dtype=torch.float32)
    bound = bind_fp8_linears(parent, {"0.weight": (weight, scale)})
    assert bound == 1
    assert parent[0].weight.dtype == torch.float8_e4m3fn
    assert bind_fp8_linears(torch.nn.Sequential(torch.nn.Linear(16, 16)), {}) == 0
    with pytest.raises(StageError) as exc:
        _require_native_bindings(0)
    assert exc.value.code == "fp8_native"


def test_estimates_print_both_gpu_tables(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_GPU", raising=False)
    monkeypatch.delenv("FORGE_COMPUTE_CAP", raising=False)
    assert main(["--root", str(ROOT), "estimates"]) == 0
    text = capsys.readouterr().out
    assert "planning gpu: a100" in text
    assert "A100 80GB" in text
    assert "H100 80GB" in text
    assert "total  1260 min" in text
    assert "total  860 min" in text
    assert "qwen-edit-2511-quant  500 min" in text
    monkeypatch.setenv("FORGE_GPU", "h100")
    assert main(["--json", "--root", str(ROOT), "estimates", "--qwen-image"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["gpu"] == "h100"
    assert payload["result"]["expected_min"] == 1470
    assert payload["result"]["gpus"]["a100"]["expected_min"] == 2270
    assert payload["result"]["gpus"]["h100"]["expected_min"] == 1470


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


def _phase0(
    path: Path,
    *,
    fits: bool = True,
    cases: tuple[str, ...] = ("c1", "c2", "c3"),
    ocr_match: bool | None = True,
    extra_fail: bool = False,
) -> Path:
    def one(case_id: str) -> dict[str, object]:
        scores: dict[str, object] = {"identity_cosine_proxy": 0.9}
        if case_id == "c3":
            scores["ocr_match"] = ocr_match
        return {"id": case_id, "seconds": 1.5, "output": f"{case_id}.webp", "scores": scores}

    arm = {
        "id": "qwen-r128",
        "rank": 128,
        "weight_file": QUANTFUNC_R128,
        "caps": {"cap16": {"ok": True, "error": "", "fits_16gb": fits, "cases": [one(case_id) for case_id in cases]}},
    }
    arms = [arm]
    if extra_fail:
        arms.append({"id": "z-image-r128", "caps": {"cap16": {"ok": True, "fits_16gb": False, "cases": []}}})
    payload = {
        "schema": "voxprint-forge-nunchaku-edit/1",
        "fits_16gb": fits and not extra_fail,
        "arms": arms,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_phase0_baseline_keeps_the_own_quant(tmp_path: Path) -> None:
    phase0 = _phase0(tmp_path / "result.json", extra_fail=True)
    ctx, done, watch = _ctx(tmp_path)
    result = run(
        ctx,
        done=done,
        watch=watch,
        own_edit=decide_own_edit("on", phase0),
        phase0_result=phase0,
    )
    assert result["winner"] == "own-r256"
    assert result["expected_min"] == 1260
    ids = [row["id"] for row in result["steps"]]
    assert "qwen-edit-2511-quant" in ids
    assert "wan-2.2-gguf" in ids
    assert "qwen-edit-2511-mirror" not in ids
    comparison = json.loads((ctx.job("qwen-image-edit-2511") / "comparison.json").read_text(encoding="utf-8"))
    baseline = comparison["phase0_baseline"]
    assert isinstance(baseline, dict)
    assert baseline["summary_fits_16gb"] is False
    arms = baseline["arms"]
    assert isinstance(arms, list)
    assert arms[0]["id"] == "qwen-r128"
    assert arms[0]["fits_16gb"] is True
    assert any("does not choose the winner" in line for line in _eval_lines(comparison))
    status = read_status(ctx.forge_out)
    assert status["own_edit"]["action"] == "quantize"
    assert status["own_edit"]["mode"] == "on"
    assert phase0.is_file()


def test_own_edit_auto_mirrors_r128_and_r32_then_runs_wan(tmp_path: Path) -> None:
    phase0 = _phase0(tmp_path / "result.json", extra_fail=True)
    ctx, done, watch = _ctx(tmp_path)
    result = run(
        ctx,
        done=done,
        watch=watch,
        own_edit=decide_own_edit("auto", phase0),
        phase0_result=phase0,
    )
    ids = [row["id"] for row in result["steps"]]
    assert "qwen-edit-2511-quant" not in ids
    assert "qwen-edit-2511-mirror" in ids
    assert "wan-2.2-publish" in ids
    assert result["expected_min"] == 220
    assert result["winner"] is None
    mirror = next(row for row in result["steps"] if row["id"] == "qwen-edit-2511-mirror")
    assert mirror["slug"] == "voxprint-qwen-image-edit-2511-quantfunc-int4"
    assert mirror["verified"] is True
    sources = [Path(str(item)) for item in mirror["sources"]]
    assert {path.name for path in sources} == {QUANTFUNC_R128, QUANTFUNC_R32}
    assert all(not path.exists() for path in sources)
    assert phase0.is_file()
    card = (ctx.job("qwen-image-edit-2511") / "publish" / "README.md").read_text(encoding="utf-8")
    assert "did not re-quantize" in card
    assert QUANTFUNC_R128 in card and QUANTFUNC_R32 in card
    status = read_status(ctx.forge_out)
    assert status["own_edit"]["action"] == "mirror"
    assert status["own_edit"]["fits_16gb"] is True
    assert status["own_edit"]["hard_cases"] == ["c1", "c2", "c3"]


def test_own_edit_auto_keeps_quant_when_the_gate_fails(tmp_path: Path) -> None:
    unfit = _phase0(tmp_path / "unfit.json", fits=False)
    assert decide_own_edit("auto", unfit).action == "quantize"
    ctx, done, watch = _ctx(tmp_path)
    result = run(ctx, done=done, watch=watch, own_edit=decide_own_edit("auto", unfit), phase0_result=unfit)
    assert result["winner"] == "own-r256"
    assert result["expected_min"] == 1260
    missing_case = _phase0(tmp_path / "smoke.json", cases=("c1",))
    passed, reason, _fits, _cases = quantfunc_r128_gate(json.loads(missing_case.read_text(encoding="utf-8")))
    assert passed is False
    assert "c2" in reason
    ocr = _phase0(tmp_path / "ocr.json", ocr_match=False)
    passed, reason, _fits, _cases = quantfunc_r128_gate(json.loads(ocr.read_text(encoding="utf-8")))
    assert passed is False
    assert "c3" in reason
    absent = tmp_path / "absent.json"
    assert decide_own_edit("auto", absent).action == "quantize"
    assert load_phase0_baseline(absent) is None
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(StageError) as exc:
        load_phase0_baseline(broken)
    assert exc.value.code == "phase0"


def test_own_edit_off_mirrors_without_a_phase0_file(tmp_path: Path) -> None:
    missing = tmp_path / "absent.json"
    ctx, done, watch = _ctx(tmp_path)
    result = run(
        ctx,
        done=done,
        watch=watch,
        own_edit=decide_own_edit("off", missing),
        phase0_result=missing,
    )
    assert result["own_edit"]["action"] == "mirror"
    assert result["phase0_baseline"] is None
    assert "qwen-edit-2511-quant" not in [row["id"] for row in result["steps"]]
    assert "wan-2.2-gguf" in [row["id"] for row in result["steps"]]


def test_cli_default_own_edit_quantizes_a_passing_phase0(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    phase0 = _phase0(tmp_path / "result.json")
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
            "--phase0-result",
            str(phase0),
        ]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["result"]["own_edit"]["mode"] == "on"
    assert payload["result"]["own_edit"]["action"] == "quantize"
    assert payload["result"]["winner"] == "own-r256"
    assert payload["result"]["phase0_baseline"]["arms"][0]["fits_16gb"] is True


def test_estimates_auto_prints_the_mirror_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    phase0 = _phase0(tmp_path / "result.json")
    assert main(["--root", str(ROOT), "estimates", "--own-edit", "auto", "--phase0-result", str(phase0)]) == 0
    text = capsys.readouterr().out
    assert "own quant is skipped" in text
    assert "does not choose the winner" in text
    assert "qwen-edit-2511-mirror" in text
    assert "total  220 min" in text
    assert "qwen-edit-2511-quant" not in text
    broken = tmp_path / "broken.json"
    broken.write_text("[]", encoding="utf-8")
    assert main(["--json", "--root", str(ROOT), "estimates", "--phase0-result", str(broken)]) == 6
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "phase0"


def test_skip_wan_drops_steps_and_the_weight_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_SKIP_WAN", "1")
    assert skip_wan() is True
    assert expected_minutes(qwen_image=False) == 1070
    assert all(not step.step_id.startswith("wan-") for step in planned_steps(qwen_image=False))
    seen: list[str] = []

    def licence(repo: str, _revision: str, _token: str | None) -> str:
        seen.append(repo)
        return "apache-2.0"

    def download(**kwargs: object) -> None:
        path = Path(str(kwargs["local_dir"])) / str(kwargs["filename"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"bytes")

    def good_hash(path: Path) -> str:
        if path.name == QUANTFUNC_R128:
            return QUANTFUNC_R128_SHA256
        if path.name == QUANTFUNC_R32:
            return QUANTFUNC_R32_SHA256
        return QUANTFUNC_R256_SHA256

    def snap(**kwargs: object) -> None:
        folder = Path(str(kwargs["local_dir"]))
        repo = str(kwargs["repo_id"])
        assert not repo.startswith("Wan")
        folder.mkdir(parents=True, exist_ok=True)
        if repo.startswith("Qwen"):
            (folder / "model_index.json").write_text("{}", encoding="utf-8")
        else:
            (folder / "transformer.safetensors").write_bytes(b"fp8")

    monkeypatch.setattr("voxprint_forge.diffusion.int4.fetch.hf_hub_download", download)
    monkeypatch.setattr("voxprint_forge.diffusion.int4.fetch.sha256_file", good_hash)
    monkeypatch.setattr("voxprint_forge.diffusion.int4.fetch.snapshot_download", snap)
    paths = fetch_weights(tmp_path / "ok", None, licence=licence)
    assert paths["FORGE_WAN_CKPT"] == ""
    assert all(not repo.startswith("Wan") for repo in seen)


def test_configs_follow_the_deepcompressor_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = tmp_path / "deepcompressor"
    monkeypatch.setenv("FORGE_DEEPCOMPRESSOR", str(checkout))
    configs = write_configs(tmp_path / "work", "qwen-image-edit-2511", rank=8)
    expected = checkout / "examples" / "diffusion" / "configs" / "voxprint" / "qwen-image-edit-2511"
    assert configs["model"].parent == expected
    assert configs["overlay"].is_file()


def test_private_calib_is_optional(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calib._PRIVATE_NOTICES.clear()
    monkeypatch.delenv("FORGE_PRIVATE_CALIB", raising=False)
    public = calibration_prompts(ROOT)
    assert [item["id"] for item in public] == [item["id"] for item in load_prompts(ROOT)]
    assert "FORGE_PRIVATE_CALIB is unset" in capsys.readouterr().err
    monkeypatch.setenv("FORGE_PRIVATE_CALIB", str(tmp_path / "missing"))
    skipped = calibration_prompts(ROOT)
    assert [item["id"] for item in skipped] == [item["id"] for item in public]
    assert "could not be loaded" in capsys.readouterr().err
    private = tmp_path / "private"
    (private / "images").mkdir(parents=True)
    (private / "images" / "hat.png").write_bytes(b"png")
    (private / "prompts.json").write_text(
        json.dumps(
            {
                "prompts": [
                    {
                        "id": "hat-1",
                        "lang": "en",
                        "kinds": ["edit"],
                        "prompt": "Add a hat",
                        "image": "images/hat.png",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("FORGE_PRIVATE_CALIB", str(private))
    rows = calibration_prompts(ROOT)
    assert rows[-1]["id"] == "hat-1"
    assert rows[-1]["prompt"] == "Add a hat"
    assert Path(str(rows[-1]["image"])).is_file()
    captured = capsys.readouterr().err
    assert "Add a hat" not in captured

    def snap(**kwargs: object) -> None:
        folder = Path(str(kwargs["local_dir"]))
        (folder / "images").mkdir(parents=True, exist_ok=True)
        (folder / "images" / "hat.png").write_bytes(b"png")
        (folder / "prompts.json").write_text(
            json.dumps(
                {
                    "prompts": [
                        {
                            "id": "hat-hf",
                            "prompt": "Add a hat",
                            "image": "images/hat.png",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["repo_id"] == "example/private-calib"

    monkeypatch.setattr("voxprint_forge.diffusion.int4.calib.snapshot_download", snap)
    monkeypatch.setenv("FORGE_PRIVATE_CALIB", "example/private-calib")
    monkeypatch.setenv("FORGE_OUT", str(tmp_path / "out"))
    remote = calibration_prompts(ROOT)
    assert remote[-1]["id"] == "hat-hf"


def test_eval_isolates_comparison_arms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    export = tmp_path / "work" / "diffusion-int4" / "qwen-image-edit-2511" / "export"
    export.mkdir(parents=True)
    (export / "model.safetensors").write_bytes(b"own")
    quant = tmp_path / "quantfunc"
    quant.mkdir()
    (quant / QUANTFUNC_R128).write_bytes(b"a")
    (quant / QUANTFUNC_R256).write_bytes(b"b")
    fp8 = tmp_path / "fp8"
    fp8.mkdir()
    monkeypatch.setenv("FORGE_QUANTFUNC", str(quant))
    monkeypatch.setenv("FORGE_FP8_EDIT", str(fp8))

    def time_pipe(
        _pipe: object,
        _cases: list[dict[str, object]],
        _images: Path,
        save_dir: Path | None = None,
        arm_id: str = "",
    ) -> tuple[float, float]:
        if arm_id == "quantfunc-r128":
            raise RuntimeError("missing smooth_factor")
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            (save_dir / f"{arm_id}-c1.png").write_bytes(b"png")
        return 1000.0, 2.0

    class _Cuda:
        def empty_cache(self) -> None:
            return None

        def is_available(self) -> bool:
            return True

    monkeypatch.setattr(measure, "require_edit_runtime", lambda: None)
    monkeypatch.setattr(measure, "QwenImageEditPlusPipeline", object())
    monkeypatch.setattr(measure, "load_edit_pipeline", lambda _path: object())
    monkeypatch.setattr(measure, "load_fp8_edit", lambda _path, mode: object())
    monkeypatch.setattr(measure, "_time_pipe", time_pipe)
    monkeypatch.setattr(measure, "_live_fp8_mode", lambda: "emulated")
    monkeypatch.setattr(measure, "torch", type("Torch", (), {"cuda": _Cuda(), "bfloat16": None})())
    ctx = StageContext(repo=ROOT, forge_out=tmp_path, dry_run=False, svd_rank=256)
    arms = measure.measure_live(ctx, "qwen-image-edit-2511")
    assert [arm.arm_id for arm in arms] == ["own-r256", "quantfunc-r256", "fp8"]
    job = ctx.job("qwen-image-edit-2511")
    assert (job / "eval-images" / "own-r256-c1.png").is_file()
    errors = json.loads((job / "eval-errors.json").read_text(encoding="utf-8"))
    assert "smooth_factor" in errors["quantfunc-r128"]

    def own_fails(
        _pipe: object,
        _cases: list[dict[str, object]],
        _images: Path,
        save_dir: Path | None = None,
        arm_id: str = "",
    ) -> tuple[float, float]:
        del save_dir
        if arm_id.startswith("own-"):
            raise RuntimeError("own export failed")
        return 1.0, 1.0

    monkeypatch.setattr(measure, "_time_pipe", own_fails)
    with pytest.raises(RuntimeError, match="own export failed"):
        measure.measure_live(ctx, "qwen-image-edit-2511")


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
