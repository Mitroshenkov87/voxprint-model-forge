"""A100 follow-ups: preflight, status, version, tokenizer patch, and the quant ladder."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from voxprint_forge.diffusion.alias import alias_key
from voxprint_forge.diffusion.planload import load_canvas_plan
from voxprint_forge.errors import StageError
from voxprint_forge.fitmath import estimate_weights_bytes, quant_rank
from voxprint_forge.llama import convert_argv
from voxprint_forge.llama_patch import (
    T_LITE_HASH,
    T_LITE_PRE,
    apply_t_lite,
    ids_match,
    parse_llama_tokenize,
    pretokenizer_hash,
    recognised,
)
from voxprint_forge.plan import load_plan
from voxprint_forge.preflight import ladder_gaps, run_preflight
from voxprint_forge.schedule import Gate
from voxprint_forge.statusfile import beating, inspect_status, update_item
from voxprint_forge.version import version_line

ROOT = Path(__file__).resolve().parents[1]
_KIMI = (
    '        if chkhsh == "81212dc7cdb7e0c1074ca62c5aeab0d43c9f52b8a737be7b12a777c953027890":\n'
    "            # ref: https://huggingface.co/moonshotai/Kimi-K2-Base\n"
    '            res = "kimi-k2"\n'
)


def test_gemma_16gb_ladder_reaches_iq3_xxs() -> None:
    plan = load_plan(ROOT / "plans" / "plan.toml")
    gemma = next(model for model in plan.models if model.id == "gemma4-31b")
    tier = next(item for item in gemma.tiers if item.name == "16gb" and item.kind == "quantize")
    assert tier.budget_gb == 13
    assert tier.quants[-2:] == ("IQ3_XXS", "IQ2_M")
    assert quant_rank("IQ3_XXS") > quant_rank("IQ2_M")
    ctx = max(plan.contexts)
    kv = gemma.arch.kv_bytes(ctx, "q8_0")
    budget = int(tier.budget_gb * (1024**3))
    assert estimate_weights_bytes(gemma.params_b, "IQ3_M") + kv > budget
    assert estimate_weights_bytes(gemma.params_b, "IQ3_XXS") + kv <= budget
    assert estimate_weights_bytes(gemma.params_b, "IQ2_M") + kv <= budget
    assert ladder_gaps(plan) == []
    for model_id in ("t-lite-it-2.1", "t-pro-it-2.1"):
        model = next(item for item in plan.models if item.id == model_id)
        assert any(item.path == "model.safetensors.index.json" for item in model.files)


def test_canvas_base_and_controlnet_min_free() -> None:
    plan = load_canvas_plan(ROOT / "plans" / "canvas-zimage-tiers-v1.toml")
    assert plan.part("base").min_free_gb == 40
    assert plan.part("controlnet").min_free_gb == 40
    assert plan.part("edit").min_free_gb == 100
    assert plan.min_free_gb == 120


def test_text_encoder_key_alias() -> None:
    current = {"layers.0.weight": object()}
    assert alias_key("model.layers.0.weight", current) == "layers.0.weight"
    assert alias_key("layers.0.weight", current) == "layers.0.weight"
    prefixed = {"model.layers.0.weight": object()}
    assert alias_key("layers.0.weight", prefixed) == "model.layers.0.weight"
    assert alias_key("other.weight", prefixed) is None


def test_t_lite_patch_and_tokenize_ids(tmp_path: Path) -> None:
    source = tmp_path / "base.py"
    source.write_text("def get_vocab_base_pre():\n" + _KIMI + "        return res\n", encoding="utf-8")
    assert apply_t_lite(source) == "applied"
    text = source.read_text(encoding="utf-8")
    assert recognised(text)
    assert T_LITE_HASH in text
    assert f'res = "{T_LITE_PRE}"' in text
    assert apply_t_lite(source) == "present"
    ids = [11, 22, 33]
    script = tmp_path / "llama-tokenize.py"
    script.write_text("print('11, 22, 33')\n", encoding="utf-8")
    parsed = parse_llama_tokenize(subprocess.check_output([sys.executable, str(script)], text=True))
    assert ids_match(ids, parsed)
    assert pretokenizer_hash(parsed) == pretokenizer_hash(ids)


def test_forge_python_falls_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = tmp_path / "convert_hf_to_gguf.py"
    script.write_text("# converter\n", encoding="utf-8")
    monkeypatch.setenv("FORGE_PYTHON", "/opt/forge/bin/python")
    argv = convert_argv(script, tmp_path / "src", tmp_path / "out.gguf")
    assert argv[0] == "/opt/forge/bin/python"
    monkeypatch.delenv("FORGE_PYTHON")
    argv = convert_argv(script, tmp_path / "src", tmp_path / "out.gguf")
    assert argv[0] == sys.executable


def test_version_line(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    from voxprint_forge.cli import main

    monkeypatch.delenv("GITHUB_RUN_NUMBER", raising=False)
    monkeypatch.delenv("FORGE_ROOT", raising=False)
    assert version_line() == '0.1.0 · build 0 "Charash"'
    code = main(["--version"])
    assert code == 0
    assert capsys.readouterr().out.strip() == '0.1.0 · build 0 "Charash"'


def test_cpu_slot_runs_beside_a_gpu_holder() -> None:
    gate = Gate(2)
    holding = threading.Event()
    release = threading.Event()

    def _gpu() -> None:
        with gate.slot(gpu=True):
            holding.set()
            assert release.wait(2)

    thread = threading.Thread(target=_gpu)
    thread.start()
    assert holding.wait(2)
    with gate.slot(gpu=False):
        overlapped = True
    release.set()
    thread.join(2)
    assert overlapped
    with pytest.raises(ValueError):
        Gate(0)


def test_status_flags_stalled_and_failed(tmp_path: Path) -> None:
    from voxprint_forge.cli import main

    forge_out = tmp_path / "forge-out"
    missing = main(["--forge-out", str(forge_out), "status"])
    assert missing == 4
    update_item(forge_out, "tiny", stage="quantize", pct=10, eta_min=4, state="RUNNING", pid=os.getpid())
    fresh = inspect_status(forge_out, alive=lambda _pid: True)
    assert fresh["exit_code"] == 0
    stalled = inspect_status(forge_out, alive=lambda _pid: False)
    assert stalled["stalled"] is True
    assert stalled["exit_code"] == 3
    path = forge_out / "status.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload["items"]
    assert isinstance(items, dict)
    row = items["tiny"]
    assert isinstance(row, dict)
    row["last_heartbeat"] = "2020-01-01T00:00:00Z"
    row["state"] = "STAGEFAIL quantize"
    path.write_text(json.dumps(payload), encoding="utf-8")
    failed = inspect_status(forge_out, now=datetime(2026, 10, 10, tzinfo=timezone.utc))
    assert failed["failed"] is True
    assert failed["exit_code"] == 4
    assert main(["--forge-out", str(forge_out), "--json", "status"]) == 4


def test_heartbeat_writes_a_progress_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGE_HEARTBEAT_S", "0.05")
    forge_out = tmp_path / "forge-out"
    with beating(forge_out, "tiny", "quantize", pct=12, eta_min=3):
        time.sleep(0.25)
    log = (forge_out / "progress.log").read_text(encoding="utf-8")
    assert log.count("quantize") >= 2
    body = json.loads((forge_out / "status.json").read_text(encoding="utf-8"))
    item = body["items"]["tiny"]
    assert item["pid"] == os.getpid()
    assert item["state"] == "RUNNING"
    assert item["pct"] == 12


class _Head:
    def __init__(self, status: int) -> None:
        self.status = status

    def close(self) -> None:
        return None


def _plan(path: Path, *, source_format: str = "gguf", budget: float = 16) -> None:
    sha = "ab" * 32
    mode = "ablate" if source_format == "safetensors" else "requant"
    index = ""
    if source_format == "safetensors":
        index = """
[[models.files]]
role = "sidecar"
quant = ""
path = "model.safetensors.index.json"
sha256 = ""
size = 0
"""
    filename = "model.safetensors" if source_format == "safetensors" else "source.gguf"
    path.write_text(
        f"""
[run]
name = "pre"
output_dir = "{path.parent.as_posix()}/work"
wall_clock_budget_min = 10
headroom_gb = 3
contexts = [512]
artifact_host = ""
[[models]]
id = "tiny"
enabled = true
mode = "{mode}"
repo = "org/tiny"
revision = "rev"
licence = "apache-2.0"
params_b = 1
base_model = "org/base"
source_format = "{source_format}"
credits = ["Org"]
[models.arch]
full_layers = 1
full_kv_heads = 1
full_head_dim = 4
[[models.files]]
role = "source"
quant = "BF16"
path = "{filename}"
sha256 = "{sha}"
size = 10
{index}
[[models.tiers]]
name = "16gb"
budget_gb = {budget}
kind = "quantize"
quants = ["IQ2_M"]
""",
        encoding="utf-8",
    )


def test_preflight_passes_and_reports_gaps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "bin"
    folder.mkdir()
    for name in ("llama-quantize", "llama-imatrix", "llama-cli", "llama-server"):
        (folder / name).write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("FORGE_LLAMA_BIN", str(folder))
    monkeypatch.setenv("HF_TOKEN", "hf-test-token-value")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "kaggle-test-token-value")
    monkeypatch.setattr("voxprint_forge.disk.free_bytes", lambda _path: 10 * 1024**3)
    plan_path = tmp_path / "plan.toml"
    _plan(plan_path)
    seen: list[str] = []

    def opener(request: object, timeout: float = 8) -> _Head:
        del timeout
        seen.append(str(getattr(request, "full_url", "")))
        return _Head(200)

    report = run_preflight(
        load_plan(plan_path),
        ROOT,
        tmp_path / "out",
        opener=opener,
        ram=lambda: 64 * 1024**3,
    )
    assert report["ok"] is True
    assert seen == []
    monkeypatch.delenv("HF_TOKEN")
    monkeypatch.delenv("KAGGLE_API_TOKEN")
    with pytest.raises(StageError):
        run_preflight(load_plan(plan_path), ROOT, tmp_path / "out", opener=opener, ram=lambda: 64 * 1024**3)
    monkeypatch.setenv("FORGE_DEFER_UPLOAD", "1")
    deferred = run_preflight(
        load_plan(plan_path),
        ROOT,
        tmp_path / "out",
        opener=opener,
        ram=lambda: 64 * 1024**3,
    )
    assert deferred["defer_upload"] is True
    _plan(plan_path, budget=0.01)
    with pytest.raises(StageError, match="fits no quant"):
        run_preflight(load_plan(plan_path), ROOT, tmp_path / "out", opener=opener, ram=lambda: 64 * 1024**3)


def test_preflight_heads_the_shard_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp_path / "bin"
    folder.mkdir()
    for name in ("llama-quantize", "llama-imatrix", "llama-cli", "llama-server"):
        (folder / name).write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("FORGE_LLAMA_BIN", str(folder))
    monkeypatch.setenv("FORGE_DEFER_UPLOAD", "1")
    monkeypatch.setattr("voxprint_forge.disk.free_bytes", lambda _path: 10 * 1024**3)
    plan_path = tmp_path / "plan.toml"
    _plan(plan_path, source_format="safetensors")
    calls: list[str] = []

    def opener(request: object, timeout: float = 8) -> _Head:
        del timeout
        url = str(getattr(request, "full_url", ""))
        calls.append(url)
        return _Head(404)

    with pytest.raises(StageError, match="shard index HEAD 404"):
        run_preflight(load_plan(plan_path), ROOT, tmp_path / "out", opener=opener, ram=lambda: 64 * 1024**3)
    assert calls and calls[0].endswith("/model.safetensors.index.json")
