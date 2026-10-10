"""Offload splits are preferred to a lower quant that only fits on the GPU."""

from __future__ import annotations

from pathlib import Path

import pytest

from voxprint_forge.fitmath import GIB, budget_bytes
from voxprint_forge.llama import Completed, Runner, server_argv
from voxprint_forge.measure import Candidate, placement_bytes
from voxprint_forge.pipeline import Host, _first_measured
from voxprint_forge.placement import diffusion_splits, first_fitting, llm_splits, options_for
from voxprint_forge.plan import load_plan
from voxprint_forge.preflight import ladder_gaps

_SHA = "ab" * 32


def test_llm_splits_prefer_gpu_then_kv_then_experts() -> None:
    plain = llm_splits(1000, 100)
    assert [item.name for item in plain] == ["gpu", "kv_ram"]
    assert plain[0].flags == ()
    assert plain[1].flags == ("--no-kv-offload",)
    assert plain[1].vram_bytes == 1000 and plain[1].ram_bytes == 100
    mixed = llm_splits(1000, 100, moe_layers=10, expert_fraction=0.25)
    assert mixed[2].name == "n_cpu_moe"
    assert mixed[2].flags == ("--n-cpu-moe", "2")
    assert mixed[2].ram_bytes == 250
    assert mixed[2].vram_bytes == 1000 - 250 + 100


def test_diffusion_splits_offload_the_encoder_or_load_one_piece() -> None:
    splits = diffusion_splits(10 * GIB, 2 * GIB, GIB)
    by_name = {item.name: item for item in splits}
    assert by_name["gpu"].vram_bytes == 13 * GIB
    assert by_name["encoder_offload"].vram_bytes == 11 * GIB
    assert by_name["encoder_offload"].ram_bytes == 2 * GIB
    assert by_name["sequential"].vram_bytes == 11 * GIB
    assert by_name["sequential"].ram_bytes == 2 * GIB


def test_options_for_keeps_kv_on_the_card_with_the_transformer() -> None:
    found = options_for(12 * GIB, GIB, encoder_gib=2)
    names = [item.name for item in found]
    assert names == ["gpu", "kv_ram", "encoder_offload", "sequential"]
    encoder = found[2]
    assert encoder.vram_bytes == 11 * GIB
    assert encoder.ram_bytes == 2 * GIB
    assert first_fitting(found, budget_gb=16, headroom_gb=3, ram_bytes=64 * GIB).name == "gpu"


def test_first_fitting_skips_gpu_when_the_card_is_short() -> None:
    weights = budget_bytes(16, 3)
    splits = llm_splits(weights, GIB)
    assert first_fitting(splits, budget_gb=16, headroom_gb=3, ram_bytes=0) is None
    chosen = first_fitting(splits, budget_gb=16, headroom_gb=3, ram_bytes=2 * GIB)
    assert chosen is not None and chosen.name == "kv_ram"


def test_placement_bytes_ignore_the_mapped_file() -> None:
    log = (
        "load_tensors:   CPU_Mapped model buffer size =   521.00 MiB\n"
        "load_tensors:        CUDA0 model buffer size =   100.00 MiB\n"
        "load_tensors:          CPU model buffer size =    40.00 MiB\n"
        "llama_kv_cache:      CUDA0 KV buffer size =    10.00 MiB\n"
        "llama_kv_cache:        CPU KV buffer size =     8.00 MiB\n"
    )
    cuda_weights, cuda_kv, host = placement_bytes(log)
    assert cuda_weights == 100 * 1024 * 1024
    assert cuda_kv == 10 * 1024 * 1024
    assert host == 48 * 1024 * 1024


def test_server_argv_inserts_split_flags_before_fit_off() -> None:
    argv = server_argv(
        "llama-server",
        Path("model.gguf"),
        ctx=8192,
        port=8080,
        cache="q8_0",
        extra=("--no-kv-offload",),
    )
    assert argv[-4:] == ["--no-kv-offload", "-v", "--fit", "off"]


class _Recording(Runner):
    def __init__(self, *, gpu_fits: bool) -> None:
        self.gpu_fits = gpu_fits
        self.argvs: list[list[str]] = []

    def capture_until(self, argv: list[str], *, pattern: str, timeout: float = 180) -> Completed:
        del pattern, timeout
        self.argvs.append(list(argv))
        offload = "--no-kv-offload" in argv or "--n-cpu-moe" in argv
        if offload or self.gpu_fits:
            log = (
                "load_tensors:        CUDA0 model buffer size =   100.00 MiB\n"
                "llama_kv_cache:        CPU KV buffer size =    10.00 MiB\n"
                "llama_kv_cache:      CUDA0 KV buffer size =    10.00 MiB\n"
            )
        else:
            log = (
                "load_tensors:        CUDA0 model buffer size = 14000.00 MiB\n"
                "llama_kv_cache:      CUDA0 KV buffer size =  4000.00 MiB\n"
            )
        return Completed(argv=argv, returncode=0, stdout=log, stderr="")


def _plan(
    path: Path,
    *,
    moe_layers: int = 0,
    expert_fraction: float = 0.0,
    encoder_gib: float = 0.0,
    params_b: float = 1,
    quant: str = "Q5_K_M",
    also: str = "IQ4_XS",
) -> None:
    path.write_text(
        f"""
[run]
name = "place"
output_dir = "{path.parent.as_posix()}/work"
headroom_gb = 3.0
contexts = [8192]
artifact_host = ""
[[models]]
id = "placed"
enabled = true
mode = "requant"
repo = "org/placed"
revision = "rev"
licence = "apache-2.0"
params_b = {params_b}
base_model = "org/base"
source_format = "gguf"
credits = ["Org"]
moe_layers = {moe_layers}
expert_fraction = {expert_fraction}
encoder_gib = {encoder_gib}
[models.arch]
full_layers = 32
full_kv_heads = 8
full_head_dim = 128
[[models.files]]
role = "source"
quant = "F16"
path = "source.gguf"
sha256 = "{_SHA}"
size = 10
[[models.tiers]]
name = "16gb"
budget_gb = 16
kind = "quantize"
quants = ["{quant}", "{also}"]
""",
        encoding="utf-8",
    )


def _bins(tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = tmp / "bin"
    folder.mkdir()
    (folder / "llama-server").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("FORGE_LLAMA_BIN", str(folder))


def _files(tmp: Path, big: int) -> list[Candidate]:
    larger = tmp / "Q5.gguf"
    smaller = tmp / "IQ4.gguf"
    larger.write_bytes(b"B" * 32)
    smaller.write_bytes(b"S" * 32)
    return [
        Candidate("Q5_K_M", big, larger, "quantize"),
        Candidate("IQ4_XS", 4096, smaller, "quantize"),
    ]


def _host(runner: Runner) -> Host:
    return Host(runner=runner, downloader=None, remote=None)  # type: ignore[arg-type]


def _fit(tmp: Path, monkeypatch: pytest.MonkeyPatch, *, weights: int, gpu_fits: bool, ram: int):
    _bins(tmp, monkeypatch)
    monkeypatch.setattr("voxprint_forge.pipeline.memory_status", lambda: (64 * GIB, ram))
    _plan(tmp / "plan.toml")
    plan = load_plan(tmp / "plan.toml")
    runner = _Recording(gpu_fits=gpu_fits)
    payload = _first_measured(
        plan,
        plan.models[0],
        plan.models[0].tiers[0],
        _host(runner),
        _files(tmp, weights),
        (8192,),
    )
    return payload, runner


def test_gpu_split_is_measured_once_when_the_formula_fits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload, runner = _fit(tmp_path, monkeypatch, weights=4096, gpu_fits=True, ram=64 * GIB)
    assert payload["quant"] == "Q5_K_M"
    assert payload["split"] == "gpu"
    assert payload["split_flags"] == []
    assert payload["ram_bytes"] == 0
    assert len(runner.argvs) == 1
    assert "--no-kv-offload" not in runner.argvs[0]
    assert runner.argvs[0][-3:] == ["-v", "--fit", "off"]


def test_kv_ram_beats_a_smaller_quant_when_gpu_formula_misses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, runner = _fit(
        tmp_path,
        monkeypatch,
        weights=budget_bytes(16, 3),
        gpu_fits=False,
        ram=64 * GIB,
    )
    assert payload["quant"] == "Q5_K_M"
    assert payload["split"] == "kv_ram"
    assert payload["split_flags"] == ["--no-kv-offload"]
    assert payload["ram_bytes"] > 0
    assert runner.argvs
    assert all("--no-kv-offload" in argv for argv in runner.argvs)


def test_measured_gpu_miss_tries_offload_before_a_smaller_quant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload, runner = _fit(tmp_path, monkeypatch, weights=4096, gpu_fits=False, ram=64 * GIB)
    assert payload["quant"] == "Q5_K_M"
    assert payload["split"] == "kv_ram"
    assert "--no-kv-offload" not in runner.argvs[0]
    assert "--no-kv-offload" in runner.argvs[1]


def test_low_ram_keeps_the_smaller_quant_on_the_gpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload, runner = _fit(tmp_path, monkeypatch, weights=budget_bytes(16, 3), gpu_fits=True, ram=0)
    assert payload["quant"] == "IQ4_XS"
    assert payload["split"] == "gpu"
    assert all("--no-kv-offload" not in argv for argv in runner.argvs)


def test_moe_offload_is_measured_before_a_smaller_quant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _bins(tmp_path, monkeypatch)
    monkeypatch.setattr("voxprint_forge.pipeline.memory_status", lambda: (64 * GIB, 64 * GIB))
    _plan(tmp_path / "plan.toml", moe_layers=40, expert_fraction=0.5)
    plan = load_plan(tmp_path / "plan.toml")
    runner = _Recording(gpu_fits=False)
    weights = budget_bytes(16, 3) + GIB
    payload = _first_measured(
        plan,
        plan.models[0],
        plan.models[0].tiers[0],
        _host(runner),
        _files(tmp_path, weights),
        (8192,),
    )
    assert payload["quant"] == "Q5_K_M"
    assert payload["split"] == "n_cpu_moe"
    assert "--n-cpu-moe" in payload["split_flags"]
    assert any("--n-cpu-moe" in argv for argv in runner.argvs)
    assert all("--no-kv-offload" not in argv for argv in runner.argvs)


def test_encoder_offload_is_the_formula_and_skips_llama_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bins(tmp_path, monkeypatch)
    monkeypatch.setattr("voxprint_forge.pipeline.memory_status", lambda: (64 * GIB, 64 * GIB))
    _plan(tmp_path / "plan.toml", encoder_gib=2)
    plan = load_plan(tmp_path / "plan.toml")
    runner = _Recording(gpu_fits=False)
    weights = budget_bytes(16, 3) + GIB
    payload = _first_measured(
        plan,
        plan.models[0],
        plan.models[0].tiers[0],
        _host(runner),
        _files(tmp_path, weights),
        (8192,),
    )
    assert payload["quant"] == "Q5_K_M"
    assert payload["split"] == "encoder_offload"
    assert payload["measurements"][0]["measured"] is False
    assert runner.argvs == []


def test_ladder_accepts_moe_offload_when_the_card_cannot_hold_the_quant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("voxprint_forge.preflight.memory_status", lambda: (64 * GIB, 64 * GIB))
    path = tmp_path / "plan.toml"
    _plan(path, moe_layers=32, expert_fraction=0.5, params_b=20, quant="Q8_0", also="Q8_0")
    assert ladder_gaps(load_plan(path)) == []
    _plan(path, params_b=20, quant="Q8_0", also="Q8_0")
    assert ladder_gaps(load_plan(path))
    monkeypatch.setattr("voxprint_forge.preflight.memory_status", lambda: (64 * GIB, 0))
    _plan(path, moe_layers=32, expert_fraction=0.5, params_b=20, quant="Q8_0", also="Q8_0")
    assert ladder_gaps(load_plan(path))
