"""H100 retune: native FP8, sm_90 gates, attention, and the synth4 setup script."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from torch import nn
from torch.nn.attention import SDPBackend

from voxprint_forge.compare.attention import attention_scope, sdpa_backends
from voxprint_forge.compare.cli import execute
from voxprint_forge.compare.codes import CompareSetup
from voxprint_forge.compare.engine import _control_file, _load_state, _place
from voxprint_forge.compare.fp8_make import produce_fp8
from voxprint_forge.compare.metrics import finish_score
from voxprint_forge.compare.plan import Arm, Job, load_compare_plan
from voxprint_forge.compare.run import gate_fp8_native, gate_requires_cc, run_compare
from voxprint_forge.diffusion.fp8 import FORMAT, quantize_weight, read_format
from voxprint_forge.diffusion.fp8_native import Fp8Linear, apply_fp8_native, self_check
from voxprint_forge.diffusion.names import QWEN3_FP8_NAME, TURBO_FP8_NAME
from voxprint_forge.errors import ToolMissingError
from voxprint_forge.kl.preflight import require_llama_arch, sass_archs

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "plans" / "compare-synth4.toml"
HOPPER = "nunchaku INT4 has no sm_90 kernels (Hopper lacks 4-bit tensor cores)"


def _arm(**overrides: object) -> Arm:
    raw: dict[str, object] = {
        "id": "c",
        "role": "candidate",
        "repo": "",
        "revision": "",
        "quant": "fp8",
        "compute": "fp8-native",
        "offload": False,
        "files": (),
        "enabled": True,
        "skip_reason": "",
        "base_repo": "",
        "base_revision": "",
        "reuse_job": "",
        "reuse_arm": "",
        "local": True,
    }
    raw.update(overrides)
    return Arm(**raw)  # type: ignore[arg-type]


def _fp8(rows: int, cols: int) -> torch.Tensor:
    payload = torch.arange(rows * cols, dtype=torch.uint8)
    return payload.view(torch.float8_e4m3fn).reshape(rows, cols)


def _score(reference: Image.Image, candidate: Image.Image, case: object) -> object:
    del reference, candidate, case
    return finish_score(
        lpips=0.02,
        ssim=0.99,
        psnr=40.0,
        psnr_saturated=False,
        dino_cosine=0.91,
        clip_i=0.95,
        clip_t_reference=0.30,
        clip_t_candidate=0.30,
    )


def test_fp8_linear_keeps_file_bytes_and_the_bf16_scale() -> None:
    payload = torch.arange(32 * 16, dtype=torch.uint8)
    weight = payload.view(torch.float8_e4m3fn).reshape(32, 16)
    scale = torch.tensor(0.5, dtype=torch.bfloat16)
    layer = Fp8Linear(weight, scale, None, "dynamic")
    assert layer.weight.dtype == torch.float8_e4m3fn
    stored = layer.weight.detach().view(torch.uint8).reshape(-1).cpu()
    assert torch.equal(stored, payload)
    assert layer.weight_scale.dtype == torch.float32
    assert layer.weight_scale.ndim == 0
    assert float(layer.weight_scale) == pytest.approx(0.5)
    output = layer(torch.randn(1, 16, dtype=torch.bfloat16))
    assert output.shape == (1, 32)
    assert output.dtype == torch.bfloat16


def test_apply_swaps_aligned_linears_and_falls_back() -> None:
    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ok = nn.Linear(16, 32, bias=False)
            self.odd = nn.Linear(15, 16, bias=False)

    block = Block()
    state = {
        "ok.weight": _fp8(32, 16),
        "ok.weight.scale_weight": torch.tensor(0.5, dtype=torch.bfloat16),
        "odd.weight": _fp8(16, 15),
        "odd.weight.scale_weight": torch.tensor(0.25, dtype=torch.bfloat16),
    }
    stats = apply_fp8_native(block, state)
    assert stats["layers_swapped"] == 1
    assert stats["layers_fallback"] == 1
    assert isinstance(block.ok, Fp8Linear)
    assert isinstance(block.odd, nn.Linear)
    assert block.odd.weight.dtype == torch.bfloat16


def test_apply_rejects_sparse_coverage() -> None:
    class Wide(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.a = nn.Linear(16, 16, bias=False)
            self.b = nn.Linear(16, 16, bias=False)
            self.c = nn.Linear(16, 16, bias=False)
            self.d = nn.Linear(16, 16, bias=False)

    state = {
        "a.weight": _fp8(16, 16),
        "a.weight.scale_weight": torch.tensor(1.0, dtype=torch.bfloat16),
    }
    with pytest.raises(CompareSetup, match="matched") as caught:
        apply_fp8_native(Wide(), state)
    assert caught.value.code == "compare_weights"


def test_self_check_passes_for_a_quantized_weight() -> None:
    weight, scale = quantize_weight(torch.randn(32, 16))
    self_check(Fp8Linear(weight, scale, None, "dynamic"))


def test_place_keeps_the_fp8_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    del monkeypatch
    weight, scale = quantize_weight(torch.randn(32, 16))
    layer = Fp8Linear(weight, scale, None, "dynamic")

    class _Pipe:
        def __init__(self) -> None:
            self.forge_fp8_layer = layer
            self.calls: list[tuple[object, ...]] = []

        def to(self, *args: object, **kwargs: object) -> _Pipe:
            if kwargs or any(isinstance(item, torch.dtype) for item in args):
                raise AssertionError("dtype cast")
            self.calls.append(args)
            return self

        def set_progress_bar_config(self, disable: bool) -> None:
            del disable

    pipe = _Pipe()
    _place(pipe, _arm())
    assert pipe.calls == [("cuda",)]
    assert layer.weight.dtype == torch.float8_e4m3fn


def test_native_loader_does_not_dequantise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "weights.safetensors"
    save_file(
        {
            "layers.0.weight": torch.zeros(16, 16, dtype=torch.float8_e4m3fn),
            "layers.0.weight.scale_weight": torch.tensor(1.0, dtype=torch.bfloat16),
        },
        path,
    )

    def boom(source: Path) -> dict[str, torch.Tensor]:
        del source
        raise AssertionError("dequant")

    monkeypatch.setattr("voxprint_forge.diffusion.fp8.load_dequant_bf16", boom)
    state = _load_state(path, native=True)
    assert state["layers.0.weight"].dtype == torch.float8_e4m3fn


def test_native_control_file_is_not_materialised(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    weights = tmp_path / "weights"
    weights.mkdir()
    path = weights / "lite.safetensors"
    save_file({"control_layers.0.weight": torch.zeros(2, 2)}, path)

    def boom(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("materialise")

    monkeypatch.setattr("voxprint_forge.diffusion.weights.materialise_bf16", boom)
    assert _control_file(_arm(), tmp_path) == path


def test_fp8_native_plan_requires_quant_fp8(tmp_path: Path) -> None:
    path = tmp_path / "plan.toml"
    path.write_text(
        "\n".join(
            [
                "[[jobs]]",
                'id = "T"',
                'kind = "turbo"',
                "steps = 8",
                "guidance = 0",
                'reference = "a"',
                "[[jobs.arms]]",
                'id = "a"',
                'role = "reference"',
                'quant = "bf16"',
                'compute = "bf16"',
                "[[jobs.arms]]",
                'id = "b"',
                'role = "candidate"',
                'quant = "bf16"',
                'compute = "fp8-native"',
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(CompareSetup, match="fp8-native requires quant fp8"):
        load_compare_plan(path)


def test_sm80_skips_native_and_keeps_dequant() -> None:
    job = load_compare_plan(PLAN).job("D1")
    gated = gate_fp8_native(job, (8, 0))
    assert gated.arm("c").enabled is False
    assert gated.arm("c").skip_reason == "fp8-native needs sm_89+"
    assert gated.arm("d").enabled is False
    assert gated.arm("c-dq").enabled is True
    assert gated.arm("c-dq").compute == "bf16-dequant"
    kept = gate_fp8_native(job, (9, 0))
    assert kept.arm("c").enabled is True
    assert kept.arm("d").enabled is True


def test_d2_on_sm90_exits_before_generation(tmp_path: Path) -> None:
    job = load_compare_plan(PLAN).job("D2")
    kept = gate_requires_cc(job, (8, 0))
    assert kept.arm("r128").enabled is True
    calls: list[str] = []

    def generate(arm: Arm, case: object, current: Job) -> Image.Image:
        del case, current
        calls.append(arm.id)
        return Image.new("RGB", (8, 8))

    with pytest.raises(CompareSetup) as caught:
        run_compare(
            job,
            tmp_path / "results",
            private_root=None,
            generate=generate,
            score=_score,
            cuda=False,
            capability=(9, 0),
        )
    assert caught.value.exit_code == 2
    assert HOPPER in str(caught.value)
    assert calls == []


def test_sdpa_never_selects_math() -> None:
    chosen = sdpa_backends("sdpa")
    assert chosen is not None
    assert SDPBackend.MATH not in chosen
    assert sdpa_backends("sdpa-math") == (SDPBackend.MATH,)
    torch.backends.cudnn.benchmark = True
    with attention_scope("sdpa"):
        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cuda.matmul.allow_tf32 is False
    assert sdpa_backends("_flash_3_hub") is None


def test_make_fp8_is_idempotent(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshot"
    for folder in ("transformer", "text_encoder"):
        (snapshot / folder).mkdir(parents=True)
        save_file({"layers.0.weight": torch.ones(16, 16)}, snapshot / folder / "model.safetensors")
    job = load_compare_plan(PLAN).job("D1")
    out = tmp_path / "fp8"
    first = produce_fp8(job, out, snapshot)
    assert first["written"] == [TURBO_FP8_NAME, QWEN3_FP8_NAME]
    assert first["uploaded"] is False
    assert read_format(out / TURBO_FP8_NAME) == FORMAT
    second = produce_fp8(job, out, snapshot)
    assert second["written"] == []
    assert second["skipped"] == [TURBO_FP8_NAME, QWEN3_FP8_NAME]


def test_make_fp8_cli_requires_out(tmp_path: Path) -> None:
    with pytest.raises(CompareSetup, match="--out"):
        execute(ROOT, tmp_path, "D1", PLAN, make_fp8=True, out=None)


def test_summary_records_the_attention_setting(tmp_path: Path) -> None:
    turbo = load_compare_plan(PLAN).job("D1")
    arms = tuple(item for item in turbo.arms if item.id in {"a", "b"})
    narrowed = replace(turbo, arms=arms)
    report = run_compare(
        narrowed,
        tmp_path / "results",
        private_root=None,
        generate=lambda arm, case, current: Image.new("RGB", (8, 8)),
        score=_score,
        fetch=lambda arm, dest: None,
        smi=lambda: None,
        capability=(9, 0),
    )
    payload = json.loads(Path(str(report["summary"])).read_text(encoding="utf-8"))
    assert payload["gpu"]["attention"] == "sdpa"
    assert payload["gpu"]["cudnn_benchmark"] is False
    assert payload["gpu"]["allow_tf32"] is False


def test_llama_preflight_requires_sass(tmp_path: Path) -> None:
    binary = tmp_path / "llama-perplexity"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    assert require_llama_arch(binary, capability=(9, 0), elf_text="ELF sm_90\n") == ""
    assert "90" in sass_archs("sm_90\ncompute_89")
    assert sass_archs("compute_90") == set()
    with pytest.raises(ToolMissingError, match="no sm_90 SASS") as caught:
        require_llama_arch(binary, capability=(9, 0), elf_text="ELF sm_80\ncompute_90\n")
    assert caught.value.exit_code == 9
    warning = require_llama_arch(binary, capability=(9, 0), cuobjdump="")
    assert "cuobjdump" in warning


def test_cuda_arch_scripts_include_sm90() -> None:
    bootstrap = (ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    int4 = (ROOT / "scripts" / "vm_run_diffusion_int4.sh").read_text(encoding="utf-8")
    assert 'FORGE_CUDA_ARCHS:-80;89;90;120' in bootstrap
    assert "WANT=" in bootstrap
    assert 'CMAKE_CUDA_ARCHITECTURES="80;89;120"' not in bootstrap
    assert 'FORGE_CUDA_ARCHS:-80;89;90;120' in int4
    assert 'CMAKE_CUDA_ARCHITECTURES="80;90"' not in int4
    assert "venv-forge" in int4


def test_vm_setup_patches_the_converter_before_cmake() -> None:
    text = (ROOT / "scripts" / "vm_setup_synth4.sh").read_text(encoding="utf-8")
    live = text.split("exit 0", 1)[1]
    patch_at = live.index('voxprint_forge.llama_patch "$LLAMA_SRC/conversion/base.py"')
    cmake_at = live.index("cmake -S")
    assert patch_at < cmake_at
    assert patch_at > live.index("BUNDLED_LLAMA")
    assert patch_at > live.index('-e "${ROOT}[compare]"')
    assert "--extra-index-url" not in live
    assert '--constraint "$PINS"' in live
    assert live.index("missing build tools") < live.index("uv python install")
    assert "-G Ninja" in live and '-DCMAKE_CUDA_COMPILER="$CUDACXX"' in live
    assert "/usr/local/cuda/bin/nvcc" in live
    assert "bundled llama.cpp commit does not match scripts/llama.rev" in live


def test_vm_setup_dry_run(monkeypatch: pytest.MonkeyPatch) -> None:
    script = ROOT / "scripts" / "vm_setup_synth4.sh"
    indexed = subprocess.run(
        ["git", "ls-files", "-s", "--", "scripts/vm_setup_synth4.sh"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert indexed.returncode == 0
    assert indexed.stdout.split()[0] == "100755"
    if sys.platform != "win32":
        assert script.stat().st_mode & 0o111
    monkeypatch.setenv("FORGE_CUDA_ARCHS", "90-real")
    done = subprocess.run([_bash(), str(script), "--dry-run"], check=False, capture_output=True, text=True)
    assert done.returncode == 0
    assert "python_fallback: 3.14 3.13 3.12" in done.stdout
    assert "extra: compare" in done.stdout
    assert "cuda_archs: 90-real" in done.stdout
    assert "FORGE_LLAMA_BIN" in done.stdout
    assert "FORGE_LLAMA_SRC" in done.stdout
    assert "FORGE_VENV" in done.stdout
    assert "HF_TOKEN" not in done.stdout
    assert "HF_TOKEN" not in done.stderr
    assert "cu130" in done.stdout or "cu128" in done.stdout


def _bash() -> str:
    found = shutil.which("bash")
    if found:
        return found
    windows = Path(r"C:\Program Files\Git\bin\bash.exe")
    if windows.is_file():
        return str(windows)
    return "bash"
