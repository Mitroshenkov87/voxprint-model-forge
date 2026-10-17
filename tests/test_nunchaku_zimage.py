"""CPU tests for the Z-Image Nunchaku matrix. No GPU and no downloads."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from voxprint_forge.diffusion import nunchaku_zimage as edit

ROOT = Path(__file__).resolve().parents[1]


def _gpu() -> edit.GpuFacts:
    return edit.GpuFacts("NVIDIA GeForce RTX 4090", 24564, "610.88", 8, 9)


def _options(tmp: Path, argv: list[str]) -> object:
    return edit.parse_args(["--out", str(tmp), *argv])


def _fit_payload(mode: str, *, peak: float = 14000.0, fraction: float | None = 16 / 24) -> dict[str, object]:
    return {
        "ok": True,
        "oom": False,
        "error": None,
        "offload_mode": mode,
        "load_seconds": 3.0,
        "peak_vram_torch_mib": 9000.0,
        "peak_vram_smi_mib": peak,
        "memory_fraction": fraction,
        "seconds_per_image": {"c1": 2.0, "mean": 2.0},
        "cases": [{"id": "c1", "seconds": 2.0, "output": "c1.webp"}],
        "versions": {
            "torch": "2.11.0",
            "diffusers": "0.37.1",
            "transformers": "4.57.6",
            "nunchaku": "1.3.0.dev20260629",
        },
    }


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("NUNCHAKU_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("NUNCHAKU_DONE", str(tmp_path / "done"))
    monkeypatch.delenv("NUNCHAKU_VM_RESULTS", raising=False)
    return tmp_path


def test_zimage_prompts() -> None:
    zipped = {case.case_id: case.prompt for case in edit.ZIMAGE_CASES}
    assert "fully clothed" in zipped["z1"]
    assert "five clearly separated fingers" in zipped["z2"]
    assert "VOXPRINT" in zipped["z3"]
    assert "печать" in zipped["z3"]
    assert all(case.image is None for case in edit.ZIMAGE_CASES)


def test_default_arms_omit_r256_and_smoke_is_one_case() -> None:
    args = edit.parse_args(["--smoke"])
    assert args.arm_ids == ["zimage-r128"]
    assert edit.cases_for("zimage-r128", None, smoke=True) == (edit.ZIMAGE_CASES[0],)
    assert edit.steps_for("zimage-r256", True) == 8
    assert edit.steps_for("zimage-fp8", False) == 8
    with pytest.raises(edit.SetupError):
        edit.steps_for("missing", False)


def test_wheel_pins_match_the_plan(tmp_path: Path) -> None:
    windows = edit.select_wheel("win32")
    linux = edit.select_wheel("linux")
    assert windows.python == "3.12"
    assert windows.sha256 == "ddc015bfa6fc6b84957532823c23418f5b986cb61e3c02c97a646cfe2a1fd955"
    assert "cu13.0" in windows.filename
    assert linux.python == "3.11"
    assert linux.sha256 == "bcdfe6fcc36cd37da34c1d0f1e0fa7c9b4201b2a70a89a3274d22420221afd2f"
    assert "cu12.8" in linux.filename
    assert "linux_x86_64" in linux.filename
    commands = edit.venv_commands(tmp_path / "venv", linux)
    assert commands[0] == ["uv", "python", "install", "3.11"]
    assert commands[1][:4] == ["uv", "venv", "--python", "3.11"]
    assert "--managed-python" in commands[1]
    with pytest.raises(edit.SetupError):
        edit.select_wheel("darwin")


def test_install_never_selects_upstream_nunchaku(tmp_path: Path) -> None:
    wheel = tmp_path / edit.WINDOWS_WHEEL.filename
    commands = edit.build_install_commands(tmp_path / "python", wheel, edit.WINDOWS_WHEEL)
    edit.reject_upstream_install(commands)
    flat = [arg for command in commands for arg in command]
    assert "diffusers==0.37.1" in flat
    assert "transformers==4.57.6" in flat
    assert "--no-deps" in flat
    assert str(wheel) in flat
    assert "nunchaku" not in flat
    with pytest.raises(edit.SetupError):
        edit.reject_upstream_install([["uv", "pip", "install", "nunchaku"]])
    with pytest.raises(edit.SetupError):
        edit.reject_upstream_install([["uv", "pip", "install", "nunchaku==1.3.0"]])


def test_hash_mismatch_deletes_the_file_and_exits_2(tmp_path: Path) -> None:
    path = tmp_path / "wheel.whl"
    path.write_bytes(b"not-the-wheel")
    with pytest.raises(edit.SetupError) as caught:
        edit.require_hash(path, "ddc015bfa6fc6b84957532823c23418f5b986cb61e3c02c97a646cfe2a1fd955")
    assert caught.value.exit_code == 2
    assert not path.exists()


def test_gpu_gate() -> None:
    edit.gate_gpu(_gpu())
    edit.gate_gpu(edit.parse_gpu_csv("NVIDIA GeForce RTX 4090, 24564, 610.88, 8.9"))
    edit.gate_gpu(edit.GpuFacts("NVIDIA A100-SXM4-80GB", 81920, "535.129.03", 8, 0))
    edit.gate_gpu(edit.GpuFacts("RTX 3090", 24576, "550.54.15", 8, 6))
    with pytest.raises(edit.GpuError):
        edit.gate_gpu(edit.GpuFacts("V100", 32768, "535.129.03", 7, 0))
    with pytest.raises(edit.GpuError):
        edit.gate_gpu(edit.GpuFacts("A100", 81920, "470.57.02", 8, 0), platform="linux")
    with pytest.raises(edit.GpuError):
        edit.gate_gpu(edit.GpuFacts("small", 12000, "550.0", 8, 9))
    assert edit.GpuFacts("ok", 16000, "610.88", 8, 9).memory_mib == 16000
    edit.gate_gpu(edit.GpuFacts("borderline", 16000, "610.88", 12, 0))
    assert edit.weight_precision(edit.GpuFacts("A100", 81920, "535.129.03", 8, 0)) == "int4"
    assert edit.linux_driver_ok("535.129.03")
    assert not edit.linux_driver_ok("470.57.02")


def test_fits_16gb_and_memory_fraction() -> None:
    assert edit.fits_16gb(15.0 * 1024)
    assert not edit.fits_16gb(15.0 * 1024 + 0.1)
    assert not edit.fits_16gb(None)
    assert edit.memory_fraction("unconstrained", 81920) is None
    assert edit.memory_fraction("cap16", 81920) == pytest.approx(16 / 80)
    assert edit.memory_fraction("cap16", 24564) == pytest.approx((16 * 1024) / 24564)
    with pytest.raises(edit.SetupError):
        edit.memory_fraction("cap16")
    with pytest.raises(edit.SetupError):
        edit.memory_fraction("cap8", 81920)


def test_offload_plan_and_block_call() -> None:
    assert edit.plan_modes("zimage-r128", "unconstrained", "auto") == (
        "resident",
        "model-offload",
        "block-offload",
    )
    assert edit.plan_modes("zimage-r256", "cap16", "auto") == ("model-offload", "block-offload")

    class Transformer:
        def __init__(self) -> None:
            self.offload: tuple[object, ...] | None = None

        def set_offload(self, flag: bool, use_pin_memory: bool = False, num_blocks_on_gpu: int = 1) -> None:
            self.offload = (flag, use_pin_memory, num_blocks_on_gpu)

    class Pipe:
        def __init__(self) -> None:
            self.transformer = Transformer()
            self._exclude_from_cpu_offload: list[str] = []
            self.calls: list[str] = []

        def enable_model_cpu_offload(self) -> None:
            self.calls.append("model")

        def enable_sequential_cpu_offload(self) -> None:
            self.calls.append("sequential")

        def to(self, device: str) -> None:
            self.calls.append(device)

    pipe = Pipe()
    assert edit.apply_offload(pipe, "block-offload") == "block-offload"
    assert pipe.transformer.offload == (True, True, 1)
    assert pipe._exclude_from_cpu_offload == ["transformer"]
    assert pipe.calls == ["sequential"]
    edit.apply_offload(pipe, "model-offload")
    edit.apply_offload(pipe, "resident")
    assert pipe.calls == ["sequential", "model", "cuda"]


def test_disk_short_is_setup(tmp_path: Path) -> None:
    with pytest.raises(edit.SetupError) as caught:
        edit.require_free(tmp_path, 50, free=10)
    assert caught.value.exit_code == 2


def test_budget_skips_r256_first() -> None:
    arms = ["zimage-r128", "zimage-r256"]
    selected, _baselines, skipped = edit.plan_arm_ids(arms, 20)
    assert "zimage-r256" not in selected
    assert skipped == [{"id": "zimage-r256", "reason": "time_budget"}]
    kept, _, none = edit.plan_arm_ids(arms, 90)
    assert kept[-1] == "zimage-r256"
    assert none == []
    assert edit.skip_for_elapsed("zimage-r256", elapsed_min=80, budget_min=90, kind="int4") == "time_budget"
    assert edit.skip_for_elapsed("zimage-r128", elapsed_min=80, budget_min=90, kind="int4") is None
    assert edit.skip_for_elapsed("zimage-fp8", elapsed_min=85, budget_min=90, kind="fp8") == "time_budget"


def test_budget_that_drops_every_arm_fails(isolated: Path) -> None:
    def spawn(job: dict[str, object]) -> dict[str, object]:
        raise AssertionError(job)

    args = _options(
        isolated / "out",
        ["--arms", "zimage-r256", "--budget-min", "10", "--caps", "cap16", "--smoke"],
    )
    code, result = edit.execute(
        args,
        spawn=spawn,
        elapsed_min=lambda: 0.0,
        gpu=_gpu(),
        wheel=edit.WINDOWS_WHEEL,
        mirror_lookup=lambda _name: [],
    )
    assert code == 1
    assert result["ok"] is False
    assert result["arms"] == []
    assert result["skipped"] == [{"id": "zimage-r256", "reason": "time_budget"}]


def test_scores_and_smi_peak() -> None:
    side = 64
    pixels = [(20, 30, 40)] * (side * side)
    scores = edit.identity_and_ssim(pixels, pixels, side)
    assert scores["identity_cosine_proxy"] == pytest.approx(1.0)
    assert scores["background_ssim_proxy"] == pytest.approx(1.0)
    left = [(255, 255, 255) if index % 2 == 0 else (0, 0, 0) for index in range(side * side)]
    right = [(0, 0, 0) if index % 2 == 0 else (255, 255, 255) for index in range(side * side)]
    assert edit.frame_cosine(left, right) == pytest.approx(0.0, abs=1e-6)
    assert edit.note_sample(10, None) == 10
    assert edit.note_sample(10, 12) == 12


def test_fp8_strategy(tmp_path: Path) -> None:
    missing = tmp_path / "base"
    assert edit.fp8_strategy([], missing, 24564) == "not_mirrored"
    transformer = missing / "transformer"
    transformer.mkdir(parents=True)
    weight = transformer / "model.safetensors"
    weight.write_bytes(b"x")
    assert edit.fp8_strategy([], missing, 24564) == "diffusers-layerwise"
    huge = tmp_path / "huge.safetensors"
    huge.write_bytes(b"y" * 1000)
    assert edit.fp8_strategy([huge], missing, 0) == "skipped_vram"
    tiny = tmp_path / "tiny.safetensors"
    tiny.write_bytes(b"z")
    assert edit.fp8_strategy([tiny], missing, 24564) == "mirror-dequant"


def test_execute_records_both_caps_and_schema(isolated: Path) -> None:
    calls: list[dict[str, object]] = []

    def spawn(job: dict[str, object]) -> dict[str, object]:
        calls.append(job)
        mode = str(job["mode"])
        total = int(job["memory_mib"])
        fraction = edit.memory_fraction(str(job["cap"]), total)
        return _fit_payload(mode, fraction=fraction)

    args = _options(isolated / "out", ["--arms", "zimage-r128", "--caps", "unconstrained,cap16", "--smoke"])
    code, result = edit.execute(
        args,
        spawn=spawn,
        elapsed_min=lambda: 0.0,
        gpu=_gpu(),
        wheel=edit.WINDOWS_WHEEL,
        versions={"torch": "2.11.0", "diffusers": "0.37.1", "transformers": "4.57.6", "nunchaku": "1.3.0.dev20260629"},
        mirror_lookup=lambda _name: [],
    )
    assert code == 0
    assert result["schema"] == "voxprint-forge-nunchaku-zimage/1"
    assert result["tag"] == "nunchaku-test-2"
    assert result["precision"] == "int4"
    assert "fp8_timing_note" not in result
    assert result["fits_16gb"] is True
    assert result["wheel"]["sha256"] == edit.WINDOWS_WHEEL.sha256
    arm = result["arms"][0]
    assert arm["weight_sha256"] == edit.ARMS["zimage-r128"].sha256
    assert set(arm["caps"]) == {"unconstrained", "cap16"}
    assert arm["caps"]["cap16"]["memory_fraction"] == pytest.approx((16 * 1024) / 24564)
    assert arm["caps"]["cap16"]["peak_vram_smi_mib"] == 14000.0
    assert result["baselines"][0]["status"] == "not_mirrored"
    text = (isolated / "out" / "MESSAGE.md").read_text(encoding="utf-8")
    assert "nunchaku-test-2" in text
    assert "fits_16gb" in text
    assert "not representative of RTX 40/50" not in text
    stored = json.loads((isolated / "out" / "result.json").read_text(encoding="utf-8"))
    assert stored["exit_code"] == 0
    assert all(len(job["cases"]) == 1 for job in calls)
    assert {job["cap"] for job in calls} == {"unconstrained", "cap16"}


def test_a100_uses_real_total_and_marks_fp8(isolated: Path) -> None:
    seen: list[dict[str, object]] = []

    def spawn(job: dict[str, object]) -> dict[str, object]:
        seen.append(job)
        fraction = edit.memory_fraction(str(job["cap"]), int(job["memory_mib"]))
        return _fit_payload(str(job["mode"]), fraction=fraction)

    transformer = isolated / "cache" / "bases" / "z-image-turbo" / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "model.safetensors").write_bytes(b"bf16")
    gpu = edit.GpuFacts("NVIDIA A100-SXM4-80GB", 81920, "535.129.03", 8, 0)
    args = _options(isolated / "out", ["--arms", "zimage-r128", "--caps", "cap16", "--smoke"])
    code, result = edit.execute(
        args,
        spawn=spawn,
        elapsed_min=lambda: 0.0,
        gpu=gpu,
        wheel=edit.LINUX_WHEEL,
        mirror_lookup=lambda _name: [],
    )
    assert code == 0
    assert result["precision"] == "int4"
    assert result["fp8_timing_note"] == "not representative of RTX 40/50"
    assert result["arms"][0]["caps"]["cap16"]["memory_fraction"] == pytest.approx(16 / 80)
    assert result["baselines"][0]["status"] == "weight-only-dequant"
    assert result["baselines"][0]["fp8_timing_note"] == "not representative of RTX 40/50"
    assert all(job.get("kind") != "fp8" for job in seen)
    text = (isolated / "out" / "MESSAGE.md").read_text(encoding="utf-8")
    stored = json.loads((isolated / "out" / "result.json").read_text(encoding="utf-8"))
    assert "not representative of RTX 40/50" in text
    assert stored["fp8_timing_note"] == "not representative of RTX 40/50"


def test_cap16_fallback_and_fail(isolated: Path) -> None:
    def spawn(job: dict[str, object]) -> dict[str, object]:
        mode = str(job["mode"])
        if mode == "model-offload":
            return {"ok": False, "oom": True, "error": "CUDA out of memory", "offload_mode": mode}
        return _fit_payload(mode, peak=12000.0)

    args = _options(isolated / "out", ["--arms", "zimage-r128", "--caps", "cap16", "--mode", "auto"])
    code, result = edit.execute(
        args,
        spawn=spawn,
        elapsed_min=lambda: 0.0,
        gpu=_gpu(),
        wheel=edit.WINDOWS_WHEEL,
        mirror_lookup=lambda _name: [],
    )
    assert code == 0
    assert result["arms"][0]["caps"]["cap16"]["offload_mode"] == "block-offload"

    def over(job: dict[str, object]) -> dict[str, object]:
        return _fit_payload(str(job["mode"]), peak=16000.0)

    fail_args = _options(isolated / "fail", ["--arms", "zimage-r256", "--caps", "cap16"])
    fail_code, fail_result = edit.execute(
        fail_args,
        spawn=over,
        elapsed_min=lambda: 0.0,
        gpu=_gpu(),
        wheel=edit.WINDOWS_WHEEL,
        mirror_lookup=lambda _name: [],
    )
    assert fail_code == 1
    assert fail_result["fits_16gb"] is False


def test_resume_skips_a_finished_cap(isolated: Path) -> None:
    calls = {"n": 0}

    def spawn(job: dict[str, object]) -> dict[str, object]:
        calls["n"] += 1
        return _fit_payload(str(job["mode"]))

    args = _options(isolated / "out", ["--arms", "zimage-r128", "--caps", "cap16", "--smoke"])
    kwargs = {
        "spawn": spawn,
        "elapsed_min": lambda: 0.0,
        "gpu": _gpu(),
        "wheel": edit.LINUX_WHEEL,
        "mirror_lookup": lambda _name: [],
    }
    assert edit.execute(args, **kwargs)[0] == 0
    assert calls["n"] == 1
    assert edit.execute(args, **kwargs)[0] == 0
    assert calls["n"] == 1


def test_elapsed_skip_drops_r256_before_fp8(isolated: Path) -> None:
    seen: list[str] = []

    def spawn(job: dict[str, object]) -> dict[str, object]:
        seen.append(str(job["arm"]))
        return _fit_payload(str(job["mode"]))

    clock = {"n": 0}

    def elapsed() -> float:
        clock["n"] += 1
        return 85.0 if clock["n"] >= 2 else 0.0

    args = _options(
        isolated / "out",
        ["--arms", "zimage-r128,zimage-r256", "--caps", "cap16", "--smoke"],
    )
    code, result = edit.execute(
        args,
        spawn=spawn,
        elapsed_min=elapsed,
        gpu=_gpu(),
        wheel=edit.WINDOWS_WHEEL,
        mirror_lookup=lambda _name: [],
    )
    assert code == 0
    assert seen == ["zimage-r128"]
    assert any(item["id"] == "zimage-r256" and item["reason"] == "time_budget" for item in result["skipped"])
    assert any(item["id"] == "zimage-fp8" and item["reason"] == "time_budget" for item in result["skipped"])


def test_cleanup_deletes_only_the_int4_file(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = isolated / "cache"
    monkeypatch.setenv("NUNCHAKU_CACHE", str(cache))
    weight = cache / "weights" / edit.ARMS["zimage-r128"].filename
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"int4")
    base = cache / "bases" / "z-image-turbo" / "model_index.json"
    base.parent.mkdir(parents=True)
    base.write_text("{}", encoding="utf-8")
    args = _options(isolated / "out", ["--arms", "zimage-r128", "--caps", "cap16", "--smoke", "--cleanup"])
    edit.execute(
        args,
        spawn=lambda job: _fit_payload(str(job["mode"])),
        elapsed_min=lambda: 0.0,
        gpu=_gpu(),
        wheel=edit.WINDOWS_WHEEL,
        mirror_lookup=lambda _name: [],
    )
    assert not weight.exists()
    assert base.is_file()


def test_aliases_and_scripts(tmp_path: Path) -> None:
    source = tmp_path / "zimage-r128-cap16-z1.webp"
    source.write_bytes(b"webp")
    edit.export_aliases(tmp_path, "zimage-r128", "cap16", ["z1"])
    assert (tmp_path / "z1.webp").read_bytes() == b"webp"
    shell = (ROOT / "scripts" / "nunchaku_zimage_test.sh").read_text(encoding="utf-8")
    powershell = (ROOT / "scripts" / "nunchaku_zimage_test.ps1").read_text(encoding="utf-8")
    assert "voxprint_forge.diffusion.nunchaku_zimage" in shell
    assert "voxprint_forge.diffusion.nunchaku_zimage" in powershell
    assert "set +x" in shell
    assert "Read-Host" not in powershell
    assert "read -s" not in shell
    assert "nunchaku==" not in shell
    assert "nunchaku==" not in powershell


def test_main_exit_codes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(edit, "read_gpu", lambda: (_ for _ in ()).throw(edit.GpuError("no CUDA GPU")))
    assert edit.main(["--out", str(tmp_path), "--json", "--arms", "zimage-r128"]) == 3
    monkeypatch.setattr(edit, "read_gpu", lambda: _gpu())
    monkeypatch.setattr(edit, "gate_gpu", lambda _gpu: None)
    def _boom(*_args: object, **_kwargs: object) -> None:
        raise edit.SetupError("hash mismatch")

    monkeypatch.setattr(edit, "prepare_runtime", _boom)
    assert edit.main(["--out", str(tmp_path / "b"), "--arms", "zimage-r128"]) == 2


def test_reexec_compares_the_venv_prefix(tmp_path: Path) -> None:
    caller = tmp_path / "caller"
    cuda = tmp_path / "cuda"
    caller.mkdir()
    cuda.mkdir()
    assert edit.should_reexec(cuda, prefix=caller) is True
    assert edit.should_reexec(caller, prefix=caller) is False
    shared = tmp_path / "uv-python"
    shared.write_bytes(b"#!/bin/sh\n")
    links: list[Path] = []
    for root in (caller, cuda):
        link = root / "bin" / "python"
        link.parent.mkdir(parents=True, exist_ok=True)
        try:
            link.symlink_to(shared)
        except OSError:
            return
        links.append(link)
    assert links[0].resolve() == links[1].resolve()
    assert edit.should_reexec(cuda, prefix=caller) is True


def test_offline_fp8_resolves_the_base_before_load(isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    family = "z-image-turbo"
    downloaded = {"n": 0}

    def boom(**_kwargs: object) -> None:
        assert "HF_HUB_OFFLINE" not in os.environ
        raise RuntimeError("hub down")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(edit, "snapshot_download", boom)
    with pytest.raises(RuntimeError, match="hub down"):
        edit._ensure_base(family)
    assert os.environ["HF_HUB_OFFLINE"] == "1"

    def snap(**kwargs: object) -> None:
        assert "HF_HUB_OFFLINE" not in os.environ
        raw = kwargs["ignore_patterns"]
        assert isinstance(raw, list)
        patterns = [str(item) for item in raw]
        assert "transformer/*.safetensors" in patterns
        assert "transformer/*" not in patterns
        dest = Path(str(kwargs["local_dir"]))
        (dest / "transformer").mkdir(parents=True, exist_ok=True)
        (dest / "model_index.json").write_text("{}", encoding="utf-8")
        (dest / "transformer" / "config.json").write_text("{}", encoding="utf-8")
        downloaded["n"] += 1

    monkeypatch.setattr(edit, "snapshot_download", snap)
    seen: dict[str, object] = {}

    class Pipe:
        @classmethod
        def from_pretrained(cls, base: str, **kwargs: object) -> Pipe:
            seen["base"] = base
            seen["local_files_only"] = kwargs.get("local_files_only")
            return cls()

    class Model:
        @classmethod
        def load_config(cls, path: str) -> dict[str, object]:
            folder = Path(path)
            assert folder.name == "transformer"
            assert (folder / "config.json").is_file()
            return {}

        @classmethod
        def from_config(cls, _config: dict[str, object]) -> Model:
            return cls()

        def load_state_dict(self, _state: dict[str, object], strict: bool = False) -> None:
            del strict
            return None

    monkeypatch.setattr(edit, "ZImagePipeline", Pipe)
    monkeypatch.setattr(edit, "ZImageTransformer2DModel", Model)
    dest = isolated / "cache" / "bases" / family
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "model_index.json").write_text("{}", encoding="utf-8")
    edit._mark(edit.done_dir(), f"base-{family}", str(edit.BASES[family]["revision"]))
    edit._load_fp8(
        {"family": family, "kind": "fp8", "strategy": "mirror-dequant", "fp8_files": []}
    )
    assert downloaded["n"] == 1
    assert seen["local_files_only"] is True
    assert Path(str(seen["base"])) == edit.cache_dir() / "bases" / family
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    edit._load_fp8({"family": family, "kind": "fp8", "strategy": "mirror-dequant", "fp8_files": []})
    assert downloaded["n"] == 1


def test_one_arm_failure_does_not_skip_the_rest(isolated: Path) -> None:
    seen: list[str] = []

    def spawn(job: dict[str, object]) -> dict[str, object]:
        seen.append(str(job["arm"]))
        if job["arm"] == "zimage-r128":
            raise edit.SetupError("missing weight")
        if job.get("kind") == "fp8":
            raise edit.SetupError("offline base")
        return _fit_payload(str(job["mode"]))

    mirror = isolated / "fp8.safetensors"
    mirror.write_bytes(b"z")
    args = _options(isolated / "out", ["--arms", "zimage-r128,zimage-r256", "--caps", "cap16", "--smoke"])
    code, result = edit.execute(
        args,
        spawn=spawn,
        elapsed_min=lambda: 0.0,
        gpu=_gpu(),
        wheel=edit.WINDOWS_WHEEL,
        mirror_lookup=lambda _name: [mirror],
    )
    assert code == 1
    assert seen == ["zimage-r128", "zimage-r256", "zimage-fp8"]
    assert [arm["id"] for arm in result["arms"]] == ["zimage-r128", "zimage-r256"]
    assert result["arms"][0]["caps"]["cap16"]["error"] == "missing weight"
    assert result["arms"][1]["caps"]["cap16"]["fits_16gb"] is True
    assert result["baselines"][0]["status"] == "error"
    assert result["baselines"][0]["error"] == "offline base"


def test_missing_cuda_stops_before_later_arms(isolated: Path) -> None:
    seen: list[str] = []

    def spawn(job: dict[str, object]) -> dict[str, object]:
        seen.append(str(job["arm"]))
        raise edit.GpuError("no CUDA GPU")

    args = _options(isolated / "fail", ["--arms", "zimage-r128,zimage-r256", "--caps", "cap16", "--smoke"])
    with pytest.raises(edit.GpuError, match="no CUDA GPU"):
        edit.execute(
            args,
            spawn=spawn,
            elapsed_min=lambda: 0.0,
            gpu=_gpu(),
            wheel=edit.WINDOWS_WHEEL,
            mirror_lookup=lambda _name: [],
        )
    assert seen == ["zimage-r128"]


def test_docs_mention_the_matrix() -> None:
    vm = (ROOT / "docs" / "VM.md").read_text(encoding="utf-8")
    cli = (ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
    for text in (vm, cli):
        assert "nunchaku-test-2" in text
        assert "8.0" in text
        assert "not representative of RTX 40/50" in text
        assert "525.60.13" in text
        assert "nunchaku_zimage_test" in text
        assert "cap16" in text
