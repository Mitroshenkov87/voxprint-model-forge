"""Compare plan, manifests, metrics, resume, and exit codes. No GPU."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from voxprint_forge.compare.cases import build_schedule
from voxprint_forge.compare.codes import CompareSetup
from voxprint_forge.compare.engine import _apply, _load_state, _overlay, seeded_generator
from voxprint_forge.compare.manifests import find_manifest, load_manifest
from voxprint_forge.compare.metrics import finish_score, passes, ssim_psnr
from voxprint_forge.compare.plan import Arm, Job, load_compare_plan
from voxprint_forge.compare.results import summarise
from voxprint_forge.compare.run import gate_local_arms, run_compare
from voxprint_forge.diffusion.prompts import BASE_PROMPTS, CONTROL_PROMPTS, CONTROL_TYPES
from voxprint_forge.diffusion.spec import CONTROL_SCALE, CREATIVE_GUIDANCE, CREATIVE_STEPS, TURBO_GUIDANCE, TURBO_STEPS

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "plans" / "compare-synth4.toml"


def _score(**overrides: float) -> dict[str, object]:
    values = {
        "lpips": 0.02,
        "ssim": 0.99,
        "psnr": 40.0,
        "psnr_saturated": False,
        "dino_cosine": 0.91,
        "clip_i": 0.95,
        "clip_t_reference": 0.30,
        "clip_t_candidate": 0.30,
    }
    values.update(overrides)
    return finish_score(
        lpips=float(values["lpips"]),
        ssim=float(values["ssim"]),
        psnr=None if values["psnr"] is None else float(values["psnr"]),
        psnr_saturated=bool(values["psnr_saturated"]),
        dino_cosine=float(values["dino_cosine"]),
        clip_i=float(values["clip_i"]),
        clip_t_reference=float(values["clip_t_reference"]),
        clip_t_candidate=float(values["clip_t_candidate"]),
    )


def _arm(arm_id: str, role: str, *, enabled: bool = True, reuse_job: str = "", reuse_arm: str = "") -> Arm:
    return Arm(
        id=arm_id,
        role=role,  # type: ignore[arg-type]
        repo="example/weights",
        revision="abc123",
        quant="bf16",
        compute="bf16",
        offload=False,
        files=(),
        enabled=enabled,
        skip_reason="" if enabled else "disabled in this test",
        base_repo="",
        base_revision="",
        reuse_job=reuse_job,
        reuse_arm=reuse_arm,
    )


def _job(arms: tuple[Arm, ...], **overrides: object) -> Job:
    raw: dict[str, object] = {
        "id": "T",
        "kind": "turbo",
        "steps": TURBO_STEPS,
        "guidance": TURBO_GUIDANCE,
        "control_scale": CONTROL_SCALE,
        "prompts": "base",
        "controls": CONTROL_TYPES,
        "public_seeds": (0, 1),
        "private_sets": (),
        "private_limit": 50,
        "width": 64,
        "height": 64,
        "reference": "a",
        "arms": arms,
        "lpips_max": 0.10,
        "clip_drop_max": 0.01,
        "note": "",
    }
    raw.update(overrides)
    return Job(**raw)  # type: ignore[arg-type]


def _fetch(arm: Arm, dest: Path) -> None:
    del arm
    (dest / "transformer.safetensors").write_bytes(b"weights")


def _picture(arm: Arm, case: object, job: Job) -> Image.Image:
    del arm, case, job
    return Image.new("RGB", (32, 32), (10, 20, 30))


def test_plan_covers_the_synth4_diffusion_jobs() -> None:
    plan = load_compare_plan(PLAN)
    assert [item.id for item in plan.jobs] == ["D1", "D2", "D3", "D4"]
    turbo = plan.job("D1")
    assert turbo.steps == TURBO_STEPS
    assert turbo.guidance == TURBO_GUIDANCE
    assert [item.id for item in turbo.enabled_arms()] == ["a", "b", "c", "c-dq", "d"]
    assert turbo.arm("a").quant == "fp32"
    assert turbo.arm("a").compute == "bf16"
    assert turbo.arm("c").compute == "fp8-native"
    assert turbo.arm("c-dq").compute == "bf16-dequant"
    assert turbo.arm("c-dq").files == turbo.arm("c").files
    assert turbo.arm("d").compute == "fp8-native"
    assert turbo.attention == "sdpa"
    assert turbo.arm("c").local is True
    assert turbo.arm("d").files == (
        "zimage-turbo-transformer-fp8.safetensors",
        "qwen3-4b-text-encoder-fp8.safetensors",
    )
    creative = plan.job("D4")
    assert creative.steps == CREATIVE_STEPS
    assert creative.guidance == CREATIVE_GUIDANCE
    assert creative.private_sets == ()
    assert creative.arm("fp8").local is True
    assert creative.arm("fp8").enabled is True
    assert creative.arm("fp8").compute == "fp8-native"
    ranked = plan.job("D2")
    assert [item.id for item in ranked.candidates()] == ["r128", "r256"]
    assert ranked.arm("r128").files == ("svdq-int4_r128-z-image-turbo.safetensors",)
    assert ranked.requires_cc == ("7.5", "8.0", "8.6", "8.9", "12.0")
    control = plan.job("D3")
    assert control.prompts == "control"
    assert control.private_sets == ()
    assert [item.id for item in control.candidates()] == ["mirror-bf16", "mirror-lite-fp8"]
    assert control.arm("mirror-lite-fp8").compute == "fp8-native"
    text = PLAN.read_text(encoding="utf-8")
    assert "HF_TOKEN" not in text
    assert "BEGIN OPENSSH" not in text
    assert "-----BEGIN" not in text


def test_case_budgets_follow_the_public_sets() -> None:
    plan = load_compare_plan(PLAN)
    turbo = build_schedule(plan.job("D1"), None)
    assert len(turbo.cases) == len(BASE_PROMPTS) * 2
    assert turbo.cases[0].warmup is True
    assert turbo.missing_sets == ("vxp_yuna", "vxp_mei")
    control = build_schedule(plan.job("D3"), None)
    assert len(control.cases) == len(CONTROL_PROMPTS) * len(CONTROL_TYPES) * 2
    assert {item.control for item in control.cases} == set(CONTROL_TYPES)
    creative = build_schedule(plan.job("D4"), None)
    assert len(creative.cases) == len(BASE_PROMPTS) * 2
    assert creative.missing_sets == ()


def test_both_manifest_layouts_keep_seed_and_size_only(tmp_path: Path) -> None:
    top = {
        "_run": {"steps": 50, "guidance_scale": 3.0, "model_id": "moss"},
        "shot": {
            "prompt": "a private row",
            "negative_prompt": "none",
            "seed": 4,
            "width": 640,
            "height": 480,
            "steps": 50,
            "guidance_scale": 3.0,
            "model_id": "moss",
        },
    }
    nested = {"other": {"prompt": "second row", "negative": "no", "seed": 9, "size": "768x512"}}
    first = load_manifest(_dump(tmp_path / "top.json", top))
    assert len(first) == 1
    assert first[0].seed == 4
    assert (first[0].width, first[0].height) == (640, 480)
    assert first[0].negative == "none"
    assert not hasattr(first[0], "steps")
    second = load_manifest(_dump(tmp_path / "nested.json", nested))
    assert second[0].seed == 9
    assert (second[0].width, second[0].height) == (768, 512)


def test_find_manifest_accepts_either_path(tmp_path: Path) -> None:
    top = tmp_path / "vxp_alina"
    top.mkdir()
    (top / "manifest.json").write_text('{"a": {"prompt": "p", "seed": 1, "size": 8}}', encoding="utf-8")
    nested = tmp_path / "vxp_lena" / "dataset"
    nested.mkdir(parents=True)
    (nested / "manifest.json").write_text('{"b": {"prompt": "q", "seed": 2, "size": 8}}', encoding="utf-8")
    assert find_manifest(top) == top / "manifest.json"
    assert find_manifest(tmp_path / "vxp_lena") == nested / "manifest.json"
    assert load_manifest(find_manifest(tmp_path / "vxp_lena")).__len__() == 1


def test_local_fp8_arms_wait_for_the_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGE_FP8_DIR", raising=False)
    job = load_compare_plan(PLAN).job("D1")
    gated = gate_local_arms(job)
    assert [item.id for item in gated.enabled_arms()] == ["a", "b"]
    folder = tmp_path / "fp8"
    folder.mkdir()
    (folder / "zimage-turbo-transformer-fp8.safetensors").write_bytes(b"t")
    (folder / "qwen3-4b-text-encoder-fp8.safetensors").write_bytes(b"e")
    monkeypatch.setenv("FORGE_FP8_DIR", str(folder))
    opened = gate_local_arms(job)
    assert [item.id for item in opened.enabled_arms()] == ["a", "b", "c", "c-dq", "d"]


def test_disabled_candidate_is_a_setup_error(tmp_path: Path) -> None:
    job = _job((_arm("a", "reference"), _arm("b", "candidate", enabled=False)))
    with pytest.raises(CompareSetup) as caught:
        run_compare(job, tmp_path / "results", private_root=None, generate=_picture, score=_score, smi=lambda: None)
    assert caught.value.exit_code == 2
    assert "no runnable candidate" in str(caught.value)


def test_run_passes_resumes_and_reclaims(tmp_path: Path) -> None:
    job = _job((_arm("a", "reference"), _arm("b", "candidate")))
    calls: list[str] = []

    def generate(arm: Arm, case: object, current: Job) -> Image.Image:
        del current
        calls.append(arm.id)
        return _picture(arm, case, job)

    results = tmp_path / "results" / "T"
    weights = results / "weights" / "b" / "transformer.safetensors"
    report = run_compare(
        job,
        results,
        private_root=None,
        generate=generate,
        score=lambda *_args: _score(),
        fetch=_fetch,
        smi=lambda: 4096,
        cuda=False,
    )
    assert report["exit_code"] == 0
    assert report["passed"] is True
    assert not weights.exists()
    assert (results / "summary.json").is_file()
    assert (results / "images" / "b").is_dir()
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    assert summary["arms"][0]["samples"] == len(BASE_PROMPTS) * 2 - 1
    assert summary["arms"][0]["seconds_per_image"] is not None
    assert summary["reference"]["id"] == "a"
    assert summary["reference"]["samples"] == len(BASE_PROMPTS) * 2 - 1
    assert summary["reference"]["seconds_per_image"] is not None
    assert "Reference s/img" in (results / "summary.md").read_text(encoding="utf-8")
    assert summary["arms"][0]["component_bytes"]["transformer.safetensors"] == len(b"weights")
    first = len(calls)
    again = run_compare(
        job,
        results,
        private_root=None,
        generate=generate,
        score=lambda *_args: _score(),
        fetch=_fetch,
        smi=lambda: None,
        cuda=False,
    )
    assert again["exit_code"] == 0
    assert len(calls) == first


def test_quality_bar_failure_exits_1(tmp_path: Path) -> None:
    job = _job((_arm("a", "reference"), _arm("b", "candidate")))
    report = run_compare(
        job,
        tmp_path / "out",
        private_root=None,
        generate=_picture,
        score=lambda *_args: _score(lpips=0.2, clip_t_candidate=0.1),
        fetch=_fetch,
        smi=lambda: None,
        cuda=False,
    )
    assert report["exit_code"] == 1
    assert report["passed"] is False
    assert report["failed_arms"] == ["b"]


def test_warmup_is_left_out_of_the_speed_mean() -> None:
    job = _job((_arm("a", "reference"), _arm("b", "candidate")))
    slow = {
        "candidate_arm": "b",
        "warmup": True,
        "seconds": 100.0,
        "vram_torch_bytes": 10,
        "vram_smi_bytes": 12,
        "metrics": _score(),
    }
    timed = {
        "candidate_arm": "b",
        "warmup": False,
        "seconds": 4.0,
        "vram_torch_bytes": 8,
        "vram_smi_bytes": 9,
        "metrics": _score(),
    }
    rows = [slow, timed]
    summary = summarise(
        job,
        rows,
        component_bytes={"b": {"transformer.safetensors": 3}},
        missing_sets=(),
        case_count=2,
        secrets=(),
    )
    assert summary["arms"][0]["seconds_per_image"] == 4.0
    assert summary["arms"][0]["samples"] == 1
    assert summary["passed"] is True


def test_ssim_and_psnr_on_rgb() -> None:
    image = torch.rand(3, 32, 32)
    ssim, psnr, saturated = ssim_psnr(image, image)
    assert ssim == pytest.approx(1.0, abs=1e-4)
    assert psnr is None
    assert saturated is True
    other = torch.rand(3, 32, 32)
    ssim_gap, psnr_gap, saturated_gap = ssim_psnr(image, other)
    assert math.isfinite(ssim_gap)
    assert ssim_gap < 0.99
    assert psnr_gap is not None
    assert saturated_gap is False


def test_pass_bar_matches_the_forge_gate() -> None:
    assert passes(0.10, 0.01, lpips_max=0.10, clip_drop_max=0.01) is True
    assert passes(0.11, 0.01, lpips_max=0.10, clip_drop_max=0.01) is False
    assert passes(0.01, 0.02, lpips_max=0.10, clip_drop_max=0.01) is False


def test_cuda_generator_uses_the_cuda_device_and_seed() -> None:
    seen: list[tuple[str, int]] = []

    class _Gen:
        def manual_seed(self, seed: int) -> _Gen:
            seen.append((self.device, seed))
            return self

        def __init__(self, device: str) -> None:
            self.device = device

    seeded_generator(17, factory=_Gen)
    assert seen == [("cuda", 17)]


def test_vm_script_requires_a_job() -> None:
    script = ROOT / "scripts" / "vm_run_compare.sh"
    indexed = subprocess.run(
        ["git", "ls-files", "-s", "--", "scripts/vm_run_compare.sh"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert indexed.returncode == 0
    assert indexed.stdout.split()[0] == "100755"
    if sys.platform != "win32":
        assert script.stat().st_mode & 0o111
    done = subprocess.run([_bash(), str(script)], check=False, capture_output=True, text=True)
    assert done.returncode == 2
    assert "HF_TOKEN" not in done.stderr


def test_layout_omits_size_and_drops_keep_false(tmp_path: Path) -> None:
    payload = {
        "kept": {
            "prompt": "studio row",
            "seed": 3,
            "model": "Tongyi-MAI/Z-Image-Turbo",
            "revision": "abc",
            "steps": 8,
            "guidance_scale": 0.0,
        },
        "aside": {"prompt": "unused row", "seed": 1, "keep": False, "size": 512},
        "blank": {"keep": False},
    }
    rows = load_manifest(_dump(tmp_path / "rows.json", payload))
    assert [item.image_key for item in rows] == ["kept"]
    assert rows[0].seed == 3
    assert (rows[0].width, rows[0].height) == (0, 0)
    folder = tmp_path / "vxp_yuna" / "dataset"
    folder.mkdir(parents=True)
    _dump(folder / "manifest.json", payload)
    job = _job(
        (_arm("a", "reference"), _arm("b", "candidate")),
        private_sets=("vxp_yuna",),
        public_seeds=(),
        width=1024,
        height=1024,
    )
    schedule = build_schedule(job, tmp_path)
    assert len(schedule.cases) == 1
    assert schedule.cases[0].seed == 3
    assert (schedule.cases[0].width, schedule.cases[0].height) == (1024, 1024)
    bad = _dump(tmp_path / "bad.json", {"shot-7": {"seed": 1, "model": "Tongyi-MAI/Z-Image"}})
    with pytest.raises(CompareSetup, match="shot-7") as caught:
        load_manifest(bad)
    assert "prompt" in str(caught.value)
    assert caught.value.exit_code == 2


def test_fp8_overlay_uses_the_a100_loader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "encoder.safetensors"
    save_file({"layers.0.weight": torch.zeros(2, 2)}, path)
    monkeypatch.setattr("voxprint_forge.diffusion.fp8.read_format", lambda _path: "voxprint-fp8-v1")
    seen: list[Path] = []

    def _load(source: Path) -> dict[str, torch.Tensor]:
        seen.append(source)
        return {"layers.0.weight": torch.ones(2, 2)}

    monkeypatch.setattr("voxprint_forge.diffusion.fp8.load_dequant_bf16", _load)
    state = _load_state(path)
    assert seen == [path]
    weight = state["layers.0.weight"]
    assert isinstance(weight, torch.Tensor)
    assert torch.equal(weight, torch.ones(2, 2))


def test_overlay_merges_shards_before_the_coverage_check(tmp_path: Path) -> None:
    text = _module(("te.0.weight", "te.1.weight", "te.2.weight", "te.3.weight"))
    transformer = _module(("tr.0.weight", "tr.1.weight", "tr.2.weight", "tr.3.weight"))
    _write_shards(tmp_path / "text_encoder", "model", ("te.0.weight", "te.1.weight"))
    _write_shards(tmp_path / "transformer", "diffusion_pytorch_model", ("tr.0.weight", "tr.1.weight"))
    shard = tmp_path / "text_encoder" / "model-00001-of-00002.safetensors"
    with pytest.raises(CompareSetup, match="matched 1 tensors"):
        _apply(text, _load_state(shard), shard.name)

    class _Pipe:
        def __init__(self) -> None:
            self.text_encoder = text
            self.transformer = transformer

    _overlay(_Pipe(), tmp_path)
    assert set(text.loaded) == {"te.0.weight", "te.1.weight"}
    assert set(transformer.loaded) == {"tr.0.weight", "tr.1.weight"}
    loose = tmp_path / "loose"
    _write_shards(loose, "model", ("te.0.weight", "te.1.weight"), index=False)
    again = _module(("te.0.weight", "te.1.weight", "te.2.weight", "te.3.weight"))

    class _Loose:
        def __init__(self) -> None:
            self.text_encoder = again

    _overlay(_Loose(), loose)
    assert set(again.loaded) == {"te.0.weight", "te.1.weight"}


def test_one_pipeline_is_loaded_at_a_time(tmp_path: Path) -> None:
    job = _job((_arm("a", "reference"), _arm("b", "candidate")))
    seen: list[tuple[str, ...]] = []

    class _Pipe:
        def __init__(self) -> None:
            self.loaded = 0
            self.max_loaded = 0

        def load_arm(self, arm: Arm, current: Job) -> None:
            del current
            assert self.loaded == 0
            self.loaded = 1
            self.max_loaded = max(self.max_loaded, self.loaded)
            seen.append(("load", arm.id))

        def release(self) -> None:
            self.loaded = 0
            seen.append(("release",))

        def __call__(self, arm: Arm, case: object, current: Job) -> Image.Image:
            del current
            assert self.loaded == 1
            seen.append(("gen", arm.id, str(getattr(case, "ordinal"))))
            return _picture(arm, case, job)

    pipe = _Pipe()
    run_compare(
        job,
        tmp_path / "results",
        private_root=None,
        generate=pipe,
        score=lambda *_args: _score(),
        fetch=_fetch,
        smi=lambda: None,
        cuda=False,
    )
    assert pipe.loaded == 0
    assert pipe.max_loaded == 1
    count = len(BASE_PROMPTS) * 2
    gens = [item for item in seen if item[0] == "gen"]
    assert [item[1] for item in gens] == ["a"] * count + ["b"] * count
    load_b = seen.index(("load", "b"))
    assert any(item == ("release",) for item in seen[:load_b])


def test_resumed_image_does_not_include_load_time(tmp_path: Path) -> None:
    job = _job((_arm("a", "reference"), _arm("b", "candidate")))
    results = tmp_path / "results"
    run_compare(
        job,
        results,
        private_root=None,
        generate=_picture,
        score=lambda *_args: _score(),
        fetch=_fetch,
        smi=lambda: None,
        cuda=False,
    )
    lines = (results / "pairs.jsonl").read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[-1])
    assert payload["warmup"] is False
    control = payload["control"] or "none"
    name = f"{payload['prompt_id']}-{payload['seed']}-{payload['width']}x{payload['height']}-{control}.png"
    (results / "images" / str(payload["candidate_arm"]) / name).unlink()
    (results / "pairs.jsonl").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    class _SlowLoad:
        def __init__(self) -> None:
            self.loads = 0
            self.calls = 0

        def load_arm(self, arm: Arm, current: Job) -> None:
            del arm, current
            self.loads += 1
            time.sleep(0.2)

        def release(self) -> None:
            return None

        def __call__(self, arm: Arm, case: object, current: Job) -> Image.Image:
            self.calls += 1
            return _picture(arm, case, current)

    slow = _SlowLoad()
    run_compare(
        job,
        results,
        private_root=None,
        generate=slow,
        score=lambda *_args: _score(),
        fetch=_fetch,
        smi=lambda: None,
        cuda=False,
    )
    fresh = json.loads((results / "pairs.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert fresh["pair_id"] == payload["pair_id"]
    assert slow.loads == 1
    assert slow.calls == 1
    assert isinstance(fresh["seconds"], float)
    assert fresh["seconds"] < 0.1


def _bash() -> str:
    found = shutil.which("bash")
    if found:
        return found
    windows = Path(r"C:\Program Files\Git\bin\bash.exe")
    if windows.is_file():
        return str(windows)
    return "bash"


def _module(keys: tuple[str, ...]) -> object:
    class _Weights:
        def state_dict(self) -> dict[str, torch.Tensor]:
            return {key: torch.zeros(2, 2) for key in keys}

        def load_state_dict(self, mapped: dict[str, torch.Tensor], strict: bool = False) -> None:
            del strict
            self.loaded = mapped

    return _Weights()


def _write_shards(folder: Path, stem: str, keys: tuple[str, str], *, index: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    save_file({keys[0]: torch.ones(2, 2)}, folder / f"{stem}-00001-of-00002.safetensors")
    save_file({keys[1]: torch.ones(2, 2)}, folder / f"{stem}-00002-of-00002.safetensors")
    if not index:
        return
    payload = {
        "metadata": {"total_size": 32},
        "weight_map": {
            keys[0]: f"{stem}-00001-of-00002.safetensors",
            keys[1]: f"{stem}-00002-of-00002.safetensors",
        },
    }
    (folder / f"{stem}.safetensors.index.json").write_text(json.dumps(payload), encoding="utf-8")


def _dump(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
