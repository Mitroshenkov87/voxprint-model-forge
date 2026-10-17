"""Compare plan, manifests, metrics, resume, and exit codes. No GPU."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest
import torch
from PIL import Image

from voxprint_forge.compare.cases import build_schedule
from voxprint_forge.compare.codes import CompareSetup
from voxprint_forge.compare.engine import seeded_generator
from voxprint_forge.compare.manifests import find_manifest, load_manifest
from voxprint_forge.compare.metrics import finish_score, passes, ssim_psnr
from voxprint_forge.compare.plan import Arm, Job, load_compare_plan
from voxprint_forge.compare.results import summarise
from voxprint_forge.compare.run import run_compare
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
    assert [item.id for item in turbo.enabled_arms()] == ["a", "b"]
    assert turbo.arm("a").quant == "fp32"
    assert turbo.arm("a").compute == "bf16"
    assert turbo.arm("c").enabled is False
    creative = plan.job("D4")
    assert creative.steps == CREATIVE_STEPS
    assert creative.guidance == CREATIVE_GUIDANCE
    assert creative.private_sets == ()
    control = plan.job("D3")
    assert control.prompts == "control"
    assert control.private_sets == ()
    assert [item.id for item in control.candidates()] == ["mirror-bf16", "mirror-lite-fp8"]
    assert control.arm("mirror-lite-fp8").compute == "bf16-dequant"
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


def test_disabled_candidate_is_a_setup_error(tmp_path: Path) -> None:
    job = load_compare_plan(PLAN).job("D2")
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
    assert script.stat().st_mode & 0o111
    done = subprocess.run(["bash", str(script)], check=False, capture_output=True, text=True)
    assert done.returncode == 2
    assert "HF_TOKEN" not in done.stderr


def _dump(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
