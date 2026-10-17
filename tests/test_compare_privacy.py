"""Private prompt text must not appear in compare outputs. CI runs this file."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from voxprint_forge.cli import main
from voxprint_forge.compare.metrics import finish_score
from voxprint_forge.compare.plan import Arm, Job, load_compare_plan
from voxprint_forge.compare.privacy import assert_tree_clean, prompt_id
from voxprint_forge.compare.run import run_compare
from voxprint_forge.diffusion.prompts import BASE_PROMPTS
from voxprint_forge.diffusion.spec import CONTROL_SCALE, TURBO_GUIDANCE, TURBO_STEPS

ROOT = Path(__file__).resolve().parents[1]


def _passing(reference: Image.Image, candidate: Image.Image, case: object) -> dict[str, object]:
    del reference, candidate, case
    return finish_score(
        lpips=0.01,
        ssim=1.0,
        psnr=40.0,
        psnr_saturated=False,
        dino_cosine=0.9,
        clip_i=0.9,
        clip_t_reference=0.4,
        clip_t_candidate=0.4,
    )


def _arm(arm_id: str, role: str) -> Arm:
    return Arm(
        id=arm_id,
        role=role,  # type: ignore[arg-type]
        repo="example/weights",
        revision="abc",
        quant="bf16",
        compute="bf16",
        offload=False,
        files=(),
        enabled=True,
        skip_reason="",
        base_repo="",
        base_revision="",
        reuse_job="",
        reuse_arm="",
    )


def test_scanner_flags_a_planted_prompt(tmp_path: Path) -> None:
    secret = "sentinel-" + uuid.uuid4().hex
    (tmp_path / "summary.md").write_text(f"leak {secret}\n", encoding="utf-8")
    hits = assert_tree_clean(tmp_path, (secret,))
    assert hits == [prompt_id(secret)]


def test_private_prompt_text_does_not_leak(tmp_path: Path, monkeypatch, capsys) -> None:
    sentinel = "sentinel-" + uuid.uuid4().hex
    negative = sentinel + "-neg"
    folder = tmp_path / "private" / "vxp_yuna"
    folder.mkdir(parents=True)
    manifest = {
        "_run": {"steps": 50, "guidance_scale": 3.0, "model_id": "moss"},
        "row": {
            "prompt": sentinel,
            "negative_prompt": negative,
            "seed": 11,
            "width": 64,
            "height": 64,
            "steps": 50,
            "guidance_scale": 3.0,
            "model_id": "moss",
        },
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("FORGE_PRIVATE_PROMPTS", str(tmp_path / "private"))
    job = Job(
        id="D1",
        kind="turbo",
        steps=TURBO_STEPS,
        guidance=TURBO_GUIDANCE,
        control_scale=CONTROL_SCALE,
        prompts="base",
        controls=("canny", "depth", "pose"),
        public_seeds=(0, 1),
        private_sets=("vxp_yuna", "vxp_mei"),
        private_limit=50,
        width=64,
        height=64,
        reference="a",
        arms=(_arm("a", "reference"), _arm("b", "candidate")),
        lpips_max=0.10,
        clip_drop_max=0.01,
        note="",
    )
    seen: list[tuple[int, float, int]] = []

    def generate(arm: Arm, case: object, current: Job) -> Image.Image:
        del arm
        seen.append((current.steps, current.guidance, int(getattr(case, "seed"))))
        return Image.new("RGB", (16, 16), (1, 2, 3))

    results = tmp_path / "results" / "D1"
    report = run_compare(
        job,
        results,
        private_root=tmp_path / "private",
        generate=generate,
        score=_passing,
        fetch=lambda _arm, dest: (dest / "transformer.safetensors").write_bytes(b"w"),
        smi=lambda: None,
        cuda=False,
    )
    captured = capsys.readouterr()
    assert report["exit_code"] == 0
    assert report["missing_private_sets"] == ["vxp_mei"]
    assert any(seed == 11 for _steps, _guidance, seed in seen)
    assert {steps for steps, _guidance, _seed in seen} == {TURBO_STEPS}
    assert {guidance for _steps, guidance, _seed in seen} == {TURBO_GUIDANCE}
    blob = (captured.out + captured.err).encode()
    for path in results.rglob("*"):
        if path.is_file():
            blob += path.name.encode() + path.read_bytes()
    assert sentinel.encode() not in blob
    assert negative.encode() not in blob
    for prompt in BASE_PROMPTS:
        assert prompt.encode() not in blob
    assert assert_tree_clean(results, (sentinel, negative, *BASE_PROMPTS)) == []
    pair = json.loads((results / "pairs.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert "prompt" not in pair
    assert "negative" not in pair
    assert "prompt_id" in pair


def test_cli_without_a_gpu_does_not_print_the_private_prompt(tmp_path: Path, monkeypatch, capsys) -> None:
    sentinel = "sentinel-" + uuid.uuid4().hex
    folder = tmp_path / "sets" / "vxp_yuna"
    folder.mkdir(parents=True)
    payload = {"row": {"prompt": sentinel, "negative_prompt": "", "seed": 3, "size": 32}}
    (folder / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setenv("FORGE_PRIVATE_PROMPTS", str(tmp_path / "sets"))
    monkeypatch.setattr("voxprint_forge.compare.run.cuda_available", lambda: False)
    code = main(
        [
            "--json",
            "--root",
            str(ROOT),
            "--forge-out",
            str(tmp_path / "forge-out"),
            "compare",
            "--job",
            "D1",
            "--plan",
            str(ROOT / "plans" / "compare-synth4.toml"),
        ]
    )
    captured = capsys.readouterr()
    assert code == 3
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert load_compare_plan(ROOT / "plans" / "compare-synth4.toml").job("D1").id == "D1"


def test_missing_private_sets_name_the_reason(tmp_path: Path, capsys) -> None:
    from voxprint_forge.compare.cases import _private_cases

    job = SimpleNamespace(private_sets=("vxp_yuna", "vxp_mei"), private_limit=5, width=64, height=64)
    assert _private_cases(job, None, start=0) == ([], ("vxp_yuna", "vxp_mei"))
    assert "FORGE_PRIVATE_PROMPTS is not set in this process" in capsys.readouterr().err
    (tmp_path / "vxp_yuna").mkdir()
    (tmp_path / "vxp_yuna" / "prompts.json").write_text("{}", encoding="utf-8")
    rows, missing = _private_cases(job, tmp_path, start=0)
    assert rows == [] and missing == ("vxp_yuna", "vxp_mei")
    err = capsys.readouterr().err
    assert str(tmp_path / "vxp_yuna" / "manifest.json") in err
    assert str(tmp_path / "vxp_mei" / "dataset" / "manifest.json") in err


def test_private_root_expands_home_and_names_a_bad_path(tmp_path: Path, monkeypatch) -> None:
    import pytest

    from voxprint_forge.compare.codes import CompareSetup
    from voxprint_forge.compare.privacy import private_root_from_env

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    (tmp_path / "compare-synth4").mkdir()
    monkeypatch.setenv("FORGE_PRIVATE_PROMPTS", "~/compare-synth4")
    assert private_root_from_env() == tmp_path / "compare-synth4"
    monkeypatch.setenv("FORGE_PRIVATE_PROMPTS", str(tmp_path / "nope"))
    with pytest.raises(CompareSetup, match="nope"):
        private_root_from_env()
