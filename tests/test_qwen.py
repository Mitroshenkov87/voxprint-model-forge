"""Part D selectors, pins, and the encoder link. No diffusers import."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from voxprint_forge.diffusion.common import Ctx
from voxprint_forge.diffusion.names import EDIT_ENCODER_FP8
from voxprint_forge.diffusion.part_edit import _convert
from voxprint_forge.diffusion.planload import load_canvas_plan
from voxprint_forge.diffusion.qwen_guard import command_is_canvas_ac
from voxprint_forge.diffusion.qwen_paths import (
    INT4_MARKER_TEXT,
    classify_encoder,
    edit_encoder_fp8,
    link_stable_file,
    write_int4_marker,
)
from voxprint_forge.diffusion.qwen_plan import load_qwen_plan
from voxprint_forge.diffusion.qwen_policy import decide_qwen_fp8, decide_qwen_int4
from voxprint_forge.diffusion.qwen_select import parse_parts, refuse_retired
from voxprint_forge.errors import PlanError, UsageError
from voxprint_forge.layout import Layout

ROOT = Path(__file__).resolve().parents[1]


def test_default_part_is_d_and_c_runs_after_d() -> None:
    assert parse_parts(None) == ("qwen",)
    assert parse_parts("") == ("qwen",)
    assert parse_parts("C,D") == ("qwen", "edit")
    assert parse_parts("E,C") == ("edit", "encoder")
    assert parse_parts("A,B,C,D,E") == ("base", "controlnet", "qwen", "edit", "encoder")
    with pytest.raises(UsageError):
        parse_parts("C,")
    with pytest.raises(UsageError):
        parse_parts("nope")


def test_qwen_plan_pins_the_shared_encoder() -> None:
    plan = load_qwen_plan(ROOT / "plans" / "canvas-qwen-image-2512-v1.toml")
    assert plan.id == "canvas-qwen-image-2512"
    assert plan.int4 == "deferred"
    assert plan.estimate_min == 120
    assert plan.estimate_min_reuse == 280
    assert plan.estimate_min_recompute == 340
    assert plan.encoder_estimate_min == 15
    assert plan.source.repo == "Qwen/Qwen-Image-2512"
    assert plan.source.revision == "25468b98e3276ca6700de15c6628e51b7de54a26"
    assert plan.edit_revision == "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
    assert plan.hf_repo == "Mitroshenkov87/voxprint-mirror-qwen-image-2512"
    assert plan.retired == ("qwen", "encoder")
    assert set(plan.encoder_pins) == {
        "text_encoder/model-00001-of-00004.safetensors",
        "text_encoder/model-00002-of-00004.safetensors",
        "text_encoder/model-00003-of-00004.safetensors",
        "text_encoder/model-00004-of-00004.safetensors",
        "text_encoder/config.json",
        "text_encoder/generation_config.json",
        "text_encoder/model.safetensors.index.json",
    }
    assert all(len(value) == 64 for value in plan.encoder_pins.values())


def test_retired_part_must_be_qwen_or_encoder(tmp_path: Path) -> None:
    path = tmp_path / "plan.toml"
    body = (ROOT / "plans" / "canvas-qwen-image-2512-v1.toml").read_text(encoding="utf-8")
    path.write_text(body.replace('retired = ["qwen", "encoder"]', 'retired = ["edit"]'), encoding="utf-8")
    with pytest.raises(PlanError):
        load_qwen_plan(path)


def test_parts_d_and_e_are_retired_unless_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("voxprint_forge.diffusion.qwen_main.canvas_ac_running", lambda: False)
    monkeypatch.setattr("voxprint_forge.diffusion.qwen_main.llm_job_running", lambda forge_out: False)
    from voxprint_forge.diffusion.qwen_main import main

    plan = load_qwen_plan(ROOT / "plans" / "canvas-qwen-image-2512-v1.toml")
    refuse_retired(parse_parts("C"), plan.retired, force=False)
    refuse_retired(parse_parts("D"), plan.retired, force=True)
    with pytest.raises(UsageError, match="retired"):
        refuse_retired(parse_parts("D"), plan.retired, force=False)
    with pytest.raises(UsageError, match="retired"):
        refuse_retired(parse_parts("E"), plan.retired, force=False)

    base = [
        "run",
        str(ROOT / "plans" / "canvas-qwen-image-2512-v1.toml"),
        "--json",
        "--forge-out",
        str(tmp_path / "forge-out"),
        "--root",
        str(ROOT),
    ]
    for extra in ([], ["--parts", "D"], ["--parts", "E"], ["--parts", "D,E"], ["--parts", "C,D"]):
        assert main([*base, *extra]) == 2
        payload = json.loads(capsys.readouterr().out)
        message = payload["error"]["message"]
        assert "retired" in message
        assert payload["error"]["code"] == "retired"
    assert main(["upload", *base[1:], "--parts", "D"]) == 2
    assert "retired" in json.loads(capsys.readouterr().out)["error"]["message"]
    from voxprint_forge.cli import main as forge_main

    assert (
        forge_main(
            [
                "--json",
                "--root",
                str(ROOT),
                "--forge-out",
                str(tmp_path / "forge-out"),
                "canvas",
                "upload",
                str(ROOT / "plans" / "canvas-qwen-image-2512-v1.toml"),
                "--parts",
                "D",
            ]
        )
        == 2
    )
    assert "retired" in json.loads(capsys.readouterr().out)["error"]["message"]


def test_part_c_converts_the_encoder_from_edit_only(tmp_path: Path) -> None:
    plan = load_canvas_plan(ROOT / "plans" / "canvas-zimage-tiers-v1.toml")
    forge_out = tmp_path / "forge-out"
    ctx = Ctx(
        repo=ROOT,
        forge_out=forge_out,
        plan=plan,
        part=plan.part("edit"),
        token=None,
        layout=Layout(forge_out),
    )
    edit = forge_out / "work" / plan.id / "sources" / "edit"
    for folder in ("transformer", "text_encoder", "vae", "processor", "tokenizer"):
        path = edit / folder / "model.safetensors"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"layers.0.attention.to_q.weight": torch.zeros(4, 4), "norm.weight": torch.zeros(4)}, path)
        (edit / folder / "config.json").write_text("{}\n", encoding="utf-8")
    (forge_out / "work" / plan.id / "sources" / "upstream-sha256.json").write_text("{}\n", encoding="utf-8")
    decoy = forge_out / "canvas" / "qwen-image-2512" / "encoder" / EDIT_ENCODER_FP8
    decoy.parent.mkdir(parents=True)
    decoy.write_bytes(b"from-part-d")
    _convert(ctx)
    produced = ctx.output / EDIT_ENCODER_FP8
    assert produced.is_file()
    assert produced.read_bytes() != b"from-part-d"
    assert decoy.read_bytes() == b"from-part-d"
    assert "qwen-image-2512" not in str(produced)


def test_int4_other_than_deferred_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "plan.toml"
    body = (ROOT / "plans" / "canvas-qwen-image-2512-v1.toml").read_text(encoding="utf-8")
    path.write_text(body.replace('int4 = "deferred"', 'int4 = "calibrate"'), encoding="utf-8")
    with pytest.raises(PlanError):
        load_qwen_plan(path)


def test_fp8_gate_matches_part_a_and_int4_stays_stricter() -> None:
    assert decide_qwen_fp8(0.10, 0.01) == "fp8"
    assert decide_qwen_fp8(0.11, 0.01) == "bf16"
    assert decide_qwen_int4(0.15, 0.02) == "int4"
    assert decide_qwen_int4(0.16, 0.01) == "bf16"


def test_canvas_job_d_does_not_match_the_ac_guard() -> None:
    assert command_is_canvas_ac("bash scripts/canvas_job.sh")
    assert command_is_canvas_ac("bash scripts/canvas_job.sh --force")
    assert not command_is_canvas_ac("bash scripts/canvas_job_d.sh")
    assert not command_is_canvas_ac("bash scripts/canvas_job_d.sh --parts C,E")


def test_classify_reuses_only_a_matching_pin_with_a_file(tmp_path: Path) -> None:
    pins = {"text_encoder/config.json": "abc"}
    present = tmp_path / "encoder.safetensors"
    present.write_bytes(b"fp8")
    missing = tmp_path / "missing.safetensors"
    assert classify_encoder(pins, pins, present) == "reuse"
    assert classify_encoder(pins, pins, missing) == "reuse-pending"
    assert classify_encoder({"text_encoder/config.json": "nope"}, pins, present) == "reuse-pending"


def test_int4_marker_has_no_weights_and_reuse_links_the_file(tmp_path: Path) -> None:
    marker = write_int4_marker(tmp_path)
    assert marker.read_text(encoding="utf-8") == INT4_MARKER_TEXT
    assert "nunchaku" in marker.read_text(encoding="utf-8")
    assert list(marker.parent.rglob("*.safetensors")) == []
    target = tmp_path / "part-c.safetensors"
    target.write_bytes(b"same-bytes")
    linked = link_stable_file(tmp_path, "encoder", target)
    if linked.is_symlink():
        assert linked.resolve() == target.resolve()
    else:
        assert linked.stat().st_ino == target.stat().st_ino
    assert "diffusers" not in sys.modules


def test_edit_encoder_path_matches_part_c(tmp_path: Path) -> None:
    path = edit_encoder_fp8(
        tmp_path,
        "canvas-zimage-tiers-v1",
        "Qwen--Qwen-Image-Edit-2511-voxprint",
        "qwen2.5-vl-7b-text-encoder-fp8.safetensors",
    )
    assert path == (
        tmp_path
        / "work"
        / "canvas-zimage-tiers-v1"
        / "Qwen--Qwen-Image-Edit-2511-voxprint"
        / "qwen2.5-vl-7b-text-encoder-fp8.safetensors"
    )
