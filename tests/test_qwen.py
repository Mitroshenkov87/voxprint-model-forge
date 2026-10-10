"""Part D selectors, pins, and the encoder link. No diffusers import."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
from voxprint_forge.diffusion.qwen_select import parse_parts
from voxprint_forge.errors import PlanError, UsageError

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
    assert classify_encoder({"text_encoder/config.json": "nope"}, pins, present) == "convert"


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
