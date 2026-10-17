"""MOSS-Image: vendored pipeline pin and audit, loader safety, plan, fit matrix, Comfy graph, preflight."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from voxprint_forge.diffusion import moss, moss_fit, moss_preflight
from voxprint_forge.errors import ChecksumError, StageError

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "plans" / "moss-synth4.toml"
FIXTURE = Path(__file__).parent / "fixtures" / "comfy_object_info_moss.json"


def test_vendored_file_is_pinned_and_clean() -> None:
    assert moss.verify_vendored() == moss.VENDORED_SHA256
    assert moss.audit_source(moss.VENDORED.read_text(encoding="utf-8")) == []
    licence = moss.VENDORED.parent / "LICENSE"
    assert hashlib.sha256(licence.read_bytes()).hexdigest() == moss.LICENSE_SHA256


def test_tampered_vendored_file_fails(tmp_path: Path) -> None:
    copy = tmp_path / "pipeline_moss_image.py"
    copy.write_bytes(moss.VENDORED.read_bytes() + b"\n")
    with pytest.raises(ChecksumError):
        moss.verify_vendored(copy)


def test_audit_flags_remote_code_and_deserialisers() -> None:
    bad = "import pickle\nimport requests\nx = eval('1')\ntorch.load(f)\nf(trust_remote_code=True)\nos.system('x')\n"
    found = moss.audit_source(bad)
    assert len(found) == 6
    assert moss.audit_source("model.eval()\njson.loads(s)\nm.load_state_dict(s)\nf(trust_remote_code=False)\n") == []


def test_loader_never_trusts_remote_code() -> None:
    src = Path(moss.__file__).read_text(encoding="utf-8")
    assert moss.audit_source(src) == []
    assert "trust_remote_code=True" not in src
    assert "custom_pipeline" not in src
    assert src.count("local_files_only") >= 2


def _snapshot(tmp_path: Path) -> Path:
    root = tmp_path / "snap"
    for sub in ("transformer", "text_encoder", "tokenizer", "vae", "scheduler"):
        (root / sub).mkdir(parents=True)
    (root / "model_index.json").write_text(json.dumps({"_diffusers_version": "0.38.0", **moss.EXPECTED_INDEX}))
    return root


def test_check_snapshot_accepts_the_pinned_layout(tmp_path: Path) -> None:
    moss.check_snapshot(_snapshot(tmp_path))


def test_check_snapshot_rejects_pickles_and_other_classes(tmp_path: Path) -> None:
    root = _snapshot(tmp_path)
    (root / "transformer" / "pytorch_model.bin").write_bytes(b"x")
    with pytest.raises(StageError, match="non-safetensors"):
        moss.check_snapshot(root)
    shutil.rmtree(root)
    root = _snapshot(tmp_path)
    index = json.loads((root / "model_index.json").read_text())
    index["transformer"] = ["evil_module", "MOSSImageTransformer"]
    (root / "model_index.json").write_text(json.dumps(index))
    with pytest.raises(StageError, match="transformer"):
        moss.check_snapshot(root)


def test_versions() -> None:
    assert moss.check_versions("2.14.1+cu130", "0.41.0", "5.20.1") == []
    assert len(moss.check_versions("2.8.0", "0.37.1", "4.57.0")) == 3


def test_plan_rows_and_matrix() -> None:
    plan = moss_fit.load_plan(PLAN)
    assert (plan.width, plan.height, plan.steps, plan.guidance) == (1024, 1024, 50, 3.0)
    assert plan.caps_gb == (16.0, 24.0)
    assert {m["key"] for m in plan.mirrors} == {"base", "sft"}
    for m in plan.mirrors:
        assert m["repo"] == moss.REPOS[m["key"]].repo
        assert m["revision"] == moss.REPOS[m["key"]].revision
        assert m["mirror"] == moss.REPOS[m["key"]].mirror
        assert len(m["mirror_revision"]) == 40
    cells = moss_fit.matrix(plan, env={})
    assert len(cells) == 6
    by = {(c["arm"], c["cap_gb"]): c for c in cells}
    assert by[("bf16-te-offload", 16.0)]["te_mode"] == "cpu"
    assert by[("bf16-te-offload", 24.0)]["te_mode"] == "swap"
    assert "FORGE_MOSS_FP8_DIR" in by[("c-dq", 16.0)]["skip"]
    assert moss_fit.matrix(plan, env={"FORGE_MOSS_FP8_DIR": "/x"})[2]["skip"] == ""
    comfy = next(a for a in plan.arms if a.engine == "comfyui")
    assert comfy.role == "reference" and all(len(s) == 64 for s in comfy.sha256)


def test_prompt_ids_are_short_hashes() -> None:
    assert moss_fit.prompt_id("a cat") == hashlib.sha256(b"a cat").hexdigest()[:12]


def test_comfy_graph_matches_the_pinned_object_info() -> None:
    plan = moss_fit.load_plan(PLAN)
    arm = next(a for a in plan.arms if a.engine == "comfyui")
    info = json.loads(FIXTURE.read_text(encoding="utf-8"))
    graph = moss_fit.comfy_workflow(plan, arm, "p", 1, "x")
    assert moss_fit.comfy_nodes_missing(info, graph) == []
    for node in graph.values():
        spec = info[node["class_type"]]["input"]
        assert set(spec["required"]) <= set(node["inputs"]), node["class_type"]
        assert set(node["inputs"]) <= set(spec["required"]) | set(spec["optional"]), node["class_type"]
    assert graph["7"]["inputs"]["steps"] == 50 and graph["7"]["inputs"]["cfg"] == 3.0
    assert graph["2"]["inputs"]["type"] == "moss_image"


def test_comfy_url_must_be_localhost() -> None:
    plan = moss_fit.load_plan(PLAN)
    arm = next(a for a in plan.arms if a.engine == "comfyui")
    with pytest.raises(Exception, match="127.0.0.1"):
        moss_fit.run_comfy(plan, arm, 16.0, "http://example.com:8188", Path("."))


def _lister(plan: moss_fit.FitPlan, *, corrupt: bool = False):
    comfy = next(a for a in plan.arms if a.engine == "comfyui")

    def lister(repo: str, rev: str) -> dict[str, tuple[int, str]]:
        if repo == comfy.repo:
            return {f: (1, ("0" * 64 if corrupt else s)) for f, s in zip(comfy.files, comfy.sha256)}
        files = {f"f{i}.json": (10, "") for i in range(29)}
        files.update({f"w{i}.safetensors": (10, "a" * 64) for i in range(8)})
        return files

    return lister


def test_preflight_passes_with_pinned_listing() -> None:
    plan = moss_fit.load_plan(PLAN)
    pins = {"torch": "2.14.1", "diffusers": "0.41.0", "transformers": "5.20.1"}
    checks = moss_preflight.run(
        plan, lister=_lister(plan), versions=lambda d: pins.get(d, "1.0"), cuda=lambda: "H100",
        disk_free_gb=lambda: 200.0, vm=True,
    )
    assert all(c["ok"] for c in checks if c["required"]), checks


def test_preflight_fails_on_sha_mismatch_and_old_diffusers() -> None:
    plan = moss_fit.load_plan(PLAN)
    checks = moss_preflight.run(
        plan, lister=_lister(plan, corrupt=True), versions=lambda d: "0.37.1" if d == "diffusers" else "9.0",
        cuda=lambda: "", disk_free_gb=lambda: 10.0, vm=True,
    )
    failed = {c["check"] for c in checks if c["required"] and not c["ok"]}
    assert {"runtime versions", "cuda", "disk"} <= failed
    assert any(name.startswith("hub Comfy-Org") for name in failed)
