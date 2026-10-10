"""Plan pins, licence gate, and KV math."""

from pathlib import Path

import pytest

from voxprint_forge.errors import LicenceError, PlanError
from voxprint_forge.fitmath import Arch, gib
from voxprint_forge.licence import decide_licence
from voxprint_forge.measure import Candidate, fit_contexts, fitting_candidates
from voxprint_forge.plan import Plan, Tier, format_estimates, load_plan
from voxprint_forge.upload import (
    assert_private_metadata,
    latest_version_number,
    parse_kaggle_version,
    require_kaggle_version,
)

ROOT = Path(__file__).resolve().parents[1]


def test_a100_plan_order_and_pins() -> None:
    plan = load_plan(ROOT / "plans" / "plan.toml")
    enabled = [model.id for model in plan.enabled_models()]
    assert enabled == ["huihui-qwen36-27b", "gemma4-31b", "t-lite-it-2.1", "t-pro-it-2.1"]
    assert plan.models[-1].id == "mistral-small-3.2-24b"
    assert plan.models[-1].enabled is False
    huihui = plan.models[0]
    assert huihui.tiers[0].kind == "reuse"
    assert huihui.tiers[1].kind == "quantize"
    assert "Q3_K_M" in huihui.tiers[1].quants
    gemma = plan.models[1]
    assert [item.quant for item in gemma.reuse_files()] == ["Q4_K_M"]
    assert "mradermacher" not in gemma.repo
    text = format_estimates(plan)
    assert "140 min" in text
    assert plan.artifact_host == ""
    required = sum(model.estimate_min for model in plan.enabled_models() if not model.optional)
    assert required + plan.estimates_min["bootstrap_build"] == 140


def test_excluded_repos() -> None:
    with pytest.raises(LicenceError):
        decide_licence("apache-2.0", repo="smkrv/T-lite-it-2.1-abl-GGUF")
    with pytest.raises(LicenceError):
        decide_licence("apache-2.0", repo="yandex/YandexGPT-5-Lite-8B-instruct")
    with pytest.raises(LicenceError):
        decide_licence("", repo="Qwen/Qwen3-8B")


def test_huihui_kv_q8_8k() -> None:
    arch = Arch(full_layers=16, full_kv_heads=4, full_head_dim=256)
    assert arch.kv_bytes(8192, "q8_0") == 285212672


def test_fit_gate_contexts() -> None:
    plan = Plan(
        name="t",
        output_dir=Path("."),
        wall_clock_budget_min=1,
        headroom_gb=3,
        imatrix_chunks=1,
        imatrix_ctx=512,
        refusal_limit=1,
        ppl_chunks=1,
        prose_samples=1,
        contexts=(8192, 16384),
        llama_cpp_tag="b11476",
        llama_cpp_commit="abc",
        estimates_min={},
        artifact_host="",
        models=(),
    )
    small = Tier("16gb", 16, "quantize", ("IQ4_XS",))
    large = Tier("24gb", 24, "reuse", ("Q5_K_M",))
    assert fit_contexts(small, plan) == (8192,)
    assert fit_contexts(large, plan) == (8192, 16384)


def test_oversized_quant_is_not_fitting() -> None:
    arch = Arch(1, 1, 1)
    huge = Candidate("Q5_K_M", 50 * 1024**3, Path("huge.gguf"), "reuse")
    small = Candidate("Q4_K_M", 1000, Path("small.gguf"), "reuse")
    fitting = fitting_candidates([huge, small], arch, (8192, 16384), budget_gb=24, headroom_gb=3, cache="q8_0")
    assert [item.quant for item in fitting] == ["Q4_K_M"]
    assert gib(1000) < 1


def test_latest_kaggle_version_is_not_the_first() -> None:
    text = "versionNumber,variationSlug\n1,a\n3,a\n2,a\n"
    assert latest_version_number(text) == 3
    assert latest_version_number("no rows\n") == 1


def test_kaggle_version_and_privacy() -> None:
    assert parse_kaggle_version("Kaggle CLI 2.2.1") == (2, 2)
    with pytest.raises(Exception):
        require_kaggle_version("kaggle 1.6.17")
    assert_private_metadata({"isPrivate": True})
    with pytest.raises(Exception):
        assert_private_metadata({"isPrivate": False})


def test_bootstrap_script_keeps_tokens_out_of_the_shell_trace() -> None:
    text = (ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    assert "set -x" not in text
    assert "set +x" in text
    assert "huggingface-cli login" not in text
    assert "hf auth login" not in text
    assert "GH_TOKEN" not in text
    assert "github.com" not in text
    for banned in ("git clone", "git fetch", "git checkout", "git archive", "git push", "git -c"):
        assert banned not in text
    assert "/dev/tty" in text
    assert "HF_TOKEN" in text
    assert "KAGGLE_API_TOKEN" in text
    assert "CMAKE_CUDA_ARCHITECTURES" in text
    assert "progress.log" in text
    assert "cleanup_secrets" in text
    assert "stored_tokens" in text
    assert ".kaggle" in text
    assert "access_token" in text
    assert "unset HF_TOKEN" in text
    assert "Ubuntu 24.04" in text
    assert "verify-logins" in text
    rev = (ROOT / "scripts" / "llama.rev").read_text(encoding="utf-8").strip()
    assert rev == "988190680d5a89fce97de3c20df2c2813731fd61"
    assert rev in (ROOT / "plans" / "plan.toml").read_text(encoding="utf-8")
    bundle = (ROOT / "scripts" / "make_vm_bundle.sh").read_text(encoding="utf-8")
    assert "archive --format=tar" in bundle
    assert "VERSION" in bundle
    assert "llama.rev" in bundle
    vm = (ROOT / "docs" / "VM.md").read_text(encoding="utf-8")
    assert "HF_TOKEN" in vm and "KAGGLE_API_TOKEN" in vm and "140" in vm
    assert "tmux new -s forge 'bash scripts/bootstrap.sh'" in vm
    assert "make_vm_bundle.sh" in vm
    assert "Ubuntu 24.04" in vm
    assert "GH_TOKEN" not in vm
    cli = (ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
    assert "private=True" in cli
    assert "isPrivate" in cli
    assert "base_model_relation" in cli


def test_bad_plan_sha(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(
        """
[run]
name = "bad"
[[models]]
id = "x"
enabled = true
mode = "requant"
repo = "org/model"
revision = "abc"
licence = "apache-2.0"
source_format = "gguf"
[[models.files]]
role = "source"
quant = "F16"
path = "a.gguf"
sha256 = "abcd"
size = 1
[[models.tiers]]
name = "16gb"
budget_gb = 16
kind = "quantize"
quants = ["IQ4_XS"]
""",
        encoding="utf-8",
    )
    with pytest.raises(PlanError):
        load_plan(path)
