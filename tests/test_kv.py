"""KV-cache argv flags, KV arm plan rows, spec/prefix bench cells, and the b11476 preflight checks."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from voxprint_forge.errors import PlanError, StageError
from voxprint_forge.kl import kv
from voxprint_forge.llama import kl_argv, kv_cache_args, perplexity_argv, with_kv_cache

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "plans" / "kv-synth4.toml"


def _flag(argv: list[str], name: str) -> str | None:
    return argv[argv.index(name) + 1] if name in argv else None


def test_default_argv_is_unchanged() -> None:
    argv = perplexity_argv("llama-perplexity", Path("m.gguf"), Path("c.txt"), chunks=1, ctx=512)
    assert "-ctk" not in argv and "-ctv" not in argv
    assert _flag(argv, "-fa") == "on"


def test_cache_flags_reach_perplexity_and_kl() -> None:
    argv = perplexity_argv("p", Path("m"), Path("c"), chunks=1, ctx=512, cache_k="q8_0", cache_v="q4_0")
    assert (_flag(argv, "-ctk"), _flag(argv, "-ctv"), _flag(argv, "-fa")) == ("q8_0", "q4_0", "on")
    kl = kl_argv("p", Path("m"), Path("c"), Path("l"), chunks=1, ctx=8192, compare=True, cache_k="q4_0", cache_v="q4_0")
    assert "--kl-divergence" in kl and _flag(kl, "-ctk") == "q4_0" and _flag(kl, "-c") == "8192"


def test_quantized_cache_adds_flash_attention_when_missing() -> None:
    assert with_kv_cache(["llama-x"], "q8_0", "q4_0")[-2:] == ["-fa", "on"]
    assert "-fa" not in with_kv_cache(["llama-x"], "f16", "f16")
    assert with_kv_cache(["x", "-fa", "on"], "q4_0", "q4_0").count("-fa") == 1


def test_unknown_cache_type_is_rejected() -> None:
    with pytest.raises(StageError):
        kv_cache_args("q2_k", None)


def test_plan_rows() -> None:
    plan = kv.load_kv_plan(PLAN)
    assert plan.contexts == ((512, 100), (8192, 12))
    assert plan.base == ("f16", "f16")
    assert plan.arms == (("q8_0", "q8_0"), ("q8_0", "q4_0"), ("q4_0", "q4_0"))
    assert [m.id for m in plan.models] == ["L1", "L2", "L3", "L4"]
    assert next(m for m in plan.models if m.id == "L2").requires


def test_kv_steps_score_each_arm_against_an_f16_base() -> None:
    plan = kv.load_kv_plan(PLAN)
    steps = kv.kv_steps(plan, Path("m.gguf"), Path("c.txt"), Path("w"))
    assert len(steps) == 2 * (1 + 3)
    for ctx in (512, 8192):
        rows = [s for s in steps if s["ctx"] == ctx]
        assert rows[0]["role"] == "base" and "--kl-divergence" not in rows[0]["argv"]
        assert _flag(rows[0]["argv"], "-ctk") == "f16"
        logits = _flag(rows[0]["argv"], "--kl-divergence-base")
        for arm in rows[1:]:
            assert "--kl-divergence" in arm["argv"] and _flag(arm["argv"], "--kl-divergence-base") == logits
            assert _flag(arm["argv"], "-fa") == "on"


def test_verdict_uses_kl_and_relative_ppl() -> None:
    plan = kv.load_kv_plan(PLAN)
    assert kv.verdict(plan, 0.004, 10.05, 10.0) == "pass"
    assert kv.verdict(plan, 0.02, 10.0, 10.0) == "fail"
    assert kv.verdict(plan, 0.004, 10.2, 10.0) == "fail"
    assert kv.verdict(plan, 0.004, 99.0, 10.0, ppl_usable=False) == "pass"


def test_bench_cells_cover_spec_and_prefix_arms() -> None:
    plan = kv.load_kv_plan(PLAN)
    cells = kv.bench_matrix(plan)
    spec = [c for c in cells if c["kind"] == "spec"]
    prefix = [c for c in cells if c["kind"] == "prefix"]
    assert len(spec) == 3 * 4 * 2 and len(prefix) == 3 * 2
    assert {c["arm"] for c in spec} == {"none", "ngram-simple", "ngram-map-k", "ngram-mod"}
    assert any("--cache-reuse" in c["args"] for c in prefix)
    for c in spec:
        if c["args"]:
            assert c["args"][0] == "--spec-type" and c["args"][1] in kv.SPEC_TYPES


def test_bad_spec_type_is_a_plan_error(tmp_path: Path) -> None:
    text = PLAN.read_text(encoding="utf-8").replace('"ngram-mod", "--spec-ngram', '"ngram-bogus", "--spec-ngram')
    bad = tmp_path / "p.toml"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(PlanError):
        kv.load_kv_plan(bad)


def test_summarize_timings() -> None:
    rows = [
        {"predicted_per_second": 40.0, "draft_n": 100, "draft_n_accepted": 60, "prompt_ms": 900.0, "cache_n": 0},
        {"predicted_per_second": 60.0, "draft_n": 100, "draft_n_accepted": 80, "prompt_ms": 100.0, "cache_n": 3000},
    ]
    s = kv.summarize(rows)
    assert s["tokens_per_second"] == 50.0 and s["accept_rate"] == 0.7 and s["prompt_ms_mean"] == 500.0
    assert kv.summarize([{"predicted_per_second": 10.0}])["accept_rate"] is None


def test_bench_url_must_be_localhost() -> None:
    plan = kv.load_kv_plan(PLAN)
    with pytest.raises(PlanError):
        kv.run_cell("http://example.com:8080", plan, ["x"])


def test_rewrite_prompts_are_fixed(tmp_path: Path) -> None:
    plan = kv.load_kv_plan(PLAN)
    text = "\n".join(["= Title ="] + [f"Paragraph {i} " + "word " * 120 for i in range(10)])
    a, b = kv.rewrite_prompts(plan, text), kv.rewrite_prompts(plan, text)
    assert a == b and len(a) == 8 and a[0].startswith(str(plan.bench["rewrite_instruction"]))


def test_corpus_sha_is_checked(tmp_path: Path) -> None:
    plan = kv.load_kv_plan(PLAN)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(plan.corpus["member"], "text")
    fake = tmp_path / "w.zip"
    fake.write_bytes(buf.getvalue())
    with pytest.raises(PlanError, match="sha256"):
        kv.fetch_corpus(plan, tmp_path / "c", fetch=lambda p, d: fake)


SERVER_HELP = (
    "--spec-type [none|draft-simple|ngram-cache|ngram-simple|ngram-map-k|ngram-map-k4v|ngram-mod]\n--cache-reuse N\n"
)
PPL_HELP = "-ctk, --cache-type-k TYPE\n-ctv, --cache-type-v TYPE\n--kl-divergence\n"


def test_help_checks() -> None:
    assert all(c["ok"] for c in kv.check_help(SERVER_HELP, PPL_HELP))
    old = kv.check_help("--draft N\n", PPL_HELP)
    assert not old[0]["ok"] and not old[1]["ok"]


def test_protobuf_pin() -> None:
    assert kv.check_protobuf(lambda d: "5.29.6")["ok"]
    assert not kv.check_protobuf(lambda d: "4.25.9")["ok"]


def test_gemma_meta_check() -> None:
    good = {"tokenizer.ggml.add_bos_token": True, "tokenizer.ggml.bos_token_id": 2,
            "tokenizer.chat_template": "{{ bos_token }}<start_of_turn>user"}
    assert kv.check_gemma_meta(good)["ok"]
    bad = dict(good, **{"tokenizer.ggml.add_bos_token": False, "tokenizer.chat_template": ""})
    result = kv.check_gemma_meta(bad)
    assert not result["ok"] and "add_bos_token" in result["detail"] and "template" in result["detail"]
