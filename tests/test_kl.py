"""KL argv, b11476 log parse, tiny-GGUF fit, and one scripted L-job."""

from __future__ import annotations

import json
from pathlib import Path

import gguf
import numpy as np
import pytest

from voxprint_forge.cli import main
from voxprint_forge.errors import StageError
from voxprint_forge.fitmath import fits_budget
from voxprint_forge.kl.arch import mtp_status, read_model_arch, require_mtp
from voxprint_forge.kl.parse import parse_kl
from voxprint_forge.kl.plan import KlJob, QuantRef, load_kl_plan
from voxprint_forge.kl.run import run_kl, select_quant_file
from voxprint_forge.llama import Completed, Runner, kl_argv

ROOT = Path(__file__).resolve().parents[1]

BASE_LOG = """\
Final estimate: PPL = 10.5
llama_perf_context_print: prompt eval time = 100.00 ms / 512 tokens ( 0.20 ms per token, 5000.00 tokens per second)
"""

KL_LOG = """\
Mean PPL(Q)                   :  12.500000 ±   0.100000
Mean PPL(base)                :  10.500000 ±   0.050000
Mean    KLD:   0.012345 ±   0.001000
99.0%   KLD:   0.045000
RMS Δp    :  1.250 ± 0.100 %
Same top p: 96.500 ± 0.400 %
llama_perf_context_print: prompt eval time = 200.00 ms / 512 tokens ( 0.39 ms per token, 3200.00 tokens per second)
"""

SERVER_LOG = """\
load_tensors:        CUDA0 model buffer size =    10.00 MiB
llama_kv_cache:      CUDA0 KV buffer size =     2.00 MiB
"""


def write_tiny(path: Path, *, blocks: int = 2, nextn: int = 0, sliding: bool = False) -> None:
    """A readable GGUF with a 4x32 embedding and the KV fields the fit stage needs."""
    writer = gguf.GGUFWriter(str(path), "qwen2")
    writer.add_block_count(blocks)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_embedding_length(32)
    writer.add_key_length(8)
    if nextn:
        writer.add_nextn_predict_layers(nextn)
    if sliding:
        writer.add_sliding_window(16)
        writer.add_full_attention_interval(2)
    writer.add_tensor("token_embd.weight", np.zeros((4, 32), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class Scripted(Runner):
    """Returns canned logs. The binaries are never started."""

    def run(self, argv: list[str], *, cwd: Path | None = None) -> Completed:
        del cwd
        if "--kl-divergence" in argv:
            return Completed(argv=argv, returncode=0, stdout=KL_LOG, stderr="")
        return Completed(argv=argv, returncode=0, stdout=BASE_LOG, stderr="")

    def capture_until(self, argv: list[str], *, pattern: str, timeout: float = 180) -> Completed:
        del pattern, timeout
        return Completed(argv=argv, returncode=0, stdout=SERVER_LOG, stderr="")


def _bins(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder.mkdir()
    for name in ("llama-perplexity", "llama-server"):
        path = folder / name
        path.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("FORGE_LLAMA_BIN", str(folder))


def test_kl_argv_writes_then_compares() -> None:
    base = kl_argv(
        "llama-perplexity",
        Path("f16.gguf"),
        Path("ppl.txt"),
        Path("logits.bin"),
        chunks=100,
        ctx=512,
        compare=False,
    )
    assert "--kl-divergence-base" in base
    assert "--kl-divergence" not in base
    assert base[base.index("--fit") + 1] == "off"
    assert base[base.index("--chunks") + 1] == "100"
    compared = kl_argv(
        "llama-perplexity",
        Path("q.gguf"),
        Path("ppl.txt"),
        Path("logits.bin"),
        chunks=100,
        ctx=512,
        compare=True,
    )
    assert compared[-1] == "--kl-divergence"
    assert "--kl-divergence-base" in compared


def test_parse_kl_lines() -> None:
    stats = parse_kl(KL_LOG)
    assert stats.ppl_quant == pytest.approx(12.5)
    assert stats.ppl_base == pytest.approx(10.5)
    assert stats.kl_mean == pytest.approx(0.012345)
    assert stats.kl_p99 == pytest.approx(0.045)
    assert stats.top1_agreement_pct == pytest.approx(96.5)
    assert stats.delta_p_rms_pct == pytest.approx(1.25)
    assert stats.tokens_per_second == pytest.approx(3200.0)


def test_tiny_gguf_arch_and_budget(tmp_path: Path) -> None:
    path = tmp_path / "tiny.gguf"
    write_tiny(path, blocks=4, sliding=True)
    arch, meta = read_model_arch(path)
    assert meta["architecture"] == "qwen2"
    assert meta["full_layers"] == 2
    assert meta["swa_layers"] == 2
    assert meta["head_dim"] == 8
    kv = arch.kv_bytes(8192, "q8_0")
    assert kv > 0
    assert fits_budget(path.stat().st_size, kv, budget_gb=16, headroom_gb=3) is True
    assert fits_budget(80 * 1024**3, kv, budget_gb=16, headroom_gb=3) is False


def test_mtp_head_must_survive_conversion() -> None:
    kept = mtp_status(2, 1, {"block_count": 3, "nextn_predict_layers": 1})
    assert kept["mtp_kept"] is True
    require_mtp(kept)
    dropped = mtp_status(2, 1, {"block_count": 2, "nextn_predict_layers": 0})
    with pytest.raises(StageError) as caught:
        require_mtp(dropped)
    assert caught.value.code == "kl_mtp"


def test_plan_names_l1_through_l4() -> None:
    plan = load_kl_plan(ROOT / "plans" / "kl-synth4.toml")
    assert [item.id for item in plan.jobs] == ["L1", "L2", "L3", "L4"]
    l2 = plan.job("L2")
    assert l2.source_format == "gguf"
    assert l2.f16_file == "gemma-4-31b-abliterated-f16.gguf"
    l3 = plan.job("L3")
    assert l3.expect_mtp is True
    assert l3.bf16_revision.startswith("27502c87")
    l4 = plan.job("L4")
    assert l4.quants[0].ours is False
    assert "voxprint-model-index" in l4.index_gap
    assert l1_chunks(plan.job("L1")) == (100, 512, 8192)


def l1_chunks(job: KlJob) -> tuple[int, int, int]:
    return job.chunks, job.ctx, job.fit_ctx


def test_scripted_job_reclaims_base_and_records_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _bins(tmp_path / "bin", monkeypatch)
    quant = tmp_path / "model-Q4_K_M.gguf"
    write_tiny(quant, blocks=2)
    corpus = tmp_path / "ppl.txt"
    corpus.write_text("one line of public text\n", encoding="utf-8")

    def fetch(repo: str, revision: str, dest: Path, patterns: list[str] | None) -> None:
        del repo, revision, patterns
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text(
            json.dumps({"num_hidden_layers": 2, "mtp_num_hidden_layers": 1}),
            encoding="utf-8",
        )

    def convert(runner: Runner, source: Path, outfile: Path) -> None:
        del runner
        assert (source / "config.json").is_file()
        write_tiny(outfile, blocks=3, nextn=1)

    job = KlJob(
        id="L3",
        bf16_repo="example/bf16",
        bf16_revision="abc",
        source_format="safetensors",
        f16_file="",
        expect_mtp=True,
        index_gap="Mitroshenkov87/voxprint-model-index has no entry for voxprint-mirror-t-pro-it-2.1-gguf.",
        note="scripted",
        quants=(QuantRef(quant="Q4_K_M", repo="", revision="", pattern=str(quant), ours=False),),
        chunks=1,
        ctx=512,
        fit_ctx=8192,
        headroom_gb=3.0,
        budgets_gb=(16.0, 24.0),
        kv_cache="q8_0",
    )
    work = tmp_path / "work"
    report = run_kl(job, work, corpus=corpus, runner=Scripted(), fetch=fetch, convert=convert)
    assert report["ppl_base"] == pytest.approx(10.5)
    assert report["base_tokens_per_second"] == pytest.approx(5000.0)
    mtp = report["mtp"]
    assert isinstance(mtp, dict)
    assert mtp["mtp_kept"] is True
    assert "voxprint-model-index" in str(report["index_gap"])
    rows = report["quants"]
    assert isinstance(rows, list)
    assert rows[0]["kl_mean"] == pytest.approx(0.012345)
    assert rows[0]["top1_agreement_pct"] == pytest.approx(96.5)
    tiers = rows[0]["tiers"]
    assert [item["budget_gb"] for item in tiers] == [16.0, 24.0]
    assert tiers[0]["fits_computed"] is True
    assert tiers[0]["fits_server"] is True
    assert tiers[0]["kv_cache"] == "q8_0"
    assert tiers[0]["ctx"] == 8192
    assert not (work / "source").exists()
    assert not (work / "f16.gguf").exists()
    assert not (work / "base-logits.bin").exists()
    summary = (work / "summary.json").read_text(encoding="utf-8")
    assert "one line of public text" not in summary
    assert "Q4_K_M" in summary


def test_missing_mtp_head_fails_and_still_reclaims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _bins(tmp_path / "bin", monkeypatch)

    def fetch(repo: str, revision: str, dest: Path, patterns: list[str] | None) -> None:
        del repo, revision, patterns
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "config.json").write_text(
            json.dumps({"num_hidden_layers": 2, "mtp_num_hidden_layers": 1}),
            encoding="utf-8",
        )

    def convert(runner: Runner, source: Path, outfile: Path) -> None:
        del runner, source
        write_tiny(outfile, blocks=2, nextn=0)

    job = KlJob(
        id="L3",
        bf16_repo="example/bf16",
        bf16_revision="abc",
        source_format="safetensors",
        f16_file="",
        expect_mtp=True,
        index_gap="",
        note="",
        quants=(QuantRef(quant="Q4_K_M", repo="example/q", revision="abc", pattern="*Q4_K_M*.gguf", ours=True),),
        chunks=1,
        ctx=512,
        fit_ctx=8192,
        headroom_gb=3.0,
        budgets_gb=(16.0, 24.0),
        kv_cache="q8_0",
    )
    work = tmp_path / "work"
    (tmp_path / "ppl.txt").write_text("x\n", encoding="utf-8")
    with pytest.raises(StageError) as caught:
        run_kl(job, work, corpus=tmp_path / "ppl.txt", runner=Scripted(), fetch=fetch, convert=convert)
    assert caught.value.code == "kl_mtp"
    assert not (work / "source").exists()
    assert not (work / "f16.gguf").exists()


def test_gguf_source_skips_conversion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _bins(tmp_path / "bin", monkeypatch)
    quant = tmp_path / "quant-IQ3_XXS.gguf"
    write_tiny(quant)

    def fetch(repo: str, revision: str, dest: Path, patterns: list[str] | None) -> None:
        del repo, revision
        dest.mkdir(parents=True, exist_ok=True)
        assert patterns == ["gemma-4-31b-abliterated-f16.gguf"]
        write_tiny(dest / "gemma-4-31b-abliterated-f16.gguf", blocks=4, sliding=True)

    def convert(runner: Runner, source: Path, outfile: Path) -> None:
        del runner, source, outfile
        raise AssertionError("an F16 source must not be reconverted")

    corpus = tmp_path / "ppl.txt"
    corpus.write_text("public\n", encoding="utf-8")
    job = KlJob(
        id="L2",
        bf16_repo="example/f16",
        bf16_revision="abc",
        source_format="gguf",
        f16_file="gemma-4-31b-abliterated-f16.gguf",
        expect_mtp=False,
        index_gap="",
        note="",
        quants=(QuantRef(quant="IQ3_XXS", repo="", revision="", pattern=str(quant), ours=True),),
        chunks=1,
        ctx=512,
        fit_ctx=8192,
        headroom_gb=3.0,
        budgets_gb=(16.0, 24.0),
        kv_cache="q8_0",
    )
    report = run_kl(job, tmp_path / "out", corpus=corpus, runner=Scripted(), fetch=fetch, convert=convert)
    assert report["mtp"] is None
    assert report["source_format"] == "gguf"


def test_two_quant_files_are_a_setup_error(tmp_path: Path) -> None:
    (tmp_path / "a-Q4_K_M.gguf").write_bytes(b"a")
    (tmp_path / "b-Q4_K_M.gguf").write_bytes(b"b")
    with pytest.raises(StageError) as caught:
        select_quant_file(tmp_path, "*Q4_K_M*.gguf", "Q4_K_M")
    assert caught.value.code == "kl_quant"


def test_cli_requires_a_job_or_a_pair() -> None:
    assert main(["kl"]) == 2
