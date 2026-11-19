"""L1 reference: stock BF16 minus our recovered ablation direction. L2: PPL flagged when base log-probs clamp."""

from __future__ import annotations

import json
from pathlib import Path

import gguf
import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from voxprint_forge.abliterate import orthogonalize_weight
from voxprint_forge.errors import PlanError, StageError
from voxprint_forge.kl.arch import gguf_model_name
from voxprint_forge.kl.cli import execute
from voxprint_forge.kl.plan import KlJob, QuantRef, load_kl_plan
from voxprint_forge.kl.recover import apply_direction, recover_ablation, recover_direction
from voxprint_forge.kl.run import PPL_INVALID_NOTE, mark_ppl_validity, rerender, run_kl
from voxprint_forge.llama import Completed, Runner

ROOT = Path(__file__).resolve().parents[1]

BASE_LOG = "Final estimate: PPL = {base}\n"
KL_LOG = """\
Mean PPL(Q)                   :  {quant} ±   0.100000
Mean PPL(base)                :  {readback} ±   0.050000
Mean    KLD:   0.012345 ±   0.001000
99.0%   KLD:   0.045000
RMS Δp    :  1.250 ± 0.100 %
Same top p: 96.500 ± 0.400 %
"""
SERVER_LOG = """\
load_tensors:        CUDA0 model buffer size =    10.00 MiB
llama_kv_cache:      CUDA0 KV buffer size =     2.00 MiB
"""


def write_tiny(path: Path, *, name: str = "") -> None:
    writer = gguf.GGUFWriter(str(path), "qwen3")
    if name:
        writer.add_name(name)
    writer.add_block_count(2)
    writer.add_head_count(4)
    writer.add_head_count_kv(2)
    writer.add_embedding_length(32)
    writer.add_key_length(8)
    writer.add_tensor("token_embd.weight", np.zeros((4, 32), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


class Logs(Runner):
    def __init__(self, base: float, readback: float, quant: float) -> None:
        super().__init__()
        self.base, self.readback, self.quant = base, readback, quant

    def run(self, argv: list[str], *, cwd: Path | None = None) -> Completed:
        del cwd
        if "--kl-divergence" in argv:
            text = KL_LOG.format(readback=self.readback, quant=self.quant)
            return Completed(argv=argv, returncode=0, stdout=text, stderr="")
        return Completed(argv=argv, returncode=0, stdout=BASE_LOG.format(base=self.base), stderr="")

    def capture_until(self, argv: list[str], *, pattern: str, timeout: float = 180) -> Completed:
        del pattern, timeout
        return Completed(argv=argv, returncode=0, stdout=SERVER_LOG, stderr="")


def _bins(folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder.mkdir()
    for name in ("llama-perplexity", "llama-server"):
        (folder / name).write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("FORGE_LLAMA_BIN", str(folder))


def _job(quant: Path, *, reference: str = "recover-ablation") -> KlJob:
    return KlJob(
        id="L1",
        bf16_repo="example/bf16",
        bf16_revision="abc",
        source_format="safetensors",
        f16_file="",
        expect_mtp=False,
        index_gap="",
        note="",
        quants=(QuantRef(quant="Q5_K_M", repo="", revision="", pattern=str(quant), ours=True),),
        chunks=1,
        ctx=512,
        fit_ctx=8192,
        headroom_gb=3.0,
        budgets_gb=(16.0, 24.0),
        kv_cache="q8_0",
        reference=reference,  # type: ignore[arg-type]
        ablation_quant="Q5_K_M" if reference == "recover-ablation" else "",
    )


def _fetch(repo: str, revision: str, dest: Path, patterns: list[str] | None) -> None:
    del repo, revision, patterns
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "config.json").write_text(json.dumps({"num_hidden_layers": 2}), encoding="utf-8")


def test_plan_l1_recovers_the_ablation_l2_stays_upstream() -> None:
    plan = load_kl_plan(ROOT / "plans" / "kl-synth4.toml")
    l1 = plan.job("L1")
    assert (l1.reference, l1.ablation_quant, l1.source_format) == ("recover-ablation", "Q5_K_M", "safetensors")
    for other in ("L2", "L3", "L4"):
        assert plan.job(other).reference == "upstream"
        assert plan.job(other).ablation_quant == ""


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            'source_format = "gguf"\nf16_file = "x.gguf"\nreference = "recover-ablation"\nablation_quant = "Q4_K_M"\n',
            "needs safetensors",
        ),
        ('reference = "recover-ablation"\nablation_quant = "Q8_0"\n', "must name one of our quants"),
        ('ablation_quant = "Q4_K_M"\n', "needs reference recover-ablation"),
        ('reference = "hf"\n', "reference must be one of"),
    ],
)
def test_plan_rejects_bad_reference(tmp_path: Path, extra: str, message: str) -> None:
    path = tmp_path / "kl.toml"
    path.write_text(
        '[[jobs]]\nid = "L9"\nbf16_repo = "a/b"\nbf16_revision = "c"\n'
        + extra
        + '[[jobs.quants]]\nquant = "Q4_K_M"\nrepo = "a/q"\nrevision = "d"\npattern = "*.gguf"\n',
        encoding="utf-8",
    )
    with pytest.raises(PlanError, match=message):
        load_kl_plan(path)


def test_recovered_reference_is_converted_and_reclaimed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _bins(tmp_path / "bin", monkeypatch)
    quant = tmp_path / "Q5_K_M.gguf"
    write_tiny(quant, name="Ablated")
    corpus = tmp_path / "ppl.txt"
    corpus.write_text("public\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def recover(found: Path, source: Path, dest: Path) -> dict[str, object]:
        seen["quant"], seen["source"] = found.name, source.name
        dest.mkdir(parents=True)
        (dest / "config.json").write_text("{}", encoding="utf-8")
        return {"from": found.name, "matrices": 72, "eigen_ratio": 0.004}

    def convert(runner: Runner, source: Path, outfile: Path) -> None:
        del runner
        seen["converted"] = source.name
        write_tiny(outfile, name="Ablated")

    work = tmp_path / "work"
    report = run_kl(
        _job(quant),
        work,
        corpus=corpus,
        runner=Logs(14.43, 14.43, 14.6),
        fetch=_fetch,
        convert=convert,
        recover=recover,
    )
    assert seen == {"quant": "Q5_K_M.gguf", "source": "source", "converted": "ablated"}
    reference = report["reference"]
    assert isinstance(reference, dict)
    assert reference["kind"] == "recover-ablation"
    assert reference["name"] == "Ablated"
    assert reference["ablation"] == {"from": "Q5_K_M.gguf", "matrices": 72, "eigen_ratio": 0.004}
    row = report["quants"][0]  # type: ignore[index]
    assert row["reference_name_match"] is True
    assert row["ppl_valid"] is True
    assert report["ppl_valid"] is True
    for gone in ("ablated", "source", "f16.gguf"):
        assert not (work / gone).exists()
    text = (work / "summary.md").read_text(encoding="utf-8")
    assert "Reference: recover-ablation, refusal direction read back from Q5_K_M.gguf (72 matrices" in text
    assert "PPL 14.6" in text


def test_recovery_failure_reclaims(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _bins(tmp_path / "bin", monkeypatch)
    quant = tmp_path / "Q5_K_M.gguf"
    write_tiny(quant)
    corpus = tmp_path / "ppl.txt"
    corpus.write_text("public\n", encoding="utf-8")

    def recover(found: Path, source: Path, dest: Path) -> dict[str, object]:
        del found, source
        dest.mkdir(parents=True)
        raise StageError("Q5_K_M.gguf has no ablated direction", code="kl_reference")

    def convert(runner: Runner, source: Path, outfile: Path) -> None:
        raise AssertionError("nothing to convert")

    work = tmp_path / "work"
    with pytest.raises(StageError) as caught:
        run_kl(_job(quant), work, corpus=corpus, runner=Logs(1, 1, 1), fetch=_fetch, convert=convert, recover=recover)
    assert caught.value.code == "kl_reference"
    assert not (work / "ablated").exists()
    assert not (work / "source").exists()


def test_upstream_reference_never_recovers_and_flags_a_name_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _bins(tmp_path / "bin", monkeypatch)
    quant = tmp_path / "Q5_K_M.gguf"
    write_tiny(quant, name="Ablated")

    def recover(found: Path, source: Path, dest: Path) -> dict[str, object]:
        raise AssertionError("upstream reference must not recover an ablation")

    def convert(runner: Runner, source: Path, outfile: Path) -> None:
        del runner
        assert source.name == "source"
        write_tiny(outfile, name="T Lite It 2.1")

    corpus = tmp_path / "ppl.txt"
    corpus.write_text("public\n", encoding="utf-8")
    work = tmp_path / "work"
    report = run_kl(
        _job(quant, reference="upstream"),
        work,
        corpus=corpus,
        runner=Logs(14.43, 14.43, 16.9),
        fetch=_fetch,
        convert=convert,
        recover=recover,
    )
    row = report["quants"][0]  # type: ignore[index]
    assert (row["name"], row["reference_name_match"]) == ("Ablated", False)
    assert "WARNING: GGUF name 'Ablated' differs from the reference" in (work / "summary.md").read_text("utf-8")


def _writer_gguf(path: Path, mats: dict[str, np.ndarray]) -> None:
    writer = gguf.GGUFWriter(str(path), "qwen3")
    for name, value in mats.items():
        writer.add_tensor(name, value.astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_direction_is_read_back_and_reapplied(tmp_path: Path) -> None:
    hidden = 16
    rng = np.random.default_rng(7)
    true = rng.standard_normal(hidden)
    true /= np.linalg.norm(true)
    stock = {
        "model.layers.0.self_attn.o_proj.weight": torch.from_numpy(rng.standard_normal((hidden, 24))).float(),
        "model.layers.0.mlp.down_proj.weight": torch.from_numpy(rng.standard_normal((hidden, 40))).float(),
        "model.layers.0.mlp.up_proj.weight": torch.from_numpy(rng.standard_normal((40, hidden))).float(),
    }
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"hidden_size": hidden}), encoding="utf-8")
    save_file(stock, str(source / "model.safetensors"))
    unit = torch.from_numpy(true).float()
    ablated = {name: orthogonalize_weight(value, unit) for name, value in stock.items()}
    quant = tmp_path / "Q5_K_M.gguf"
    _writer_gguf(
        quant,
        {
            "blk.0.attn_output.weight": ablated["model.layers.0.self_attn.o_proj.weight"].numpy(),
            "blk.0.ffn_down.weight": ablated["model.layers.0.mlp.down_proj.weight"].numpy(),
            "blk.0.ffn_up.weight": ablated["model.layers.0.mlp.up_proj.weight"].numpy(),
        },
    )
    direction, info = recover_direction(quant, hidden)
    assert info["matrices"] == 2
    assert abs(float(direction @ true)) > 0.9999
    record = recover_ablation(quant, source, tmp_path / "ablated")
    assert record["from"] == "Q5_K_M.gguf"
    rebuilt = load_file(str(tmp_path / "ablated" / "model.safetensors"))
    for name, value in ablated.items():
        assert torch.allclose(rebuilt[name], value, atol=1e-5)
    assert (tmp_path / "ablated" / "config.json").is_file()


def test_a_stock_quant_has_no_ablated_direction(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    quant = tmp_path / "Q5_K_M.gguf"
    _writer_gguf(quant, {"blk.0.attn_output.weight": rng.standard_normal((16, 64))})
    with pytest.raises(StageError, match="no ablated direction"):
        recover_direction(quant, 16)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert apply_direction(empty, tmp_path / "out", np.ones(16) / 4, 16) == 0


def test_clamped_base_readback_marks_ppl_unusable() -> None:
    """L2 on synth4: base run 6827, KL read-back 3867 (log-probs floored at max-16)."""
    payload: dict[str, object] = {
        "ppl_base": 6827.0307,
        "quants": [{"ppl_base": 3867.615064, "ppl": 2677.4}, {"ppl_base": 3867.615064, "ppl": 14207.9}],
    }
    mark_ppl_validity(payload)
    assert payload["ppl_valid"] is False
    assert payload["ppl_note"] == PPL_INVALID_NOTE
    assert [row["ppl_valid"] for row in payload["quants"]] == [False, False]  # type: ignore[index]
    healthy: dict[str, object] = {"ppl_base": 14.4288, "quants": [{"ppl_base": 14.429058}]}
    mark_ppl_validity(healthy)
    assert healthy["ppl_valid"] is True
    assert healthy["ppl_note"] == ""


def test_render_rewrites_an_existing_summary_without_a_gpu(tmp_path: Path) -> None:
    work = tmp_path / "L2"
    work.mkdir()
    old = {
        "job": "L2",
        "ppl_base": 6827.0307,
        "gpu_note": "tokens/s is the NVIDIA H100 NVL figure from this box.",
        "note": "",
        "index_gap": "",
        "quants": [
            {
                "quant": "IQ3_XXS",
                "ppl": 14207.932744,
                "ppl_base": 3867.615064,
                "kl_mean": 2.293518,
                "kl_p99": 13.004377,
                "top1_agreement_pct": 44.314,
                "delta_p_rms_pct": 23.436,
                "file_bytes": 12077502368,
            }
        ],
    }
    (work / "summary.json").write_text(json.dumps(old), encoding="utf-8")
    payload = execute(
        ROOT, tmp_path, job_id=None, plan_path=None, bf16=None, quants=None, chunks=None, ctx=None, render=work
    )
    assert payload["ppl_valid"] is False
    text = (work / "summary.md").read_text(encoding="utf-8")
    assert "Base PPL: 6827.0307 (not usable, see below)" in text
    assert "- IQ3_XXS: PPL n/a, KL mean 2.293518" in text
    assert json.loads((work / "summary.json").read_text("utf-8"))["quants"][0]["ppl_valid"] is False
    with pytest.raises(StageError):
        rerender(tmp_path / "missing")


def test_gguf_model_name_reads_general_name(tmp_path: Path) -> None:
    path = tmp_path / "a.gguf"
    write_tiny(path, name="Ablated")
    assert gguf_model_name(path) == "Ablated"
    bare = tmp_path / "b.gguf"
    write_tiny(bare)
    assert gguf_model_name(bare) == ""
