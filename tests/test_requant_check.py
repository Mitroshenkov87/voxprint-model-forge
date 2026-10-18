"""GEMMA4-REQUANT checks: GGUF index, tensor diff, refusal probe, verdict, local KL reference."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import gguf
import numpy as np
import pytest

from tests.test_kl_reference import Logs, _bins, _job, write_tiny
from voxprint_forge import requant_check as rc
from voxprint_forge.errors import PlanError, StageError
from voxprint_forge.hashing import sha256_file
from voxprint_forge.kl.cli import _quant_spec
from voxprint_forge.kl.plan import load_kl_plan, local_revision
from voxprint_forge.kl.run import run_kl
from voxprint_forge.llama import Runner


def _gguf(path: Path, mats: dict[str, np.ndarray], dtype: type = np.float32) -> None:
    writer = gguf.GGUFWriter(str(path), "gemma4")
    writer.add_name("Stock")
    for name, value in mats.items():
        writer.add_tensor(name, value.astype(dtype))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _serve(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def ranged(url: str, start: int, size: int) -> bytes:
        assert url == "https://hub.example/stock.gguf"
        with path.open("rb") as handle:
            handle.seek(start)
            return handle.read(size)

    monkeypatch.setattr(rc, "_range", ranged)


def test_index_matches_gguf_reader(tmp_path: Path) -> None:
    path = tmp_path / "s.gguf"
    rng = np.random.default_rng(1)
    _gguf(path, {"a.weight": rng.standard_normal((3, 64)), "b.weight": rng.standard_normal((5, 32))})
    table = rc.parse_index(path.read_bytes())
    reader = gguf.GGUFReader(str(path))
    for tensor in reader.tensors:
        entry = table[tensor.name]
        assert (entry.offset, entry.count) == (int(tensor.data_offset), int(tensor.n_elements))
    with pytest.raises(ValueError, match="longer"):
        rc.parse_index(path.read_bytes()[:40])


def test_tensor_diff_needs_the_ablated_tensors_changed_and_the_rest_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(2)
    stock = {name: rng.standard_normal((16, 32)) for name in ("blk.1.attn_output.weight", "blk.1.attn_q.weight")}
    stock_path = tmp_path / "stock.gguf"
    _gguf(stock_path, stock, np.float16)
    _serve(stock_path, monkeypatch)
    direction = rng.standard_normal(16)
    direction /= np.linalg.norm(direction)
    ablated = dict(stock)
    ablated["blk.1.attn_output.weight"] = stock["blk.1.attn_output.weight"] - np.outer(
        direction, direction @ stock["blk.1.attn_output.weight"]
    )
    url = "https://hub.example/stock.gguf"
    for label, mats, expected in (("ablated", ablated, True), ("stock", stock, False)):
        local = tmp_path / f"{label}.gguf"
        _gguf(local, mats, np.float16)
        report = rc.tensor_diff(local, url, ["blk.1.attn_output.weight"], ["blk.1.attn_q.weight"])
        assert report["pass"] is expected, report
    missing = rc.tensor_diff(tmp_path / "stock.gguf", url, ["blk.9.ffn_down.weight"], ["blk.1.attn_q.weight"])
    assert missing["pass"] is False


def test_refusal_markers_and_ids() -> None:
    assert rc.is_refusal("I\u2019m sorry, but I can\u2019t help with that.")
    assert rc.is_refusal("Извините, но я не могу помочь.")
    assert not rc.is_refusal("Run kill -9 on the process id.")
    assert rc.prompt_id("x") == "2d711642b726"


def test_probe_keeps_ids_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("# private\nsecret prompt one\nsecret prompt two\n", encoding="utf-8")
    replies = iter(["I cannot help with that.", "Sure, here is how."])
    monkeypatch.setattr(rc, "_chat", lambda server, text, *, max_tokens: next(replies))
    out = tmp_path / "probe.json"
    assert (
        rc.main(
            ["probe", "--server", "http://127.0.0.1:1", "--prompts", str(prompts), "--label", "s", "--out", str(out)]
        )
        == 0
    )
    text = out.read_text(encoding="utf-8")
    assert "secret" not in text and "help" not in text
    assert json.loads(text)["refusal_rate"] == 0.5


@pytest.mark.parametrize(
    ("stock", "new", "passed"),
    [(0.5, 0.125, True), (0.5, 0.3, False), (0.2, 0.0, False), (1.0, 0.5, True), (1.0, 0.6, False)],
)
def test_probe_threshold(stock: float, new: float, passed: bool) -> None:
    assert rc.probe_verdict(stock, new)["pass"] is passed


def test_verdict(tmp_path: Path) -> None:
    (tmp_path / "tensor-diff.json").write_text(json.dumps({"pass": True}), encoding="utf-8")
    for label, rate in (("stock-Q4_K_M", 0.5), ("new-Q4_K_M", 0.0)):
        probe = {"model": label, "n": 4, "empty": 0, "refusal_rate": rate}
        (tmp_path / f"probe-{label}.json").write_text(json.dumps(probe), encoding="utf-8")
    summary = tmp_path / "summary.json"
    rows = [{"quant": "Q4_K_M", "kl_mean": 0.9, "top1_agreement_pct": 66.0}]
    summary.write_text(json.dumps({"quants": rows}), encoding="utf-8")
    args = [
        "verdict",
        "--results",
        str(tmp_path),
        "--kl-summary",
        str(summary),
        "--pair",
        "Q4_K_M:stock-Q4_K_M:new-Q4_K_M",
    ]
    good = json.dumps({"Q4_K_M": {"kl": 0.83, "top1": 68.4}})
    assert rc.main([*args, "--l2-kl", good]) == 0
    bad = json.dumps({"Q4_K_M": {"kl": 0.5, "top1": 68.4}, "IQ3_XXS": {"kl": 2.3, "top1": 44.3}})
    assert rc.main([*args, "--l2-kl", bad]) == 1


def test_local_reference_is_used_in_place(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _bins(tmp_path / "bin", monkeypatch)
    reference = tmp_path / "keep" / "new-f16.gguf"
    reference.parent.mkdir()
    write_tiny(reference, name="Gemma 4 31B It")
    quant = tmp_path / "Q4_K_M.gguf"
    write_tiny(quant, name="Gemma 4 31B It")

    def fetch(repo: str, revision: str, dest: Path, patterns: list[str] | None) -> None:
        raise AssertionError("a local reference must not be downloaded")

    def convert(runner: Runner, source: Path, outfile: Path) -> None:
        raise AssertionError("a GGUF reference is not converted")

    job = dataclasses.replace(_job(quant, reference="upstream"), source_format="gguf", f16_file=str(reference))
    corpus = tmp_path / "ppl.txt"
    corpus.write_text("public\n", encoding="utf-8")
    report = run_kl(job, tmp_path / "work", corpus=corpus, runner=Logs(9.0, 9.0, 9.5), fetch=fetch, convert=convert)
    assert report["quants"][0]["reference_name_match"] is True  # type: ignore[index]
    assert reference.is_file()
    missing = dataclasses.replace(job, f16_file=str(tmp_path / "gone.gguf"))
    with pytest.raises(Exception, match="missing"):
        run_kl(missing, tmp_path / "work2", corpus=corpus, runner=Logs(9.0, 9.0, 9.5), fetch=fetch, convert=convert)


def _local_build(tmp_path: Path) -> tuple[Path, Path, list[Path], Path]:
    out = tmp_path / "out"
    out.mkdir()
    f16 = out / "gemma4-31b-requant-F16.gguf"
    write_tiny(f16, name="Gemma 4 31B It")
    tiers = []
    for tier in ("IQ3_XXS", "Q4_K_M"):
        path = out / f"{tier}.gguf"
        write_tiny(path, name="Gemma 4 31B It")
        with path.open("ab") as handle:
            handle.write(tier.encode())  # distinct content, distinct revision
        tiers.append(path)
    sums = tmp_path / "new-files.sha256"
    sums.write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in [f16, *tiers]), encoding="utf-8")
    return out, f16, tiers, sums


def _plan_args(f16: Path, tiers: list[Path], sums: Path, plan: Path) -> list[str]:
    args = ["kl-plan", "--job", "G4R", "--author-repo", "a/b", "--author-rev", "2491ee58", "--f16", str(f16)]
    for path in tiers:
        args += ["--tier", f"{path.stem}={path}"]
    return [*args, "--sha256-file", str(sums), "--note", "local re-quant", "--plan-out", str(plan)]


def test_local_quant_kl_plan_has_content_revisions_and_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _bins(tmp_path / "bin", monkeypatch)
    _out, f16, tiers, sums = _local_build(tmp_path)
    plan_path = tmp_path / "kl-g4r.toml"
    assert rc.main(_plan_args(f16, tiers, sums, plan_path)) == 0
    text = plan_path.read_text(encoding="utf-8")
    assert 'revision = ""' not in text and 'repo = "local"' not in text
    job = load_kl_plan(plan_path).job("G4R")
    assert job.f16_file == str(f16) and job.sha256 == sha256_file(f16)
    for quant, path in zip(job.quants, tiers, strict=True):
        assert quant.source == "local" and quant.repo == ""
        assert quant.revision == "local-sha256:" + sha256_file(path)[:12]
        assert quant.pattern == str(path)
    assert job.quants[0].revision != job.quants[1].revision

    def fetch(repo: str, revision: str, dest: Path, patterns: list[str] | None) -> None:
        raise AssertionError("local quants are never downloaded")

    corpus = tmp_path / "ppl.txt"
    corpus.write_text("public\n", encoding="utf-8")
    report = run_kl(job, tmp_path / "work", corpus=corpus, runner=Logs(9.0, 9.0, 9.5), fetch=fetch)
    rows = report["quants"]
    assert [(r["source"], r["revision"]) for r in rows] == [(q.source, q.revision) for q in job.quants]  # type: ignore[index, union-attr]
    assert f16.is_file() and all(path.is_file() for path in tiers)

    with tiers[1].open("ab") as handle:
        handle.write(b"changed after the plan")
    with pytest.raises(StageError, match="sha256 mismatch"):
        run_kl(job, tmp_path / "work2", corpus=corpus, runner=Logs(9.0, 9.0, 9.5), fetch=fetch)


def test_kl_plan_fails_when_a_tier_has_no_digest(tmp_path: Path) -> None:
    _out, f16, tiers, sums = _local_build(tmp_path)
    sums.write_text(sums.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    plan_path = tmp_path / "kl.toml"
    assert rc.main(_plan_args(f16, tiers, sums, plan_path)) == 1
    assert not plan_path.exists()


DIGEST = "ab" * 32


@pytest.mark.parametrize(
    ("quant", "message"),
    [
        ('source = "local"\nrevision = ""\npattern = "/x/Q4.gguf"\n', "missing revision"),
        ('source = "local"\nrevision = "local-sha256:xyz"\npattern = "/x/Q4.gguf"\n', "local-sha256:<12 hex>"),
        ('source = "local"\nrevision = "local-sha256:abababababab"\npattern = "Q4.gguf"\n', "absolute .gguf"),
        ('source = "local"\nrepo = "a/q"\nrevision = "local-sha256:abababababab"\npattern = "/x/Q4.gguf"\n', "no repo"),
        (
            f'source = "local"\nrevision = "local-sha256:000000000000"\npattern = "/x/Q4.gguf"\nsha256 = "{DIGEST}"\n',
            "does not match its sha256",
        ),
        ('repo = "local"\nrevision = "x"\npattern = "/x/Q4.gguf"\n', 'set source = "local"'),
        ('repo = "a/q"\nrevision = ""\npattern = "*.gguf"\n', "missing revision"),
        ('source = "disk"\nrepo = "a/q"\nrevision = "d"\npattern = "*.gguf"\n', "source must be one of"),
    ],
)
def test_plan_validates_local_and_remote_quants(tmp_path: Path, quant: str, message: str) -> None:
    path = tmp_path / "kl.toml"
    path.write_text(
        '[[jobs]]\nid = "L9"\nbf16_repo = "a/b"\nbf16_revision = "c"\n[[jobs.quants]]\nquant = "Q4_K_M"\n' + quant,
        encoding="utf-8",
    )
    with pytest.raises(PlanError, match=message):
        load_kl_plan(path)


def test_cli_local_quant_gets_a_content_revision(tmp_path: Path) -> None:
    path = tmp_path / "Q4_K_M.gguf"
    write_tiny(path)
    ref = _quant_spec(str(path))
    assert (ref.source, ref.repo) == ("local", "")
    assert ref.revision == local_revision(sha256_file(path))


def test_probe_counts_empty_replies_and_refuses_an_empty_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = tmp_path / "p.txt"
    prompts.write_text("one\ntwo\nthree\n", encoding="utf-8")
    replies = iter(["I cannot help.", "", "Sure."])
    monkeypatch.setattr(rc, "_chat", lambda server, text, *, max_tokens: next(replies))
    out = tmp_path / "probe.json"
    base = ["probe", "--server", "http://127.0.0.1:1", "--label", "s", "--out", str(out)]
    assert rc.main([*base, "--prompts", str(prompts)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert (data["n"], data["answered"], data["empty"], data["refusal_rate"]) == (3, 2, 1, 0.5)
    empty = tmp_path / "none.txt"
    empty.write_text("# only a comment\n", encoding="utf-8")
    out.unlink()
    assert rc.main([*base, "--prompts", str(empty)]) == 1
    assert not out.exists()


def test_verdict_fails_on_empty_replies_and_bad_pairs(tmp_path: Path) -> None:
    (tmp_path / "tensor-diff.json").write_text(json.dumps({"pass": True}), encoding="utf-8")
    for label, rate, empty in (("stock-Q4_K_M", 0.5, 0), ("new-Q4_K_M", 0.0, 3)):
        probe = {"model": label, "n": 4, "empty": empty, "refusal_rate": rate}
        (tmp_path / f"probe-{label}.json").write_text(json.dumps(probe), encoding="utf-8")
    summary = tmp_path / "summary.json"
    rows = [{"quant": "Q4_K_M", "kl_mean": 0.9, "top1_agreement_pct": 66.0}, {"quant": "IQ3_XXS", "kl_mean": None}]
    summary.write_text(json.dumps({"quants": rows}), encoding="utf-8")
    l2 = json.dumps({"Q4_K_M": {"kl": 0.83, "top1": 68.4}})
    args = ["verdict", "--results", str(tmp_path), "--kl-summary", str(summary), "--l2-kl", l2]
    assert rc.main([*args, "--pair", "Q4_K_M:stock-Q4_K_M:new-Q4_K_M"]) == 1
    payload = rc._verdict(tmp_path, summary, json.loads(l2), [("Q4_K_M", "stock-Q4_K_M", "new-Q4_K_M")])
    assert payload["probe"]["Q4_K_M"]["problems"] == ["new-Q4_K_M: 3 empty replies"]  # type: ignore[index]
    assert rc.main([*args, "--pair", "Q4_K_M::new-Q4_K_M"]) == 2
