"""G4R rerun orchestrator: argv, reference check, preflight and KL stage with fake llama.cpp tools, Windows pins."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from voxprint_forge import g4r_rerun as g
from voxprint_forge import requant_check as rc

ROOT = Path(__file__).resolve().parents[1]
PS1 = ROOT / "scripts" / "win_g4r_rerun.ps1"
posix_only = pytest.mark.skipif(os.name == "nt", reason="fake llama.cpp tools are shell scripts")


def test_argv_keeps_reference_and_tiers_on_the_same_settings(tmp_path: Path) -> None:
    base = g.perplexity_argv(Path("pp"), Path("ref.gguf"), Path("c.txt"), Path("l.bin"), compare=False, fit=False)
    tier = g.perplexity_argv(Path("pp"), Path("q.gguf"), Path("c.txt"), Path("l.bin"), compare=True, fit=False)
    assert base[3:] == tier[3:-1] and tier[-1] == "--kl-divergence"
    assert ["--chunks", "100", "-c", "512"] == base[5:9]
    fit = g.perplexity_argv(Path("pp"), Path("r"), Path("c"), Path("l"), compare=False, fit=True)
    assert "-ngl" not in fit and fit[fit.index("--fit") + 1] == "on"
    server = g.server_argv(Path("srv"), Path("m.gguf"), 8097, fit=False)
    assert server[server.index("--reasoning") + 1] == "off" and "--jinja" in server


def test_reference_check() -> None:
    rows: list[dict[str, object]] = [
        {"quant": "Q4_K_M", "ppl_base": 19581.7},
        {"quant": "IQ3_XXS", "ppl_base": 19581.7},
    ]
    assert g.reference_check(rows, "a" * 64, "a" * 64) == []
    assert "not the pinned" in g.reference_check(rows, "b" * 64, "a" * 64)[0]
    rows[1]["ppl_base"] = 20000.0
    assert "different reference logits" in g.reference_check(rows, "a" * 64, "a" * 64)[0]
    assert "no PPL(base)" in g.reference_check([{"quant": "Q4_K_M"}], "a" * 64, "a" * 64)[0]


def test_windows_pins_are_b11476_and_sha256() -> None:
    text = PS1.read_text(encoding="utf-8")
    block = text[text.index("# PINS begin") : text.index("# PINS end")]
    shas = re.findall(r'Sha = "([0-9a-f]+)"', block)
    urls = re.findall(r'Url = "([^"]+)"', block)
    assert len(shas) == len(urls) == 3 and all(len(sha) == 64 for sha in shas)
    assert all("b11476" in url or "988190680d5a89fce97de3c20df2c2813731fd61" in url for url in urls)
    assert "cuda-12.4-x64" in urls[0] and "cudart" in urls[1]
    assert "voxprint_forge.g4r_rerun" in text and "q8_0-upstream" in text and "--fit" in text


@pytest.mark.skipif(os.name != "nt", reason="PowerShell parser is on Windows")
def test_windows_entry_point_parses() -> None:
    command = (
        "$e=$null; [System.Management.Automation.Language.Parser]::ParseFile("
        f"'{PS1}', [ref]$null, [ref]$e) | Out-Null; if ($e.Count) {{ $e | ForEach-Object {{ $_.Message }}; exit 1 }}"
    )
    done = subprocess.run(["powershell", "-NoProfile", "-Command", command], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr


def _script(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


KL_BLOCK = """====== Perplexity statistics ======
Mean PPL(Q)                   : 30.000000 ±   1.0
Mean PPL(base)                : 19581.738233 ±   1.0
====== KL divergence statistics ======
Mean    KLD:   0.020000 ±   0.001
99.0%   KLD:   0.300000
====== Token probability statistics ======
RMS Δp    : 2.500 ± 0.1 %
Same top p: 95.000 ± 0.5 %
"""


def _fixture(tmp_path: Path) -> tuple[Path, list[str], dict[str, str]]:
    if shutil.which("sh") is None:
        pytest.skip("no sh")
    bins = tmp_path / "bin"
    bins.mkdir()
    version = 'case "$1" in --version) echo "version: 0.6.0-dev (build 11476, commit 988190680)"; exit 0;; esac\n'
    _script(bins / "llama-server", version + 'echo "-rea,  --reasoning [on|off|auto]  thinking"\n')
    kl = tmp_path / "kl.txt"
    kl.write_text(KL_BLOCK, encoding="utf-8")
    _script(
        bins / "llama-perplexity",
        version + f'for a in "$@"; do [ "$a" = --kl-divergence ] && {{ cat "{kl}"; exit 0; }}; done\n'
        'echo "Final estimate: PPL = 50898.1926 +/- 1.0"\n',
    )
    tiers = tmp_path / "out"
    tiers.mkdir()
    pins: dict[str, str] = {}
    for name in ("Q4_K_M", "IQ3_XXS", "F16"):
        (tiers / f"{name}.gguf").write_bytes(name.encode())
        pins[name] = hashlib.sha256(name.encode()).hexdigest()
    stock = tmp_path / "shipped"
    stock.mkdir()
    for item in g.STOCK.values():
        (stock / item.filename).write_bytes(item.filename.encode())
    prompts = tmp_path / "p.txt"
    prompts.write_text("one\ntwo\n", encoding="utf-8")
    prompts.chmod(0o600)
    diff = tmp_path / "tensor-diff.json"
    diff.write_text(json.dumps({"pass": True}), encoding="utf-8")
    argv = [
        "--llama-bin", str(bins), "--work", str(tmp_path / "work"), "--tiers", str(tiers), "--prompts", str(prompts),
        "--prompt-sha12", hashlib.sha256(prompts.read_bytes()).hexdigest()[:12], "--stock-dir", str(stock),
        "--reference", "f16-local", "--f16", str(tiers / "F16.gguf"), "--results", str(tmp_path / "rr"),
        "--private", str(tmp_path / "private"), "--tensor-diff", str(diff), "--offline",
    ]
    return tmp_path, argv, pins


@posix_only
def test_dry_run_touches_no_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, argv, _ = _fixture(tmp_path)
    monkeypatch.setattr(g.shutil, "which", lambda name: None)
    monkeypatch.setattr(g, "sha256_path", _guard_sha(tmp_path))
    assert g.main([*argv, "--dry-run"]) == 0
    log = (tmp_path / "rr" / "run.log").read_text(encoding="utf-8")
    assert "preflight OK" in log and "--reasoning off supported" in log
    assert not (tmp_path / "rr" / "kl-summary.json").exists()


def _guard_sha(tmp_path: Path):  # noqa: ANN202 - test helper
    real = g.sha256_path

    def guarded(path: Path) -> str:
        assert path.suffix != ".gguf", f"dry run hashed {path.name}"
        return real(path)

    return guarded


@posix_only
def test_run_keeps_logs_checks_reference_and_gates_on_stock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, argv, pins = _fixture(tmp_path)
    monkeypatch.setattr(g, "F16_SHA256", pins["F16"])
    monkeypatch.setattr(g, "NEW_TIERS", {"Q4_K_M": pins["Q4_K_M"], "IQ3_XXS": pins["IQ3_XXS"]})
    stock = {
        quant: g.Stock(s.repo, s.revision, s.filename, hashlib.sha256(s.filename.encode()).hexdigest(), 1)
        for quant, s in g.STOCK.items()
    }
    monkeypatch.setattr(g, "STOCK", stock)
    monkeypatch.setattr(g.Rerun, "_gpu", lambda self: [])
    arms: list[str] = []

    def fake_arm(self: g.Rerun, label: str, model: Path) -> dict[str, object]:
        arms.append(label)
        empty = 24 if label == "stock-Q4_K_M" else 0
        payload = {"model": label, "n": 24, "empty": empty, "refusal_rate": 0.5 if label.startswith("stock") else 0.0}
        (self.out / f"probe-{label}.json").write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(g.Rerun, "probe_arm", fake_arm)
    assert g.main(argv) == rc.HARNESS_INVALID_EXIT
    assert arms == ["stock-Q4_K_M"]  # quant arms never ran
    rr = tmp_path / "rr"
    assert "Final estimate" in (rr / "ppl-base.log").read_text(encoding="utf-8")
    header = (rr / "ppl-Q4_K_M.log").read_text(encoding="utf-8").splitlines()[0]
    assert header.startswith("# argv: ") and "--kl-divergence" in header
    summary = json.loads((rr / "kl-summary.json").read_text(encoding="utf-8"))
    assert summary["reference_problems"] == [] and summary["reference_sha256"] == pins["F16"]
    assert [row["quant"] for row in summary["quants"]] == ["Q4_K_M", "IQ3_XXS"]
    assert not (tmp_path / "work" / "base-logits.bin").exists()


@posix_only
def test_wrong_reference_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, argv, _ = _fixture(tmp_path)
    monkeypatch.setattr(g.Rerun, "_gpu", lambda self: [])
    assert g.main(argv) == g.EXIT_PIN  # the fixture tiers are not the synth4 files


def test_cli_needs_tensor_diff_for_a_real_run(tmp_path: Path) -> None:
    argv = ["--llama-bin", "x", "--work", str(tmp_path), "--tiers", "x", "--prompts", "x"]
    assert g.main(argv) == g.EXIT_USAGE
    assert g.main([*argv, "--reference", "f16-local", "--dry-run"]) == g.EXIT_USAGE


def test_module_runs_as_script() -> None:
    done = subprocess.run([sys.executable, "-m", "voxprint_forge.g4r_rerun", "--help"], capture_output=True, text=True)
    assert done.returncode == 0 and "--dry-run" in done.stdout
