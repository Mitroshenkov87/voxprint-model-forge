"""G4R rerun on any host (Windows 4090 or the Linux VM): KL against a reference, fixed refusal probe, verdict.

``python -m voxprint_forge.g4r_rerun --llama-bin DIR --work DIR --tiers DIR --prompts FILE [--dry-run]``

* No re-quant and no upload. The new tiers (``Q4_K_M.gguf`` required, ``IQ3_XXS.gguf`` if present) are read from
  ``--tiers`` and must match the sha256 the synth4 quant stage wrote.
* Reference: ``--reference q8_0-upstream`` (default) rebuilds a Q8_0 GGUF from the author's safetensors at the pinned
  revision with the pinned ``convert_hf_to_gguf.py`` (``--llama-src``), checks every author file's sha256 first, and
  deletes the safetensors after. ``--reference f16-local --f16 FILE`` uses the synth4 F16 (sha256 pinned) instead.
* KL: the reference and every tier run through the same ``llama-perplexity`` binary, corpus, chunks and ctx. The full
  stdout of each run is kept (``ppl-<name>.log``, argv and sha256s in its header). The reference check fails the
  stage when the tiers did not read the same reference logits (different ``PPL(base)`` read back) or when the
  reference file is not the pinned one.
* Probe: ``llama-server --jinja --reasoning off`` and ``enable_thinking: false`` in every request; the stock arms run
  first and a stock arm with more than 2 empty replies stops the run with exit 3 (probe harness invalid). Raw
  requests and responses go to ``--private`` (owner-only), never to the results directory.

Exit codes: 0 PASS, 1 FAIL, 2 usage, 3 probe harness invalid, 4 preflight, 5 a pinned file did not match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from voxprint_forge import requant_check as rc

AUTHOR_REPO = "paperscarecrow/Gemma-4-31B-it-abliterated"
AUTHOR_REV = "2491ee58472b82a2372d288f982943b9d5e6a82c"
AUTHOR_DIR = "gemma-4-31b-abliterated"
AUTHOR_MANIFEST = {
    "model-00001-of-00002.safetensors": "d1ac1e9509e5492b9924c8f1f6d28d4a4d9c10c0f7f410890dd852647eae8b75",
    "model-00002-of-00002.safetensors": "cbd223bb460ee1c92fd286f51880acd12955f987ba2cfa83a563c375010f864e",
    "tokenizer.json": "a2619fe11b50dbed06ac443c51d757b354d0b62d64baa514404d4e84e6713519",
    "chat_template.jinja": "2dfbfc7d538912f4ea11d29d85b4e25d7bc26386e53f57529f1d707c28b5828c",
    "config.json": "15e57c2d2945536d2c2fef5d33090b73de7291654a6072e69c90f25980e6e5b0",
    "gemma4_31b_abliterator.py": "afb77136430424e6c4bfdf6d2006c706dc7a4affb45d7edc3472729baf114b7b",
    "generation_config.json": "5439541d3bf0ba9ade7c1122dc14703f85f32a520150351b344b14f48c574cc2",
    "model.safetensors.index.json": "390b7f9a0a502e4526cf20862d22954c911cc318d8aba2cdb745c6fbd98264f6",
    "processor_config.json": "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
    "tokenizer_config.json": "b9686f02a0bb2d5de25e5bed7949d5858109e62cbb1111ce4cea731885e779e9",
}
AUTHOR_BYTES = 62_580_000_000
REFERENCE_BYTES = 34_000_000_000
LLAMA_BUILD = "build 11476, commit 9881906"
CONVERT_SHA256 = "e9a1da876330bbce9687541ab31736542a01b4ac43c6686126514a50f122fb7f"  # convert_hf_to_gguf.py @ 988190680
F16_SHA256 = "a44474f8b23cee19c2b5e4236f7cfeb591c090c53d28135633abffac87e55e49"
REFERENCE_NAME = "gemma4-31b-author-Q8_0.gguf"
NEW_TIERS = {
    "Q4_K_M": "7da4874785f66be9689edf70d8adf339bb7b0b18905c1789f6e8f11b85f1f805",
    "IQ3_XXS": "60a470bba516deb046ed1ed6580ce9c559a45fb19cdc58a65e78048516f653a9",
}
REQUIRED_TIERS = ("Q4_K_M",)


@dataclass(frozen=True)
class Stock:
    """A shipped tier used as the stock arm of the probe, pinned by revision and sha256."""

    repo: str
    revision: str
    filename: str
    sha256: str
    size: int


STOCK = {
    "Q4_K_M": Stock(
        AUTHOR_REPO,
        AUTHOR_REV,
        "gemma-4-31b-abliterated-Q4_K_M.gguf",
        "380f2f50f441225e669ab1541220985387bf3933573cd2f4f7e667f3790e73a3",
        18_700_000_000,
    ),
    "IQ3_XXS": Stock(
        "Mitroshenkov87/voxprint-gemma4-31b",
        "0d84d5f0d1ea740c85c16de0a7c7a844a8bee587",
        "IQ3_XXS.gguf",
        "c7bb8f9f4bdb396fb40a5beb617de27132cda6727b885e82f635340f353c5340",
        12_100_000_000,
    ),
}
# L2 on synth4 (2026-10-17): shipped-lineage quants against the stock-identical F16, same corpus, chunks and ctx.
L2_KL = {"IQ3_XXS": {"kl": 2.293518, "top1": 44.314}, "Q4_K_M": {"kl": 0.827463, "top1": 68.366}}
PROMPT_SHA12 = "ed863e1e3462"
CHUNKS, CTX = 100, 512
SERVER_CTX = 8192
# Two runs that read the same base-logits file print the same "Mean PPL(base)"; anything else is a reference mismatch.
READBACK_TOLERANCE = 1e-6
EXIT_FAIL, EXIT_USAGE, EXIT_PREFLIGHT, EXIT_PIN = 1, 2, 4, 5


class RerunError(RuntimeError):
    """A stage failed; ``code`` is the process exit code."""

    def __init__(self, message: str, code: int = EXIT_FAIL) -> None:
        super().__init__(message)
        self.code = code


def exe(bin_dir: Path, name: str) -> Path:
    """``name`` or ``name.exe`` in the llama.cpp binary directory."""
    for candidate in (bin_dir / name, bin_dir / f"{name}.exe"):
        if candidate.is_file():
            return candidate
    return bin_dir / (f"{name}.exe" if os.name == "nt" else name)


def sha256_path(path: Path) -> str:
    """sha256 of a file, streamed."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Log:
    """Timestamped lines to stdout and ``run.log``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, text: str) -> None:
        line = f"[g4r-rerun {datetime.now(UTC).strftime('%H:%M:%SZ')}] {text}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def private_dir(path: Path) -> Path:
    """Create the raw-probe directory readable by the owner only (mode 700; on Windows an owner-only ACL)."""
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        user = os.environ.get("USERNAME", "")
        if user:
            subprocess.run(
                ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:(OI)(CI)F"],
                check=False,
                capture_output=True,
            )
    else:
        path.chmod(0o700)
    return path


def perplexity_argv(binary: Path, model: Path, corpus: Path, logits: Path, *, compare: bool, fit: bool) -> list[str]:
    """One llama-perplexity run. ``fit`` lets llama.cpp choose the GPU layers (a Q8_0 31B does not fit 24 GB)."""
    argv = [str(binary), "-m", str(model), "-f", str(corpus), "--chunks", str(CHUNKS), "-c", str(CTX), "-fa", "on"]
    argv += ["--fit", "on"] if fit else ["-ngl", "99", "--fit", "off"]
    argv += ["--kl-divergence-base", str(logits)]
    if compare:
        argv.append("--kl-divergence")
    return argv


def server_argv(binary: Path, model: Path, port: int, *, fit: bool) -> list[str]:
    """llama-server for the probe: chat template on, thinking off."""
    argv = [str(binary), "-m", str(model), "-c", str(SERVER_CTX), "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"]
    argv += ["--fit", "on"] if fit else ["-ngl", "99"]
    return argv + ["--jinja", "--reasoning", "off", "--host", "127.0.0.1", "--port", str(port)]


def reference_check(rows: list[dict[str, object]], reference_sha: str, expected_sha: str) -> list[str]:
    """Problems that make the KL figures meaningless: wrong reference file, or tiers that read different logits."""
    problems: list[str] = []
    if reference_sha != expected_sha:
        problems.append(f"reference sha256 {reference_sha[:12]} is not the pinned {expected_sha[:12]}")
    readbacks = [float(v) for row in rows if isinstance(v := row.get("ppl_base"), (int, float))]
    if len(readbacks) != len(rows):
        problems.append("a tier log has no PPL(base) read back")
    elif readbacks and max(readbacks) - min(readbacks) > READBACK_TOLERANCE * max(readbacks):
        problems.append(f"tiers read different reference logits (PPL(base) {min(readbacks)} vs {max(readbacks)})")
    return problems


class Rerun:
    """One rerun; see the module docstring."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.bin = Path(args.llama_bin)
        self.work = Path(args.work)
        self.out = Path(args.results or self.work / f"rerun-{datetime.now(UTC).strftime('%Y%m%dT%H%MZ')}")
        self.private = Path(args.private or self.work / "private")
        self.log = Log(self.out / "run.log")
        candidates = {quant: Path(args.tiers) / f"{quant}.gguf" for quant in NEW_TIERS}
        self.tiers = {quant: path for quant, path in candidates.items() if path.is_file()}
        self.stock_dir = Path(args.stock_dir) if args.stock_dir else self.work / "stock"
        self.reference = Path(args.f16) if args.reference == "f16-local" else self.work / "ref" / REFERENCE_NAME

    # -- preflight ---------------------------------------------------------------------------------------------
    def preflight(self) -> None:
        """Binaries, flags, tiers, prompt set, disk, hub revisions, GPU. No hashing of GGUFs, no download."""
        log = self.log
        bad = self._binaries()
        for quant in REQUIRED_TIERS:
            if quant not in self.tiers:
                bad.append(f"tier missing: {quant}.gguf in {self.args.tiers}")
        for quant in NEW_TIERS:
            if quant in self.tiers:
                log(f"tier {quant}: {self.tiers[quant].name}, {self.tiers[quant].stat().st_size} bytes")
            else:
                log(f"tier {quant}: not present, its KL row and probe pair are skipped")
        bad += self._prompts()
        need = 15e9
        if self.args.reference == "q8_0-upstream":
            if self.reference.is_file():
                log(f"reference: {self.reference.name} present, reused after its sha256 check")
            else:
                need += AUTHOR_BYTES + REFERENCE_BYTES
                bad += self._converter()
        elif not self.reference.is_file():
            bad.append(f"reference F16 missing: {self.reference}")
        for quant in self.tiers:
            if not (self.stock_dir / STOCK[quant].filename).is_file():
                need += STOCK[quant].size
        self.work.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.work).free
        log(f"disk: {free / 1e9:.0f} GB free under work, need {need / 1e9:.0f}")
        if free < need:
            bad.append("not enough disk")
        bad += self._hub_check()
        bad += self._gpu()
        offload = "--fit on" if self.args.fit else "-ngl 99"
        log(
            f"plan: reference {self.args.reference}; KL {', '.join(self.tiers)} (chunks {CHUNKS}, ctx {CTX}, "
            f"{offload}); probe stock arms first (gate <= {rc.STOCK_MAX_EMPTY} empty), then new arms; "
            "raw replies in the private dir"
        )
        for line in bad:
            log(f"preflight: {line}")
        if bad:
            raise RerunError("preflight", EXIT_PREFLIGHT)
        log("preflight OK")

    def _binaries(self) -> list[str]:
        bad: list[str] = []
        for name in ("llama-perplexity", "llama-server"):
            path = exe(self.bin, name)
            if not path.is_file():
                bad.append(f"binary missing: {path.name}")
                continue
            done = subprocess.run([str(path), "--version"], capture_output=True, text=True, check=False)
            if LLAMA_BUILD not in done.stdout + done.stderr:
                bad.append(f"{path.name} is not {LLAMA_BUILD}")
        server = exe(self.bin, "llama-server")
        if server.is_file():
            helptext = subprocess.run([str(server), "--help"], capture_output=True, text=True, check=False)
            if "--reasoning [on|off|auto]" in helptext.stdout + helptext.stderr:
                self.log("llama-server: b11476, --reasoning off supported")
            else:
                bad.append("llama-server has no --reasoning flag")
        return bad

    def _prompts(self) -> list[str]:
        prompts = Path(self.args.prompts)
        if not prompts.is_file():
            return ["prompt set missing"]
        bad: list[str] = []
        if os.name != "nt" and prompts.stat().st_mode & 0o077:
            bad.append("prompt set is readable by others; chmod 600")
        sha12 = sha256_path(prompts)[:12]
        if sha12 != self.args.prompt_sha12:
            bad.append(f"prompt set sha256[:12] {sha12}, expected {self.args.prompt_sha12}")
        else:
            self.log(f"prompt set: {len(rc.read_prompts(prompts))} prompts, sha256[:12] {sha12}")
        return bad

    def _converter(self) -> list[str]:
        convert = Path(self.args.llama_src or "") / "convert_hf_to_gguf.py"
        if not convert.is_file():
            return ["convert_hf_to_gguf.py missing (--llama-src)"]
        if sha256_path(convert) != CONVERT_SHA256:
            return ["convert_hf_to_gguf.py is not the pinned 988190680 copy"]
        self.log("convert_hf_to_gguf.py: pinned 988190680 copy")
        missing = []
        for module in ("torch", "safetensors", "transformers", "sentencepiece", "numpy", "gguf"):
            try:
                __import__(module)
            except ImportError:
                missing.append(f"python module missing for the conversion: {module}")
        return missing

    def _hub_check(self) -> list[str]:
        """Pinned revisions resolve (metadata only, nothing downloaded)."""
        if self.args.offline:
            return []
        from huggingface_hub import HfApi

        api = HfApi()
        bad: list[str] = []
        missing = [STOCK[q] for q in self.tiers if not (self.stock_dir / STOCK[q].filename).is_file()]
        repos = {(stock.repo, stock.revision) for stock in missing}
        if self.args.reference == "q8_0-upstream" and not self.reference.is_file():
            repos.add((AUTHOR_REPO, AUTHOR_REV))
        for repo, revision in sorted(repos):
            try:
                info = api.model_info(repo, revision=revision)
            except Exception as exc:  # noqa: BLE001 - any hub error fails the preflight with its type
                bad.append(f"hub: {repo} @ {revision[:8]}: {type(exc).__name__}")
                continue
            self.log(f"hub: {repo} @ {info.sha}")
            if info.sha != revision:
                bad.append(f"hub: {repo} resolved to {info.sha}")
        return bad

    def _gpu(self) -> list[str]:
        smi = shutil.which("nvidia-smi")
        if smi is None:
            return [] if self.args.dry_run else ["nvidia-smi missing"]
        query = subprocess.run(
            [smi, "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = query.stdout.strip().splitlines()
        self.log(f"gpu: {lines[0] if lines else 'none'}")
        apps = subprocess.run(
            [smi, "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True, check=False
        )
        if apps.stdout.strip():
            if self.args.dry_run:
                self.log("gpu busy now (fine for a dry run; the real run needs it free)")
                return []
            return ["gpu busy"]
        return []

    # -- stages ------------------------------------------------------------------------------------------------
    def verify_tiers(self) -> None:
        """The new tiers are byte-identical to the synth4 quant stage output."""
        for quant, path in self.tiers.items():
            if sha256_path(path) != NEW_TIERS[quant]:
                raise RerunError(f"{quant}.gguf sha256 is not the synth4 tier", EXIT_PIN)
            self.log(f"tier {quant}: sha256 {NEW_TIERS[quant][:12]} ok")

    def fetch_stock(self) -> dict[str, Path]:
        """Shipped stock tiers at pinned revisions; fetched only when missing, always sha256-checked."""
        paths: dict[str, Path] = {}
        for quant in self.tiers:
            stock = STOCK[quant]
            path = self.stock_dir / stock.filename
            if not path.is_file():
                from huggingface_hub import hf_hub_download

                self.log(f"fetch stock {quant}: {stock.repo} @ {stock.revision[:8]} {stock.filename}")
                hf_hub_download(stock.repo, stock.filename, revision=stock.revision, local_dir=str(self.stock_dir))
            if sha256_path(path) != stock.sha256:
                raise RerunError(f"stock {quant} sha256 mismatch", EXIT_PIN)
            self.log(f"stock {quant}: sha256 {stock.sha256[:12]} ok")
            paths[quant] = path
        return paths

    def build_reference(self) -> str:
        """sha256 of the reference GGUF, building the Q8_0 from the author's pinned safetensors when needed."""
        if self.args.reference == "f16-local":
            digest = sha256_path(self.reference)
            if digest != F16_SHA256:
                raise RerunError("F16 reference sha256 is not the synth4 F16", EXIT_PIN)
            return digest
        stamp = self.reference.with_suffix(".sha256")
        if self.reference.is_file() and stamp.is_file():
            digest = sha256_path(self.reference)
            if digest == stamp.read_text(encoding="utf-8").split()[0]:
                self.log(f"reference {self.reference.name} reused, sha256 {digest[:12]}")
                return digest
        from huggingface_hub import snapshot_download

        source = self.work / "author"
        self.log(f"download {AUTHOR_REPO} @ {AUTHOR_REV[:8]} ({AUTHOR_DIR}/, about 62.6 GB)")
        snapshot_download(AUTHOR_REPO, revision=AUTHOR_REV, local_dir=str(source), allow_patterns=[f"{AUTHOR_DIR}/*"])
        for name, expected in AUTHOR_MANIFEST.items():
            if sha256_path(source / AUTHOR_DIR / name) != expected:
                raise RerunError(f"author file {name} sha256 mismatch", EXIT_PIN)
        self.log("author files: 10/10 sha256 ok")
        self.reference.parent.mkdir(parents=True, exist_ok=True)
        convert = Path(self.args.llama_src) / "convert_hf_to_gguf.py"
        argv = [sys.executable, str(convert), str(source / AUTHOR_DIR), "--outtype", "q8_0"]
        argv += ["--outfile", str(self.reference)]
        self.log("convert to Q8_0 (llama.cpp 988190680)")
        with (self.out / "convert.log").open("w", encoding="utf-8", newline="\n") as handle:
            done = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if done.returncode != 0 or not self.reference.is_file():
            raise RerunError("conversion failed (see convert.log)")
        shutil.rmtree(source)
        digest = sha256_path(self.reference)
        stamp.write_text(f"{digest}  {self.reference.name}\n", encoding="utf-8", newline="\n")
        self.log(f"reference {self.reference.name}: sha256 {digest[:12]}, author safetensors deleted")
        return digest

    def kl(self, reference_sha: str) -> Path:
        """Reference logits, then every tier against them; full stdout kept; reference check; kl-summary.json."""
        from voxprint_forge.evaluate import data_dir, parse_ppl
        from voxprint_forge.kl.parse import parse_kl

        corpus = data_dir() / "ppl.txt"
        perplexity = exe(self.bin, "llama-perplexity")
        logits = self.work / "base-logits.bin"
        header = {
            "binary_version": LLAMA_BUILD,
            "corpus_sha256": sha256_path(corpus),
            "chunks": CHUNKS,
            "ctx": CTX,
            "reference": self.reference.name,
            "reference_sha256": reference_sha,
        }

        def run(name: str, model: Path, digest: str, *, compare: bool) -> str:
            argv = perplexity_argv(perplexity, model, corpus, logits, compare=compare, fit=self.args.fit)
            path = self.out / f"ppl-{name}.log"
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(f"# argv: {' '.join(argv)}\n# model sha256: {digest}\n# {json.dumps(header)}\n")
                handle.flush()
                done = subprocess.run(argv, stdout=handle, stderr=subprocess.STDOUT, check=False)
            if done.returncode != 0:
                raise RerunError(f"llama-perplexity {name} exited {done.returncode} (see {path.name})")
            return path.read_text(encoding="utf-8", errors="replace")

        self.log(f"KL reference: {self.reference.name} base logits")
        base_ppl = parse_ppl(run("base", self.reference, reference_sha, compare=False))
        rows: list[dict[str, object]] = []
        for quant, path in self.tiers.items():
            self.log(f"KL {quant}")
            stats = parse_kl(run(quant, path, NEW_TIERS[quant], compare=True))
            rows.append(
                {
                    "quant": quant,
                    "kl_mean": stats.kl_mean,
                    "kl_p99": stats.kl_p99,
                    "top1_agreement_pct": stats.top1_agreement_pct,
                    "delta_p_rms_pct": stats.delta_p_rms_pct,
                    "ppl": stats.ppl_quant,
                    "ppl_base": stats.ppl_base,
                }
            )
            self.log(f"KL {quant}: mean {stats.kl_mean}, p99 {stats.kl_p99}, top-1 {stats.top1_agreement_pct}%")
        logits.unlink(missing_ok=True)
        expected = F16_SHA256 if self.args.reference == "f16-local" else reference_sha
        problems = reference_check(rows, reference_sha, expected)
        summary = {"job": "G4R-rerun", **header, "ppl_base": base_ppl, "quants": rows, "reference_problems": problems}
        path = self.out / "kl-summary.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        if problems:
            raise RerunError("KL reference check: " + "; ".join(problems))
        self.log(f"KL reference check OK (base PPL {base_ppl}; every tier read the same logits)")
        return path

    def probe_arm(self, label: str, model: Path) -> dict[str, object]:
        """Serve one model, probe it, stop the server. Summary to results, raw text to the private dir."""
        port = self.args.port
        log_path = self.out / f"server-{label}.log"
        with log_path.open("w", encoding="utf-8", newline="\n") as handle:
            server = subprocess.Popen(
                server_argv(exe(self.bin, "llama-server"), model, port, fit=self.args.fit),
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            try:
                self._wait_healthy(server, port, label)
                payload = rc.probe(
                    f"http://127.0.0.1:{port}",
                    rc.read_prompts(Path(self.args.prompts)),
                    label,
                    raw_log=self.private / f"probe-raw-{label}.jsonl",
                )
            finally:
                server.terminate()
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    server.kill()
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        (self.out / f"probe-{label}.json").write_text(text, encoding="utf-8", newline="\n")
        self.log(
            f"probe {label}: refusal rate {payload['refusal_rate']}, {payload['answered']}/{payload['n']} answered, "
            f"{payload['empty']} empty ({payload['thinking_only']} thinking-only), finish {payload['finish_reasons']}"
        )
        return payload

    def _wait_healthy(self, server: subprocess.Popen[bytes], port: int, label: str) -> None:
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RerunError(f"llama-server for {label} exited {server.returncode} (see server-{label}.log)")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                    if response.status == 200:
                        return
            except OSError:
                pass
            time.sleep(2)
        raise RerunError(f"llama-server for {label} not ready in 600 s")

    def run(self) -> int:
        """Preflight, then (unless dry) tiers, stock, reference, KL, stock-gated probe, verdict."""
        self.preflight()
        if self.args.dry_run:
            return 0
        private_dir(self.private)
        self.verify_tiers()
        stock = self.fetch_stock()
        summary = self.kl(self.build_reference())
        for quant in self.tiers:
            reason = rc.stock_gate(self.probe_arm(f"stock-{quant}", stock[quant]))
            if reason:
                self.log(reason + "; the quant arms are not run")
                return rc.HARNESS_INVALID_EXIT
        for quant, path in self.tiers.items():
            self.probe_arm(f"new-{quant}", path)
        diff = Path(self.args.tensor_diff)
        (self.out / "tensor-diff.json").write_text(diff.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        pairs = [(quant, f"stock-{quant}", f"new-{quant}") for quant in self.tiers]
        verdict = rc._verdict(self.out, summary, {quant: L2_KL[quant] for quant in self.tiers}, pairs)
        (self.out / "verdict.json").write_text(
            json.dumps(verdict, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        self.log(f"verdict: {verdict['verdict']}")
        if verdict["verdict"] == "INVALID":
            return rc.HARNESS_INVALID_EXIT
        return 0 if verdict["verdict"] == "PASS" else EXIT_FAIL


def parser() -> argparse.ArgumentParser:
    """Command-line options; see the module docstring."""
    p = argparse.ArgumentParser(prog="g4r_rerun")
    p.add_argument("--llama-bin", required=True, help="llama.cpp b11476 binaries (llama-perplexity, llama-server)")
    p.add_argument("--work", required=True, help="work directory: stock tiers, reference, private dir")
    p.add_argument("--tiers", required=True, help="directory with the synth4 Q4_K_M.gguf (and IQ3_XXS.gguf)")
    p.add_argument("--prompts", required=True, help="private refusal-probe set")
    p.add_argument("--prompt-sha12", default=PROMPT_SHA12)
    p.add_argument("--tensor-diff", help="tensor-diff.json from synth4 (needed for the verdict)")
    p.add_argument("--stock-dir", help="directory with the shipped stock tiers (default: work/stock, fetched pinned)")
    p.add_argument("--reference", choices=("q8_0-upstream", "f16-local"), default="q8_0-upstream")
    p.add_argument("--f16", help="synth4 F16 for --reference f16-local")
    p.add_argument("--llama-src", help="llama.cpp source at 988190680 (convert_hf_to_gguf.py)")
    p.add_argument("--results", help="results directory (default: work/rerun-<UTC stamp>)")
    p.add_argument("--private", help="owner-only directory for raw probe requests and replies")
    p.add_argument("--port", type=int, default=8097)
    p.add_argument("--fit", action="store_true", help="let llama.cpp place layers (needed for a Q8_0 31B on 24 GB)")
    p.add_argument("--offline", action="store_true", help="skip the hub revision check in the preflight")
    p.add_argument("--dry-run", action="store_true", help="preflight only: no hashing of GGUFs, no download, no GPU")
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point."""
    args = parser().parse_args(argv)
    if args.reference == "f16-local" and not args.f16:
        print("--reference f16-local needs --f16", file=sys.stderr)
        return EXIT_USAGE
    if not args.dry_run and not args.tensor_diff:
        print("a real run needs --tensor-diff", file=sys.stderr)
        return EXIT_USAGE
    rerun = Rerun(args)
    try:
        return rerun.run()
    except RerunError as exc:
        rerun.log(f"FAIL: {exc}")
        return exc.code


if __name__ == "__main__":
    sys.exit(main())
