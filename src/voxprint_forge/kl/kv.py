"""KV-cache arms, a speculative/prefix bench for llama-server, and their preflight (llama.cpp b11476).

KV arms: each GGUF is scored against itself with an f16 KV cache, using llama-perplexity
``--kl-divergence-base`` and then ``--kl-divergence`` with ``-ctk``/``-ctv`` at 512 and 8k context.
Bench: llama-server with no draft and with ``--spec-type ngram-*``, plus prompt caching with
and without ``--cache-reuse``. It reports tokens/s, the draft accept rate, and prompt_ms.
``python -m voxprint_forge.kl.kv --plan plans/kv-synth4.toml --dry-run`` lists every step.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import sys
import time
import tomllib
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from voxprint_forge.errors import PlanError
from voxprint_forge.llama import KV_CACHE_TYPES, kl_argv

SPEC_TYPES = ("ngram-cache", "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod")
PROTOBUF_PIN = "5.29.6"


@dataclass(frozen=True)
class KvModel:
    """One GGUF under test."""

    id: str
    repo: str
    revision: str
    pattern: str
    dense: bool
    requires: str = ""
    note: str = ""


@dataclass(frozen=True)
class KvPlan:
    """Parsed plans/kv-synth4.toml."""

    corpus: dict[str, str]
    contexts: tuple[tuple[int, int], ...]
    base: tuple[str, str]
    arms: tuple[tuple[str, str], ...]
    kl_max: float
    ppl_rel_max: float
    models: tuple[KvModel, ...]
    bench: dict[str, object] = field(default_factory=dict)


def _pair(value: object, label: str) -> tuple[str, str]:
    if not (isinstance(value, list) and len(value) == 2 and all(v in KV_CACHE_TYPES for v in value)):
        raise PlanError(f"{label}: expected [K, V] from {KV_CACHE_TYPES}", code="plan")
    return str(value[0]), str(value[1])


def load_kv_plan(path: Path) -> KvPlan:
    """Parse and validate the plan."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    kv = raw.get("kv", {})
    contexts = tuple((int(c), int(n)) for c, n in kv.get("contexts", []))
    if not contexts or any(c < 256 or n < 1 for c, n in contexts):
        raise PlanError("kv.contexts must be [[ctx, chunks], ...]", code="plan")
    models = tuple(
        KvModel(str(m["id"]), str(m["repo"]), str(m["revision"]), str(m["pattern"]), bool(m.get("dense", True)),
                str(m.get("requires", "")), str(m.get("note", "")))
        for m in raw.get("models", [])
    )
    if not models:
        raise PlanError("plan has no models", code="plan")
    bench = dict(raw.get("bench", {}))
    ids = {m.id for m in models}
    for mid in bench.get("models", []):
        if mid not in ids:
            raise PlanError(f"bench model {mid} is not in [[models]]", code="plan")
    for arm in bench.get("spec_arms", []):
        args = list(arm.get("args", []))
        if args and (args[0] != "--spec-type" or args[1] not in SPEC_TYPES):
            raise PlanError(f"spec arm {arm.get('id')}: --spec-type must be one of {SPEC_TYPES}", code="plan")
    corpus = {k: str(v) for k, v in raw.get("corpus", {}).items()}
    if len(corpus.get("sha256", "")) != 64 or len(corpus.get("revision", "")) != 40:
        raise PlanError("corpus needs a 40-hex revision and a sha256", code="plan")
    return KvPlan(
        corpus=corpus,
        contexts=contexts,
        base=_pair(kv.get("base", ["f16", "f16"]), "kv.base"),
        arms=tuple(_pair(a, "kv.arms") for a in kv.get("arms", [])),
        kl_max=float(kv.get("kl_max", 0.01)),
        ppl_rel_max=float(kv.get("ppl_rel_max", 0.01)),
        models=models,
        bench=bench,
    )


def arm_id(pair: tuple[str, str]) -> str:
    """``q8_0/q4_0`` -> ``k-q8_0-v-q4_0``."""
    return f"k-{pair[0]}-v-{pair[1]}"


def kv_steps(plan: KvPlan, model: Path, corpus: Path, work: Path, binary: str = "llama-perplexity") -> list[dict]:
    """Every llama-perplexity call for one model: per context, one f16 base then one call per arm."""
    steps = []
    for ctx, chunks in plan.contexts:
        logits = work / f"base-{ctx}.kld"
        steps.append({"ctx": ctx, "arm": arm_id(plan.base), "role": "base",
                      "argv": kl_argv(binary, model, corpus, logits, chunks=chunks, ctx=ctx, compare=False,
                                      cache_k=plan.base[0], cache_v=plan.base[1])})
        for pair in plan.arms:
            steps.append({"ctx": ctx, "arm": arm_id(pair), "role": "arm",
                          "argv": kl_argv(binary, model, corpus, logits, chunks=chunks, ctx=ctx, compare=True,
                                          cache_k=pair[0], cache_v=pair[1])})
    return steps


def verdict(plan: KvPlan, kl_mean: float, ppl: float, ppl_base: float, *, ppl_usable: bool = True) -> str:
    """'pass' or 'fail' against the plan bar. Without a usable PPL, KL alone decides."""
    if kl_mean > plan.kl_max:
        return "fail"
    if ppl_usable and ppl_base > 0 and (ppl - ppl_base) / ppl_base > plan.ppl_rel_max:
        return "fail"
    return "pass"


# ---- corpus ----------------------------------------------------------------------------------

def fetch_corpus(plan: KvPlan, dest: Path, fetch: Callable[[KvPlan, Path], Path] | None = None) -> Path:
    """Download the pinned zip, check sha256, extract the member. Returns the text path."""
    out = dest / Path(plan.corpus["member"]).name
    if out.is_file():
        return out
    dest.mkdir(parents=True, exist_ok=True)
    if fetch is None:
        from huggingface_hub import hf_hub_download

        zpath = Path(hf_hub_download(plan.corpus["repo"], plan.corpus["file"], revision=plan.corpus["revision"],
                                     repo_type=plan.corpus.get("repo_type", "dataset"), local_dir=dest))
    else:
        zpath = fetch(plan, dest)
    got = hashlib.sha256(zpath.read_bytes()).hexdigest()
    if got != plan.corpus["sha256"]:
        raise PlanError(f"corpus sha256 {got} != {plan.corpus['sha256']}", code="checksum")
    with zipfile.ZipFile(zpath) as z:
        out.write_bytes(z.read(plan.corpus["member"]))
    return out


# ---- bench -----------------------------------------------------------------------------------

def server_cmd(binary: str, model: Path, *, ctx: int, port: int, extra: list[str]) -> list[str]:
    """llama-server argv for the bench: f16 KV, one slot, prompt cache on (b11476 default)."""
    return [binary, "-m", str(model), "-c", str(ctx), "-ngl", "99", "-fa", "on", "--host", "127.0.0.1",
            "--port", str(port), "-np", "1", "--fit", "off", "--cache-prompt", *extra]


def bench_matrix(plan: KvPlan) -> list[dict[str, object]]:
    """(model, arm, set) cells of the bench."""
    b = plan.bench
    cells = []
    for mid in b.get("models", []):
        for arm in b.get("spec_arms", []):
            for kind in ("rewrite", "free"):
                cells.append({"model": mid, "kind": "spec", "arm": arm["id"], "set": kind, "args": list(arm["args"])})
        for arm in b.get("prefix_arms", []):
            cells.append({"model": mid, "kind": "prefix", "arm": arm["id"], "set": "shared-system",
                          "args": list(arm["args"])})
    return cells


def rewrite_prompts(plan: KvPlan, corpus_text: str) -> list[str]:
    """Fixed rewrite prompts: the first N wikitext paragraphs of 400-1200 characters."""
    lines = (p.strip() for p in corpus_text.split("\n"))
    paras = [p for p in lines if 400 <= len(p) <= 1200 and not p.startswith("=")]
    n = int(plan.bench.get("rewrite_passages", 8))
    if len(paras) < n:
        raise PlanError("corpus has too few paragraphs for the rewrite set", code="plan")
    instr = str(plan.bench["rewrite_instruction"])
    return [f"{instr}\n\n{p}" for p in paras[:n]]


def summarize(results: list[dict[str, object]]) -> dict[str, object]:
    """Mean tokens/s, accept rate and prompt_ms over the /completion ``timings`` of one cell."""
    tps = [float(r["predicted_per_second"]) for r in results if r.get("predicted_per_second")]
    drafted = sum(int(r.get("draft_n", 0) or 0) for r in results)
    accepted = sum(int(r.get("draft_n_accepted", 0) or 0) for r in results)
    prompt_ms = [float(r["prompt_ms"]) for r in results if "prompt_ms" in r]
    cache_n = [int(r.get("cache_n", 0) or 0) for r in results]
    return {
        "n": len(results),
        "tokens_per_second": round(sum(tps) / len(tps), 2) if tps else None,
        "draft_n": drafted,
        "accept_rate": round(accepted / drafted, 3) if drafted else None,
        "prompt_ms_mean": round(sum(prompt_ms) / len(prompt_ms), 1) if prompt_ms else None,
        "prompt_ms": prompt_ms,
        "cache_n": cache_n,
    }


def _post(url: str, body: dict[str, object], timeout: float = 600.0) -> dict[str, object]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # localhost only
        return json.loads(r.read())


def run_cell(url: str, plan: KvPlan, prompts: list[str], *, system: str = "") -> list[dict[str, object]]:
    """Send each prompt to /completion (chat template applied server-side via /apply-template)."""
    if not url.startswith("http://127.0.0.1:"):
        raise PlanError("bench URL must be http://127.0.0.1:<port>", code="plan")
    b = plan.bench
    out = []
    for p in prompts:
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": p}]
        text = _post(f"{url}/apply-template", {"messages": msgs})["prompt"]
        res = _post(f"{url}/completion", {"prompt": text, "n_predict": int(b.get("n_predict", 256)),
                                          "temperature": float(b.get("temperature", 0.0)),
                                          "seed": int(b.get("seed", 7)), "cache_prompt": True})
        t = dict(res.get("timings", {}))
        out.append(t)
    return out


# ---- preflight -------------------------------------------------------------------------------

def check_help(server_help: str, ppl_help: str) -> list[dict[str, object]]:
    """b11476 flags the plan relies on: --spec-type with each ngram type, --cache-reuse, -ctk/-ctv."""
    checks = []
    have_spec = "--spec-type" in server_help
    missing = [t for t in SPEC_TYPES if t not in server_help]
    checks.append({"check": "llama-server --spec-type", "ok": have_spec and not missing,
                   "detail": "lists " + ", ".join(SPEC_TYPES) if have_spec and not missing
                   else f"missing: {'--spec-type' if not have_spec else missing}"})
    checks.append({"check": "llama-server --cache-reuse", "ok": "--cache-reuse" in server_help, "detail": ""})
    ok = all(f in ppl_help for f in ("--cache-type-k", "--cache-type-v", "--kl-divergence"))
    checks.append({"check": "llama-perplexity -ctk/-ctv/--kl-divergence", "ok": ok, "detail": ""})
    return checks


def check_protobuf(version: Callable[[str], str] | None = None) -> dict[str, object]:
    """protobuf must be exactly 5.29.6 (4.25.9 crashes the converter on Py3.14)."""
    get = version or (lambda d: md.version(d))
    try:
        got = get("protobuf")
    except md.PackageNotFoundError:
        got = "missing"
    return {"check": f"protobuf=={PROTOBUF_PIN}", "ok": got == PROTOBUF_PIN, "detail": got}


def check_gemma_meta(meta: dict[str, object]) -> dict[str, object]:
    """gemma4 GGUF: BOS added, a BOS id, and a Gemma chat template (start_of_turn) in the file."""
    template = str(meta.get("tokenizer.chat_template", ""))
    problems = []
    if meta.get("tokenizer.ggml.add_bos_token") is not True:
        problems.append("add_bos_token is not true")
    if meta.get("tokenizer.ggml.bos_token_id") is None:
        problems.append("no bos_token_id")
    if not template:
        problems.append("no chat template")
    elif "<start_of_turn>" not in template and "<|turn>" not in template:
        problems.append("chat template has no Gemma turn markers")
    return {"check": "gemma4 template/BOS", "ok": not problems, "detail": "; ".join(problems) or "ok"}


def read_gguf_meta(path: Path, keys: tuple[str, ...]) -> dict[str, object]:
    """Selected GGUF metadata fields (header only)."""
    import gguf

    reader = gguf.GGUFReader(str(path))
    out: dict[str, object] = {}
    for key in keys:
        f = reader.get_field(key)
        if f is not None:
            out[key] = f.contents()
    return out


GEMMA_KEYS = ("tokenizer.ggml.add_bos_token", "tokenizer.ggml.bos_token_id", "tokenizer.chat_template")


def main(argv: list[str] | None = None) -> int:
    """CLI: --dry-run prints every step; --preflight runs the checks; otherwise see scripts/vm_run_kv.sh."""
    ap = argparse.ArgumentParser(prog="python -m voxprint_forge.kl.kv", description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--server-help", type=Path)
    ap.add_argument("--ppl-help", type=Path)
    ap.add_argument("--gemma-gguf", type=Path)
    ap.add_argument("--kv-model", help="run the KV arms for one model id")
    ap.add_argument("--gguf", type=Path)
    ap.add_argument("--corpus-dir", type=Path)
    ap.add_argument("--work", type=Path)
    ap.add_argument("--bench-model", help="run the bench cells for one model id against --url")
    ap.add_argument("--url", default="")
    ap.add_argument("--cell-arm", default="")
    args = ap.parse_args(argv)
    plan = load_kv_plan(args.plan)
    if args.dry_run:
        demo = kv_steps(plan, Path("<model>.gguf"), Path("<corpus>"), Path("<work>"))
        print(json.dumps({"kv_steps_per_model": [{"ctx": s["ctx"], "arm": s["arm"], "argv": s["argv"][1:]}
                                                 for s in demo],
                          "models": [m.id for m in plan.models],
                          "bench_cells": bench_matrix(plan)}, indent=1))
        return 0
    if args.preflight:
        checks = [check_protobuf()]
        if args.server_help and args.ppl_help:
            checks += check_help(args.server_help.read_text(), args.ppl_help.read_text())
        else:
            checks.append({"check": "llama.cpp help", "ok": False, "detail": "pass --server-help and --ppl-help"})
        if args.gemma_gguf:
            checks.append(check_gemma_meta(read_gguf_meta(args.gemma_gguf, GEMMA_KEYS)))
        for c in checks:
            print(("ok   " if c["ok"] else "FAIL ") + f"{c['check']}: {c['detail']}")
        return 0 if all(c["ok"] for c in checks) else 1
    if args.kv_model:
        from voxprint_forge.kl.parse import parse_kl
        from voxprint_forge.llama import Runner, tool

        corpus = fetch_corpus(plan, args.corpus_dir)
        work = args.work / args.kv_model
        work.mkdir(parents=True, exist_ok=True)
        runner = Runner()
        rows = []
        for step in kv_steps(plan, args.gguf, corpus, work, tool("llama-perplexity")):
            done = runner.run(step["argv"])
            log = done.stdout + done.stderr
            (work / f"{step['arm']}-{step['ctx']}.log").write_text(log, encoding="utf-8")
            if step["role"] == "arm":
                s = parse_kl(log)
                rows.append({"ctx": step["ctx"], "arm": step["arm"], "ppl": s.ppl_quant, "ppl_base": s.ppl_base,
                             "kl_mean": s.kl_mean, "kl_p99": s.kl_p99, "top1_pct": s.top1_agreement_pct,
                             "delta_p_rms_pct": s.delta_p_rms_pct, "tokens_per_second": s.tokens_per_second,
                             "verdict": verdict(plan, s.kl_mean, s.ppl_quant, s.ppl_base)})
        for logits in work.glob("base-*.kld"):
            logits.unlink()
        (work / "kv.json").write_text(json.dumps({"model": args.kv_model, "rows": rows}, indent=2) + "\n")
        print(json.dumps(rows))
        return 0
    if args.bench_model:
        corpus = fetch_corpus(plan, args.corpus_dir).read_text(encoding="utf-8")
        cells = [c for c in bench_matrix(plan) if c["model"] == args.bench_model and c["arm"] == args.cell_arm]
        out = []
        for c in cells:
            t0 = time.perf_counter()
            if c["kind"] == "prefix":
                system = corpus[: int(plan.bench.get("prefix_system_chars", 12000))]
                turns = [f"Question {i + 1}: summarise paragraph {i + 1} of the text in one sentence."
                         for i in range(int(plan.bench.get("prefix_turns", 4)))]
                res = run_cell(args.url, plan, turns, system=system)
            else:
                prompts = rewrite_prompts(plan, corpus) if c["set"] == "rewrite" else list(plan.bench["free_prompts"])
                res = run_cell(args.url, plan, prompts)
            out.append({**{k: c[k] for k in ("model", "kind", "arm", "set")}, **summarize(res),
                        "wall_s": round(time.perf_counter() - t0, 1)})
        print(json.dumps(out))
        return 0
    ap.error("choose --dry-run, --preflight, --kv-model or --bench-model")
    return 2


if __name__ == "__main__":
    sys.exit(main())
