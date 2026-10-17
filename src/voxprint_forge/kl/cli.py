"""``forge kl`` command."""

from __future__ import annotations

import argparse
from pathlib import Path

from voxprint_forge.errors import UsageError
from voxprint_forge.kl.plan import KlJob, QuantRef, load_kl_plan
from voxprint_forge.kl.run import run_kl


def add_command(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the KL subcommand."""
    command = sub.add_parser(
        "kl",
        help="Perplexity and KL of GGUF quants against an F16 base.",
    )
    command.add_argument("--job", default=None, help="L-job id from the KL plan, for example L1.")
    command.add_argument("--plan", type=Path, default=None, help="KL plan. Default: plans/kl-synth4.toml.")
    command.add_argument("--bf16", default=None, help="Hugging Face repo@revision of the BF16 safetensors.")
    command.add_argument(
        "--quant",
        action="append",
        default=None,
        help="Repeatable. A local .gguf, or repo@revision:*.QUANT*.gguf.",
    )
    command.add_argument("--chunks", type=int, default=None, help="Perplexity chunks. Default: the plan, or 100.")
    command.add_argument("--ctx", type=int, default=None, help="Perplexity context. Default: the plan, or 512.")


def execute(
    root: Path,
    forge_out: Path,
    *,
    job_id: str | None,
    plan_path: Path | None,
    bf16: str | None,
    quants: list[str] | None,
    chunks: int | None,
    ctx: int | None,
) -> dict[str, object]:
    """Run one KL job. Results land in ``forge-out/results/kl/<id>``."""
    job = _resolve(root, job_id, plan_path, bf16, quants, chunks, ctx)
    return run_kl(job, forge_out / "results" / "kl" / job.id)


def _resolve(
    root: Path,
    job_id: str | None,
    plan_path: Path | None,
    bf16: str | None,
    quants: list[str] | None,
    chunks: int | None,
    ctx: int | None,
) -> KlJob:
    if job_id and (bf16 or quants):
        raise UsageError("pass either --job or --bf16 with --quant", code="usage")
    if job_id:
        path = plan_path or (root / "plans" / "kl-synth4.toml")
        job = load_kl_plan(path).job(job_id)
        return _override(job, chunks, ctx)
    if not bf16 or not quants:
        raise UsageError("kl needs --job, or --bf16 repo@rev and at least one --quant", code="usage")
    repo, revision = _repo_rev(bf16)
    parsed = tuple(_quant_spec(item) for item in quants)
    slug = repo.split("/")[-1]
    return KlJob(
        id=slug,
        bf16_repo=repo,
        bf16_revision=revision,
        source_format="safetensors",
        f16_file="",
        expect_mtp=False,
        index_gap="",
        note="",
        quants=parsed,
        chunks=chunks if chunks is not None else 100,
        ctx=ctx if ctx is not None else 512,
        fit_ctx=8192,
        headroom_gb=3.0,
        budgets_gb=(16.0, 24.0),
        kv_cache="q8_0",
    )


def _override(job: KlJob, chunks: int | None, ctx: int | None) -> KlJob:
    if chunks is None and ctx is None:
        return job
    return KlJob(
        id=job.id,
        bf16_repo=job.bf16_repo,
        bf16_revision=job.bf16_revision,
        source_format=job.source_format,
        f16_file=job.f16_file,
        expect_mtp=job.expect_mtp,
        index_gap=job.index_gap,
        note=job.note,
        quants=job.quants,
        chunks=job.chunks if chunks is None else chunks,
        ctx=job.ctx if ctx is None else ctx,
        fit_ctx=job.fit_ctx,
        headroom_gb=job.headroom_gb,
        budgets_gb=job.budgets_gb,
        kv_cache=job.kv_cache,
    )


def _repo_rev(text: str) -> tuple[str, str]:
    if "@" not in text:
        raise UsageError("--bf16 must be repo@revision", code="usage")
    repo, revision = text.rsplit("@", 1)
    if not repo or not revision:
        raise UsageError("--bf16 must be repo@revision", code="usage")
    return repo, revision


def _quant_spec(text: str) -> QuantRef:
    path = Path(text)
    if path.is_file():
        return QuantRef(quant=path.stem, repo="", revision="", pattern=str(path), ours=True)
    if "@" not in text or ":" not in text.split("@", 1)[-1]:
        raise UsageError("--quant must be a local .gguf or repo@revision:pattern", code="usage")
    repo_rev, pattern = text.rsplit(":", 1)
    repo, revision = _repo_rev(repo_rev)
    label = _label(pattern)
    return QuantRef(quant=label, repo=repo, revision=revision, pattern=pattern, ours=True)


def _label(pattern: str) -> str:
    name = Path(pattern).name
    for token in ("IQ4_XS", "IQ3_XXS", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q5_K"):
        if token in name:
            return token
    stem = Path(name).stem.replace("*", "")
    if stem:
        return stem
    raise UsageError(f"--quant pattern {pattern} has no quant name", code="usage")
