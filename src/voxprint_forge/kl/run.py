"""Run one L-job: F16 base, KL against each GGUF, then the 16/24 GB fit."""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Never

import torch
from huggingface_hub import snapshot_download

from voxprint_forge.errors import DownloadError, StageError
from voxprint_forge.evaluate import data_dir, parse_ppl
from voxprint_forge.fitmath import Arch, fits_budget
from voxprint_forge.hashing import sha256_file
from voxprint_forge.kl.arch import (
    gguf_model_name,
    gguf_reference_mtp,
    mtp_status,
    read_model_arch,
    require_mtp,
    source_mtp_layers,
)
from voxprint_forge.kl.parse import parse_kl, tokens_per_second
from voxprint_forge.kl.plan import KlJob, QuantRef, SourceFormat, local_revision
from voxprint_forge.kl.preflight import require_llama_arch
from voxprint_forge.kl.recover import recover_ablation
from voxprint_forge.llama import Runner, convert_argv, convert_script, kl_argv, server_argv, tool
from voxprint_forge.measure import parse_buffers
from voxprint_forge.scrub import active_secrets, scrub


def gpu_note(name: str) -> str:
    """Tokens/s sentence for the GPU that produced the figure."""
    label = name or "this GPU"
    return (
        f"tokens/s is the {label} figure from this box. It is not a consumer-GPU number. "
        "The 16 GB and 24 GB rows are weights plus q8_0 KV at context 8192 with 3 GB headroom."
    )


def speed_facts(name: str, capability: tuple[int, int] | None) -> dict[str, object]:
    """``tokens_per_second_is_a100`` is true only on compute capability 8.0."""
    return {
        "tokens_per_second_is_a100": capability == (8, 0),
        "tokens_per_second_gpu": name,
        "gpu_note": gpu_note(name),
    }
SERVER_PATTERN = "KV buffer size"
FetchFn = Callable[[str, str, Path, list[str] | None], None]
ConvertFn = Callable[[Runner, Path, Path], None]
RecoverFn = Callable[[Path, Path, Path], dict[str, object]]

# llama-perplexity stores base log-probs floored at (max logit - 16). When the
# KL run's "Mean PPL(base)" (read back from that file) and the base run's own
# final estimate disagree by more than this, many target tokens sit more than
# 16 nats below the argmax: the corpus is far outside what the model predicts
# (Gemma 4 instruct on raw prose does this) and PPL is not a usable figure.
PPL_CLAMP_TOLERANCE = 0.02
PPL_INVALID_NOTE = (
    "PPL is not a usable figure for this model on this corpus: the base PPL read back from the KL file "
    "differs from the base run by more than 2%, so many target tokens are more than 16 nats below the "
    "top token. Compare quants by KL, top-1 and delta-p only."
)


def run_kl(
    job: KlJob,
    work: Path,
    *,
    corpus: Path | None = None,
    runner: Runner | None = None,
    fetch: FetchFn | None = None,
    convert: ConvertFn | None = None,
    recover: RecoverFn | None = None,
) -> dict[str, object]:
    """Score ``job`` and write ``summary.json``. BF16 and F16 files are deleted afterwards.

    The logits file is deleted with them. It is an intermediate of the base run,
    and the next L-job needs that disk. Quant GGUFs under ``work/quants`` stay.
    """
    text = corpus or (data_dir() / "ppl.txt")
    if not text.is_file():
        raise StageError(f"missing perplexity corpus {text.name}", code="kl_corpus")
    if runner is None:
        _preflight_llama()
    active = runner or Runner()
    fetch_fn = fetch or fetch_snapshot
    work.mkdir(parents=True, exist_ok=True)
    source = work / "source"
    f16 = work / "f16.gguf"
    logits = work / "base-logits.bin"
    ablated = work / "ablated"
    try:
        f16_path, reference = _prepare_base(job, work, source, f16, active, fetch_fn, convert, recover, ablated)
        arch, meta = read_model_arch(f16_path)
        reference["name"] = gguf_model_name(f16_path)
        mtp = _mtp(job, source, meta)
        base_done = active.run(
            kl_argv(
                tool("llama-perplexity"),
                f16_path,
                text,
                logits,
                chunks=job.chunks,
                ctx=job.ctx,
                compare=False,
            )
        )
        base_log = base_done.stdout + base_done.stderr
        keep_log(work, "base", base_done.argv, base_log)
        quants = [
            _one_quant(job, quant, work, text, logits, arch, active, fetch_fn, str(reference["name"]))
            for quant in job.quants
        ]
        payload: dict[str, object] = {
            "job": job.id,
            "bf16_repo": job.bf16_repo,
            "bf16_revision": job.bf16_revision,
            "source_format": job.source_format,
            "chunks": job.chunks,
            "ctx": job.ctx,
            "ppl_base": parse_ppl(base_log),
            "base_tokens_per_second": tokens_per_second(base_log),
            **speed_facts(*_detected_gpu()),
            "architecture": meta,
            "mtp": mtp,
            "reference": reference,
            "index_gap": job.index_gap,
            "note": job.note,
            "quants": quants,
            "base_reclaimed": True,
        }
        mark_ppl_validity(payload)
        _write(work, payload)
        return payload
    finally:
        reclaim_base(source, f16, logits)
        if ablated.exists():
            shutil.rmtree(ablated)


def keep_log(work: Path, name: str, argv: Sequence[str], text: str) -> Path:
    """Full llama-perplexity output next to summary.json (G4R 07:29 kept only the parsed figures)."""
    path = work / f"perplexity-{name}.log"
    path.write_text("# argv: " + " ".join(str(item) for item in argv) + "\n" + text, encoding="utf-8", newline="\n")
    return path


def mark_ppl_validity(payload: dict[str, object]) -> None:
    """Set ``ppl_valid`` on the job and on each quant row (see ``PPL_CLAMP_TOLERANCE``)."""
    base = payload.get("ppl_base")
    rows = payload.get("quants")
    verdicts: list[bool] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            readback = row.get("ppl_base")
            valid = (
                isinstance(base, (int, float))
                and isinstance(readback, (int, float))
                and base > 0
                and abs(float(readback) / float(base) - 1.0) <= PPL_CLAMP_TOLERANCE
            )
            row["ppl_valid"] = valid
            verdicts.append(valid)
    payload["ppl_valid"] = bool(verdicts) and all(verdicts)
    payload["ppl_note"] = "" if payload["ppl_valid"] else PPL_INVALID_NOTE


def rerender(work: Path) -> dict[str, object]:
    """Re-mark PPL validity and rewrite ``summary.json`` and ``summary.md`` from an existing run. No GPU."""
    path = work / "summary.json"
    if not path.is_file():
        raise StageError(f"no summary.json in {work.name}", code="kl_render")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StageError("summary.json is not an object", code="kl_render")
    mark_ppl_validity(payload)
    _write(work, payload)
    return payload


def _detected_gpu() -> tuple[str, tuple[int, int] | None]:
    if not torch.cuda.is_available():
        return "", None
    major, minor = torch.cuda.get_device_capability()
    return torch.cuda.get_device_name(0), (int(major), int(minor))


def _preflight_llama() -> None:
    """Fail with exit 9 when llama-perplexity has no SASS for this GPU."""
    warning = require_llama_arch(Path(tool("llama-perplexity")), capability=_cuda_capability())
    if warning:
        print(warning, file=sys.stderr)


def _cuda_capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability()
    return (int(major), int(minor))


def fetch_snapshot(repo: str, revision: str, dest: Path, patterns: list[str] | None) -> None:
    """Download one public repo. No token is passed or printed."""
    dest.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, object] = {"repo_id": repo, "local_dir": str(dest)}
    if revision:
        kwargs["revision"] = revision
    if patterns:
        kwargs["allow_patterns"] = patterns
    try:
        snapshot_download(**kwargs)
    except Exception as exc:
        message = scrub(str(exc), active_secrets())
        raise DownloadError(f"download failed for {repo}: {message}", code="kl_download") from exc


def convert_f16(runner: Runner, source: Path, outfile: Path) -> None:
    """``convert_hf_to_gguf.py --outtype f16`` from the pinned llama.cpp tree."""
    runner.run(convert_argv(convert_script(), source, outfile))


def reclaim_base(source: Path, f16: Path, logits: Path) -> None:
    """Delete the BF16 tree, the F16 GGUF, and the base logits."""
    if source.exists():
        shutil.rmtree(source)
    for path in (f16, logits):
        if path.is_file():
            path.unlink()


def select_quant_file(folder: Path, pattern: str, quant: str) -> Path:
    """One GGUF under ``folder`` matching ``pattern``. Two equally good matches fail."""
    if Path(pattern).is_file():
        return Path(pattern)
    files = [path for path in sorted(folder.rglob(pattern)) if path.is_file() and path.suffix == ".gguf"]
    if not files:
        raise StageError(f"no GGUF matched {pattern} for {quant}", code="kl_quant")
    preferred = [path for path in files if quant in path.name]
    chosen = preferred or files
    if len(chosen) == 1:
        return chosen[0]
    names = ", ".join(path.name for path in chosen)
    raise StageError(f"{quant} matched more than one file: {names}", code="kl_quant")


def tier_rows(
    weights_bytes: int,
    arch: Arch,
    job: KlJob,
    measured_weights: int | None,
    measured_kv: int | None,
) -> list[dict[str, object]]:
    """Computed fit and, when the server log was parsed, the same fit on real buffers."""
    kv = arch.kv_bytes(job.fit_ctx, job.kv_cache)
    rows: list[dict[str, object]] = []
    for budget in job.budgets_gb:
        computed = fits_budget(weights_bytes, kv, budget_gb=budget, headroom_gb=job.headroom_gb)
        server_fits: bool | None = None
        if measured_weights is not None and measured_kv is not None:
            server_fits = fits_budget(
                measured_weights,
                measured_kv,
                budget_gb=budget,
                headroom_gb=job.headroom_gb,
            )
        rows.append(
            {
                "budget_gb": budget,
                "headroom_gb": job.headroom_gb,
                "ctx": job.fit_ctx,
                "kv_cache": job.kv_cache,
                "weights_bytes": weights_bytes,
                "kv_bytes_computed": kv,
                "fits_computed": computed,
                "server_weights_bytes": measured_weights,
                "server_kv_bytes": measured_kv,
                "fits_server": server_fits,
            }
        )
    return rows


def _prepare_base(
    job: KlJob,
    work: Path,
    source: Path,
    f16: Path,
    runner: Runner,
    fetch: FetchFn,
    convert: ConvertFn | None,
    recover: RecoverFn | None,
    ablated: Path,
) -> tuple[Path, dict[str, object]]:
    fmt: SourceFormat = job.source_format
    record: dict[str, object] = {"kind": job.reference, "ablation": None}
    if fmt == "safetensors":
        if job.sha256:
            raise StageError(f"{job.id} sha256 needs one GGUF source file", code="kl_download")
        fetch(job.bf16_repo, job.bf16_revision, source, None)
        tree = source
        if job.reference == "recover-ablation":
            quant = _fetch_quant(job, _ablation_ref(job), work, fetch)
            record["ablation"] = (recover or recover_ablation)(quant, source, ablated)
            if not ablated.is_dir():
                raise StageError("ablation recovery wrote no model directory", code="kl_reference")
            tree = ablated
        (convert or convert_f16)(runner, tree, f16)
        if not f16.is_file():
            raise StageError("F16 conversion produced no file", code="kl_convert")
        if tree != source and tree.exists():
            shutil.rmtree(tree)
        return f16, record
    if fmt == "gguf":
        local = Path(job.f16_file)
        if local.is_absolute():
            # A reference built on this machine (GEMMA4-REQUANT): not downloaded and never deleted.
            if not local.is_file():
                raise StageError(f"{job.id} reference {local.name} is missing", code="kl_download")
            _check_sha256(local, job.sha256, f"{job.id} reference")
            return local, record
        fetch(job.bf16_repo, job.bf16_revision, source, [job.f16_file])
        path = select_quant_file(source, job.f16_file, "F16")
        _check_sha256(path, job.sha256, f"{job.id} reference")
        return path, record
    return _unexpected_format(fmt)


def _ablation_ref(job: KlJob) -> QuantRef:
    for quant in job.quants:
        if quant.quant == job.ablation_quant:
            return quant
    raise StageError(f"{job.id} has no quant {job.ablation_quant}", code="kl_plan")


def _fetch_quant(job: KlJob, quant: QuantRef, work: Path, fetch: FetchFn) -> Path:
    if quant.source == "local":
        return _local_quant(job, quant)
    if Path(quant.pattern).is_file():
        path = Path(quant.pattern)
    else:
        folder = work / "quants" / quant.quant
        fetch(quant.repo, quant.revision, folder, [quant.pattern])
        path = select_quant_file(folder, quant.pattern, quant.quant)
    _check_sha256(path, quant.sha256, f"{job.id} {quant.quant}")
    return path


def _local_quant(job: KlJob, quant: QuantRef) -> Path:
    """A GGUF built on this machine: never downloaded; its content must match the plan's revision."""
    path = Path(quant.pattern)
    label = f"{job.id} {quant.quant}"
    if not path.is_file():
        raise StageError(f"{label} local file {path.name} is missing", code="kl_quant")
    digest = sha256_file(path)
    if quant.sha256 and digest != quant.sha256:
        raise StageError(f"{label} sha256 mismatch", code="kl_download")
    if local_revision(digest) != quant.revision:
        raise StageError(f"{label} content does not match revision {quant.revision}", code="kl_download")
    return path


def _unexpected_format(fmt: Never) -> Never:
    raise StageError(f"unhandled KL source format {fmt}", code="kl_plan")


def _check_sha256(path: Path, expected: str, label: str) -> None:
    if not expected:
        return
    actual = sha256_file(path)
    if actual != expected:
        raise StageError(f"{label} sha256 mismatch", code="kl_download")


def _mtp(job: KlJob, source: Path, meta: dict[str, object]) -> dict[str, object] | None:
    if not job.expect_mtp:
        return None
    fmt: SourceFormat = job.source_format
    if fmt == "safetensors":
        layers, mtp_layers = source_mtp_layers(source)
        status = mtp_status(layers, mtp_layers, meta)
        require_mtp(status)
        return status
    if fmt == "gguf":
        status = gguf_reference_mtp(meta)
        require_mtp(status)
        return status
    return _unexpected_format(fmt)


def _one_quant(
    job: KlJob,
    quant: QuantRef,
    work: Path,
    corpus: Path,
    logits: Path,
    arch: Arch,
    runner: Runner,
    fetch: FetchFn,
    reference_name: str,
) -> dict[str, object]:
    path = _fetch_quant(job, quant, work, fetch)
    name = gguf_model_name(path)
    done = runner.run(
        kl_argv(
            tool("llama-perplexity"),
            path,
            corpus,
            logits,
            chunks=job.chunks,
            ctx=job.ctx,
            compare=True,
        )
    )
    keep_log(work, quant.quant, done.argv, done.stdout + done.stderr)
    stats = parse_kl(done.stdout + done.stderr)
    server = runner.capture_until(
        server_argv(tool("llama-server"), path, ctx=job.fit_ctx, port=8088, cache=job.kv_cache),
        pattern=SERVER_PATTERN,
    )
    measured = parse_buffers(server.stdout + server.stderr)
    return {
        "quant": quant.quant,
        "source": quant.source,
        "repo": quant.repo,
        "revision": quant.revision,
        "ours": quant.ours,
        "file": path.name,
        "file_bytes": path.stat().st_size,
        "name": name,
        "reference_name_match": name == reference_name,
        "ppl": stats.ppl_quant,
        "ppl_base": stats.ppl_base,
        "kl_mean": stats.kl_mean,
        "kl_p99": stats.kl_p99,
        "top1_agreement_pct": stats.top1_agreement_pct,
        "delta_p_rms_pct": stats.delta_p_rms_pct,
        "tokens_per_second": stats.tokens_per_second,
        "tiers": tier_rows(path.stat().st_size, arch, job, measured.weights_bytes, measured.kv_bytes),
    }


def _write(work: Path, payload: dict[str, object]) -> None:
    (work / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (work / "summary.md").write_text(_markdown(payload), encoding="utf-8", newline="\n")


def _markdown(payload: dict[str, object]) -> str:
    valid = payload.get("ppl_valid", True) is not False
    lines = [
        f"# {payload['job']}",
        "",
        f"Base PPL: {payload['ppl_base']}" + ("" if valid else " (not usable, see below)"),
        "",
        str(payload["gpu_note"]),
        "",
    ]
    if not valid:
        lines.extend([str(payload.get("ppl_note") or PPL_INVALID_NOTE), ""])
    reference = payload.get("reference")
    if isinstance(reference, dict):
        line = f"Reference: {reference.get('kind', 'upstream')}"
        ablation = reference.get("ablation")
        if isinstance(ablation, dict):
            line += (
                f", refusal direction read back from {ablation.get('from')} "
                f"({ablation.get('matrices')} matrices, eigenvalue ratio {float(ablation.get('eigen_ratio') or 0):.4f})"
            )
        lines.extend([line + ".", ""])
    if payload.get("index_gap"):
        lines.extend([str(payload["index_gap"]), ""])
    if payload.get("note"):
        lines.extend([str(payload["note"]), ""])
    quants = payload["quants"]
    if isinstance(quants, Sequence):
        for item in quants:
            if not isinstance(item, dict):
                continue
            ppl = item["ppl"] if item.get("ppl_valid", True) is not False else "n/a"
            mismatch = "" if item.get("reference_name_match", True) is not False else (
                f" WARNING: GGUF name '{item.get('name')}' differs from the reference"
            )
            lines.append(
                f"- {item['quant']}: PPL {ppl}, KL mean {item['kl_mean']}, "
                f"KL p99 {item['kl_p99']}, top-1 {item['top1_agreement_pct']}%, "
                f"delta-p RMS {item['delta_p_rms_pct']}%, {item['file_bytes']} bytes{mismatch}"
            )
    lines.append("")
    return "\n".join(lines)
