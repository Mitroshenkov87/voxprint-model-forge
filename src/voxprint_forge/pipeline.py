"""Run fetch, ablation, quantize, fit, eval, card, and private upload for one model."""

from __future__ import annotations

import json
import os
import shutil
import socket
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from voxprint_forge.abliterate import abliterate_directory
from voxprint_forge.calibration import ensure_calibration
from voxprint_forge.cards import render_card
from voxprint_forge.diffusion.defer import defer_requested
from voxprint_forge.disk import ensure_free
from voxprint_forge.drive import upload_drive
from voxprint_forge.errors import ForgeError, StageError, UsageError
from voxprint_forge.evaluate import data_dir, evaluate_model, read_lines
from voxprint_forge.fetch import Downloader, ensure_pinned
from voxprint_forge.hashing import sha256_file, write_sha256
from voxprint_forge.layout import Layout, read_marker, write_marker
from voxprint_forge.llama import (
    Runner,
    convert_argv,
    convert_script,
    imatrix_argv,
    quantize_argv,
    server_argv,
    tool,
)
from voxprint_forge.manifest import summary_lines, write_run_manifest
from voxprint_forge.measure import (
    Candidate,
    choose_cache,
    fit_contexts,
    measurement_fits,
    parse_buffers,
    tier_min_vram,
)
from voxprint_forge.outcome import summarise
from voxprint_forge.plan import ModelPlan, Plan, Tier
from voxprint_forge.progress import Progress
from voxprint_forge.schedule import Gate
from voxprint_forge.scrub import log
from voxprint_forge.statusfile import beating, current_stage, set_stage, update_item
from voxprint_forge.upload import Artifact, Remote, publish_private

ORDER = ("fetch", "abliterate", "quantize", "fit", "eval", "card", "publish-prep", "upload")
_GPU = frozenset({"abliterate", "fit", "eval"})
_SILENT = frozenset({"imatrix", "quantize", "fit", "eval", "upload"})


def _no_probe() -> int | None:
    return None


@dataclass
class Host:
    runner: Runner
    downloader: Downloader
    remote: Remote
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic
    free_vram: Callable[[], int | None] = _no_probe
    nvidia_used: Callable[[], int | None] = _no_probe
    ablate: Callable[[Path, Path, int, int | None], dict[str, object]] | None = None
    attempts: int = 4
    stamped: str = ""


def _tracker(plan: Plan, layout: Layout, host: Host, progress: Progress | None) -> Progress:
    if progress is not None:
        return progress
    build = float(plan.estimates_min.get("bootstrap_build", 0))
    total = build + sum(float(item.estimate_min) for item in plan.enabled_models())
    return Progress(
        layout.root / "progress.log",
        total_min=total,
        consumed_min=build,
        now=host.now,
    )


def _slot(gate: Gate | None, *, gpu: bool) -> Iterator[None]:
    if gate is None:
        return nullcontext()
    return gate.slot(gpu=gpu)


def _beat(layout: Layout, model: ModelPlan, stage: str) -> Iterator[None]:
    if stage not in _SILENT:
        return nullcontext()
    return beating(layout.root, model.id, stage, eta_min=float(model.estimate_min))


def _skip_for_time(model: ModelPlan, remaining: float) -> dict[str, object]:
    log(f"skip {model.id}: {remaining:.1f} min left, needs {model.estimate_min}")
    return {
        "id": model.id,
        "base_model": model.base_model,
        "revision": model.revision,
        "licence": model.licence,
        "skipped": True,
        "reason": "time",
        "state": "DEFERRED",
        "files": [],
    }


def _fail(layout: Layout, model: ModelPlan, exc: ForgeError) -> dict[str, object]:
    stage = current_stage()
    payload = {
        "id": model.id,
        "base_model": model.base_model,
        "state": f"STAGEFAIL {stage}",
        "stage": stage,
        "error": str(exc),
        "skipped": False,
        "files": [],
    }
    update_item(
        layout.root,
        model.id,
        stage=stage,
        pct=0,
        eta_min=float(model.estimate_min),
        state=str(payload["state"]),
        error=str(exc),
    )
    log(f"{model.id} STAGEFAIL {stage}: {exc}")
    return payload


def run_all(
    plan: Plan,
    host: Host,
    repo: Path,
    layout: Layout,
    *,
    jobs: int = 1,
) -> dict[str, object]:
    start = host.now()
    progress = _tracker(plan, layout, host, None)
    enabled = list(plan.enabled_models())
    models: list[dict[str, object]] = []
    if jobs <= 1:
        for model in enabled:
            remaining = plan.wall_clock_budget_min - (host.now() - start) / 60.0
            if model.optional and remaining < model.estimate_min:
                skipped = _skip_for_time(model, remaining)
                write_marker(layout.stage(model.id, "skipped"), skipped)
                progress.advance(model.id, "skipped", float(model.estimate_min))
                models.append(skipped)
                continue
            try:
                models.append(run_model(plan, model, host, repo, layout, stop_after="upload", progress=progress))
            except ForgeError as exc:
                models.append(_fail(layout, model, exc))
    else:
        gate = Gate(jobs)

        def _worker(model: ModelPlan) -> dict[str, object]:
            remaining = plan.wall_clock_budget_min - (host.now() - start) / 60.0
            if model.optional and remaining < model.estimate_min:
                skipped = _skip_for_time(model, remaining)
                write_marker(layout.stage(model.id, "skipped"), skipped)
                return skipped
            try:
                return run_model(
                    plan,
                    model,
                    host,
                    repo,
                    layout,
                    stop_after="upload",
                    progress=progress,
                    gate=gate,
                )
            except ForgeError as exc:
                return _fail(layout, model, exc)

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            models.extend(pool.map(_worker, enabled))
    stamped = host.stamped or datetime.now(timezone.utc).date().isoformat()
    text, ok = summarise(models)
    log(text)
    payload: dict[str, object] = {"date": stamped, "plan": plan.name, "models": models}
    written = write_run_manifest(layout, payload, date=stamped)
    card = "\n".join(summary_lines(payload))
    log(card)
    if written.get("instructions"):
        log(str(written["instructions"]))
    return {"models": models, "manifest": written, "summary": card, "ok": ok, "states": text}


def _stage_share(model: ModelPlan) -> float:
    return float(model.estimate_min) / len(ORDER)


def _open_stage(progress: Progress, layout: Layout, model: ModelPlan, stage: str) -> None:
    set_stage(stage)
    update_item(
        layout.root,
        model.id,
        stage=stage,
        pct=0,
        eta_min=float(model.estimate_min),
        state="RUNNING",
    )
    if read_marker(layout.stage(model.id, stage)) is None:
        progress.begin(model.id, stage)


def run_model(
    plan: Plan,
    model: ModelPlan,
    host: Host,
    repo: Path,
    layout: Layout,
    *,
    stop_after: str,
    budgets: set[float] | None = None,
    progress: Progress | None = None,
    gate: Gate | None = None,
) -> dict[str, object]:
    if stop_after not in ORDER:
        raise UsageError(f"unknown stage {stop_after}", code="usage")
    tracker = _tracker(plan, layout, host, progress)
    share = _stage_share(model)
    done = read_marker(layout.stage(model.id, "upload"))
    if done is not None:
        tracker.advance(model.id, "upload", float(model.estimate_min))
        return done
    target = ORDER.index(stop_after)
    work = layout.model_dir(model.id)
    work.mkdir(parents=True, exist_ok=True)
    _reserve_disk(plan, model, work)
    with _slot(gate, gpu=False):
        _open_stage(tracker, layout, model, "fetch")
        fetched = _fetch(model, host, layout)
    tracker.advance(model.id, "fetch", share)
    if target == 0:
        return fetched
    with _slot(gate, gpu=True):
        _open_stage(tracker, layout, model, "abliterate")
        prepared = _prepare(model, host, layout)
    tracker.advance(model.id, "abliterate", share)
    if target == 1:
        return prepared
    gguf = Path(str(prepared["gguf"]))
    with _slot(gate, gpu=False):
        set_stage("imatrix")
        with _beat(layout, model, "imatrix"):
            imatrix = _imatrix(plan, model, host, layout, gguf)
        _open_stage(tracker, layout, model, "quantize")
        with _beat(layout, model, "quantize"):
            quantized = _quantize(model, host, layout, gguf, Path(str(imatrix["path"])))
    tracker.advance(model.id, "quantize", share)
    if target == 2:
        return quantized
    imatrix_path = Path(str(imatrix["path"]))
    with _slot(gate, gpu=True):
        _open_stage(tracker, layout, model, "fit")
        with _beat(layout, model, "fit"):
            fitted = _fit(plan, model, host, layout, gguf, imatrix_path, budgets)
        tracker.advance(model.id, "fit", share)
        if target == 3:
            return fitted
        _open_stage(tracker, layout, model, "eval")
        with _beat(layout, model, "eval"):
            evaluated = _eval(plan, model, host, layout, fitted)
    tracker.advance(model.id, "eval", share)
    if target == 4:
        return evaluated
    with _slot(gate, gpu=False):
        _open_stage(tracker, layout, model, "card")
        card = _card(plan, model, host, repo, layout, evaluated, prepared, imatrix)
        tracker.advance(model.id, "card", share)
        if target == 5:
            return card
        _open_stage(tracker, layout, model, "publish-prep")
        drafted = _publish_prep(model, layout, card)
        tracker.advance(model.id, "publish-prep", share)
        if target == 6:
            return drafted
        _open_stage(tracker, layout, model, "upload")
        _delete_intermediates(work)
        with _beat(layout, model, "upload"):
            uploaded = _upload(model, host, layout, card, imatrix)
    tracker.advance(model.id, "upload", share)
    return uploaded


def default_ablate(source: Path, dest: Path, weight_bytes: int, free_vram: int | None) -> dict[str, object]:
    root = data_dir()
    result = abliterate_directory(
        source,
        dest,
        read_lines(root / "harmful.txt", 32),
        read_lines(root / "harmless.txt", 32),
        weight_bytes=weight_bytes,
        free_vram_bytes=free_vram,
    )
    return {
        "layer": result.layer,
        "matrices": result.matrices,
        "mode": result.mode,
        "direction_norm": result.direction_norm,
    }


def _reserve_disk(plan: Plan, model: ModelPlan, work: Path) -> None:
    del plan
    need = sum(item.size for item in model.files)
    if model.mode == "ablate":
        need += sum(item.size for item in model.source_files()) * 2
    ensure_free(work, need, model.id)


def _fetch(model: ModelPlan, host: Host, layout: Layout) -> dict[str, object]:
    marker = layout.stage(model.id, "fetch")
    cached = read_marker(marker)
    if cached is not None:
        return cached
    folder = layout.model_dir(model.id) / "source"
    files: list[dict[str, object]] = []
    for pinned in model.files:
        if pinned.role == "reuse":
            continue
        dest = folder / Path(pinned.path).name
        digest = ensure_pinned(host.downloader, pinned, dest)
        files.append(
            {
                "role": pinned.role,
                "path": str(dest),
                "sha256": digest,
                "quant": pinned.quant,
                "name": Path(pinned.path).name,
            }
        )
    payload: dict[str, object] = {"files": files}
    write_marker(marker, payload)
    return payload


def _prepare(model: ModelPlan, host: Host, layout: Layout) -> dict[str, object]:
    marker = layout.stage(model.id, "abliterate")
    cached = read_marker(marker)
    if cached is not None:
        return cached
    work = layout.model_dir(model.id)
    if model.mode == "requant":
        payload = {"skipped": True, "gguf": str(_source_gguf(model, work)), "ablation": None}
        write_marker(marker, payload)
        return payload
    if model.mode != "ablate":
        raise StageError(f"unhandled mode {model.mode}", code="abliterate_mode")
    source = work / "source"
    dest = work / "ablated"
    weight = sum(item.size for item in model.source_files())
    ablate = host.ablate or default_ablate
    info = ablate(source, dest, weight, host.free_vram())
    gguf = work / "f16.gguf"
    if not gguf.is_file():
        host.runner.run(convert_argv(convert_script(), dest, gguf))
    if not gguf.is_file():
        raise StageError("convert did not write f16.gguf", code="convert")
    payload = {"skipped": False, "gguf": str(gguf), "ablation": info}
    write_marker(marker, payload)
    return payload


def _source_gguf(model: ModelPlan, work: Path) -> Path:
    sources = list(model.source_files())
    if not sources:
        raise StageError(f"{model.id} has no source file", code="plan_source")
    preferred = [item for item in sources if item.quant.upper() in {"F16", "BF16"}]
    chosen = (preferred or sources)[0]
    return work / "source" / Path(chosen.path).name


def _imatrix_corpus(layout: Layout, model_id: str) -> Path:
    """Default corpus is the pinned gist file. FORGE_IMATRIX_RU=1 appends the public-domain slice."""
    base = ensure_calibration()
    if os.environ.get("FORGE_IMATRIX_RU", "").strip() != "1":
        return base
    extra = data_dir() / "imatrix-ru-pd.txt"
    if not extra.is_file():
        return base
    combined = layout.model_dir(model_id) / "imatrix-corpus.txt"
    combined.parent.mkdir(parents=True, exist_ok=True)
    combined.write_text(
        base.read_text(encoding="utf-8") + "\n" + extra.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return combined


def _imatrix(plan: Plan, model: ModelPlan, host: Host, layout: Layout, gguf: Path) -> dict[str, object]:
    marker = layout.stage(model.id, "imatrix")
    dest = layout.model_dir(model.id) / "imatrix.gguf"
    cached = read_marker(marker)
    if cached is not None and dest.is_file():
        return cached
    if not dest.is_file():
        host.runner.run(
            imatrix_argv(
                tool("llama-imatrix"),
                gguf,
                _imatrix_corpus(layout, model.id),
                dest,
                chunks=plan.imatrix_chunks,
                ctx=plan.imatrix_ctx,
            )
        )
    if not dest.is_file():
        raise StageError("imatrix was not written", code="imatrix")
    digest = sha256_file(dest)
    write_sha256(dest, digest)
    payload: dict[str, object] = {"path": str(dest), "sha256": digest, "size": dest.stat().st_size}
    write_marker(marker, payload)
    return payload


def _quantize(
    model: ModelPlan,
    host: Host,
    layout: Layout,
    gguf: Path,
    imatrix: Path,
) -> dict[str, object]:
    marker = layout.stage(model.id, "quantize")
    cached = read_marker(marker)
    if cached is not None:
        return cached
    produced: list[dict[str, object]] = []
    for tier in model.tiers:
        if tier.kind != "quantize":
            continue
        for quant in tier.quants:
            dest = layout.model_dir(model.id) / "quants" / tier.name / f"{quant}.gguf"
            _run_quantize(host, gguf, imatrix, dest, quant)
            produced.append(_file_record(dest, tier.name, quant))
    payload: dict[str, object] = {"files": produced}
    write_marker(marker, payload)
    return payload


def _file_record(dest: Path, tier: str, quant: str) -> dict[str, object]:
    digest = sha256_file(dest)
    sidecar = write_sha256(dest, digest)
    return {
        "path": str(dest),
        "sha256": digest,
        "size": dest.stat().st_size,
        "tier": tier,
        "quant": quant,
        "sha_file": str(sidecar),
    }


def _run_quantize(host: Host, gguf: Path, imatrix: Path, dest: Path, quant: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    sidecar = Path(str(dest) + ".sha256")
    # A failed llama-quantize leaves a partial GGUF. The next attempt must not keep it.
    if dest.is_file() and sidecar.is_file():
        return
    if dest.exists():
        dest.unlink()
    try:
        host.runner.run(quantize_argv(tool("llama-quantize"), gguf, dest, quant, imatrix))
    except StageError:
        if dest.exists():
            dest.unlink()
        raise
    if not dest.is_file():
        raise StageError(f"quantize did not write {dest.name}", code="quantize")
    write_sha256(dest, sha256_file(dest))


def _fit(
    plan: Plan,
    model: ModelPlan,
    host: Host,
    layout: Layout,
    gguf: Path,
    imatrix: Path,
    budgets: set[float] | None,
) -> dict[str, object]:
    marker = layout.stage(model.id, "fit")
    cached = read_marker(marker)
    if cached is not None:
        return cached
    tiers: list[dict[str, object]] = []
    for tier in model.tiers:
        if budgets and tier.budget_gb not in budgets:
            continue
        selected = _select_tier(plan, model, tier, host, layout, gguf, imatrix)
        tiers.append(selected)
    if not tiers:
        raise StageError(f"{model.id} produced no tier", code="fit_empty")
    payload: dict[str, object] = {"tiers": tiers}
    write_marker(marker, payload)
    return payload


def _select_tier(
    plan: Plan,
    model: ModelPlan,
    tier: Tier,
    host: Host,
    layout: Layout,
    gguf: Path,
    imatrix: Path,
) -> dict[str, object]:
    contexts = fit_contexts(tier, plan)
    if tier.kind == "reuse":
        chosen = _select_reuse(plan, model, tier, host, layout, contexts)
        if chosen is not None:
            return chosen
        log(f"{model.id} {tier.name}: no reused GGUF fit; quantizing {list(tier.quants)}")
    elif tier.kind != "quantize":
        raise StageError(f"unhandled tier kind {tier.kind}", code="plan_tier")
    candidates = _quant_candidates(model, tier, host, layout, gguf, imatrix)
    return _first_measured(plan, model, tier, host, candidates, contexts)


def _select_reuse(
    plan: Plan,
    model: ModelPlan,
    tier: Tier,
    host: Host,
    layout: Layout,
    contexts: tuple[int, ...],
) -> dict[str, object] | None:
    pins = [item for item in model.reuse_files() if item.quant in tier.quants]
    rough = [
        Candidate(item.quant, item.size, layout.model_dir(model.id) / "reuse" / Path(item.path).name, "reuse")
        for item in pins
    ]
    _cache, fitting = choose_cache(
        rough,
        model.arch,
        contexts,
        budget_gb=tier.budget_gb,
        headroom_gb=plan.headroom_gb,
    )
    ordered = []
    for cand in fitting:
        pin = next(item for item in pins if item.quant == cand.quant and item.size == cand.weights)
        ordered.append(pin)
    for pin in ordered:
        dest = layout.model_dir(model.id) / "reuse" / Path(pin.path).name
        ensure_pinned(host.downloader, pin, dest)
        candidate = Candidate(pin.quant, dest.stat().st_size, dest, "reuse")
        cache, still = choose_cache(
            [candidate],
            model.arch,
            contexts,
            budget_gb=tier.budget_gb,
            headroom_gb=plan.headroom_gb,
        )
        if not still:
            continue
        measured = _measure(plan, tier, host, candidate, cache)
        if _gate_ok(measured, contexts):
            return _tier_payload(tier, candidate, cache, measured, plan.headroom_gb, contexts)
    return None


def _quant_candidates(
    model: ModelPlan,
    tier: Tier,
    host: Host,
    layout: Layout,
    gguf: Path,
    imatrix: Path,
) -> list[Candidate]:
    found: list[Candidate] = []
    for quant in tier.quants:
        dest = layout.model_dir(model.id) / "quants" / tier.name / f"{quant}.gguf"
        if not dest.is_file():
            _run_quantize(host, gguf, imatrix, dest, quant)
        found.append(Candidate(quant, dest.stat().st_size, dest, "quantize"))
    return found


def _first_measured(
    plan: Plan,
    model: ModelPlan,
    tier: Tier,
    host: Host,
    candidates: list[Candidate],
    contexts: tuple[int, ...],
) -> dict[str, object]:
    cache, fitting = choose_cache(
        candidates,
        model.arch,
        contexts,
        budget_gb=tier.budget_gb,
        headroom_gb=plan.headroom_gb,
    )
    for candidate in fitting:
        measured = _measure(plan, tier, host, candidate, cache)
        if _gate_ok(measured, contexts):
            return _tier_payload(tier, candidate, cache, measured, plan.headroom_gb, contexts)
    raise StageError(f"{model.id} tier {tier.name} has no quant inside the budget", code="fit_miss")


def _measure(
    plan: Plan,
    tier: Tier,
    host: Host,
    candidate: Candidate,
    cache: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ctx in plan.contexts:
        argv = server_argv(
            tool("llama-server"),
            candidate.path,
            ctx=ctx,
            port=_free_port(),
            cache=cache,
        )
        done = host.runner.capture_until(argv, pattern="KV buffer size")
        report = parse_buffers(done.stdout + done.stderr)
        fits = measurement_fits(
            report.weights_bytes,
            report.kv_bytes,
            budget_gb=tier.budget_gb,
            headroom_gb=plan.headroom_gb,
        )
        rows.append(
            {
                "ctx": ctx,
                "cache": cache,
                "weights_bytes": report.weights_bytes,
                "kv_bytes": report.kv_bytes,
                "devices": report.devices,
                "fits": fits,
                "nvidia_used_bytes": host.nvidia_used(),
            }
        )
    return rows


def _gate_ok(rows: list[dict[str, object]], contexts: tuple[int, ...]) -> bool:
    by_ctx = {int(row["ctx"]): row for row in rows}
    return all(bool(by_ctx[ctx]["fits"]) for ctx in contexts)


def _tier_payload(
    tier: Tier,
    candidate: Candidate,
    cache: str,
    measured: list[dict[str, object]],
    headroom_gb: float,
    contexts: tuple[int, ...],
) -> dict[str, object]:
    digest = sha256_file(candidate.path)
    write_sha256(candidate.path, digest)
    gated = [row for row in measured if int(row["ctx"]) in contexts]
    peak = max(int(row["weights_bytes"]) + int(row["kv_bytes"]) for row in gated)
    return {
        "name": tier.name,
        "budget_gb": tier.budget_gb,
        "quant": candidate.quant,
        "origin": candidate.origin,
        "path": str(candidate.path),
        "sha256": digest,
        "size": candidate.path.stat().st_size,
        "cache": cache,
        "min_vram_gb": tier_min_vram(peak, 0, headroom_gb),
        "measurements": measured,
    }


def _eval(
    plan: Plan,
    model: ModelPlan,
    host: Host,
    layout: Layout,
    fitted: dict[str, object],
) -> dict[str, object]:
    marker = layout.stage(model.id, "eval")
    cached = read_marker(marker)
    if cached is not None:
        return cached
    tiers = _tier_rows(fitted)
    evaluated: list[dict[str, object]] = []
    for tier in tiers:
        stats = evaluate_model(
            host.runner,
            Path(str(tier["path"])),
            cache=str(tier["cache"]),
            refusal_limit=plan.refusal_limit,
            ppl_chunks=plan.ppl_chunks,
            prose_samples=plan.prose_samples,
        )
        merged = dict(tier)
        merged.update(stats)
        evaluated.append(merged)
    payload: dict[str, object] = {"tiers": evaluated}
    write_marker(marker, payload)
    return payload


def _card(
    plan: Plan,
    model: ModelPlan,
    host: Host,
    repo: Path,
    layout: Layout,
    evaluated: dict[str, object],
    prepared: dict[str, object],
    imatrix: dict[str, object],
) -> dict[str, object]:
    del plan
    marker = layout.stage(model.id, "card")
    cached = read_marker(marker)
    if cached is not None:
        return cached
    tiers = _tier_rows(evaluated)
    ablation = prepared.get("ablation")
    ablation_dict = ablation if isinstance(ablation, dict) else None
    text = render_card(
        repo,
        model,
        tiers,
        ablation=ablation_dict,
        imatrix_note=(
            f"Importance matrix {Path(str(imatrix['path'])).name} "
            f"sha256 {imatrix['sha256']}, built with llama.cpp from the same source."
        ),
        stamped=host.stamped or None,
    )
    path = layout.model_dir(model.id) / "README.md"
    path.write_text(text, encoding="utf-8")
    digest = sha256_file(path)
    write_sha256(path, digest)
    payload: dict[str, object] = {
        "path": str(path),
        "sha256": digest,
        "size": path.stat().st_size,
        "tiers": tiers,
    }
    write_marker(marker, payload)
    return payload


def _publish_prep(model: ModelPlan, layout: Layout, card: dict[str, object]) -> dict[str, object]:
    marker = layout.stage(model.id, "publish-prep")
    cached = read_marker(marker)
    if cached is not None:
        return cached
    tiers = _tier_rows(card)
    draft = {
        "id": model.id,
        "base_model": model.base_model,
        "revision": model.revision,
        "licence": model.licence,
        "files": [
            {
                "name": Path(str(tier["path"])).name,
                "sha256": tier["sha256"],
                "size": tier["size"],
                "tier": tier["name"],
                "quant": tier["quant"],
                "min_vram_gb": tier["min_vram_gb"],
                "sources": {"hf": "", "kaggle": "", "torrent": "placeholder"},
            }
            for tier in tiers
        ],
    }
    path = layout.model_dir(model.id) / "registry-draft.json"
    path.write_text(json.dumps(draft, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload: dict[str, object] = {"path": str(path), "draft": draft, "tiers": tiers, "card": card}
    write_marker(marker, payload)
    return payload


def _upload(
    model: ModelPlan,
    host: Host,
    layout: Layout,
    card: dict[str, object],
    imatrix: dict[str, object],
) -> dict[str, object]:
    marker = layout.stage(model.id, "upload")
    cached = read_marker(marker)
    if cached is not None:
        body = dict(cached)
        body.setdefault("state", "UPLOADED")
        return body
    tiers = _tier_rows(card)
    artifacts = _artifacts(layout.model_dir(model.id), tiers, card, imatrix)
    if defer_requested(False):
        payload = {
            "id": model.id,
            "base_model": model.base_model,
            "revision": model.revision,
            "licence": model.licence,
            "skipped": False,
            "deferred_upload": True,
            "state": "READY",
            "files": [
                {"name": item.name, "sha256": item.sha256, "size": item.size, "tier": item.tier}
                for item in artifacts
            ],
            "tiers": tiers,
        }
        write_marker(layout.stage(model.id, "upload-deferred"), payload)
        update_item(layout.root, model.id, stage="upload", pct=100, eta_min=0, state="READY")
        return payload
    date = host.stamped or datetime.now(timezone.utc).date().isoformat()
    try:
        records = publish_private(
            host.remote,
            artifacts,
            model_id=model.id,
            title=f"Voxprint {model.id}",
            description=Path(str(card["path"])).read_text(encoding="utf-8"),
            attempts=host.attempts,
            sleep=host.sleep,
            work=layout.model_dir(model.id) / "upload-staging",
        )
    except ForgeError as exc:
        try:
            records = upload_drive(artifacts, model_id=model.id, date=date)
        except ForgeError as drive_exc:
            raise StageError(
                f"HF/Kaggle failed ({exc}); Drive failed ({drive_exc})",
                code="upload",
            ) from drive_exc
    vram = {str(tier["name"]): tier["min_vram_gb"] for tier in tiers}
    for record in records:
        tier_name = str(record.get("tier", ""))
        if record.get("kind") == "gguf" and tier_name in vram:
            record["min_vram_gb"] = vram[tier_name]
    payload = {
        "id": model.id,
        "base_model": model.base_model,
        "revision": model.revision,
        "licence": model.licence,
        "skipped": False,
        "state": "UPLOADED",
        "files": records,
        "tiers": tiers,
    }
    write_marker(marker, payload)
    deferred = layout.stage(model.id, "upload-deferred")
    if deferred.is_file():
        deferred.unlink()
    update_item(layout.root, model.id, stage="upload", pct=100, eta_min=0, state="UPLOADED")
    return payload


def _artifacts(
    work: Path,
    tiers: list[dict[str, object]],
    card: dict[str, object],
    imatrix: dict[str, object],
) -> list[Artifact]:
    del work
    artifacts: list[Artifact] = []
    imatrix_path = Path(str(imatrix["path"]))
    artifacts.append(
        Artifact(
            imatrix_path,
            imatrix_path.name,
            str(imatrix["sha256"]),
            int(imatrix["size"]),
            "shared",
            "",
            "imatrix",
        )
    )
    artifacts.append(_sha_artifact(imatrix_path, "shared"))
    card_path = Path(str(card["path"]))
    artifacts.append(
        Artifact(card_path, card_path.name, str(card["sha256"]), int(card["size"]), "shared", "", "card")
    )
    artifacts.append(_sha_artifact(card_path, "shared"))
    for tier in tiers:
        path = Path(str(tier["path"]))
        artifacts.append(
            Artifact(
                path,
                path.name,
                str(tier["sha256"]),
                int(tier["size"]),
                str(tier["name"]),
                str(tier["quant"]),
                "gguf",
            )
        )
        artifacts.append(_sha_artifact(path, str(tier["name"])))
    return artifacts


def _sha_artifact(path: Path, tier: str) -> Artifact:
    sidecar = Path(str(path) + ".sha256")
    if not sidecar.is_file():
        write_sha256(path)
    return Artifact(sidecar, sidecar.name, sha256_file(sidecar), sidecar.stat().st_size, tier, "", "sha256")


def _tier_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    raw = payload.get("tiers")
    if not isinstance(raw, list) or not raw:
        nested = payload.get("card")
        if isinstance(nested, dict):
            return _tier_rows(nested)
        raise StageError("stage result has no tiers", code="stage_shape")
    rows: list[dict[str, object]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise StageError("tier result is not an object", code="stage_shape")
        rows.append(item)
    return rows


def _delete_intermediates(work: Path) -> None:
    for relative in ("source", "ablated", "f16.gguf", "offload"):
        path = work / relative
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
