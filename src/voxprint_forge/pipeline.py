"""Run fetch, ablation, quantize, fit, eval, card, and upload for one model."""

from __future__ import annotations

import json
import os
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
from voxprint_forge.cards import hub_title, render_card
from voxprint_forge.components import place_weight
from voxprint_forge.diffusion.defer import defer_requested
from voxprint_forge.disk import ensure_free
from voxprint_forge.drive import upload_drive
from voxprint_forge.errors import ForgeError, StageError, UsageError
from voxprint_forge.evaluate import data_dir, evaluate_model, read_lines
from voxprint_forge.fetch import Downloader, ensure_pinned
from voxprint_forge.fitmath import budget_bytes, quant_rank
from voxprint_forge.hashing import sha256_file, write_sha256
from voxprint_forge.layout import Layout, read_marker, write_marker
from voxprint_forge.llama import (
    TAG,
    Runner,
    convert_argv,
    convert_script,
    imatrix_argv,
    quantize_argv,
    server_argv,
    tool,
)
from voxprint_forge.manifest import summary_lines, write_run_manifest
from voxprint_forge.mdl import (
    MANIFEST_NAME,
    build_document,
    card_url_for,
    drive_mirror,
    hub_mirrors,
    manifest_artifact,
    refresh_manifest,
    retarget_mirrors,
    rows_from_artifacts,
    write_manifest,
)
from voxprint_forge.measure import (
    Candidate,
    fit_contexts,
    measurement_fits,
    parse_buffers,
    placement_bytes,
    tier_min_vram,
)
from voxprint_forge.outcome import summarise
from voxprint_forge.placement import Split, options_for, split_fits
from voxprint_forge.plan import ModelPlan, Plan, Tier
from voxprint_forge.preflight import memory_status
from voxprint_forge.progress import Progress
from voxprint_forge.reclaim import hub_cache, reclaim_model_work
from voxprint_forge.schedule import Gate
from voxprint_forge.scrub import log
from voxprint_forge.statusfile import beating, current_stage, set_stage, update_item
from voxprint_forge.upload import (
    Artifact,
    Remote,
    blocks_public_upload,
    hf_user,
    kaggle_model_slug,
    kaggle_user,
    publish_private,
    repo_slug,
    selected_targets,
)

ORDER = ("fetch", "abliterate", "quantize", "fit", "eval", "card", "publish-prep", "upload")


def retry_stage(stage: str) -> str:
    """Stage command that restarts `stage`. `imatrix` restarts at quantize."""
    if stage in {"", "start"}:
        return "fetch"
    if stage == "imatrix":
        return "quantize"
    if stage in ORDER:
        return stage
    raise UsageError(f"unknown stage {stage}", code="usage")


def clear_from_stage(layout: Layout, model_id: str, stage: str) -> list[str]:
    """Delete markers from `stage` through upload so the stage command runs that work again."""
    command = retry_stage(stage)
    start = ORDER.index(command)
    names = list(ORDER[start:])
    if start <= ORDER.index("quantize"):
        names.append("imatrix")
    if "upload" in names:
        names.append("upload-deferred")
    names.append("skipped")
    removed: list[str] = []
    for name in names:
        path = layout.stage(model_id, name)
        if path.is_file():
            path.unlink()
            removed.append(name)
    return removed


def _target_flags(
    records: list[dict[str, object]],
    targets: tuple[str, ...],
    *,
    hub_failed: bool,
    deferred: bool = False,
) -> dict[str, str]:
    if deferred:
        return {"hf": "deferred", "kaggle": "deferred", "drive": "deferred"}
    drive_used = any(row.get("target") == "drive" for row in records)
    flags: dict[str, str] = {}
    for name in ("hf", "kaggle", "drive"):
        if name not in targets:
            flags[name] = "off"
            continue
        if name == "drive":
            flags[name] = "uploaded" if drive_used else "standby"
            continue
        if hub_failed or drive_used:
            flags[name] = "failed"
            continue
        present = False
        for row in records:
            sources = row.get("sources")
            if isinstance(sources, dict) and str(sources.get(name, "")).strip():
                present = True
                break
        flags[name] = "uploaded" if present else "failed"
    return flags


_GPU = frozenset({"abliterate", "fit", "eval"})
_SILENT = frozenset({"imatrix", "quantize", "fit", "eval", "upload"})


def _no_probe() -> int | None:
    return None


@dataclass
class Host:
    """Runner, downloader, and remote used by one pipeline process."""
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
    """Run every enabled model and write the manifest."""
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
    """Run one model from fetch through ``stop_after``."""
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
    _reserve_disk(plan, model, work, layout.root)
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
        with _beat(layout, model, "upload"):
            uploaded = _upload(model, host, layout, card, imatrix, llama_tag=plan.llama_cpp_tag)
    tracker.advance(model.id, "upload", share)
    return uploaded


def default_ablate(source: Path, dest: Path, weight_bytes: int, free_vram: int | None) -> dict[str, object]:
    """Abliterate ``source`` into ``dest`` and return the ablation record."""
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


def _reserve_disk(plan: Plan, model: ModelPlan, work: Path, forge_out: Path) -> None:
    del plan
    need = sum(item.size for item in model.files)
    if model.mode == "ablate":
        need += sum(item.size for item in model.source_files()) * 2
    ensure_free(work, need, model.id, forge_out=forge_out, cache_dir=hub_cache())


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
    ram = memory_status()[1]
    ordered = sorted(pins, key=lambda item: quant_rank(item.quant), reverse=True)
    for pin in ordered:
        if not _declares_fit(plan, model, tier, pin.size, contexts, ram):
            continue
        dest = layout.model_dir(model.id) / "reuse" / Path(pin.path).name
        ensure_pinned(host.downloader, pin, dest)
        candidate = Candidate(pin.quant, dest.stat().st_size, dest, "reuse")
        chosen = _try_candidate(plan, model, tier, host, candidate, contexts, ram)
        if chosen is not None:
            return chosen
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
    ram = memory_status()[1]
    ranked = sorted(candidates, key=lambda item: quant_rank(item.quant), reverse=True)
    for candidate in ranked:
        chosen = _try_candidate(plan, model, tier, host, candidate, contexts, ram)
        if chosen is not None:
            return chosen
    raise StageError(f"{model.id} tier {tier.name} has no quant inside the budget", code="fit_miss")


def _declares_fit(
    plan: Plan,
    model: ModelPlan,
    tier: Tier,
    weights: int,
    contexts: tuple[int, ...],
    ram_bytes: int,
) -> bool:
    for cache in ("q8_0", "q4_0"):
        if _fitting_splits(plan, model, tier, weights, contexts, cache, ram_bytes):
            return True
    return False


def _fitting_splits(
    plan: Plan,
    model: ModelPlan,
    tier: Tier,
    weights: int,
    contexts: tuple[int, ...],
    cache: str,
    ram_bytes: int,
) -> tuple[Split, ...]:
    kv = max(model.arch.kv_bytes(ctx, cache) for ctx in contexts)
    return tuple(
        split
        for split in options_for(
            weights,
            kv,
            moe_layers=model.moe_layers,
            expert_fraction=model.expert_fraction,
            encoder_gib=model.encoder_gib,
        )
        if split_fits(
            split,
            budget_gb=tier.budget_gb,
            headroom_gb=plan.headroom_gb,
            ram_bytes=ram_bytes,
        )
    )


def _try_candidate(
    plan: Plan,
    model: ModelPlan,
    tier: Tier,
    host: Host,
    candidate: Candidate,
    contexts: tuple[int, ...],
    ram_bytes: int,
) -> dict[str, object] | None:
    """Best cache whose split fits. GPU is measured alone when the formula says it fits.

    A measured miss, or a formula miss, tries the later offloads of that same quant
    before the caller moves on to a lower quant. Encoder offload and sequential
    loading have no llama-server flag, so those two are the byte formula.
    """
    for cache in ("q8_0", "q4_0"):
        fitting = list(_fitting_splits(plan, model, tier, candidate.weights, contexts, cache, ram_bytes))
        if not fitting:
            continue
        pending = fitting
        if fitting[0].name == "gpu":
            measured = _measure(plan, tier, host, candidate, cache, ram_budget=ram_bytes)
            if _gate_ok(measured, contexts):
                return _tier_payload(tier, candidate, cache, measured, plan.headroom_gb, contexts, fitting[0])
            pending = fitting[1:]
        for split in pending:
            if not split.flags:
                rows = _formula_rows(plan, cache, split)
                return _tier_payload(tier, candidate, cache, rows, plan.headroom_gb, contexts, split)
            measured = _measure(
                plan,
                tier,
                host,
                candidate,
                cache,
                extra=split.flags,
                split_name=split.name,
                ram_budget=ram_bytes,
            )
            if _gate_ok(measured, contexts):
                return _tier_payload(tier, candidate, cache, measured, plan.headroom_gb, contexts, split)
    return None


def _formula_rows(plan: Plan, cache: str, split: Split) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ctx in plan.contexts:
        rows.append(
            {
                "ctx": ctx,
                "cache": cache,
                "weights_bytes": split.vram_bytes,
                "kv_bytes": 0,
                "devices": {},
                "fits": True,
                "nvidia_used_bytes": None,
                "split": split.name,
                "ram_bytes": split.ram_bytes,
                "measured": False,
            }
        )
    return rows


def _measure(
    plan: Plan,
    tier: Tier,
    host: Host,
    candidate: Candidate,
    cache: str,
    *,
    extra: tuple[str, ...] = (),
    split_name: str = "gpu",
    ram_budget: int = 0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ctx in plan.contexts:
        argv = server_argv(
            tool("llama-server"),
            candidate.path,
            ctx=ctx,
            port=_free_port(),
            cache=cache,
            extra=extra,
        )
        done = host.runner.capture_until(argv, pattern="KV buffer size")
        text = done.stdout + done.stderr
        if split_name == "gpu":
            report = parse_buffers(text)
            weights = report.weights_bytes
            kv = report.kv_bytes
            devices: dict[str, dict[str, float]] = report.devices
            host_bytes = 0
            fits = measurement_fits(
                weights,
                kv,
                budget_gb=tier.budget_gb,
                headroom_gb=plan.headroom_gb,
            )
        else:
            weights, kv, host_bytes = placement_bytes(text)
            devices = {}
            fits = _offload_fits(
                weights,
                kv,
                host_bytes,
                budget_gb=tier.budget_gb,
                headroom_gb=plan.headroom_gb,
                ram_budget=ram_budget,
            )
        rows.append(
            {
                "ctx": ctx,
                "cache": cache,
                "weights_bytes": weights,
                "kv_bytes": kv,
                "devices": devices,
                "fits": fits,
                "nvidia_used_bytes": host.nvidia_used(),
                "split": split_name,
                "ram_bytes": host_bytes,
                "measured": True,
            }
        )
    return rows


def _offload_fits(
    cuda_weights: int,
    cuda_kv: int,
    host_bytes: int,
    *,
    budget_gb: float,
    headroom_gb: float,
    ram_budget: int,
) -> bool:
    if cuda_weights == 0 and cuda_kv == 0 and host_bytes == 0:
        return False
    if cuda_weights + cuda_kv > budget_bytes(budget_gb, headroom_gb):
        return False
    return host_bytes <= ram_budget


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
    split: Split,
) -> dict[str, object]:
    digest = sha256_file(candidate.path)
    write_sha256(candidate.path, digest)
    component = place_weight(candidate.path, digest=digest)
    gated = [row for row in measured if int(row["ctx"]) in contexts]
    peak = max(int(row["weights_bytes"]) + int(row["kv_bytes"]) for row in gated)
    return {
        "name": tier.name,
        "budget_gb": tier.budget_gb,
        "quant": candidate.quant,
        "origin": candidate.origin,
        "path": str(candidate.path),
        "component": str(component) if component is not None else "",
        "sha256": digest,
        "size": candidate.path.stat().st_size,
        "cache": cache,
        "split": split.name,
        "split_flags": list(split.flags),
        "ram_bytes": split.ram_bytes,
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
                "split": tier.get("split", "gpu"),
                "split_flags": list(tier.get("split_flags", [])),
                "ram_bytes": tier.get("ram_bytes", 0),
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
    *,
    llama_tag: str = TAG,
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
        flags = _target_flags([], (), hub_failed=False, deferred=True)
        payload = {
            "id": model.id,
            "base_model": model.base_model,
            "revision": model.revision,
            "licence": model.licence,
            "skipped": False,
            "deferred_upload": True,
            "state": "READY",
            "uploads": flags,
            "files": [
                {"name": item.name, "sha256": item.sha256, "size": item.size, "tier": item.tier}
                for item in artifacts
            ],
            "tiers": tiers,
        }
        write_marker(layout.stage(model.id, "upload-deferred"), payload)
        update_item(layout.root, model.id, stage="upload", pct=100, eta_min=0, state="READY", uploads=flags)
        return payload
    date = host.stamped or datetime.now(timezone.utc).date().isoformat()
    targets = selected_targets()
    hub = tuple(name for name in targets if name in {"hf", "kaggle"})
    records: list[dict[str, object]] = []
    hub_failed = False
    model_dir = layout.model_dir(model.id)

    def _on_verified(target: str) -> None:
        reclaim_model_work(model_dir, drop_staging=target != "hf" or "kaggle" not in hub)

    if hub:
        artifacts = _with_manifest(
            model,
            model_dir,
            artifacts,
            tiers,
            llama_tag=llama_tag,
            mirrors=_hub_mirror_rows(model.id, hub),
            date=date,
        )
        try:
            records = publish_private(
                host.remote,
                artifacts,
                model_id=model.id,
                title=hub_title(model.id),
                description=Path(str(card["path"])).read_text(encoding="utf-8"),
                licence=model.licence,
                attempts=host.attempts,
                sleep=host.sleep,
                work=model_dir / "upload-staging",
                targets=hub,
                on_verified=_on_verified,
            )
        except ForgeError as exc:
            if blocks_public_upload(exc):
                raise
            hub_failed = True
            if "drive" not in targets:
                raise
            try:
                manifest_path = model_dir / MANIFEST_NAME
                retarget_mirrors(manifest_path, [drive_mirror(model.id, date)])
                artifacts = refresh_manifest(artifacts, manifest_path)
                records = upload_drive(artifacts, model_id=model.id, date=date)
            except ForgeError as drive_exc:
                raise StageError(
                    f"HF/Kaggle failed ({exc}); Drive failed ({drive_exc})",
                    code="upload",
                ) from drive_exc
            _on_verified("drive")
    elif "drive" in targets:
        artifacts = _with_manifest(
            model,
            model_dir,
            artifacts,
            tiers,
            llama_tag=llama_tag,
            mirrors=[drive_mirror(model.id, date)],
            date=date,
        )
        records = upload_drive(artifacts, model_id=model.id, date=date)
        _on_verified("drive")
    flags = _target_flags(records, targets, hub_failed=hub_failed)
    by_tier = {str(tier["name"]): tier for tier in tiers}
    for record in records:
        tier_name = str(record.get("tier", ""))
        chosen = by_tier.get(tier_name)
        if record.get("kind") == "gguf" and chosen is not None:
            record["min_vram_gb"] = chosen["min_vram_gb"]
            record["split"] = chosen.get("split", "gpu")
            record["split_flags"] = list(chosen.get("split_flags", []))
            record["ram_bytes"] = chosen.get("ram_bytes", 0)
    payload = {
        "id": model.id,
        "base_model": model.base_model,
        "revision": model.revision,
        "licence": model.licence,
        "skipped": False,
        "state": "UPLOADED",
        "uploads": flags,
        "files": records,
        "tiers": tiers,
    }
    write_marker(marker, payload)
    deferred = layout.stage(model.id, "upload-deferred")
    if deferred.is_file():
        deferred.unlink()
    update_item(layout.root, model.id, stage="upload", pct=100, eta_min=0, state="UPLOADED", uploads=flags)
    return payload


def _with_manifest(
    model: ModelPlan,
    model_dir: Path,
    artifacts: list[Artifact],
    tiers: list[dict[str, object]],
    *,
    llama_tag: str,
    mirrors: list[dict[str, str]],
    date: str,
) -> list[Artifact]:
    budgets = [float(tier["min_vram_gb"]) for tier in tiers]
    flags = ("least-censored",) if model.mode == "ablate" or model.already_abliterated else ()
    document = build_document(
        model_id=model.id,
        display_name=hub_title(model.id),
        app_ids=("plotweaver", "audiobook-builder"),
        task="text-generation",
        license=model.licence,
        files=rows_from_artifacts(artifacts),
        mirrors=mirrors,
        min_vram_gb=min(budgets),
        recommended_vram_gb=max(budgets),
        gpu_min_cc=6.0,
        engine="llama.cpp",
        version_req=llama_tag,
        base_model=model.base_model,
        card_url=card_url_for(mirrors),
        flags=flags,
        on_demand=True,
        date=date,
    )
    path = write_manifest(model_dir, document)
    return [*artifacts, manifest_artifact(path)]


def _hub_mirror_rows(model_id: str, hub: tuple[str, ...]) -> list[dict[str, str]]:
    hf_repo = f"{hf_user()}/{repo_slug(model_id)}" if "hf" in hub else ""
    kaggle_ref = f"{kaggle_user()}/{kaggle_model_slug(model_id)}" if "kaggle" in hub else ""
    return hub_mirrors(hf_repo=hf_repo, kaggle_ref=kaggle_ref)


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


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
