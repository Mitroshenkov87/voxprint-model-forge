"""Unattended laptop smoke for the Windows vitoom wheel.

The PowerShell entry is ``scripts/smoke_laptop.ps1``. This module is what that
script runs after it has installed torch cu130 and the pinned cp312 wheel.
Importing it does not import diffusers. The GPU steps import that stack on
first use so a CPU test process stays free of it.

Windows cannot run the Linux deepcompressor quant entry. That substep is
recorded as ``skip: deepcompressor linux-only`` and the tiny checkpoint is
built in-process instead.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Never

import torch

from voxprint_forge.diffusion.nunchaku_edit import (
    ARMS,
    NEGATIVE_PROMPT,
    QWEN_BASE_REPO,
    QWEN_BASE_REVISION,
    QWEN_CASES,
    QWEN_STEPS,
    SEED,
    TRUE_CFG,
    TURBO_GUIDANCE,
    WHEEL_REPO,
    WHEEL_REVISION,
    WINDOWS_WHEEL,
    ZIMAGE_BASE_REPO,
    ZIMAGE_BASE_REVISION,
    ZIMAGE_CASES,
    ZIMAGE_STEPS,
    asset_dir,
)
from voxprint_forge.diffusion.smoke_prefetch import DownloadError
from voxprint_forge.errors import UsageError

SCHEMA = "voxprint-forge-smoke-laptop/1"
DEEPCOMPRESSOR_WINDOWS_SKIP = "skip: deepcompressor linux-only"
TIME_CAP_SKIP = "skip: time cap"
DEFAULT_MINUTES = 90
MIN_CC = (8, 6)
MIN_VRAM_MIB = 14_000
FITS_16GB_MIB = 15.0 * 1024
StepId = Literal["smoke-tiny", "quantfunc", "zimage", "calib-quant-timing"]
STEP_ORDER: tuple[StepId, ...] = ("smoke-tiny", "quantfunc", "zimage", "calib-quant-timing")
Runner = Callable[[], dict[str, object]]
WHEEL_URL = f"https://huggingface.co/{WHEEL_REPO}/resolve/{WHEEL_REVISION}/{WINDOWS_WHEEL.filename}"


def parse_steps(text: str) -> tuple[StepId, ...]:
    """Return the requested steps in smoke order. Empty means every step."""
    raw = [item.strip() for item in text.split(",") if item.strip()]
    if not raw:
        return STEP_ORDER
    unknown = [item for item in raw if item not in STEP_ORDER]
    if unknown:
        raise UsageError(f"unknown smoke step {unknown[0]}", code="usage")
    selected = [step for step in STEP_ORDER if step in raw]
    if not selected:
        raise UsageError("no smoke steps selected", code="usage")
    return tuple(selected)


def deepcompressor_skip(platform: str) -> str | None:
    """The Linux deepcompressor entry does not run on Windows."""
    if platform == "win32":
        return DEEPCOMPRESSOR_WINDOWS_SKIP
    return None


def max_abs_rel(reference: object, actual: object) -> tuple[float, float]:
    """Maximum absolute error and maximum relative error of two tensors."""
    if not isinstance(reference, torch.Tensor) or not isinstance(actual, torch.Tensor):
        raise TypeError("max_abs_rel expects two tensors")
    if reference.shape != actual.shape:
        raise ValueError(f"shape {tuple(reference.shape)} != {tuple(actual.shape)}")
    if reference.numel() == 0:
        raise ValueError("empty tensor")
    delta = (reference.detach().float().cpu() - actual.detach().float().cpu()).abs()
    max_abs = float(delta.max())
    denom = reference.detach().float().cpu().abs().clamp_min(1e-8)
    max_rel = float((delta / denom).max())
    return max_abs, max_rel


def errors_finite(max_abs: float, max_rel: float) -> bool:
    """True when both reported errors are finite numbers."""
    return math.isfinite(max_abs) and math.isfinite(max_rel)


def suitable_gpu(memory_mib: int, major: int, minor: int) -> bool:
    """True for a 16 GB-class GPU at compute capability 8.6 or newer."""
    if (major, minor) < MIN_CC:
        return False
    return memory_mib >= MIN_VRAM_MIB


def parse_gpu_csv(line: str) -> dict[str, object]:
    """Parse one ``nvidia-smi`` csv row: name, memory MiB, driver, compute cap."""
    parts = [item.strip() for item in line.split(",")]
    if len(parts) < 4 or "." not in parts[3]:
        raise ValueError(f"unusable nvidia-smi row {line!r}")
    major_text, minor_text = parts[3].split(".", 1)
    return {
        "name": parts[0],
        "memory_mib": int(float(parts[1])),
        "driver": parts[2],
        "compute_major": int(major_text),
        "compute_minor": int(minor_text),
        "compute_capability": f"{int(major_text)}.{int(minor_text)}",
    }


def run_plan(
    selected: tuple[StepId, ...],
    runners: dict[StepId, Runner],
    *,
    deadline: float,
    now: Callable[[], float],
) -> list[dict[str, object]]:
    """Run each step. A failure is recorded and the next step still runs."""
    records: list[dict[str, object]] = []
    for step_id in selected:
        if now() >= deadline:
            records.append(_skipped(step_id, TIME_CAP_SKIP))
            continue
        started = now()
        try:
            record = runners[step_id]()
        except DownloadError:
            raise
        except Exception as exc:
            record = {
                "id": step_id,
                "status": "fail",
                "error": f"{type(exc).__name__}: {exc}",
                "notes": [],
                "images": [],
            }
        if "id" not in record:
            record["id"] = step_id
        if "seconds" not in record:
            record["seconds"] = round(now() - started, 3)
        records.append(record)
    return records


def exit_code(records: list[dict[str, object]]) -> int:
    """0 when every step passed. 1 when any step failed or was skipped."""
    if records and all(item.get("status") == "pass" for item in records):
        return 0
    return 1


def result_document(
    *,
    records: list[dict[str, object]],
    code: int,
    results_dir: Path,
    minutes_cap: float,
    seconds_cap: float,
    selected: tuple[StepId, ...],
    gpu: dict[str, object] | None,
    versions: dict[str, str],
    started: str,
    error: str | None = None,
    nunchaku: dict[str, object] | None = None,
    extrapolation: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the ``voxprint-forge-smoke-laptop/1`` document."""
    return {
        "schema": SCHEMA,
        "ok": code == 0,
        "exit_code": code,
        "error": error,
        "started": started,
        "results_dir": str(results_dir),
        "minutes_cap": minutes_cap,
        "seconds_cap": seconds_cap,
        "steps_requested": list(selected),
        "versions": versions,
        "nunchaku": nunchaku,
        "gpu": gpu,
        "extrapolation": extrapolation,
        "steps": records,
    }


def write_result(path: Path, payload: dict[str, object]) -> None:
    """Atomically replace ``result.json``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    """Run the laptop smoke or the Linux QuantFunc preflight."""
    parser = argparse.ArgumentParser(description="Laptop smoke or H100 preflight.")
    parser.add_argument("command", choices=("run", "preflight"))
    parser.add_argument("--results", type=Path, default=None)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--steps", default=",".join(STEP_ORDER))
    parser.add_argument("--minutes", type=float, default=DEFAULT_MINUTES)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--forge-out", type=Path, default=None)
    parser.add_argument("--quantfunc", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            return _preflight(args)
        if args.command == "run":
            return _run(args)
    except UsageError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except DownloadError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    unexpected: Never = args.command
    raise UsageError(f"unknown command {unexpected}", code="usage")


def _preflight(args: argparse.Namespace) -> int:
    repo = args.repo
    quantfunc = args.quantfunc
    forge_out = args.forge_out
    if repo is None or quantfunc is None or forge_out is None:
        print("preflight needs --repo, --quantfunc, and --forge-out", file=sys.stderr)
        return 2
    weight = Path(quantfunc) / ARMS["qwen-r128"].filename
    if not weight.is_file():
        print(f"missing QuantFunc weight {weight}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("preflight needs a CUDA GPU", file=sys.stderr)
        return 3
    results = Path(forge_out) / "preflight"
    results.mkdir(parents=True, exist_ok=True)
    tiny = _step_smoke_tiny(_Ctx(repo=Path(repo), results=results, cache=results, deadline=time.monotonic() + 600))
    _print_step(tiny)
    if tiny.get("status") != "pass":
        print("preflight smoke-tiny failed. The long quant was not started.", file=sys.stderr)
        return 1
    try:
        _load_quantfunc_transformer(weight)
    except Exception as exc:
        print(f"preflight QuantFunc load failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("PREFLIGHT-OK")
    return 0


def _run(args: argparse.Namespace) -> int:
    if args.results is None or args.repo is None or args.cache is None:
        print("run needs --results, --repo, and --cache", file=sys.stderr)
        return 2
    if int(args.blocks) < 1:
        print("blocks must be at least 1", file=sys.stderr)
        return 2
    selected = parse_steps(str(args.steps))
    seconds_cap = float(args.minutes) * 60.0 if args.seconds is None else float(args.seconds)
    results = Path(args.results)
    results.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    gpu, gpu_error = _query_gpu()
    nunchaku = probe_nunchaku()
    versions = _versions(nunchaku)
    path = results / "result.json"
    previous = _load_previous(args.resume_from)

    def _store(
        partial: list[dict[str, object]],
        code: int,
        error: str | None = None,
        *,
        deadline_seconds: float = seconds_cap,
    ) -> None:
        write_result(
            path,
            result_document(
                records=partial,
                code=code,
                results_dir=results,
                minutes_cap=deadline_seconds / 60.0,
                seconds_cap=deadline_seconds,
                selected=selected,
                gpu=gpu,
                versions=versions,
                started=started,
                error=error,
                nunchaku=nunchaku,
                extrapolation=_extrapolation(partial),
            ),
        )

    if gpu_error is not None or gpu is None or not suitable_gpu(
        int(gpu["memory_mib"]),
        int(gpu["compute_major"]),
        int(gpu["compute_minor"]),
    ):
        reason = gpu_error or "GPU is below compute capability 8.6 or 14000 MiB"
        _store([], 3, reason)
        print(reason, file=sys.stderr)
        print(f"RESULTS {results}")
        return 3
    from voxprint_forge.diffusion.smoke_timing import reuse_step

    pending: list[StepId] = []
    carried: dict[StepId, dict[str, object]] = {}
    for step_id in selected:
        reused = reuse_step(previous, step_id, force=bool(args.force))
        if reused is None:
            pending.append(step_id)
        else:
            carried[step_id] = reused
    try:
        if pending:
            from voxprint_forge.diffusion.smoke_prefetch import prefetch_weights

            prefetch_weights(Path(args.cache), pending)
    except DownloadError as exc:
        carried_records = [carried[step_id] for step_id in selected if step_id in carried]
        _store(carried_records, 2, str(exc))
        print(str(exc), file=sys.stderr)
        print(f"RESULTS {results}")
        return 2
    os.environ["HF_HUB_OFFLINE"] = "1"
    deadline = time.monotonic() + seconds_cap
    ctx = _Ctx(
        repo=Path(args.repo),
        results=results,
        cache=Path(args.cache),
        deadline=deadline,
        blocks=int(args.blocks),
    )
    runners: dict[StepId, Runner] = {
        "smoke-tiny": lambda: _step_smoke_tiny(ctx),
        "quantfunc": lambda: _step_quantfunc(ctx),
        "zimage": lambda: _step_zimage(ctx),
        "calib-quant-timing": lambda: _step_calib_quant(ctx),
    }

    def clock() -> float:
        return time.monotonic()

    records: list[dict[str, object]] = []
    for step_id in selected:
        reused = carried.get(step_id)
        if reused is not None:
            records.append(reused)
            _print_step(reused)
            _store(records, exit_code(records) if len(records) == len(selected) else 1)
            continue
        if clock() >= deadline:
            records.append(_skipped(step_id, TIME_CAP_SKIP))
            _store(records, 1)
            continue
        try:
            batch = run_plan((step_id,), runners, deadline=deadline, now=clock)
        except DownloadError as exc:
            records.append(
                {
                    "id": step_id,
                    "status": "fail",
                    "seconds": 0.0,
                    "error": str(exc),
                    "notes": [str(exc)],
                    "images": [],
                }
            )
            _store(records, 2, str(exc))
            print(str(exc), file=sys.stderr)
            print(f"RESULTS {results}")
            return 2
        records.extend(batch)
        _print_step(batch[0])
        _store(records, exit_code(records) if len(records) == len(selected) else 1)
    code = exit_code(records)
    _store(records, code)
    print(f"RESULTS {results}")
    return code


class _Ctx:
    """Paths and deadline for one laptop run."""

    def __init__(self, *, repo: Path, results: Path, cache: Path, deadline: float, blocks: int = 2) -> None:
        self.repo = repo
        self.results = results
        self.cache = cache
        self.deadline = deadline
        self.blocks = blocks


def _skipped(step_id: str, reason: str) -> dict[str, object]:
    return {
        "id": step_id,
        "status": "skip",
        "seconds": 0.0,
        "error": reason,
        "notes": [reason],
        "images": [],
    }


def _versions(nunchaku: dict[str, object] | None = None) -> dict[str, str]:
    """Package versions. diffusers is imported here, on the smoke path only."""
    found: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("torch", "diffusers", "transformers"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            found[name] = "missing"
            continue
        found[name] = str(getattr(module, "__version__", "missing"))
    wheel = None if nunchaku is None else nunchaku.get("version")
    found["nunchaku"] = str(wheel) if wheel else "missing"
    return found


def probe_nunchaku() -> dict[str, object]:
    """Wheel version plus whether the Qwen, zero_cond_t, and smooth_factor paths import."""
    version = "missing"
    try:
        version = importlib.metadata.version("nunchaku")
    except importlib.metadata.PackageNotFoundError:
        version = "missing"
    qwen = False
    zero_cond_t = False
    smooth_factor = False
    try:
        module = importlib.import_module("nunchaku.models.transformers.transformer_qwenimage")
    except ImportError:
        module = None
    if module is not None:
        qwen = hasattr(module, "NunchakuQwenImageTransformer2DModel")
        source_path = getattr(module, "__file__", None)
        if isinstance(source_path, str):
            text = Path(source_path).read_text(encoding="utf-8", errors="replace")
            zero_cond_t = "zero_cond_t" in text
    from voxprint_forge.diffusion.int4.quantfunc_smooth import install_quantfunc_shim

    smooth_factor = install_quantfunc_shim()
    return {
        "version": version,
        "qwen": qwen,
        "zero_cond_t": zero_cond_t,
        "smooth_factor": smooth_factor,
    }


def _load_previous(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    result = path / "result.json" if path.is_dir() else path
    if not result.is_file():
        return None
    payload = json.loads(result.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    return payload


def _extrapolation(records: list[dict[str, object]]) -> dict[str, object] | None:
    for record in records:
        value = record.get("extrapolation")
        if record.get("id") == "calib-quant-timing" and isinstance(value, dict):
            return value
    return None


def _query_gpu() -> tuple[dict[str, object] | None, str | None]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"nvidia-smi failed: {exc}"
    line = proc.stdout.strip().splitlines()
    if proc.returncode != 0 or not line or not line[0].strip():
        detail = proc.stderr.strip() or "nvidia-smi returned no GPU"
        return None, detail
    try:
        return parse_gpu_csv(line[0]), None
    except ValueError as exc:
        return None, str(exc)


def _print_step(record: dict[str, object]) -> None:
    print(f"STEP {record.get('id')} {record.get('status')} {record.get('error') or ''}".rstrip())


def _step_smoke_tiny(ctx: _Ctx) -> dict[str, object]:
    """In-process tiny INT4, then a nunchaku CUDA forward against the float block."""
    notes: list[str] = []
    skip = deepcompressor_skip(sys.platform)
    if skip is not None:
        notes.append(skip)
    started = time.perf_counter()
    # diffusers and nunchaku stay out of the import graph until this step runs.
    from safetensors.torch import save_file

    from voxprint_forge.diffusion.int4.quantfunc_smooth import install_quantfunc_shim
    from voxprint_forge.diffusion.int4.qwen_tiny import (
        TINY_RANK,
        _checkpoint_metadata,
        _checkpoint_tensors,
        _edit_batch,
        _run_tiny,
    )

    _report, model, converted, images, prompts = _run_tiny(ctx.repo, rank=TINY_RANK, device=torch.device("cpu"))
    dest = ctx.results / "tiny.safetensors"
    save_file(_checkpoint_tensors(model, converted), dest, metadata=_checkpoint_metadata(TINY_RANK))
    if not torch.cuda.is_available():
        raise RuntimeError("nunchaku CUDA forward needs a CUDA device")
    model = model.to(device="cuda", dtype=torch.bfloat16)
    prompt = str(prompts[0].get("prompt", "edit"))
    batch = _edit_batch(model, images[0], prompt)
    with torch.no_grad():
        reference = model(**batch).sample.detach()
    if not install_quantfunc_shim():
        raise RuntimeError("nunchaku Qwen transformer is not importable")
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
    from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel

    from voxprint_forge.diffusion.nunchaku_config import bind_qwen_config_signature

    bind_qwen_config_signature(NunchakuQwenImageTransformer2DModel, QwenImageTransformer2DModel)
    loaded = NunchakuQwenImageTransformer2DModel.from_pretrained(
        str(dest),
        device="cuda",
        torch_dtype=torch.bfloat16,
    )
    moved = _move_batch(batch)
    with torch.no_grad():
        output = loaded(**moved)
    sample = output.sample if hasattr(output, "sample") else output[0]
    max_abs, max_rel = max_abs_rel(reference, sample.detach())
    if not errors_finite(max_abs, max_rel):
        raise RuntimeError(f"non-finite kernel error abs={max_abs} rel={max_rel}")
    return {
        "id": "smoke-tiny",
        "status": "pass",
        "seconds": round(time.perf_counter() - started, 3),
        "error": None,
        "notes": notes,
        "images": [],
        "max_abs": max_abs,
        "max_rel": max_rel,
        "checkpoint": "tiny.safetensors",
        "cuda_kernels": True,
    }


def _step_quantfunc(ctx: _Ctx) -> dict[str, object]:
    """Load QuantFunc r128 through the shim and run one c1 edit with cpu offload."""
    started = time.perf_counter()
    case = QWEN_CASES[0]
    weight = _ensure_weight(ctx, "qwen-r128")
    base = _ensure_base(ctx, "qwen-edit-2511", QWEN_BASE_REPO, QWEN_BASE_REVISION)
    image_path = asset_dir() / str(case.image)
    if not image_path.is_file():
        image_path = ctx.repo / "tests" / "assets" / "nunchaku_edit" / str(case.image)
    if not image_path.is_file():
        raise RuntimeError(f"missing c1 image {case.image}")

    def generate() -> object:
        pipe = _load_edit_pipe(weight, base)
        return _one_edit(pipe, image_path, case.prompt, case.width, case.height, QWEN_STEPS, TRUE_CFG)

    image, peak = _peak_mib(generate)
    filename = "c1.webp"
    _save_webp(image, ctx.results / filename)
    elapsed = time.perf_counter() - started
    return {
        "id": "quantfunc",
        "status": "pass",
        "seconds": round(elapsed, 3),
        "seconds_per_image": round(elapsed, 3),
        "error": None,
        "notes": ["offload: model", f"wheel: {WINDOWS_WHEEL.filename}"],
        "images": [filename],
        "peak_vram_mib": peak,
        "fits_16gb": None if peak is None else peak <= FITS_16GB_MIB,
        "case": case.case_id,
    }


def _step_zimage(ctx: _Ctx) -> dict[str, object]:
    """Load Z-Image-Turbo INT4 r128 and save one or two images at the 16 GB budget."""
    started = time.perf_counter()
    weight = _ensure_weight(ctx, "zimage-r128")
    base = _ensure_base(ctx, "z-image-turbo", ZIMAGE_BASE_REPO, ZIMAGE_BASE_REVISION)
    pipe = _load_zimage_pipe(weight, base)
    names: list[str] = []
    peaks: list[float] = []
    generate_seconds = 0.0
    for case in ZIMAGE_CASES[:2]:
        if time.monotonic() >= ctx.deadline and names:
            break
        began = time.perf_counter()

        def generate(prompt: str = case.prompt, width: int = case.width, height: int = case.height) -> object:
            return _one_zimage(pipe, prompt, width, height)

        image, peak = _peak_mib(generate)
        generate_seconds += time.perf_counter() - began
        filename = f"{case.case_id}.webp"
        _save_webp(image, ctx.results / filename)
        names.append(filename)
        if peak is not None:
            peaks.append(peak)
    if not names:
        raise RuntimeError("Z-Image produced no images")
    elapsed = time.perf_counter() - started
    peak = max(peaks) if peaks else None
    return {
        "id": "zimage",
        "status": "pass",
        "seconds": round(elapsed, 3),
        "seconds_per_image": round(generate_seconds / len(names), 3),
        "error": None,
        "notes": ["offload: model"],
        "images": names,
        "peak_vram_mib": peak,
        "fits_16gb": None if peak is None else peak <= FITS_16GB_MIB,
    }


def _ensure_weight(ctx: _Ctx, arm_id: str) -> Path:
    spec = ARMS[arm_id]
    dest = ctx.cache / "weights" / arm_id
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / spec.filename
    if not path.is_file():
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id=spec.repo,
            filename=spec.filename,
            revision=spec.revision,
            local_dir=str(dest),
            token=False,
        )
    digest = _sha256(path)
    if digest != spec.sha256:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"{spec.filename} sha256 {digest} != {spec.sha256}")
    return path


def _ensure_base(ctx: _Ctx, name: str, repo: str, revision: str) -> Path:
    dest = ctx.cache / "weights" / name
    marker = dest / "model_index.json"
    config = dest / "transformer" / "config.json"
    if marker.is_file() and config.is_file():
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo,
        revision=revision,
        ignore_patterns=["transformer/*.safetensors", "transformer/*.bin", "transformer/*.pt"],
        local_dir=str(dest),
        token=False,
    )
    return dest


def _load_quantfunc_transformer(weight: Path) -> object:
    from voxprint_forge.diffusion.int4.quantfunc_smooth import install_quantfunc_shim

    if not install_quantfunc_shim():
        raise RuntimeError("nunchaku Qwen transformer is not importable")
    from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
    from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel

    from voxprint_forge.diffusion.nunchaku_config import bind_qwen_config_signature

    bind_qwen_config_signature(NunchakuQwenImageTransformer2DModel, QwenImageTransformer2DModel)
    return NunchakuQwenImageTransformer2DModel.from_pretrained(str(weight), torch_dtype=torch.bfloat16)


def _load_edit_pipe(weight: Path, base: Path) -> object:
    from voxprint_forge.diffusion.smoke_prefetch import weights_ready

    weights_ready(weight, base)
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline

    transformer = _load_quantfunc_transformer(weight)
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        str(base),
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    _offload_16gb(pipe, transformer)
    return pipe


def _load_zimage_pipe(weight: Path, base: Path) -> object:
    from voxprint_forge.diffusion.smoke_prefetch import weights_ready

    weights_ready(weight, base)
    from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
    from nunchaku.models.transformers.transformer_zimage import NunchakuZImageTransformer2DModel

    transformer = NunchakuZImageTransformer2DModel.from_pretrained(str(weight), torch_dtype=torch.bfloat16)
    pipe = ZImagePipeline.from_pretrained(
        str(base),
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=False,
    )
    _offload_16gb(pipe, transformer)
    return pipe


def _offload_16gb(pipe: object, transformer: object) -> None:
    """Keep the text encoder and VAE on CPU until the transformer file is loaded.

    The caller finishes the transformer ``from_pretrained`` before this runs.
    One block stays on the GPU. The text encoder and VAE move only for a forward.
    """
    if hasattr(transformer, "set_offload"):
        transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)  # type: ignore[attr-defined]
        exclude = getattr(pipe, "_exclude_from_cpu_offload", None)
        if isinstance(exclude, list) and "transformer" not in exclude:
            exclude.append("transformer")
        pipe.enable_sequential_cpu_offload()  # type: ignore[attr-defined]
        return
    pipe.enable_model_cpu_offload()  # type: ignore[attr-defined]


def _step_calib_quant(ctx: _Ctx) -> dict[str, object]:
    """Time a short syh quant on the first N real blocks and extrapolate to 60 and an H100."""
    started = time.perf_counter()
    snapshot = ctx.cache / "weights" / "qwen-edit-2511"
    config = snapshot / "transformer" / "config.json"
    if not config.is_file():
        raise RuntimeError(f"setup: download failed: missing {config}")

    def measure() -> tuple[dict[str, object], list[str]]:
        notes: list[str] = []
        timed: dict[str, object] | None = None
        remaining = max(1.0, ctx.deadline - time.monotonic())
        if torch.cuda.is_available():
            try:
                from voxprint_forge.diffusion.smoke_timing import try_deepcompressor

                timed = try_deepcompressor(
                    ctx.cache,
                    snapshot,
                    ctx.results / "calib-quant",
                    repo=ctx.repo,
                    blocks=ctx.blocks,
                    timeout=remaining,
                )
            except Exception as exc:
                notes.append(f"cpu fallback: deepcompressor could not use CUDA: {type(exc).__name__}: {exc}")
        else:
            notes.append("cpu fallback: deepcompressor could not use CUDA")
        if timed is None:
            from voxprint_forge.diffusion.smoke_timing import CPU_FALLBACK, time_real_blocks

            if not any(str(note).startswith(CPU_FALLBACK) for note in notes):
                notes.append(CPU_FALLBACK)
            timed = time_real_blocks(ctx.repo, snapshot, blocks=ctx.blocks, rank=8)
        return timed, notes

    timed, peak = _peak_mib(measure)
    if not isinstance(timed, tuple):
        raise RuntimeError("calibration timing returned nothing")
    payload, notes = timed
    extrapolation = payload["extrapolation"]
    if not isinstance(extrapolation, dict):
        raise RuntimeError("calibration timing did not report an extrapolation")
    extra_notes = payload.get("notes")
    if isinstance(extra_notes, list):
        notes = [str(item) for item in extra_notes] + notes
    sample_seconds = payload["sample_seconds"]
    block_seconds = payload["block_seconds"]
    per_sample = extrapolation["seconds_per_calibration_sample"]
    per_block = extrapolation["seconds_per_block"]
    return {
        "id": "calib-quant-timing",
        "status": "pass",
        "seconds": round(time.perf_counter() - started, 3),
        "error": None,
        "notes": notes,
        "images": [],
        "blocks": ctx.blocks,
        "seconds_per_calibration_sample": per_sample,
        "seconds_per_block": per_block,
        "sample_seconds": sample_seconds,
        "block_seconds": block_seconds,
        "peak_vram_mib": peak,
        "extrapolation": extrapolation,
    }


def _one_edit(
    pipe: object,
    image_path: Path,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    true_cfg: float,
) -> object:
    from PIL import Image

    source = Image.open(image_path).convert("RGB")
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    output = pipe(  # type: ignore[operator]
        image=source,
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        true_cfg_scale=true_cfg,
        num_inference_steps=steps,
        height=height,
        width=width,
        generator=generator,
        output_type="pil",
    )
    return output.images[0]


def _one_zimage(pipe: object, prompt: str, width: int, height: int) -> object:
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    output = pipe(  # type: ignore[operator]
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=ZIMAGE_STEPS,
        guidance_scale=TURBO_GUIDANCE,
        generator=generator,
    )
    return output.images[0]


def _save_webp(image: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="WEBP", quality=95, method=6)  # type: ignore[attr-defined]


def _move_batch(batch: dict[str, object]) -> dict[str, object]:
    moved: dict[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device="cuda")
        else:
            moved[key] = value
    return moved


def _peak_mib(fn: Callable[[], object]) -> tuple[object, float | None]:
    stop = threading.Event()
    peaks: list[float] = []

    def poll() -> None:
        while not stop.is_set():
            value = _smi_mib()
            if value is not None:
                peaks.append(value)
            stop.wait(0.25)

    thread = threading.Thread(target=poll, name="smoke-smi", daemon=True)
    thread.start()
    try:
        result = fn()
    finally:
        stop.set()
        thread.join(timeout=2)
    return result, (max(peaks) if peaks else None)


def _smi_mib() -> float | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return float(proc.stdout.strip().splitlines()[0].split()[0])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
