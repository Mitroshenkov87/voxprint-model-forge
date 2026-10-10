"""Seconds-long checks before a download, a quant, or a render."""

from __future__ import annotations

import ctypes
import importlib.util
import os
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from voxprint_forge import disk
from voxprint_forge.errors import StageError
from voxprint_forge.fitmath import Arch, estimate_weights_bytes, gib
from voxprint_forge.llama import bin_dir
from voxprint_forge.llama_patch import T_LITE_HASH, recognised
from voxprint_forge.plan import ModelPlan, Plan

Opener = Callable[..., object]
_BINARIES = ("llama-quantize", "llama-imatrix", "llama-cli", "llama-server")
_IMPORTS = ("torch", "transformers", "gguf")


def _deferring() -> bool:
    return os.environ.get("FORGE_DEFER_UPLOAD", "").strip().lower() in {"1", "true", "yes", "on"}


def missing_imports() -> list[str]:
    return [name for name in _IMPORTS if importlib.util.find_spec(name) is None]


def ram_bytes() -> int:
    if os.name == "nt":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.ullTotalPhys)
    page = os.sysconf("SC_PAGE_SIZE")
    pages = os.sysconf("SC_PHYS_PAGES")
    return int(page) * int(pages)


def _arch(model: ModelPlan) -> Arch:
    arch = model.arch
    return Arch(
        full_layers=arch.full_layers,
        full_kv_heads=arch.full_kv_heads,
        full_head_dim=arch.full_head_dim,
        swa_layers=arch.swa_layers,
        swa_kv_heads=arch.swa_kv_heads,
        swa_head_dim=arch.swa_head_dim,
        swa_window=arch.swa_window,
    )


def ladder_gaps(plan: Plan) -> list[str]:
    """A tier fails when no listed quant's weights + KV fit its budget."""
    ctx = max(plan.contexts) if plan.contexts else 8192
    gaps: list[str] = []
    for model in plan.enabled_models():
        if model.params_b <= 0:
            continue
        kv = _arch(model).kv_bytes(ctx, "q8_0")
        for tier in model.tiers:
            if tier.kind != "quantize":
                continue
            budget = int(tier.budget_gb * (1024**3))
            fits = [
                quant
                for quant in tier.quants
                if estimate_weights_bytes(model.params_b, quant) + kv <= budget
            ]
            if not fits:
                gaps.append(
                    f"{model.id} {tier.name} budget {tier.budget_gb:g} GB fits no quant "
                    f"in {', '.join(tier.quants)} at ctx {ctx}"
                )
    return gaps


def peak_download_bytes(plan: Plan) -> int:
    total = 0
    for model in plan.enabled_models():
        total += sum(item.size for item in model.files)
    return total


def index_targets(plan: Plan) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for model in plan.enabled_models():
        if model.source_format != "safetensors":
            continue
        for item in model.files:
            if item.path == "model.safetensors.index.json":
                found.append((model.id, model.repo, model.revision))
    return found


def _head(url: str, opener: Opener) -> int:
    request = urllib.request.Request(url, method="HEAD")
    try:
        response = opener(request, timeout=8)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except Exception as exc:
        raise StageError(f"HEAD {url} failed: {exc}", code="preflight") from exc
    status = getattr(response, "status", None)
    close = getattr(response, "close", None)
    if callable(close):
        close()
    if status is None:
        return 200
    return int(status)


def converter_recognises_t_lite(repo: Path) -> bool:
    env = os.environ.get("FORGE_LLAMA_SRC", "")
    if env:
        path = Path(env) / "conversion" / "base.py"
        if path.is_file():
            return recognised(path.read_text(encoding="utf-8"))
        return False
    # Sources are applied at build time from this registration.
    return T_LITE_HASH in (repo / "src" / "voxprint_forge" / "llama_patch.py").read_text(encoding="utf-8")


def binaries_ready() -> tuple[bool, str]:
    folder = bin_dir()
    if folder is not None and all((folder / name).is_file() for name in _BINARIES):
        return True, str(folder)
    missing = [name for name in ("cmake", "ninja", "nvcc") if shutil.which(name) is None]
    if not missing:
        return True, "buildable"
    return False, "missing " + ", ".join(missing)


def run_preflight(
    plan: Plan,
    repo: Path,
    forge_out: Path,
    *,
    opener: Opener | None = None,
    ram: Callable[[], int] | None = None,
) -> dict[str, object]:
    problems: list[str] = []
    missing = missing_imports()
    if missing:
        problems.append("converter Python is missing " + ", ".join(missing))
    fetch = opener or urllib.request.urlopen
    indexes: list[dict[str, object]] = []
    for model_id, model_repo, revision in index_targets(plan):
        url = f"https://huggingface.co/{model_repo}/resolve/{revision}/model.safetensors.index.json"
        try:
            status = _head(url, fetch)
        except StageError as exc:
            problems.append(str(exc))
            continue
        indexes.append({"id": model_id, "status": status})
        if status >= 400:
            problems.append(f"{model_id} shard index HEAD {status} at {revision}")
    if any(model.id == "t-lite-it-2.1" for model in plan.enabled_models()):
        if not converter_recognises_t_lite(repo):
            problems.append("T-lite pre-tokenizer hash is not registered on the pinned converter")
    gaps = ladder_gaps(plan)
    problems.extend(gaps)
    need = peak_download_bytes(plan)
    try:
        free = disk.free_bytes(forge_out)
    except OSError as exc:
        problems.append(f"disk check failed: {exc}")
        free = 0
    if free < need:
        problems.append(f"need {gib(need):.1f} GiB for the plan, have {gib(free):.1f} GiB")
    memory = (ram or ram_bytes)()
    resident = 0
    for model in plan.enabled_models():
        if model.params_b <= 0:
            continue
        quants = [quant for tier in model.tiers if tier.kind == "quantize" for quant in tier.quants]
        if quants:
            resident = max(resident, min(estimate_weights_bytes(model.params_b, quant) for quant in quants))
    if resident and memory < resident:
        problems.append(f"need {gib(resident):.1f} GiB RAM for the smallest quant, have {gib(memory):.1f} GiB")
    tokens_ok = _deferring() or (
        bool(os.environ.get("HF_TOKEN", "").strip()) and bool(os.environ.get("KAGGLE_API_TOKEN", "").strip())
    )
    if not tokens_ok:
        problems.append("HF_TOKEN and KAGGLE_API_TOKEN are required unless FORGE_DEFER_UPLOAD=1")
    ready, detail = binaries_ready()
    if not ready:
        problems.append(f"llama.cpp binaries are absent and not buildable ({detail})")
    report = {
        "ok": not problems,
        "problems": problems,
        "imports": [name for name in _IMPORTS if name not in missing],
        "indexes": indexes,
        "disk_free": free,
        "disk_need": need,
        "ram": memory,
        "llama": detail,
        "defer_upload": _deferring(),
    }
    if problems:
        raise StageError("; ".join(problems), code="preflight")
    return report
