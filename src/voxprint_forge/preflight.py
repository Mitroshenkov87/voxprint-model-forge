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
from voxprint_forge.placement import options_for
from voxprint_forge.plan import ModelPlan, Plan

Opener = Callable[..., object]
_BINARIES = ("llama-quantize", "llama-imatrix", "llama-cli", "llama-server")
_IMPORTS = ("torch", "transformers", "gguf")


def _deferring() -> bool:
    return os.environ.get("FORGE_DEFER_UPLOAD", "").strip().lower() in {"1", "true", "yes", "on"}


def missing_imports() -> list[str]:
    """Required converter imports that this interpreter cannot find."""
    return [name for name in _IMPORTS if importlib.util.find_spec(name) is None]


class _MemoryStatus(ctypes.Structure):
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


def memory_status() -> tuple[int, int]:
    """Physical RAM as (total, available) bytes."""
    if os.name == "nt":
        status = _MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return int(status.ullTotalPhys), int(status.ullAvailPhys)
    page = int(os.sysconf("SC_PAGE_SIZE"))
    total = page * int(os.sysconf("SC_PHYS_PAGES"))
    try:
        available = page * int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        available = total
    return total, available


def ram_bytes() -> int:
    """Total physical RAM in bytes."""
    return memory_status()[0]


_FIX_IMPORTS = (
    "Install torch, transformers, and gguf into the forge interpreter "
    "(python -m pip install torch transformers gguf)."
)
_FIX_INDEX = (
    "Confirm the repo, the pinned revision, and that model.safetensors.index.json is listed in the plan."
)
_FIX_TOKENIZER = (
    "Run python -m voxprint_forge.llama_patch on the pinned conversion/base.py and rebuild llama.cpp."
)
_FIX_LADDER = "Add IQ3_XXS or IQ2_M to that tier, or raise its budget_gb, so weights plus q8_0 KV fit."
_FIX_DISK = "Free disk space or remove files from the plan until the free space covers the download."
_FIX_RAM = "Use a machine with more RAM, or leave this model out of the run."
_FIX_TOKENS = (
    "Enter HF_TOKEN and KAGGLE_API_TOKEN in the GUI (they are stored in the OS keyring), "
    "or set FORGE_DEFER_UPLOAD=1."
)
_FIX_BINARIES = (
    "Install cmake, ninja, and nvcc, or set FORGE_LLAMA_BIN to a directory with "
    "llama-quantize, llama-imatrix, llama-cli, and llama-server."
)


def _check(name: str, ok: bool, detail: str, fix: str) -> dict[str, object]:
    return {"name": name, "ok": ok, "detail": detail, "fix": "" if ok else fix}


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
    """A tier fails when no listed quant fits on the card or with an offload split."""
    ctx = max(plan.contexts) if plan.contexts else 8192
    ram = memory_status()[1]
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
                if _quant_fits(model, quant, kv, budget, ram)
            ]
            if not fits:
                gaps.append(
                    f"{model.id} {tier.name} budget {tier.budget_gb:g} GB fits no quant "
                    f"in {', '.join(tier.quants)} at ctx {ctx}"
                )
    return gaps


def _quant_fits(model: ModelPlan, quant: str, kv: int, budget: int, ram: int) -> bool:
    weights = estimate_weights_bytes(model.params_b, quant)
    return any(
        split.vram_bytes <= budget and split.ram_bytes <= ram
        for split in options_for(
            weights,
            kv,
            moe_layers=model.moe_layers,
            expert_fraction=model.expert_fraction,
            encoder_gib=model.encoder_gib,
        )
    )


def peak_download_bytes(plan: Plan) -> int:
    """Sum of pinned file sizes for enabled models."""
    total = 0
    for model in plan.enabled_models():
        total += sum(item.size for item in model.files)
    return total


def index_targets(plan: Plan) -> list[tuple[str, str, str]]:
    """Safetensors models whose shard index must answer HEAD."""
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
    """True when the pinned converter contains the T-lite pre-tokenizer hash."""
    env = os.environ.get("FORGE_LLAMA_SRC", "")
    if env:
        path = Path(env) / "conversion" / "base.py"
        if path.is_file():
            return recognised(path.read_text(encoding="utf-8"))
        return False
    # Sources are applied at build time from this registration.
    return T_LITE_HASH in (repo / "src" / "voxprint_forge" / "llama_patch.py").read_text(encoding="utf-8")


def binaries_ready() -> tuple[bool, str]:
    """Whether llama.cpp binaries exist, and a short reason when they do not."""
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
    """Run the seconds-long checks and return the report."""
    problems: list[str] = []
    checks: list[dict[str, object]] = []
    missing = missing_imports()
    if missing:
        detail = "converter Python is missing " + ", ".join(missing)
        problems.append(detail)
        checks.append(_check("imports", False, detail, _FIX_IMPORTS))
    else:
        checks.append(_check("imports", True, "torch, transformers, and gguf import", ""))
    fetch = opener or urllib.request.urlopen
    indexes: list[dict[str, object]] = []
    targets = index_targets(plan)
    if not targets:
        checks.append(_check("indexes", True, "no safetensors shard index in this plan", ""))
    for model_id, model_repo, revision in targets:
        url = f"https://huggingface.co/{model_repo}/resolve/{revision}/model.safetensors.index.json"
        try:
            status = _head(url, fetch)
        except StageError as exc:
            detail = str(exc)
            problems.append(detail)
            checks.append(_check("indexes", False, detail, _FIX_INDEX))
            continue
        indexes.append({"id": model_id, "status": status})
        if status >= 400:
            detail = f"{model_id} shard index HEAD {status} at {revision}"
            problems.append(detail)
            checks.append(_check("indexes", False, detail, _FIX_INDEX))
        else:
            checks.append(_check("indexes", True, f"{model_id} shard index HEAD {status}", ""))
    if any(model.id == "t-lite-it-2.1" for model in plan.enabled_models()):
        if converter_recognises_t_lite(repo):
            checks.append(_check("tokenizer", True, "T-lite pre-tokenizer hash is registered", ""))
        else:
            detail = "T-lite pre-tokenizer hash is not registered on the pinned converter"
            problems.append(detail)
            checks.append(_check("tokenizer", False, detail, _FIX_TOKENIZER))
    else:
        checks.append(_check("tokenizer", True, "T-lite is not enabled", ""))
    gaps = ladder_gaps(plan)
    problems.extend(gaps)
    if gaps:
        checks.append(_check("quant-ladder", False, "; ".join(gaps), _FIX_LADDER))
    else:
        checks.append(_check("quant-ladder", True, "every quantize tier has a quant that fits", ""))
    need = peak_download_bytes(plan)
    disk_notes: list[str] = []
    try:
        free = disk.free_bytes(forge_out)
    except OSError as exc:
        free = 0
        disk_notes.append(f"disk check failed: {exc}")
    if free < need:
        disk_notes.append(f"need {gib(need):.1f} GiB for the plan, have {gib(free):.1f} GiB")
    problems.extend(disk_notes)
    if disk_notes:
        checks.append(_check("disk", False, "; ".join(disk_notes), _FIX_DISK))
    else:
        checks.append(_check("disk", True, f"{gib(free):.1f} GiB free, need {gib(need):.1f} GiB", ""))
    memory = (ram or ram_bytes)()
    resident = 0
    for model in plan.enabled_models():
        if model.params_b <= 0:
            continue
        quants = [quant for tier in model.tiers if tier.kind == "quantize" for quant in tier.quants]
        if quants:
            resident = max(resident, min(estimate_weights_bytes(model.params_b, quant) for quant in quants))
    if resident and memory < resident:
        detail = f"need {gib(resident):.1f} GiB RAM for the smallest quant, have {gib(memory):.1f} GiB"
        problems.append(detail)
        checks.append(_check("ram", False, detail, _FIX_RAM))
    else:
        checks.append(_check("ram", True, f"{gib(memory):.1f} GiB RAM", ""))
    tokens_ok = _deferring() or (
        bool(os.environ.get("HF_TOKEN", "").strip()) and bool(os.environ.get("KAGGLE_API_TOKEN", "").strip())
    )
    if tokens_ok:
        checks.append(_check("tokens", True, "upload tokens are set or upload is deferred", ""))
    else:
        detail = "HF_TOKEN and KAGGLE_API_TOKEN are required unless FORGE_DEFER_UPLOAD=1"
        problems.append(detail)
        checks.append(_check("tokens", False, detail, _FIX_TOKENS))
    ready, llama_detail = binaries_ready()
    if ready:
        checks.append(_check("llama", True, llama_detail, ""))
    else:
        detail = f"llama.cpp binaries are absent and not buildable ({llama_detail})"
        problems.append(detail)
        checks.append(_check("llama", False, detail, _FIX_BINARIES))
    report: dict[str, object] = {
        "ok": not problems,
        "problems": problems,
        "checks": checks,
        "imports": [name for name in _IMPORTS if name not in missing],
        "indexes": indexes,
        "disk_free": free,
        "disk_need": need,
        "ram": memory,
        "llama": llama_detail,
        "defer_upload": _deferring(),
    }
    if problems:
        raise StageError("; ".join(problems), code="preflight", report=report)
    return report


# ----- Build a set from references (non-VM) --------------------------------------------------------
REFSET_DIFFUSERS = ((0, 37, 1), (0, 38, 0))
REFSET_MIN_VRAM_GB = 16
REFSET_MIN_RAM_GB = 32
REFSET_MIN_DISK_GB = 2.0


def _version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for piece in text.split(".")[:3]:
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts + [0] * (3 - len(parts)))


def _probe_cuda() -> int | None:
    """Total memory of GPU 0 in bytes, or ``None`` without CUDA."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.get_device_properties(0).total_memory)


def _mivolo_imports(python: str) -> bool:
    import subprocess

    try:
        done = subprocess.run([python, "-c", "import mivolo, timm, cv2"], capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return done.returncode == 0


def refset_preflight(
    *,
    home: Path | None = None,
    diffusers_version: Callable[[], str | None] | None = None,
    img2img: Callable[[], bool] | None = None,
    installed: Callable[[str], bool] | None = None,
    vram: Callable[[], int | None] | None = None,
    ram: Callable[[], int] | None = None,
    mivolo: Callable[[str], bool] | None = None,
) -> dict[str, object]:
    """Seconds-long checks for "Build a set from references" on this computer (no GPU work, no downloads).

    diffusers in [0.37.1, 0.38) with ``ZImageImg2ImgPipeline``; Z-Image-Turbo, the judge VLM, MiVOLO and the
    face detector installed; the MiVOLO worker env imports; CUDA with at least 16 GB; at least 32 GB RAM
    (model CPU offload); disk for the rounds. DINOv2 is reported but optional (no near-duplicate marks).
    """
    from importlib import metadata

    from voxprint_forge.lora import refset
    from voxprint_forge.lora.backends import (
        ESTIMATOR_ID,
        FACE_DETECTOR_ID,
        JUDGE_ID,
        installed_dir,
        mivolo_python,
        model_dir,
    )

    def version() -> str | None:
        try:
            return metadata.version("diffusers")
        except metadata.PackageNotFoundError:
            return None

    def has_img2img() -> bool:
        try:
            return importlib.util.find_spec("diffusers.pipelines.z_image.pipeline_z_image_img2img") is not None
        except (ImportError, ValueError):
            return False

    present = installed or (lambda mdl_id: installed_dir(mdl_id) is not None)
    checks: list[dict[str, object]] = []
    problems: list[str] = []

    def add(name: str, ok: bool, detail: str, fix: str, *, required: bool = True) -> None:
        checks.append({**_check(name, ok, detail, fix), "required": required})
        if required and not ok:
            problems.append(detail)

    found = (diffusers_version or version)()
    low, high = REFSET_DIFFUSERS
    ok = found is not None and low <= _version_tuple(found) < high
    add("diffusers", ok, f"diffusers {found or 'missing'} (need >=0.37.1,<0.38)",
        'Install the app extras: pip install "diffusers>=0.37.1,<0.38".')
    pipeline = ok and (img2img or has_img2img)()
    add("img2img", pipeline, f"ZImageImg2ImgPipeline {'found' if pipeline else 'missing'}",
        "Reinstall diffusers 0.37.1.")
    for mdl_id, label in ((refset.TEACHER_ID, "Z-Image-Turbo"), (JUDGE_ID, "judge VLM")):
        ready = present(mdl_id)
        add(f"model:{mdl_id}", ready, f"{label} ({mdl_id}) {'installed' if ready else 'missing'}",
            "Install it in Manage models.")
    detector = model_dir(FACE_DETECTOR_ID) / "face_detection_yunet_2023mar.onnx"
    estimator_ok = (installed(ESTIMATOR_ID) if installed else model_dir(ESTIMATOR_ID).is_dir()) and (
        installed(FACE_DETECTOR_ID) if installed else detector.is_file())
    add(f"model:{ESTIMATOR_ID}", estimator_ok,
        f"estimator and face detector {'installed' if estimator_ok else 'missing'}",
        "Install them in Manage models.")
    python = mivolo_python()
    worker = (mivolo or _mivolo_imports)(python)
    add("mivolo-env", worker, f"estimator worker env {'imports' if worker else 'does not import'}",
        "Repair the venv-mivolo env (installer/requirements-mivolo.txt).")
    dino = present(refset.EMBEDDER_ID)
    add(f"model:{refset.EMBEDDER_ID}", dino, f"DINOv2-base {'installed' if dino else 'missing'} (near-duplicate marks)",
        "Install dinov2-base in Manage models; without it rounds show no near-duplicate marks.", required=False)
    total = (vram or _probe_cuda)()
    add("cuda", total is not None and total >= (REFSET_MIN_VRAM_GB - 0.5) * 2**30,
        f"CUDA GPU {gib(total):.0f} GB" if total else "no CUDA GPU",
        f"A CUDA GPU with at least {REFSET_MIN_VRAM_GB} GB is needed.")
    memory = (ram or ram_bytes)()
    add("ram", memory >= (REFSET_MIN_RAM_GB - 1) * 2**30, f"{gib(memory):.0f} GB RAM",
        f"At least {REFSET_MIN_RAM_GB} GB RAM: the models wait in RAM between stages.")
    where = home or Path.home()
    try:
        free = disk.free_bytes(where)
    except OSError:
        free = 0
    add("disk", free >= REFSET_MIN_DISK_GB * 2**30, f"{gib(free):.1f} GiB free",
        f"Free at least {REFSET_MIN_DISK_GB:.0f} GiB for the rounds.")
    return {"ok": not problems, "problems": problems, "checks": checks}
