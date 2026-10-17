"""``forge preflight --vm``: check a fresh VM image before any paid GPU job.

One PASS/FAIL line per check with a fix hint. Nothing is installed or
downloaded, and the Hugging Face token is only reported as set or missing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

MIN_PYTHON = (3, 12)
MIN_FREE_GB = 150.0
SCOPES = ("all", "lora")
# Modules convert_hf_to_gguf.py and its bundled gguf-py import at the llama.cpp pin
# (scripts/llama.rev). Some are imported lazily inside set_vocab, so a missing one
# only shows after the BF16 download. Importing them here catches that up front.
CONVERT_IMPORTS = (
    "numpy",
    "torch",
    "safetensors",
    "transformers",
    "sentencepiece",
    "sentencepiece.sentencepiece_model_pb2",
    "google.protobuf",
    "yaml",
    "tqdm",
    "requests",
)


def pinned_common(root: Path) -> str:
    """The voxprint-common version this checkout pins (the ``==`` pin in pyproject.toml)."""
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    found = re.findall(r"voxprint-common(?:\[[^\]]*\])?\s*==\s*([0-9][0-9A-Za-z.]*)", text)
    if not found:
        found = re.findall(r"voxprint-common(?:\[[^\]]*\])?\s*>=\s*([0-9][0-9A-Za-z.]*)", text)
    return found[0] if found else ""


def wheel_path(root: Path) -> str:
    """Vendored wheel for the pinned version, relative to the checkout."""
    return f"vendor/voxprint_common-{pinned_common(root)}-py3-none-any.whl"
# Minimum driver major for each CUDA runtime major.minor of the torch build.
DRIVER_FOR_CUDA = {"13.0": 580, "12.8": 570, "12.6": 560, "12.4": 550}


@dataclass
class Probe:
    """Everything the checks read from the machine. Tests replace the callables."""

    which: Callable[[str], str | None] = shutil.which
    env: dict[str, str] = field(default_factory=lambda: dict(os.environ))
    version: Callable[[str], str | None] = None  # type: ignore[assignment]
    run: Callable[[list[str]], str] = None  # type: ignore[assignment]
    torch_info: Callable[[], dict[str, object]] = None  # type: ignore[assignment]
    free_gb: Callable[[Path], float] = None  # type: ignore[assignment]
    python: tuple[int, int] = (sys.version_info.major, sys.version_info.minor)
    executable: Callable[[str], bool] = lambda path: os.path.isfile(path) and os.access(path, os.X_OK)
    import_error: Callable[[str], str | None] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.version = self.version or _dist_version
        self.run = self.run or _run
        self.torch_info = self.torch_info or _torch_info
        self.free_gb = self.free_gb or (lambda path: shutil.disk_usage(path).free / 1e9)
        self.import_error = self.import_error or _import_error


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _import_error(name: str) -> str | None:
    """None when ``name`` imports, else the exception type and message."""
    import importlib

    try:
        importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - reported as a failed check
        return f"{type(exc).__name__}: {exc}"
    return None


def _run(argv: list[str]) -> str:
    try:
        return subprocess.run(argv, check=False, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _torch_info() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - reported as a failed check
        return {"error": str(exc)}
    return {"version": torch.__version__, "cuda": torch.version.cuda, "available": bool(torch.cuda.is_available())}


def _line(name: str, ok: bool, detail: str, fix: str = "") -> dict[str, object]:
    return {"name": name, "ok": ok, "detail": detail, "fix": "" if ok else fix}


def check_python(probe: Probe) -> dict[str, object]:
    """Python 3.12 or newer (the setup tries 3.14, 3.13, 3.12)."""
    major, minor = probe.python
    return _line(
        "python", (major, minor) >= MIN_PYTHON, f"{major}.{minor}", "run scripts/vm_setup_synth4.sh (uv installs 3.14)"
    )


def check_torch(probe: Probe) -> dict[str, object]:
    """Torch imports, sees a CUDA GPU, and its CUDA build fits the driver."""
    info = probe.torch_info()
    if "error" in info:
        return _line("torch", False, "torch does not import", "run scripts/vm_setup_synth4.sh")
    out = probe.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).strip()
    driver = int(out.split(".")[0]) if re.match(r"^\d+", out) else None
    cuda = str(info.get("cuda") or "")
    detail = f"torch {info.get('version')} cuda {cuda or 'none'} driver {driver if driver is not None else 'none'}"
    if not cuda:
        return _line("torch", False, detail, "torch is a CPU build; reinstall from the cu130 index")
    need = DRIVER_FOR_CUDA.get(".".join(cuda.split(".")[:2]))
    if driver is None:
        return _line("torch", False, detail, "nvidia-smi found no driver; use a GPU image")
    if need is not None and driver < need:
        return _line("torch", False, detail, f"driver {driver} is too old for CUDA {cuda}; use cu128 or a newer image")
    if not info.get("available"):
        return _line("torch", False, detail, "torch.cuda.is_available() is False; check the driver and CUDA libs")
    return _line("torch", True, detail)


def requirements(root: Path, extra: str = "compare") -> list[str]:
    """Base dependencies plus one optional extra from ``pyproject.toml``."""
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    return list(project.get("dependencies", [])) + list(project.get("optional-dependencies", {}).get(extra, []))


def check_requirements(root: Path, probe: Probe, extra: str = "compare") -> dict[str, object]:
    """Every requirement installed at a matching version, e.g. torchmetrics>=1.6."""
    from packaging.requirements import Requirement
    from packaging.version import Version

    bad: list[str] = []
    for text in requirements(root, extra):
        req = Requirement(text)
        if req.marker is not None and not req.marker.evaluate():
            continue
        found = probe.version(req.name)
        if found is None:
            bad.append(f"{req.name} missing")
        elif req.specifier and not req.specifier.contains(Version(found).public, prereleases=True):
            bad.append(f"{req.name} {found} (needs {req.specifier})")
    detail = "all match" if not bad else "; ".join(bad)
    return _line(
        f"requirements[{extra}]",
        not bad,
        detail,
        f'uv pip install -e ".[{extra}]" from PyPI only, torch pinned by --constraint (no --extra-index-url)',
    )


def check_tools(probe: Probe) -> list[dict[str, object]]:
    """cmake, ninja, nvcc via CUDACXX, and a C++ compiler."""
    lines = [
        _line("cmake", probe.which("cmake") is not None, probe.which("cmake") or "missing", "apt-get install -y cmake"),
        _line(
            "ninja",
            probe.which("ninja") is not None,
            probe.which("ninja") or "missing",
            "apt-get install -y ninja-build",
        ),
    ]
    cudacxx = probe.env.get("CUDACXX", "").strip()
    if cudacxx:
        ok, detail = probe.executable(cudacxx), f"CUDACXX={cudacxx}"
    elif probe.executable("/usr/local/cuda/bin/nvcc"):
        ok, detail = True, "/usr/local/cuda/bin/nvcc (CUDACXX unset; the setup scripts export it)"
    else:
        found = probe.which("nvcc")
        ok, detail = found is not None, found or "missing"
    lines.append(_line("nvcc", ok, detail, "install the CUDA toolkit or export CUDACXX=/usr/local/cuda/bin/nvcc"))
    cxx = probe.env.get("CXX", "").strip() or probe.which("g++") or probe.which("c++")
    lines.append(_line("c++", bool(cxx), cxx or "missing", "apt-get install -y build-essential"))
    return lines


def check_llama(root: Path) -> dict[str, object]:
    """third_party/llama.cpp is present and its commit matches scripts/llama.rev."""
    want = (root / "scripts" / "llama.rev").read_text(encoding="utf-8").strip()
    src = root / "third_party" / "llama.cpp" / "CMakeLists.txt"
    stamp = root / "third_party" / "LLAMA_COMMIT"
    if not src.is_file() or not stamp.is_file():
        return _line(
            "llama.cpp", False, "third_party/llama.cpp missing", "build the bundle with scripts/make_vm_bundle.sh"
        )
    have = stamp.read_text(encoding="utf-8").strip()
    return _line(
        "llama.cpp",
        have == want,
        f"{have[:12]} (want {want[:12]})",
        "rebuild the bundle with scripts/make_vm_bundle.sh",
    )


def check_convert_imports(probe: Probe) -> dict[str, object]:
    """Every module convert_hf_to_gguf.py needs imports in this interpreter (the one FORGE_PYTHON runs)."""
    bad = []
    for name in CONVERT_IMPORTS:
        error = probe.import_error(name)
        if error is not None:
            bad.append(f"{name} ({error})")
    detail = f"{len(CONVERT_IMPORTS)} modules import" if not bad else "; ".join(bad)
    return _line(
        "convert_imports",
        not bad,
        detail,
        'uv pip install -e "." from PyPI (sentencepiece==0.2.2, protobuf==5.29.6 are base dependencies)',
    )


def check_wheel(root: Path, probe: Probe) -> dict[str, object]:
    """The vendored wheel for the pinned voxprint_common exists and is the installed version."""
    want = pinned_common(root)
    wheel = wheel_path(root)
    present = bool(want) and (root / wheel).is_file()
    installed = probe.version("voxprint-common")
    ok = present and installed == want
    return _line(
        "voxprint_common",
        ok,
        f"pinned {want or 'none'}, {wheel} {'present' if present else 'missing'}, installed {installed or 'none'}",
        f"uv pip install {wheel}",
    )


def check_token(probe: Probe) -> dict[str, object]:
    """HF_TOKEN is set. The value is never printed."""
    ok = bool(probe.env.get("HF_TOKEN", "").strip())
    return _line("hf_token", ok, "set" if ok else "missing", "export HF_TOKEN in the job environment (never in git)")


def check_disk(path: Path, probe: Probe, min_free_gb: float = MIN_FREE_GB) -> dict[str, object]:
    """Free space at ``path`` (the forge output root)."""
    target = path if path.exists() else path.parent
    free = probe.free_gb(target)
    return _line(
        "disk", free >= min_free_gb, f"{free:.0f} GB free at {target} (need {min_free_gb:.0f})", "attach a larger disk"
    )


def run_vm_checks(
    root: Path,
    forge_out: Path,
    probe: Probe | None = None,
    min_free_gb: float = MIN_FREE_GB,
    scope: str = "all",
) -> dict:
    """Checks for ``scope``. ``lora`` skips the build tools, llama.cpp and the converter imports
    (LoRA training builds and converts nothing).

    ``ok`` is False when any check fails.
    """
    if scope not in SCOPES:
        raise ValueError(f"unknown preflight scope {scope}")
    probe = probe or Probe()
    checks = [check_python(probe), check_torch(probe), check_requirements(root, probe)]
    if scope == "all":
        checks += check_tools(probe)
        checks.append(check_llama(root))
        checks.append(check_convert_imports(probe))
    checks += [
        check_wheel(root, probe),
        check_token(probe),
        check_disk(forge_out, probe, min_free_gb),
    ]
    ok = all(item["ok"] for item in checks)
    return {"ok": ok, "scope": scope, "checks": checks, "text": render(checks)}


def render(checks: list[dict[str, object]]) -> str:
    """``PASS name: detail`` or ``FAIL name: detail. Fix: hint`` per line."""
    lines = []
    for item in checks:
        word = "PASS" if item["ok"] else "FAIL"
        tail = f". Fix: {item['fix']}" if not item["ok"] and item["fix"] else ""
        lines.append(f"{word} {item['name']}: {item['detail']}{tail}")
    return "\n".join(lines)


def to_json(report: dict) -> str:
    """Report without the text block."""
    return json.dumps({key: value for key, value in report.items() if key != "text"}, indent=2, sort_keys=True)
