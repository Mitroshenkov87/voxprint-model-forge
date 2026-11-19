"""One attention setting for every arm of a compare job.

Z-Image calls diffusers ``dispatch_attention_fn``. The default backend is torch
SDPA. A boolean mask is passed when caption lengths differ, which
FlashAttention-2 through SDPA cannot take, so the plan chooses the kernel list
and both arms of a pair use it.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Never

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from voxprint_forge.compare.codes import CompareSetup

_SDPA = (SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION)
_MATH = (SDPBackend.MATH,)


def _unexpected_attention(name: Never) -> Never:
    raise CompareSetup(f"unhandled attention setting {name}", code="compare_plan")


def sdpa_backends(name: str) -> tuple[SDPBackend, ...] | None:
    """Kernel list for ``sdpa`` and ``sdpa-math``. Any other name is a diffusers backend."""
    if name == "sdpa":
        return _SDPA
    if name == "sdpa-math":
        return _MATH
    return None


@contextmanager
def attention_scope(name: str) -> Iterator[None]:
    """Fix cuDNN, TF32, and the SDPA kernel list for one job.

    ``sdpa`` allows flash, cuDNN, and efficient kernels and never MATH.
    ``sdpa-math`` is MATH only. A diffusers backend name is applied on the
    pipeline separately; this scope still turns the benchmark and TF32 off.
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backends = sdpa_backends(name)
    if backends is None:
        if not name:
            _unexpected_attention(name)
        yield
        return
    try:
        context = sdpa_kernel(list(backends), set_priority=True)
    except TypeError:
        context = sdpa_kernel(list(backends))
    with context:
        yield


def set_diffusers_backend(pipe: object, name: str) -> None:
    """Pass a non-SDPA plan value to ``transformer.set_attention_backend``."""
    if sdpa_backends(name) is not None:
        return
    transformer = getattr(pipe, "transformer", None)
    setter = getattr(transformer, "set_attention_backend", None)
    if setter is None:
        raise CompareSetup(f"attention backend {name} is not available", code="compare_attention")
    setter(name)


def gpu_record(attention: str) -> dict[str, object]:
    """Device facts for ``summary.json``. Empty strings when CUDA is absent."""
    name = ""
    capability = ""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability()
        capability = f"{major}.{minor}"
    return {
        "name": name,
        "compute_capability": capability,
        "driver": _driver_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "",
        "attention": attention,
        "diffusers": _package_version("diffusers"),
        "cudnn_benchmark": False,
        "allow_tf32": False,
    }


def _driver_version() -> str:
    try:
        done = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if done.returncode != 0:
        return ""
    return done.stdout.strip().splitlines()[0].strip() if done.stdout.strip() else ""


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""
