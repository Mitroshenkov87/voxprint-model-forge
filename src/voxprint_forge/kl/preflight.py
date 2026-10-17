"""Check that llama-perplexity contains SASS for the GPU that will run it."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from voxprint_forge.errors import ToolMissingError

_ARCH = re.compile(r"\b(?:sm|compute)_(\d+)\b")


def arch_token(capability: tuple[int, int]) -> str:
    """``(9, 0)`` becomes ``90``. That is the cuobjdump ``sm_90`` suffix."""
    major, minor = capability
    return f"{major}{minor}"


def sass_archs(text: str) -> set[str]:
    """``sm_`` architectures in ``cuobjdump --list-elf`` output. PTX ``compute_`` is ignored."""
    found: set[str] = set()
    for line in text.splitlines():
        if "compute_" in line and "sm_" not in line:
            continue
        for match in _ARCH.finditer(line):
            if f"sm_{match.group(1)}" in line:
                found.add(match.group(1))
    return found


def require_llama_arch(
    binary: Path,
    *,
    capability: tuple[int, int] | None,
    elf_text: str | None = None,
    cuobjdump: str | None = None,
) -> str:
    """Fail with exit 9 when the binary has no SASS for this GPU.

    ``capability`` ``None`` means there is no CUDA device to check. When
    ``cuobjdump`` is missing the return value is a warning and the run continues.
    ``elf_text`` skips the subprocess so tests stay on CPU.
    """
    if capability is None:
        return ""
    if not binary.is_file():
        raise ToolMissingError(f"{binary} is missing", code="kl_llama_arch")
    token = arch_token(capability)
    if elf_text is None and cuobjdump is None:
        _probe_version(binary)
    if elf_text is None:
        tool = cuobjdump if cuobjdump is not None else shutil.which("cuobjdump")
        if not tool:
            return "cuobjdump is not installed; llama.cpp GPU arch was not checked"
        done = subprocess.run([tool, "--list-elf", str(binary)], check=False, capture_output=True, text=True)
        if done.returncode != 0:
            raise ToolMissingError(
                f"cuobjdump could not read {binary.name}",
                code="kl_llama_arch",
            )
        elf_text = done.stdout + done.stderr
    found = sass_archs(elf_text)
    if token not in found:
        listed = ", ".join(sorted(found)) or "none"
        raise ToolMissingError(
            f"{binary.name} has no sm_{token} SASS (found: {listed}). "
            f"Rebuild llama.cpp with FORGE_CUDA_ARCHS including {token}.",
            code="kl_llama_arch",
        )
    return ""


def _probe_version(binary: Path) -> None:
    """Run ``--version``, or ``--list-devices`` when version is missing."""
    for flag in ("--version", "--list-devices"):
        try:
            done = subprocess.run([str(binary), flag], check=False, capture_output=True, text=True)
        except OSError as exc:
            raise ToolMissingError(f"{binary.name} could not start", code="kl_llama_arch") from exc
        if done.returncode == 0:
            return
    raise ToolMissingError(
        f"{binary.name} did not answer --version or --list-devices",
        code="kl_llama_arch",
    )
