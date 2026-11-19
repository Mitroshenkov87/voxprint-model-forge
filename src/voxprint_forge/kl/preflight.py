"""Check that the llama.cpp CUDA device code has SASS for the GPU that will run it.

A shared llama.cpp build (the default, ``BUILD_SHARED_LIBS=ON``) keeps every CUDA kernel in
``libggml-cuda.so``; ``llama-perplexity`` itself has no device code. The check therefore reads the
CUDA backend library: the one ``ldd`` resolves for the binary, else ``libggml-cuda.so*`` next to it
or in a sibling ``lib``/``lib64``. Only when no such library exists is the binary itself read
(static build). When nothing can be read the check fails; it never passes silently.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from voxprint_forge.errors import ToolMissingError

_ARCH = re.compile(r"\b(?:sm|compute)_(\d+)\b")
_LDD_CUDA = re.compile(r"^\s*(libggml-cuda[^\s]*\.so[^\s]*)\s+=>\s+(\S+)")
CUDA_LIB_GLOB = "libggml-cuda*.so*"

# (cuobjdump flag, file) -> (returncode, text)
Reader = Callable[[str, Path], tuple[int, str]]


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


def ptx_archs(text: str) -> set[str]:
    """``compute_`` architectures in ``cuobjdump --list-ptx`` output."""
    return {match.group(1) for match in re.finditer(r"\bcompute_(\d+)\b", text)}


def cuda_backend(binary: Path, *, ldd_text: str | None = None) -> tuple[Path, str]:
    """The file that holds the CUDA device code for ``binary`` and how it was found.

    Order: the ``libggml-cuda`` that ``ldd`` resolves, then ``libggml-cuda*.so*`` in the binary's
    folder, ``../lib`` or ``../lib64``, then the binary itself (a static build).
    """
    text = ldd_text if ldd_text is not None else _ldd(binary)
    for line in text.splitlines():
        match = _LDD_CUDA.match(line)
        if match and match.group(2) != "not" and Path(match.group(2)).is_file():
            return Path(match.group(2)), "ldd"
    folder = binary.resolve().parent
    for candidate_dir in (folder, folder.parent / "lib", folder.parent / "lib64"):
        if candidate_dir.is_dir():
            libs = sorted(path for path in candidate_dir.glob(CUDA_LIB_GLOB) if path.is_file())
            if libs:
                return libs[0], f"next to {binary.name}"
    return binary, "static build"


def require_llama_arch(
    binary: Path,
    *,
    capability: tuple[int, int] | None,
    elf_text: str | None = None,
    cuobjdump: str | None = None,
    ldd_text: str | None = None,
    reader: Reader | None = None,
) -> str:
    """Fail with exit 9 when the llama.cpp CUDA device code has no SASS for this GPU.

    ``capability`` ``None`` means there is no CUDA device to check. When ``cuobjdump``
    is not installed the return value is a warning that the caller prints. ``elf_text``
    skips the lookup and the subprocess; ``ldd_text`` and ``reader`` replace ``ldd`` and
    ``cuobjdump`` so tests stay on CPU.
    """
    if capability is None:
        return ""
    if not binary.is_file():
        raise ToolMissingError(f"{binary} is missing", code="kl_llama_arch")
    token = arch_token(capability)
    target, how = binary, "given"
    ptx = ""
    if elf_text is None:
        if reader is None and cuobjdump is None:
            _probe_version(binary)
        if reader is None:
            tool = cuobjdump if cuobjdump is not None else shutil.which("cuobjdump")
            if not tool:
                return "cuobjdump is not installed; llama.cpp GPU arch was not checked"
            reader = _cuobjdump_reader(tool)
        target, how = cuda_backend(binary, ldd_text=ldd_text)
        code, elf_text = reader("--list-elf", target)
        if code != 0 or not sass_archs(elf_text):
            tried = f"{target} ({how})"
            if target != binary:
                tried += f"; {binary.name} itself is not checked because the CUDA code lives in that library"
            raise ToolMissingError(
                f"cuobjdump could not read CUDA device code for {binary.name}: tried {tried}. "
                f"Point FORGE_LLAMA_BIN at a b11476 CUDA build (libggml-cuda.so next to the binaries) "
                f"or a static build with device code.",
                code="kl_llama_arch",
            )
        ptx_code, ptx = reader("--list-ptx", target)
        if ptx_code != 0:
            ptx = ""
    found = sass_archs(elf_text)
    if token not in found:
        listed = ", ".join(sorted(found)) or "none"
        ptx_listed = ", ".join(sorted(ptx_archs(ptx))) or "none"
        raise ToolMissingError(
            f"{target.name} has no sm_{token} SASS (found: {listed}; PTX: {ptx_listed}). "
            f"Rebuild llama.cpp with FORGE_CUDA_ARCHS including {token}.",
            code="kl_llama_arch",
        )
    return ""


def _cuobjdump_reader(tool: str) -> Reader:
    def read(flag: str, path: Path) -> tuple[int, str]:
        try:
            done = subprocess.run([tool, flag, str(path)], check=False, capture_output=True, text=True)
        except OSError as exc:
            return 127, str(exc)
        return done.returncode, done.stdout + done.stderr

    return read


def _ldd(binary: Path) -> str:
    tool = shutil.which("ldd")
    if not tool:
        return ""
    try:
        done = subprocess.run([tool, str(binary)], check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return done.stdout if done.returncode == 0 else ""


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
