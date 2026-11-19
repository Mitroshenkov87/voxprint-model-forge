"""llama.cpp arch preflight: the CUDA backend library is read, not the shared-build binary."""

from __future__ import annotations

from pathlib import Path

import pytest

from voxprint_forge.errors import ToolMissingError
from voxprint_forge.kl.preflight import cuda_backend, require_llama_arch

ELF_90 = "ELF file    1: libggml-cuda.1.sm_90.cubin\nELF file    2: libggml-cuda.2.sm_90.cubin\n"
PTX_90 = "PTX file    1: libggml-cuda.1.sm_90.ptx compute_90\n"


def _reader(device: dict[Path, tuple[str, str]]):  # type: ignore[no-untyped-def]
    calls: list[tuple[str, Path]] = []

    def read(flag: str, path: Path) -> tuple[int, str]:
        calls.append((flag, path))
        if path not in device:
            return 255, f"cuobjdump info    : File '{path}' does not contain device code"
        elf, ptx = device[path]
        return 0, elf if flag == "--list-elf" else ptx

    return read, calls


def _bin(folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    binary = folder / "llama-perplexity"
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    return binary


def test_shared_build_reads_libggml_cuda_from_ldd(tmp_path: Path) -> None:
    binary = _bin(tmp_path / "build" / "bin")
    lib = tmp_path / "elsewhere" / "libggml-cuda.so"
    lib.parent.mkdir()
    lib.write_bytes(b"\x7fELF")
    ldd = f"\tlibggml-base.so => /x/libggml-base.so (0x1)\n\tlibggml-cuda.so => {lib} (0x2)\n"
    read, calls = _reader({lib: (ELF_90, PTX_90)})
    assert cuda_backend(binary, ldd_text=ldd) == (lib, "ldd")
    assert require_llama_arch(binary, capability=(9, 0), ldd_text=ldd, reader=read) == ""
    assert all(path == lib for _, path in calls)


def test_shared_build_finds_the_library_next_to_the_binary(tmp_path: Path) -> None:
    binary = _bin(tmp_path / "build" / "bin")
    lib = binary.parent / "libggml-cuda.so"
    lib.write_bytes(b"\x7fELF")
    read, _ = _reader({lib: (ELF_90, PTX_90)})
    assert require_llama_arch(binary, capability=(9, 0), ldd_text="", reader=read) == ""
    sibling = _bin(tmp_path / "b2" / "bin")
    (tmp_path / "b2" / "lib").mkdir()
    lib2 = tmp_path / "b2" / "lib" / "libggml-cuda.so.0"
    lib2.write_bytes(b"\x7fELF")
    assert cuda_backend(sibling, ldd_text="")[0] == lib2


def test_shared_build_without_sm90_names_sass_and_ptx(tmp_path: Path) -> None:
    binary = _bin(tmp_path / "bin")
    lib = binary.parent / "libggml-cuda.so"
    lib.write_bytes(b"\x7fELF")
    read, _ = _reader({lib: ("ELF file 1: x.sm_80.cubin\n", "PTX file 1: x.sm_80.ptx compute_80\n")})
    with pytest.raises(ToolMissingError, match=r"libggml-cuda.so has no sm_90 SASS \(found: 80; PTX: 80\)") as caught:
        require_llama_arch(binary, capability=(9, 0), ldd_text="", reader=read)
    assert caught.value.exit_code == 9


def test_static_build_reads_the_binary(tmp_path: Path) -> None:
    binary = _bin(tmp_path / "bin")
    read, calls = _reader({binary: (ELF_90, PTX_90)})
    assert cuda_backend(binary, ldd_text="\tlibc.so.6 => /lib/libc.so.6 (0x1)\n") == (binary, "static build")
    assert require_llama_arch(binary, capability=(9, 0), ldd_text="", reader=read) == ""
    assert {path for _, path in calls} == {binary}


def test_unreadable_device_code_fails_with_the_paths_tried(tmp_path: Path) -> None:
    binary = _bin(tmp_path / "bin")
    read, _ = _reader({})
    with pytest.raises(ToolMissingError, match="could not read CUDA device code for llama-perplexity") as caught:
        require_llama_arch(binary, capability=(9, 0), ldd_text="", reader=read)
    assert "static build" in str(caught.value) and caught.value.exit_code == 9
    lib = binary.parent / "libggml-cuda.so"
    lib.write_bytes(b"not cuda")
    with pytest.raises(ToolMissingError, match="libggml-cuda.so"):
        require_llama_arch(binary, capability=(9, 0), ldd_text="", reader=read)


def test_empty_listing_is_not_a_pass(tmp_path: Path) -> None:
    binary = _bin(tmp_path / "bin")
    read, _ = _reader({binary: ("", "")})
    with pytest.raises(ToolMissingError, match="could not read CUDA device code"):
        require_llama_arch(binary, capability=(9, 0), ldd_text="", reader=read)
