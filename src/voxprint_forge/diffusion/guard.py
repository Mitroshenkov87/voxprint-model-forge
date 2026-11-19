"""Refuse to start while the LLM bootstrap process is still alive."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

# PROCESS_QUERY_LIMITED_INFORMATION. Enough to tell that a pid still exists.
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_ACCESS_DENIED = 5


def _windows_pid_alive(pid: int) -> bool:
    """OpenProcess, because os.kill(pid, 0) is CTRL_C_EVENT on Windows.

    signal.CTRL_C_EVENT is 0, so the POSIX "signal 0" probe raises
    KeyboardInterrupt in the calling console instead of testing the pid.
    """
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if handle:
        kernel32.CloseHandle(handle)
        return True
    return ctypes.get_last_error() == _ERROR_ACCESS_DENIED


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


_SHELLS = {"bash", "bash.exe", "sh", "dash", "zsh", "zsh.exe"}
_PYTHONS = {"python", "python3", "python.exe", "python3.exe", "forge", "forge.exe"}


def _basename(token: str) -> str:
    name = token.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return name


def cmdline_is_job(cmdline: str) -> bool:
    """True for a forge or bootstrap process. A tmux server line is not a job."""
    text = cmdline.replace("\0", " ").strip()
    if not text:
        return False
    first = text.split()[0]
    base = _basename(first)
    if base == "tmux" or base.startswith("tmux"):
        return False
    bootstrap = "scripts/bootstrap.sh" in text or "scripts\\bootstrap.sh" in text
    if bootstrap and (base in _SHELLS or base == "bootstrap.sh"):
        return True
    lowered = text.lower()
    if base not in _PYTHONS and "python" not in base:
        return False
    if "voxprint_forge" in lowered:
        return True
    padded = f" {lowered} "
    return " forge " in padded and "forge-gui" not in lowered


def powershell_exe() -> str:
    """PowerShell 7 (``pwsh``) when it is on PATH, else Windows PowerShell."""
    return shutil.which("pwsh") or "powershell"


def _process_cmdlines() -> list[str]:
    if sys.platform == "win32":
        found = subprocess.run(
            [
                powershell_exe(),
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | Select-Object -ExpandProperty CommandLine",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        return [line.strip() for line in found.stdout.splitlines() if line.strip()]
    root = Path("/proc")
    if not root.is_dir():
        return []
    lines: list[str] = []
    for entry in root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if raw:
            lines.append(raw.replace(b"\0", b" ").decode("utf-8", "replace").strip())
    return lines


def llm_job_running(forge_out: Path, *, processes: Callable[[], list[str]] | None = None) -> bool:
    """True when bootstrap.lock names a live pid, or a real forge/python job is running.

    A stale lock whose pid is gone does not block. The tmux server command line
    can still contain ``scripts/bootstrap.sh`` after the pane exits, so a process
    whose executable is tmux is not a job.
    """
    lock = forge_out / "bootstrap.lock"
    if lock.is_file():
        pid_text = lock.read_text(encoding="utf-8").strip()
        if pid_text.isdigit() and _pid_alive(int(pid_text)):
            return True
    lines = (processes or _process_cmdlines)()
    return any(cmdline_is_job(line) for line in lines)
