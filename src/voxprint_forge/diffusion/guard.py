"""Refuse to start while the LLM bootstrap process is still alive."""

from __future__ import annotations

import ctypes
import os
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


def _pgrep_bootstrap() -> bool:
    # The bracket keeps this pgrep from matching its own command line.
    found = subprocess.run(
        ["pgrep", "-f", "[s]cripts/bootstrap.sh"],
        check=False,
        capture_output=True,
        text=True,
    )
    return found.returncode == 0


def llm_job_running(forge_out: Path, *, pgrep: Callable[[], bool] | None = None) -> bool:
    """True when bootstrap.lock names a live pid, or bootstrap.sh is in the process list.

    A stale lock whose pid is gone does not block. A tmux session named forge
    can linger after the script exits, so the session name is not the signal.
    """
    lock = forge_out / "bootstrap.lock"
    if lock.is_file():
        pid_text = lock.read_text(encoding="utf-8").strip()
        if pid_text.isdigit() and _pid_alive(int(pid_text)):
            return True
    if pgrep is not None:
        return pgrep()
    return _pgrep_bootstrap()
