"""Refuse to start while the LLM bootstrap process is still alive."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
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
