"""Refuse to start part D while the A–C canvas job is still alive."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable

# `canvas_job.sh` is a prefix of `canvas_job_d.sh` only before the dot.
# Require `.sh` and then either the end or a character other than `_`.
CANVAS_JOB_PATTERN = r"scripts/canvas_job\.sh([^_]|$)"


def command_is_canvas_ac(cmdline: str) -> bool:
    return re.search(CANVAS_JOB_PATTERN, cmdline) is not None


def _pgrep_canvas() -> bool:
    # The bracket keeps this pgrep from matching its own command line.
    # Windows CI has no pgrep. A missing tool is not a running canvas job.
    try:
        found = subprocess.run(
            ["pgrep", "-f", "[s]cripts/canvas_job\\.sh([^_]|$)"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return False
    return found.returncode == 0


def canvas_ac_running(*, pgrep: Callable[[], bool] | None = None) -> bool:
    """True when scripts/canvas_job.sh is in the process list. The D script does not match."""
    if pgrep is not None:
        return pgrep()
    return _pgrep_canvas()
