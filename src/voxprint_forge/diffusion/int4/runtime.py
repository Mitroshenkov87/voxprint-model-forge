"""Run a pinned tool. The disk floor is checked while the process is alive."""

from __future__ import annotations

import os
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

from voxprint_forge.diffusion.int4.disk import guard_disk
from voxprint_forge.errors import StageError, ToolMissingError


def execute(
    argv: list[str],
    *,
    cwd: Path,
    disk_path: Path,
    dry_run: bool,
    ratio: Callable[[Path], float] | None = None,
    reclaim: Callable[[], int] | None = None,
    poll_s: float | None = None,
) -> dict[str, object]:
    """Run ``argv`` unless this is a dry run. A disk-floor miss terminates it."""
    record = {"argv": argv, "cwd": str(cwd), "dry_run": dry_run}
    if dry_run:
        return record
    if not cwd.is_dir():
        raise ToolMissingError(f"missing working directory {cwd}", code="tool_missing")
    guard_disk(disk_path, ratio=ratio, reclaim=reclaim)
    raw = os.environ.get("FORGE_DISK_POLL_S", "")
    interval = float(raw) if raw else (30.0 if poll_s is None else poll_s)
    proc = subprocess.Popen(argv, cwd=str(cwd))
    floor_error: list[BaseException] = []
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(interval):
            try:
                guard_disk(disk_path, ratio=ratio, reclaim=reclaim)
            except BaseException as exc:
                floor_error.append(exc)
                proc.terminate()
                return

    thread = threading.Thread(target=watch, name="diffusion-int4-disk", daemon=True)
    thread.start()
    code = proc.wait()
    stop.set()
    thread.join(timeout=interval + 1)
    if floor_error:
        raise floor_error[0]
    if code != 0:
        raise StageError(f"command failed ({code}): {argv[0]}", code="tool_failed")
    return record
