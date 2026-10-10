"""Append human-readable lines to ~/forge-out/progress.log.

Each line is: timestamp, model, stage, percent, ETA in minutes.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path


class Progress:
    def __init__(
        self,
        path: Path,
        *,
        total_min: float,
        consumed_min: float,
        now: Callable[[], float],
    ) -> None:
        self.path = path
        self.total_min = total_min
        self.consumed_min = consumed_min
        self.now = now
        self._start = now()

    def begin(self, model: str, stage: str) -> None:
        self._write(model, stage)

    def advance(self, model: str, stage: str, minutes: float) -> None:
        self.consumed_min += minutes
        self._write(model, stage)

    def _write(self, model: str, stage: str) -> None:
        total = self.total_min
        consumed = self.consumed_min
        if total <= 0:
            pct = 0
        else:
            pct = min(100, int(100 * consumed / total))
        remaining = max(0.0, total - consumed)
        elapsed_min = max(0.0, (self.now() - self._start) / 60.0)
        if consumed > 0 and elapsed_min > 0:
            eta = elapsed_min * remaining / consumed
        else:
            eta = remaining
        append_progress(self.path, model=model, stage=stage, pct=pct, eta_min=eta)


def append_progress(path: Path, *, model: str, stage: str, pct: int, eta_min: float) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    line = f"{stamp}  {model}  {stage}  {pct}%  eta {eta_min:.0f} min\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
