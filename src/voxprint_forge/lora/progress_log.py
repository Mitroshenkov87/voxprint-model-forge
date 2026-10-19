"""Live training progress on disk.

``train.json`` holds the latest step, total, loss, elapsed time and ETA.
``train.log`` gains one line and ``vram.csv`` one sample on the same cadence:
at least every 60 seconds, and at each stage boundary. Every write is flushed
and fsynced so a crash keeps the last report.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path

from voxprint_forge.lora.i18n import text

INTERVAL_SECONDS = 60.0
TRAIN_FILE = "train.json"
LOG_FILE = "train.log"
VRAM_FILE = "vram.csv"

VramSample = tuple[float | None, float | None]
Clock = Callable[[], float]
VramReader = Callable[[], VramSample]


def sample_vram() -> VramSample:
    """Allocated and reserved GiB, or ``(None, None)`` when CUDA is absent."""
    try:
        import torch
    except ImportError:
        return None, None
    if not torch.cuda.is_available():
        return None, None
    return (
        round(torch.cuda.memory_allocated() / 2**30, 3),
        round(torch.cuda.memory_reserved() / 2**30, 3),
    )


class ProgressWriter:
    """Write progress, a log line and a VRAM sample together."""

    def __init__(
        self,
        root: Path,
        *,
        interval: float = INTERVAL_SECONDS,
        clock: Clock = time.monotonic,
        vram: VramReader | None = None,
    ) -> None:
        self.root = Path(root)
        self.interval = interval
        self.clock = clock
        self.vram = vram or sample_vram
        self.started = clock()
        self._last: float | None = None
        self._stage: str | None = None

    def note(
        self,
        *,
        step: int,
        total: int,
        loss: float | None,
        eta: int | None,
        stage: str,
        extra: dict[str, object] | None = None,
    ) -> bool:
        """Record one update. Returns whether this call wrote the files.

        ``extra`` is merged into ``train.json``. Round 1 uses it for the
        first-pictures ETA and, once measured, the seconds per sample.
        """
        now = self.clock()
        boundary = stage != self._stage
        due = self._last is None or (now - self._last) >= self.interval
        if not boundary and not due:
            return False
        self._stage = stage
        self._last = now
        elapsed = max(0.0, now - self.started)
        row: dict[str, object] = {
            "step": int(step),
            "total": int(total),
            "loss": _loss(loss),
            "elapsed_seconds": round(elapsed, 3),
            "eta_seconds": None if eta is None else int(eta),
            "stage": stage,
        }
        if extra:
            row.update(extra)
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic_fsync(self.root / TRAIN_FILE, json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        loss_text = "none" if row["loss"] is None else f"{row['loss']:.4f}"
        eta_text = "none" if row["eta_seconds"] is None else str(row["eta_seconds"])
        pictures = ""
        if "first_pictures_seconds" in row:
            pictures = f" first_pictures={row['first_pictures_seconds']}"
        line = (
            f"stage={stage} step={row['step']}/{row['total']} loss={loss_text} "
            f"elapsed={row['elapsed_seconds']:.0f}s eta={eta_text}{pictures}\n"
        )
        _append_fsync(self.root / LOG_FILE, line)
        allocated, reserved = self.vram()
        _append_vram(self.root / VRAM_FILE, elapsed, allocated, reserved)
        return True


def format_span(seconds: float, lang: str | None = None) -> str:
    """A duration in hours and minutes, for status text."""
    whole = max(0, int(seconds))
    if whole == 0:
        return text("lora.progress.span.minutes", lang, count=0)
    if whole < 30:
        return text("lora.progress.span.under", lang)
    minutes = (whole + 30) // 60
    if minutes < 60:
        return text("lora.progress.span.minutes", lang, count=minutes)
    hours, rest = divmod(minutes, 60)
    if rest == 0:
        return text("lora.progress.span.hours", lang, count=hours)
    return text("lora.progress.span.hours_minutes", lang, hours=hours, minutes=rest)


def first_pictures_text(row: dict[str, object], lang: str | None = None) -> str | None:
    """'First pictures in about N min', plus the per-sample time once it is measured."""
    raw = row.get("first_pictures_seconds")
    if not isinstance(raw, int | float):
        return None
    eta = format_span(float(raw), lang)
    sample = row.get("seconds_per_sample")
    estimated = bool(row.get("sample_seconds_estimated", True))
    if isinstance(sample, int | float) and not estimated:
        return text(
            "lora.status.first_pictures_rate", lang, eta=eta, seconds=max(1, round(float(sample))),
        )
    return text("lora.status.first_pictures", lang, eta=eta)


def read_progress(root: Path) -> dict[str, object] | None:
    """The latest ``train.json``, or ``None`` when training has not reported yet."""
    path = Path(root) / TRAIN_FILE
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None


def _loss(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):  # NaN or non-finite
        return None
    return round(number, 6)


def _atomic_fsync(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _append_fsync(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _append_vram(path: Path, elapsed: float, allocated: float | None, reserved: float | None) -> None:
    new = not path.is_file()
    with path.open("a", encoding="utf-8") as handle:
        if new:
            handle.write("elapsed_seconds,allocated_gb,reserved_gb\n")
        alloc = "" if allocated is None else f"{allocated:.3f}"
        hold = "" if reserved is None else f"{reserved:.3f}"
        handle.write(f"{elapsed:.3f},{alloc},{hold}\n")
        handle.flush()
        os.fsync(handle.fileno())
