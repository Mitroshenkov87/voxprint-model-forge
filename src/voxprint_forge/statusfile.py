"""Atomic ~/forge-out/status.json and a heartbeat during long silent stages."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path

from voxprint_forge.diffusion.guard import _pid_alive
from voxprint_forge.progress import append_progress

STALL_S = 180
HEARTBEAT_S = 60
_LOCK = threading.Lock()
_STAGE: ContextVar[str] = ContextVar("forge_stage", default="start")


def set_stage(name: str) -> None:
    """Remember the stage name heartbeats should report."""
    _STAGE.set(name)


def current_stage() -> str:
    """Stage name set by ``set_stage``, or an empty string."""
    return _STAGE.get()


def status_path(forge_out: Path) -> Path:
    """Path of ``status.json`` under forge-out."""
    return forge_out / "status.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"updated": _now(), "items": {}}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return {"updated": _now(), "items": {}}
    items = loaded.get("items")
    if not isinstance(items, dict):
        loaded["items"] = {}
    return loaded


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def update_item(
    forge_out: Path,
    item_id: str,
    *,
    stage: str,
    pct: int,
    eta_min: float,
    state: str,
    error: str = "",
    pid: int | None = None,
    uploads: dict[str, str] | None = None,
) -> None:
    """Update one model's row in ``status.json``."""
    path = status_path(forge_out)
    with _LOCK:
        payload = _read(path)
        items = payload.get("items")
        if not isinstance(items, dict):
            items = {}
            payload["items"] = items
        previous = items.get(item_id)
        kept: dict[str, str] | None = None
        if isinstance(previous, dict) and isinstance(previous.get("uploads"), dict):
            kept = {str(key): str(value) for key, value in previous["uploads"].items()}
        entry: dict[str, object] = {
            "stage": stage,
            "pct": pct,
            "eta_min": eta_min,
            "last_heartbeat": _now(),
            "pid": os.getpid() if pid is None else pid,
            "state": state,
            "error": error,
        }
        chosen = uploads if uploads is not None else kept
        if chosen is not None:
            entry["uploads"] = chosen
        items[item_id] = entry
        payload["updated"] = _now()
        _write(path, payload)


def read_status(forge_out: Path) -> dict[str, object]:
    """Load ``status.json``, or an empty document when it is missing."""
    return _read(status_path(forge_out))


def _age_s(stamp: str, now: datetime) -> float:
    try:
        seen = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return STALL_S + 1
    return (now - seen).total_seconds()


def inspect_status(
    forge_out: Path,
    *,
    now: datetime | None = None,
    alive: Callable[[int], bool] | None = None,
) -> dict[str, object]:
    """Flag a running item as stalled when its heartbeat is older than 180 s or its pid is gone."""
    clock = now or datetime.now(timezone.utc)
    probe = alive or _pid_alive
    payload = read_status(forge_out)
    items = payload.get("items")
    rows: list[dict[str, object]] = []
    stalled = False
    failed = False
    if isinstance(items, dict):
        for item_id, raw in items.items():
            if not isinstance(raw, dict):
                continue
            state = str(raw.get("state", ""))
            stamp = str(raw.get("last_heartbeat", ""))
            pid = int(raw.get("pid", 0) or 0)
            age = _age_s(stamp, clock)
            live = probe(pid) if pid else False
            flagged = state
            if state in {"RUNNING", "READY"} and (age > STALL_S or not live):
                flagged = "stalled"
            if flagged == "stalled":
                stalled = True
            if state.startswith("STAGEFAIL") or state == "FAILED":
                failed = True
            rows.append({**raw, "id": item_id, "state": flagged, "pid_alive": live, "heartbeat_age_s": age})
    exit_code = 4 if failed else 3 if stalled else 0
    return {"items": rows, "stalled": stalled, "failed": failed, "exit_code": exit_code}


class Heartbeat:
    """Write a progress line and status.json at least every 60 s while a stage is silent."""

    def __init__(
        self,
        forge_out: Path,
        item_id: str,
        stage: str,
        *,
        pct: int = 0,
        eta_min: float = 0,
        interval_s: float | None = None,
    ) -> None:
        self.forge_out = forge_out
        self.item_id = item_id
        self.stage = stage
        self.pct = pct
        self.eta_min = eta_min
        raw = os.environ.get("FORGE_HEARTBEAT_S", "")
        self.interval_s = float(raw) if raw else (HEARTBEAT_S if interval_s is None else interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def pulse(self) -> None:
        """Rewrite the status row so a watcher can see the process is alive."""
        update_item(
            self.forge_out,
            self.item_id,
            stage=self.stage,
            pct=self.pct,
            eta_min=self.eta_min,
            state="RUNNING",
        )
        append_progress(
            self.forge_out / "progress.log",
            model=self.item_id,
            stage=self.stage,
            pct=self.pct,
            eta_min=self.eta_min,
        )

    def __enter__(self) -> Heartbeat:
        self.pulse()
        self._thread = threading.Thread(target=self._loop, name=f"heartbeat-{self.item_id}", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.pulse()


@contextmanager
def beating(
    forge_out: Path,
    item_id: str,
    stage: str,
    *,
    pct: int = 0,
    eta_min: float = 0,
) -> Iterator[None]:
    """Context manager that pulses the status row until the stage ends."""
    with Heartbeat(forge_out, item_id, stage, pct=pct, eta_min=eta_min):
        yield
