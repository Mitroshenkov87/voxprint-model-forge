"""Crash-safe marker for one ``lora prepare`` safety check.

``gate_run.json`` is written before the first photo and set to ``complete`` only when
prepare finishes. Each trace row and each safety drop is appended and synced on its own.
Training will not start while the marker is still open.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from voxprint_forge.lora.gate import RULE_VERSION
from voxprint_forge.lora.job_gate import MANIFEST
from voxprint_forge.lora.project import Project


def start(project: Project) -> dict[str, object]:
    """Write ``gate_run.json`` with ``state`` ``running`` before the first photo."""
    project.root.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": uuid.uuid4().hex,
        "rule_version": RULE_VERSION,
        "gate_sha": gate_sha(project),
        "started": _now(),
        "state": "running",
    }
    _atomic_json(project.gate_run, payload)
    return payload


def complete(project: Project) -> None:
    """Mark the current run ``complete`` and store the sha256 of ``gate.json``."""
    current = _read(project)
    payload = {
        "run_id": str(current.get("run_id") or uuid.uuid4().hex),
        "rule_version": RULE_VERSION,
        "gate_sha": gate_sha(project),
        "started": str(current.get("started") or _now()),
        "state": "complete",
    }
    _atomic_json(project.gate_run, payload)


def open_gate_run(project: Project) -> bool:
    """True when prepare did not finish.

    A missing marker is an older project and is not an open run. ``running`` and
    ``interrupted`` are open. A photo with no trace row from that run was not checked
    and must not be treated as kept. Callers refuse training and write nothing.
    """
    if not project.gate_run.is_file():
        return False
    try:
        data = json.loads(project.gate_run.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return True
    if not isinstance(data, dict):
        return True
    return data.get("state") != "complete"


def gate_sha(project: Project) -> str:
    """sha256 of the ``gate.json`` pin, or ``""`` when prepare has not pinned yet."""
    path = project.root / MANIFEST
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append_trace_row(project: Project, row: dict[str, object]) -> None:
    """Append one ``gate_trace.jsonl`` row and sync it to disk."""
    append_line(project.gate_trace, json.dumps(row, ensure_ascii=False) + "\n")


def append_drop(project: Project, row: dict[str, object]) -> None:
    """Append one ``drops.jsonl`` row and sync it to disk."""
    append_line(project.drops, json.dumps(row, ensure_ascii=False) + "\n")


def trace_line_count(project: Project) -> int:
    """How many lines ``gate_trace.jsonl`` has, or 0 when the file is absent."""
    path = project.gate_trace
    if not path.is_file():
        return 0
    return len(path.read_text(encoding="utf-8").splitlines())


def rewrite_trace_tail(project: Project, start: int, rows: Sequence[dict[str, object]]) -> None:
    """Replace trace lines from ``start`` with ``rows``, keeping every earlier line.

    Used when the cross-photo average changes a tier after each photo was already
    appended. Earlier runs' rows stay. The write is a temp file plus ``os.replace``.
    """
    path = project.gate_trace
    existing = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    lines = existing[:start] + [json.dumps(row, ensure_ascii=False) for row in rows]
    atomic_text(path, "".join(line + "\n" for line in lines))


def append_line(path: Path, text: str) -> None:
    """Append ``text`` and ``fsync`` so a crash keeps the line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_text(path: Path, text: str) -> None:
    """Write ``text`` to a temp file, sync it, and ``os.replace`` onto ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read(project: Project) -> dict[str, object]:
    if not project.gate_run.is_file():
        return {}
    try:
        data = json.loads(project.gate_run.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
