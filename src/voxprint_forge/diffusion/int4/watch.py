"""``~/watch.json`` lists the expected jobs, which are done, and the planning time."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from voxprint_forge.diffusion.int4.pins import SCHEMA, StepEstimate, expected_minutes, planned_steps


def watch_path(explicit: Path | None = None) -> Path:
    """Watch file. ``FORGE_WATCH`` overrides ``~/watch.json``."""
    if explicit is not None:
        return explicit
    env = os.environ.get("FORGE_WATCH", "").strip()
    if env:
        return Path(env)
    return Path.home() / "watch.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_watch(
    *,
    qwen_image: bool,
    done: set[str],
    running: str = "",
) -> dict[str, object]:
    """Watch document for the selected steps."""
    steps = planned_steps(qwen_image=qwen_image)
    jobs: list[dict[str, object]] = []
    for step in steps:
        state = "done" if step.step_id in done else "running" if step.step_id == running else "pending"
        jobs.append(
            {
                "id": step.step_id,
                "expected_min": step.expected_min,
                "state": state,
                "note": step.note,
            }
        )
    finished = [step.step_id for step in steps if step.step_id in done]
    return {
        "schema": SCHEMA,
        "updated": _now(),
        "expected_min": expected_minutes(qwen_image=qwen_image),
        "done": finished,
        "jobs": jobs,
    }


def write_watch(path: Path, payload: dict[str, object]) -> None:
    """Atomically replace the watch file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def estimates_text(steps: tuple[StepEstimate, ...]) -> str:
    """Plain-text planning table."""
    lines = [f"{step.step_id}  {step.expected_min} min  {step.note}" for step in steps]
    total = sum(step.expected_min for step in steps)
    lines.append(f"total  {total} min")
    return "\n".join(lines)
