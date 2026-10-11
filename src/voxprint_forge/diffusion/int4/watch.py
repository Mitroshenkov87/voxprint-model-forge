"""``~/watch.json`` lists the expected jobs, which are done, and the planning time."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from voxprint_forge.diffusion.int4.gpu import planning_gpu, step_minutes, total_minutes
from voxprint_forge.diffusion.int4.pins import SCHEMA, StepEstimate, planned_steps


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
    gpu: str | None = None,
    mirror: bool = False,
) -> dict[str, object]:
    """Watch document for the selected steps on one planning GPU."""
    chosen = gpu if gpu in {"a100", "h100"} else planning_gpu()
    steps = planned_steps(qwen_image=qwen_image, mirror=mirror)
    jobs: list[dict[str, object]] = []
    for step in steps:
        state = "done" if step.step_id in done else "running" if step.step_id == running else "pending"
        jobs.append(
            {
                "id": step.step_id,
                "expected_min": step_minutes(step, chosen),
                "state": state,
                "note": step.note,
            }
        )
    finished = [step.step_id for step in steps if step.step_id in done]
    return {
        "schema": SCHEMA,
        "updated": _now(),
        "gpu": chosen,
        "expected_min": total_minutes(steps, chosen),
        "done": finished,
        "jobs": jobs,
    }


def write_watch(path: Path, payload: dict[str, object]) -> None:
    """Atomically replace the watch file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


_GPU_LABEL = {"a100": "A100 80GB", "h100": "H100 80GB"}


def estimates_text(steps: tuple[StepEstimate, ...], gpu: str) -> str:
    """Plain-text planning table for one GPU."""
    lines = [_GPU_LABEL.get(gpu, gpu)]
    lines.extend(f"{step.step_id}  {step_minutes(step, gpu)} min  {step.note}" for step in steps)
    lines.append(f"total  {total_minutes(steps, gpu)} min")
    return "\n".join(lines)


def estimates_report(steps: tuple[StepEstimate, ...]) -> str:
    """A100 and H100 tables, plus which table this process will follow."""
    return "\n\n".join(
        [
            f"planning gpu: {planning_gpu()}",
            estimates_text(steps, "a100"),
            estimates_text(steps, "h100"),
        ]
    )
