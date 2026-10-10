"""Per-item state lines. ALL-DONE is included only when nothing failed."""

from __future__ import annotations

from voxprint_forge.diffusion.deadline import DEFERRED


def state_line(payload: dict[str, object]) -> str:
    state = payload.get("state")
    if isinstance(state, str) and state:
        return state
    status = str(payload.get("status", ""))
    if payload.get("skipped") or status in {DEFERRED, "DEFERRED"}:
        return "DEFERRED"
    if payload.get("deferred_upload") or status == "upload-deferred":
        return "READY"
    if status.startswith("STAGEFAIL"):
        return status
    return "UPLOADED"


def summarise(items: list[dict[str, object]]) -> tuple[str, bool]:
    lines = [f"{item.get('id')} {state_line(item)}" for item in items]
    failed = any(line.split(" ", 1)[-1].startswith("STAGEFAIL") for line in lines)
    if not failed:
        lines.append("ALL-DONE")
    return "\n".join(lines), not failed
