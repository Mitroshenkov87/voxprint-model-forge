"""Stage order and the marker reset the operator window uses.

Importing this module does not import torch. :data:`ORDER` is the stage
commands from fetch through upload. ``imatrix`` is a marker, not a command.
"""

from __future__ import annotations

from voxprint_forge.errors import UsageError
from voxprint_forge.layout import Layout

ORDER = ("fetch", "abliterate", "quantize", "fit", "eval", "card", "publish-prep", "upload")


def retry_stage(stage: str) -> str:
    """Stage command that restarts `stage`. `imatrix` restarts at quantize."""
    if stage in {"", "start"}:
        return "fetch"
    if stage == "imatrix":
        return "quantize"
    if stage in ORDER:
        return stage
    raise UsageError(f"unknown stage {stage}", code="usage")


def clear_from_stage(layout: Layout, model_id: str, stage: str) -> list[str]:
    """Delete markers from `stage` through upload so the stage command runs that work again."""
    command = retry_stage(stage)
    start = ORDER.index(command)
    names = list(ORDER[start:])
    if start <= ORDER.index("quantize"):
        names.append("imatrix")
    if "upload" in names:
        names.append("upload-deferred")
    names.append("skipped")
    removed: list[str] = []
    for name in names:
        path = layout.stage(model_id, name)
        if path.is_file():
            path.unlink()
            removed.append(name)
    return removed
