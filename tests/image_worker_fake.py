"""Stand-in image-check handlers for the worker process.

The window names these as ``module:function`` strings. This module does not
import the checker or torch. The mode comes from the environment the child
inherited when it was spawned.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from voxprint_common.worker import JobCancelled, JobContext


def load_model(model_id: str) -> str:
    """Pretend to keep one checker loaded for the batch."""
    del model_id
    return "fake-checker"


def unload_model(model_id: str, loaded: object) -> None:
    """Note that the child unloaded, when a test asked for a marker file."""
    del model_id, loaded
    marker = os.environ.get("VOXPRINT_TEST_UNLOAD_MARKER", "")
    if marker:
        Path(marker).write_text("unloaded", encoding="utf-8")


def run(ctx: JobContext) -> object:
    """Reject a job that is not an image check."""
    del ctx
    raise RuntimeError("image checks use the check handler")


def check(ctx: JobContext) -> dict[str, object]:
    """Return the rows the test mode asked for, or raise, or leave the process."""
    mode = os.environ.get("VOXPRINT_TEST_IMAGE_MODE", "")
    if mode == "fail":
        raise RuntimeError("check failed")
    if mode == "crash":
        os._exit(91)
    if mode == "wait":
        while not ctx.cancelled():
            ctx.beat()
            time.sleep(0.05)
        raise JobCancelled()
    payload = ctx.payload
    if not isinstance(payload, dict):
        raise TypeError("image check payload must be an object")
    items = payload.get("items")
    if not isinstance(items, list):
        raise TypeError("image check items must be a list")
    refuse = {item for item in os.environ.get("VOXPRINT_TEST_IMAGE_REFUSE", "").split(",") if item}
    results: list[dict[str, object]] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        if ctx.cancelled():
            raise JobCancelled()
        if not isinstance(item, dict):
            raise TypeError("image check item must be an object")
        photo_id = item.get("id")
        if not isinstance(photo_id, str) or not photo_id:
            raise TypeError("image check item needs an id")
        if mode == "bad":
            results.append({"id": photo_id, "ok": 0, "message": "lora.drop.collage"})
        elif photo_id in refuse:
            results.append({"id": photo_id, "ok": False, "message": "lora.drop.collage"})
        else:
            results.append({"id": photo_id, "ok": True, "message": ""})
        ctx.progress(step=index, total_steps=total, fraction=(index / total) if total else 1.0)
    return {"results": results}
