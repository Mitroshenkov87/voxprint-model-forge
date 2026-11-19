"""Image-check handlers for the worker process.

The window names these callables as ``module:function`` strings. The child
imports this module, so the checker and torch stay out of the UI process.
A handler error raises. It is not reported as a photo result.
"""

from __future__ import annotations

from pathlib import Path

from voxprint_common.worker import JobCancelled, JobContext

from voxprint_forge.lora.image_check import ImageChecker


def load_model(model_id: str) -> ImageChecker:
    """Create the checker kept loaded for one batch of photos."""
    del model_id
    checker = ImageChecker()
    checker.ensure_loaded()
    return checker


def unload_model(model_id: str, loaded: object) -> None:
    """Drop the checker and any device memory it staged."""
    del model_id
    if isinstance(loaded, ImageChecker):
        loaded.unload()


def run(ctx: JobContext) -> object:
    """Reject a job that is not an image check."""
    del ctx
    raise RuntimeError("image checks use the check handler")


def check(ctx: JobContext) -> dict[str, object]:
    """Classify each photo in the payload.

    The return value is one row per photo. Cancelled work raises
    :class:`~voxprint_common.worker.JobCancelled` and returns nothing.
    """
    payload = ctx.payload
    if not isinstance(payload, dict):
        raise TypeError("image check payload must be an object")
    items = payload.get("items")
    if not isinstance(items, list):
        raise TypeError("image check items must be a list")
    checker = ctx.model_object if isinstance(ctx.model_object, ImageChecker) else ImageChecker()
    token = _Token(ctx)
    results: list[dict[str, object]] = []
    total = len(items)
    for index, item in enumerate(items, start=1):
        if ctx.cancelled():
            raise JobCancelled()
        photo_id, path = _item(item)
        ok, message = checker.classify(path, token)
        if ctx.cancelled():
            raise JobCancelled()
        results.append({"id": photo_id, "ok": ok, "message": message})
        ctx.progress(step=index, total_steps=total, fraction=(index / total) if total else 1.0)
    return {"results": results}


def _item(item: object) -> tuple[str, Path]:
    """Return the photo id and path, or raise when the row is not usable."""
    if not isinstance(item, dict):
        raise TypeError("image check item must be an object")
    photo_id = item.get("id")
    raw_path = item.get("path")
    if not isinstance(photo_id, str) or not photo_id or not isinstance(raw_path, str) or not raw_path:
        raise TypeError("image check item needs an id and a path")
    return photo_id, Path(raw_path)


class _Token:
    """The cancel bit :class:`~voxprint_forge.lora.image_check.ImageChecker` polls."""

    def __init__(self, ctx: JobContext) -> None:
        self._ctx = ctx

    @property
    def is_cancelled(self) -> bool:
        """True after the host has asked this job to stop."""
        return self._ctx.cancelled()
