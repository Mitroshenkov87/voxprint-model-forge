"""Original-versus-compressed compare harness for the synth4 queue.

Private prompt text stays in memory. Results, logs, and summaries record
``sha256(prompt)[:12]`` only.
"""

from __future__ import annotations

from typing import Any

__all__ = ["load_compare_plan", "prompt_id", "run_compare"]


def __getattr__(name: str) -> Any:
    """Return a public name without importing the compare runner at package import time."""
    if name == "load_compare_plan":
        from voxprint_forge.compare.plan import load_compare_plan

        return load_compare_plan
    if name == "prompt_id":
        from voxprint_forge.compare.privacy import prompt_id

        return prompt_id
    if name == "run_compare":
        from voxprint_forge.compare.run import run_compare

        return run_compare
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
