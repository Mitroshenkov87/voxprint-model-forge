"""Perplexity and KL stage for the synth4 L-jobs.

The base is an F16 GGUF. Each quant is scored with llama.cpp b11476
``--kl-divergence``. Summaries record metrics and the 16/24 GB fit. They do
not record prompts.
"""

from __future__ import annotations

from typing import Any

__all__ = ["load_kl_plan", "run_kl"]


def __getattr__(name: str) -> Any:
    """Return a public name without importing the KL runner at package import time."""
    if name == "load_kl_plan":
        from voxprint_forge.kl.plan import load_kl_plan

        return load_kl_plan
    if name == "run_kl":
        from voxprint_forge.kl.run import run_kl

        return run_kl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
