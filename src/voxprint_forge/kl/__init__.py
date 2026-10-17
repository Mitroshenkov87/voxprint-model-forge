"""Perplexity and KL stage for the synth4 L-jobs.

The base is an F16 GGUF. Each quant is scored with llama.cpp b11476
``--kl-divergence``. Summaries record metrics and the 16/24 GB fit. They do
not record prompts.
"""

from __future__ import annotations

from voxprint_forge.kl.plan import load_kl_plan
from voxprint_forge.kl.run import run_kl

__all__ = ["load_kl_plan", "run_kl"]
