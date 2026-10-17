"""Original-versus-compressed compare harness for the synth4 queue.

Private prompt text stays in memory. Results, logs, and summaries record
``sha256(prompt)[:12]`` only.
"""

from __future__ import annotations

from voxprint_forge.compare.plan import load_compare_plan
from voxprint_forge.compare.privacy import prompt_id
from voxprint_forge.compare.run import run_compare

__all__ = ["load_compare_plan", "prompt_id", "run_compare"]
