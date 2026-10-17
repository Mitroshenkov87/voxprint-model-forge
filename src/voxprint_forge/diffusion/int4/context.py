"""Shared paths for one diffusion-int4 run."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class StageContext:
    """Inputs for the step functions. Callables are tests' seams."""

    repo: Path
    forge_out: Path
    dry_run: bool
    svd_rank: int
    free_ratio: Callable[[Path], float] | None = None
    hub: Callable[..., list[dict[str, object]]] | None = None
    drive: Callable[..., list[dict[str, object]]] | None = None

    def job(self, name: str) -> Path:
        """Work directory for one model."""
        return self.forge_out / "work" / "diffusion-int4" / name

    def tool(self, env_name: str, *parts: str) -> Path:
        """A checkout path from the environment, or under forge-out/src."""
        env = os.environ.get(env_name, "").strip()
        if env:
            return Path(env)
        return self.forge_out.joinpath("src", *parts)
