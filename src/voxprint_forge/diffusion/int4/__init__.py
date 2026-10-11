"""SVDQuant INT4 for Qwen-Image-Edit and GGUF/FP8 for Wan 2.2.

``run`` is loaded on first use. Importing this package must not import
``stage``, which pulls ``voxprint_common``. The laptop smoke venv is
Python 3.12 for the Windows nunchaku wheel and does not install that library.
"""

from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    """Return ``run`` without importing it at package import time."""
    if name == "run":
        from voxprint_forge.diffusion.int4.stage import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["run"]
