"""Preview picture shape: vertical 2:3 at the training resolution.

Test versions read as phone-style full-figure pictures rather than squares: the
height stays at the training resolution and the width is 2/3 of it, rounded to a
multiple of 16 so the latent grid stays aligned. A 688x1024 picture is smaller
than the 1024x1024 square the memory plan budgets, so the VRAM estimates stay
safe. No torch: this module imports anywhere.
"""

from __future__ import annotations

PREVIEW_ASPECT = (2, 3)


def preview_size(resolution: int) -> tuple[int, int]:
    """``(width, height)`` for one preview picture at ``resolution``: vertical 2:3, width a multiple of 16."""
    height = int(resolution)
    width = round(height * PREVIEW_ASPECT[0] / PREVIEW_ASPECT[1] / 16) * 16
    return width, height
