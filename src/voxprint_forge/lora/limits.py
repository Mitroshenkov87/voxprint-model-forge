"""How many photos a LoRA trains on: the one place the numbers live.

``RECOMMENDED_RANGE`` is what every UI string, the CLI help and the manual call the
recommended set size (Simple mode enforces it). ``EXPERT_RANGE`` is what Expert mode
allows; the UI names it ("Allowed in Expert") only when it really differs.
"""

from __future__ import annotations

RECOMMENDED_RANGE = (10, 30)
EXPERT_FLOOR = 5
EXPERT_CAP = 60
EXPERT_RANGE = (EXPERT_FLOOR, EXPERT_CAP)
MIN_SIDE = 512


def expert_range_differs() -> bool:
    """True when Expert mode allows a range other than the recommended one."""
    return EXPERT_RANGE != RECOMMENDED_RANGE


def text_params() -> dict[str, int]:
    """Default placeholders every catalog string may use: ``{rec_low}``–``{rec_high}``, the Expert range, MIN_SIDE."""
    return {
        "rec_low": RECOMMENDED_RANGE[0],
        "rec_high": RECOMMENDED_RANGE[1],
        "expert_low": EXPERT_RANGE[0],
        "expert_high": EXPERT_RANGE[1],
        "min_side": MIN_SIDE,
    }
