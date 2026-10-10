"""Select canvas parts for the part D entrypoint. Unselected parts are omitted."""

from __future__ import annotations

from typing import Literal

from voxprint_forge.errors import UsageError

QwenPart = Literal["base", "controlnet", "qwen", "edit", "encoder"]
# Rental order: A, B, then D, then C, then E. `--parts C,D` therefore runs D first.
PART_ORDER: tuple[QwenPart, ...] = ("base", "controlnet", "qwen", "edit", "encoder")

_LETTERS: dict[str, QwenPart] = {
    "a": "base",
    "b": "controlnet",
    "c": "edit",
    "d": "qwen",
    "e": "encoder",
}
_IDS: dict[str, QwenPart] = {part: part for part in PART_ORDER}


def _one(token: str) -> QwenPart:
    key = token.strip().lower()
    if key in _LETTERS:
        return _LETTERS[key]
    if key in _IDS:
        return _IDS[key]
    raise UsageError(
        f"unknown part {token!r}; use A,B,C,D,E or base,controlnet,edit,qwen,encoder",
        code="parts",
    )


def parse_parts(text: str | None) -> tuple[QwenPart, ...]:
    """Default is D only. Letters and ids may be mixed. Order follows PART_ORDER."""
    raw = (text or "").strip()
    if not raw:
        return ("qwen",)
    chosen: set[QwenPart] = set()
    for token in raw.split(","):
        if not token.strip():
            raise UsageError(f"empty part in {text!r}", code="parts")
        chosen.add(_one(token))
    return tuple(part for part in PART_ORDER if part in chosen)
