"""Review-sample prompts: trigger, a framing, and one attire word.

Framings rotate ``close-up portrait``, ``waist-up``, ``full body``. The start
index is ``(seed + round number) % 3``. ``auto`` resolves to ``unclothed`` when
any caption sets ``unclothed``, and to ``as_photos`` otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, Never

from voxprint_forge.lora.captioner import text_marks_unclothed
from voxprint_forge.lora.project import Project

ATTIRE_MODES = ("auto", "as_photos", "clothed", "unclothed")
FRAMINGS = ("close-up portrait", "waist-up", "full body")
CLOTHED = "clothed"
UNCLOTHED = "unclothed"
SAMPLE_ATTIRE = "sample_attire"

AttireMode = Literal["auto", "as_photos", "clothed", "unclothed"]
ResolvedAttire = Literal["as_photos", "clothed", "unclothed"]


def marks_unclothed(caption: Mapping[str, object]) -> bool:
    """True when this caption record marks unclothed.

    A boolean ``unclothed`` field wins. When that field is absent, the long and
    short text are checked.
    """
    flag = caption.get("unclothed")
    if isinstance(flag, bool):
        return flag
    return text_marks_unclothed(str(caption.get("long") or ""), str(caption.get("short") or ""))


def resolve_attire(mode: str, captions: Sequence[Mapping[str, object]]) -> ResolvedAttire:
    """``auto`` becomes ``unclothed`` or ``as_photos``. The other modes stay."""
    choice = _mode(mode)
    marked = any(marks_unclothed(caption) for caption in captions)
    match choice:
        case "auto":
            return "unclothed" if marked else "as_photos"
        case "as_photos":
            return "as_photos"
        case "clothed":
            return "clothed"
        case "unclothed":
            return "unclothed"
        case _ as unexpected:
            return _unexpected_mode(unexpected)


def attire_word(resolved: ResolvedAttire, captions: Sequence[Mapping[str, object]]) -> str:
    """One attire word. ``as_photos`` does not add clothing the captions do not have."""
    match resolved:
        case "unclothed":
            return UNCLOTHED
        case "clothed":
            return CLOTHED
        case "as_photos":
            return UNCLOTHED if any(marks_unclothed(caption) for caption in captions) else CLOTHED
        case _ as unexpected:
            return _unexpected_resolved(unexpected)


def build_sample_prompts(
    trigger: str,
    *,
    count: int,
    seed: int,
    round_no: int,
    mode: str,
    captions: Sequence[Mapping[str, object]],
) -> list[str]:
    """``count`` prompts for one round. The same seed and round repeat the list."""
    word = attire_word(resolve_attire(mode, captions), captions)
    total = max(0, int(count))
    if total == 0:
        return []
    start = (int(seed) + int(round_no)) % len(FRAMINGS)
    prompts = []
    for index in range(total):
        framing = FRAMINGS[(start + index) % len(FRAMINGS)]
        prompts.append(_compose(trigger, framing, word))
    return prompts


def read_attire(data: Mapping[str, object]) -> str:
    """The project's sample-attire mode, or ``auto`` when it is missing."""
    raw = data.get(SAMPLE_ATTIRE)
    if isinstance(raw, str) and raw in ATTIRE_MODES:
        return raw
    return "auto"


def store_attire(project: Project, mode: str) -> str:
    """Write ``sample_attire`` on ``project.json`` and return the mode."""
    choice = _mode(mode)
    data = project.read()
    if data.get(SAMPLE_ATTIRE) != choice:
        project.write({**data, SAMPLE_ATTIRE: choice})
    return choice


def caption_rows(project: Project) -> list[dict[str, object]]:
    """Prepared captions, including ``unclothed`` when prepare stored it."""
    rows: list[dict[str, object]] = []
    for item in project.items():
        row: dict[str, object] = {"long": item.long, "short": item.short}
        if isinstance(item.unclothed, bool):
            row["unclothed"] = item.unclothed
        rows.append(row)
    return rows


def _compose(trigger: str, framing: str, word: str) -> str:
    parts = [part.strip() for part in (trigger, framing, word) if part and part.strip()]
    return ", ".join(parts)


def _mode(mode: str) -> AttireMode:
    if mode not in ATTIRE_MODES:
        raise ValueError(mode)
    match mode:
        case "auto" | "as_photos" | "clothed" | "unclothed":
            return mode
        case _ as unexpected:
            return _unexpected_mode(unexpected)


def _unexpected_mode(mode: Never) -> Never:
    raise AssertionError(mode)


def _unexpected_resolved(mode: Never) -> Never:
    raise AssertionError(mode)
