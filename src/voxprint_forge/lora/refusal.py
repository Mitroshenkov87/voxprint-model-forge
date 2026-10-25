"""Why simple mode set a photo aside, and whether Expert mode would keep the set.

Age and minors stay on the safety wording. Those refusals never grow an Expert hint:
Expert uses the same gate, so it cannot accept a photo the gate refused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from voxprint_forge.lora.events import Progress
from voxprint_forge.lora.i18n import text
from voxprint_forge.lora.limits import EXPERT_CAP, EXPERT_FLOOR, MIN_SIDE, RECOMMENDED_RANGE
from voxprint_forge.lora.photos import BLUR_THRESHOLD, DUP_THRESHOLD, Drop, filter_reason, open_photo

SAFETY = "lora.drop.safety"
FIX = {
    "lora.drop.blurry": "lora.refuse.fix.blurry",
    "lora.drop.collage": "lora.refuse.fix.collage",
    "lora.drop.duplicate": "lora.refuse.fix.duplicate",
    "lora.drop.small": "lora.refuse.fix.small",
    "lora.drop.unreadable": "lora.refuse.fix.unreadable",
}


@dataclass(frozen=True)
class Note:
    """One photo simple mode will not train on, and the ``lora.drop.*`` key."""

    name: str
    reason: str


@dataclass(frozen=True)
class Report:
    """Quality notes, safety notes, and how many photos can still be used."""

    notes: tuple[Note, ...]
    safety_notes: tuple[Note, ...]
    kept: int
    total: int
    blocked: bool = False

    def params(self) -> dict[str, int]:
        """Placeholders for the summary and the Expert sentence."""
        return {
            "dropped": len(self.notes) + len(self.safety_notes),
            "total": self.total,
            "kept": self.kept,
        }

    def summary_key(self) -> str:
        """Count line when photos were set aside, otherwise the simple-mode range line."""
        if self.notes or self.safety_notes:
            return "lora.refuse.line"
        return "lora.refuse.short"

    def fixes(self) -> tuple[str, ...]:
        """One plain fix per quality reason, in the order the photos were seen."""
        seen: list[str] = []
        for note in self.notes:
            key = FIX.get(note.reason)
            if key and key not in seen:
                seen.append(key)
        return tuple(seen)

    def show_expert(self) -> bool:
        """True when Expert would train on the photos simple mode is refusing.

        A safety drop or a blocked set never qualifies: Expert does not take those photos.
        """
        if self.blocked or self.safety_notes:
            return False
        low, high = RECOMMENDED_RANGE
        if low <= self.kept <= high:
            return False
        return EXPERT_FLOOR <= self.kept <= EXPERT_CAP

    def notable(self) -> bool:
        """Something new to say. A pure age or minors refusal stays on its own wording."""
        if self.blocked or (self.safety_notes and not self.notes):
            return False
        if self.notes:
            return True
        low, high = RECOMMENDED_RANGE
        return not (low <= self.kept <= high)


def outside_simple(count: int) -> bool:
    """True when ``count`` is outside the range simple mode accepts."""
    low, high = RECOMMENDED_RANGE
    return count < low or count > high


def visible_on_photos_step(report: Report, offered: int, *, expert: bool) -> bool:
    """The photos step explains a simple-mode count refusal, not a set simple mode still accepts."""
    if expert or offered <= 0 or report.blocked:
        return False
    if not outside_simple(offered):
        return False
    return bool(report.notes) or report.show_expert()


def scan_files(
    files: Sequence[Path],
    *,
    min_side: int = MIN_SIDE,
    blur_threshold: float = BLUR_THRESHOLD,
    dup_threshold: int = DUP_THRESHOLD,
) -> Report:
    """Local size, collage, blur and duplicate checks. The safety gate is not run."""
    seen: list[int] = []
    notes: list[Note] = []
    kept = 0
    for path in files:
        reason = filter_reason(
            open_photo(path),
            seen,
            min_side=min_side,
            blur_threshold=blur_threshold,
            dup_threshold=dup_threshold,
        )
        if reason:
            notes.append(Note(Path(path).name, reason))
        else:
            kept += 1
    return Report(tuple(notes), (), kept, len(files), False)


def report_from_drops(
    drops: Sequence[Drop | Mapping[str, object]],
    kept: int,
    total: int,
    *,
    blocked: bool = False,
) -> Report:
    """A report from ``drops.jsonl`` rows or :class:`Drop` values already recorded."""
    notes: list[Note] = []
    safety: list[Note] = []
    for item in drops:
        name, reason = _name_reason(item)
        if not reason:
            continue
        note = Note(name, reason)
        if reason == SAFETY:
            safety.append(note)
        else:
            notes.append(note)
    return Report(tuple(notes), tuple(safety), kept, total, blocked)


def cli_lines(report: Report, lang: str | None = None) -> tuple[str, ...]:
    """A summary line and, when there is a way forward, one hint line."""
    if not report.notable():
        return ()
    values = report.params()
    summary = text(report.summary_key(), lang, **values)
    parts = [text(key, lang, **values) for key in report.fixes()]
    if report.show_expert():
        parts.append(text("lora.refuse.expert", lang, **values))
    hint = " ".join(part for part in parts if part)
    if hint:
        return (summary, hint)
    return (summary,)


def announce(progress: Progress, report: Report) -> None:
    """Print the summary and the hint. A quiet report prints nothing."""
    spoken = cli_lines(report)
    if not spoken:
        return
    progress.emit("prepare.refuse", report.summary_key(), report.params())
    if len(spoken) > 1:
        progress.emit("prepare.refuse", "lora.refuse.hint", {"hint": spoken[1]})


def _name_reason(item: Drop | Mapping[str, object]) -> tuple[str, str]:
    if isinstance(item, Drop):
        return item.name, item.reason
    name = str(item.get("name") or "")
    if not name:
        name = Path(str(item.get("file") or "")).name
    return name, str(item.get("reason") or "")
