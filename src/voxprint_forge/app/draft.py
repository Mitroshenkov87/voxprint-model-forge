"""Headless state of the "Train a model" steps. Qt is not imported.

Each step has a validator that returns a :class:`Check`: whether Next is
allowed and, if not, the catalog key and parameters of the plain reason.
Photo limits follow plan §9.7: Simple mode is a hard 10-30; Expert mode warns
outside 10-30 and has a hard floor of 5 and a cap of ``EXPERT_CAP``. The
limits are the engine's own (:mod:`voxprint_forge.lora.photos`).
The safety result comes from the engine and cannot be changed here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from voxprint_forge.lora.photos import EXPERT_CAP, EXPERT_FLOOR, EXTENSIONS, SIMPLE_RANGE

SIMPLE_MIN, SIMPLE_MAX = SIMPLE_RANGE
PHOTO_SUFFIXES = EXTENSIONS
KINDS = ("person", "object", "style")
__all__ = ["EXPERT_CAP", "EXPERT_FLOOR", "SIMPLE_MAX", "SIMPLE_MIN", "Check", "Draft", "Prepared"]
STEPS = ("photos", "subject", "prepare", "train", "result")


@dataclass(frozen=True)
class Check:
    """Whether a step may continue, plus a reason or warning key."""

    ok: bool
    key: str = ""
    params: dict[str, object] = field(default_factory=dict)
    warning: bool = False


@dataclass
class Prepared:
    """What ``lora prepare`` reported: kept photos, drops with reasons, and a block."""

    kept: list[dict[str, object]] = field(default_factory=list)
    drops: list[dict[str, object]] = field(default_factory=list)
    blocked: bool = False


@dataclass
class Draft:
    """Everything the user entered, in plan §3 terms."""

    expert: bool = False
    photos: list[Path] = field(default_factory=list)
    kind: str = ""
    name: str = ""
    stage_name: str = ""
    trigger: str = ""
    trigger_edited: bool = False
    consent: bool = False
    adult: bool = False
    rights: bool = False
    release: Path | None = None
    note: str = ""
    crop: str = "subject"
    prepared: Prepared | None = None
    trained: bool = False
    checkpoint: int | None = None
    extend_from: Path | None = None

    def add_photos(self, paths: list[Path]) -> int:
        """Add image files (folders are expanded). Returns how many were new."""
        seen = {path.resolve() for path in self.photos}
        added = 0
        for path in paths:
            items = sorted(path.iterdir()) if path.is_dir() else [path]
            for item in items:
                if item.suffix.lower() not in PHOTO_SUFFIXES or not item.is_file():
                    continue
                resolved = item.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                self.photos.append(item)
                added += 1
        if added:
            self.prepared = None
            self.trained = False
        return added

    def remove_photo(self, path: Path) -> None:
        """Drop one photo from training. The file on disk stays."""
        self.photos = [item for item in self.photos if item != path]
        self.prepared = None
        self.trained = False

    def set_names(self, name: str, stage_name: str) -> None:
        """Store the names and pre-fill the trigger word unless the user edited it."""
        self.name, self.stage_name = name.strip(), stage_name.strip()
        if not self.trigger_edited:
            self.trigger = suggest_trigger(self.stage_name or self.name)

    def check_photos(self) -> Check:
        """Photo count and kind (step 1)."""
        count = len(self.photos)
        if self.expert:
            if count < EXPERT_FLOOR:
                return Check(False, "lora.err.below_floor")
            if count > EXPERT_CAP:
                return Check(False, "lora.err.above_cap", {"cap": EXPERT_CAP})
        else:
            if count < SIMPLE_MIN:
                return Check(False, "lora.err.too_few", {"need": SIMPLE_MIN - count})
            if count > SIMPLE_MAX:
                return Check(False, "lora.err.too_many", {"extra": count - SIMPLE_MAX})
        if self.kind not in KINDS:
            return Check(False, "forge.app.err.no_kind")
        if self.expert and not SIMPLE_MIN <= count <= SIMPLE_MAX:
            return Check(True, "lora.warn.count", {"count": count}, warning=True)
        return Check(True)

    def check_subject(self) -> Check:
        """Names and the consent box for the kind (step 2)."""
        if not (self.name or self.stage_name):
            return Check(False, "lora.err.no_name")
        if not self.trigger.strip():
            return Check(False, "forge.app.err.no_trigger")
        if self.kind == "person" and not (self.consent and self.adult):
            return Check(False, "lora.err.no_consent")
        if self.kind in ("object", "style") and not self.rights:
            return Check(False, "lora.err.no_rights")
        return Check(True)

    def check_prepare(self) -> Check:
        """The engine's preparation result (step 3). A safety block is final."""
        if self.prepared is None:
            return Check(False, "forge.app.err.not_prepared")
        if self.prepared.blocked:
            return Check(False, "lora.err.blocked")
        kept = len(self.prepared.kept)
        need = EXPERT_FLOOR if self.expert else SIMPLE_MIN
        if kept < need:
            return Check(False, "lora.err.too_few", {"need": need - kept})
        return Check(True)

    def check_train(self) -> Check:
        """Training finished (step 4)."""
        return Check(True) if self.trained else Check(False, "forge.app.err.not_trained")

    def check(self, step: str) -> Check:
        """The validator for ``step``; every earlier step must pass too."""
        validators = {
            "photos": self.check_photos,
            "subject": self.check_subject,
            "prepare": self.check_prepare,
            "train": self.check_train,
            "result": lambda: Check(True),
        }
        for name in STEPS:
            result = validators[name]()
            if name == step or not result.ok:
                return result
        raise ValueError(f"unknown step {step!r}")



def suggest_trigger(name: str) -> str:
    """A lower-case one-word trigger from a name: ``"Mira Stone"`` -> ``"mira_stone"``."""
    word = re.sub(r"[^\w]+", "_", name.strip().lower(), flags=re.UNICODE).strip("_")
    return word[:40]
