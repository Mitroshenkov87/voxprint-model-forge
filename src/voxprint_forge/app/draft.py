"""Headless state of the "Train a model" steps. Qt is not imported.

Each step has a validator that returns a :class:`Check`: whether Next is
allowed and, if not, the catalog key and parameters of the plain reason.
Photo limits follow plan §9.7: Simple mode is a hard ``RECOMMENDED_RANGE``; Expert
mode warns outside it and has a hard floor ``EXPERT_FLOOR`` and a cap ``EXPERT_CAP``.
The numbers live in :mod:`voxprint_forge.lora.limits`.
The safety result comes from the engine and cannot be changed here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from voxprint_forge.lora.attr import accepted
from voxprint_forge.lora.limits import need_more
from voxprint_forge.lora.photos import EXPERT_CAP, EXPERT_FLOOR, EXTENSIONS, RECOMMENDED_RANGE
from voxprint_forge.lora.refset import REF_RANGE
from voxprint_forge.lora.trigger import looks_like_words, rare_trigger

SIMPLE_MIN, SIMPLE_MAX = RECOMMENDED_RANGE
PHOTO_SUFFIXES = EXTENSIONS
KINDS = ("person", "object", "style")
SOURCES = ("photos", "refs")
REF_MIN, REF_MAX = REF_RANGE
__all__ = ["EXPERT_CAP", "EXPERT_FLOOR", "SIMPLE_MAX", "SIMPLE_MIN", "Check", "Draft", "Prepared", "photo_key"]
STEPS = ("photos", "subject", "refset", "prepare", "train", "result")


def photo_key(path: Path) -> str:
    """Stable id of one photo file, used by the dataset check and the include switch."""
    return str(Path(path).resolve())


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
    verbal: bool = False
    adult: bool = False
    adult_content: bool = False
    attr: int | None = None
    rights: bool = False
    release: Path | None = None
    note: str = ""
    crop: str = "subject"
    prepared: Prepared | None = None
    trained: bool = False
    checkpoint: int | None = None
    extend_from: Path | None = None
    source: str = "photos"
    approved: list[Path] = field(default_factory=list)
    inactive: set[str] = field(default_factory=set)
    photo_notes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fill an empty trigger once. Later name edits do not replace it."""
        if not self.trigger.strip():
            self.trigger = rare_trigger()

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
            self.approved = []
        return added

    def remove_photo(self, path: Path) -> None:
        """Drop one photo from training. The file on disk stays."""
        key = photo_key(path)
        self.photos = [item for item in self.photos if item != path and photo_key(item) != key]
        self.inactive.discard(key)
        self.photo_notes.pop(key, None)
        self.prepared = None
        self.trained = False
        self.approved = []

    def active_photos(self) -> list[Path]:
        """Photos still in the set. Deactivated ones stay listed but are left out of training."""
        return [path for path in self.photos if photo_key(path) not in self.inactive]

    def activate(self, path: Path) -> None:
        """Put one photo back into the set. The dataset check must be run again."""
        self.inactive.discard(photo_key(path))
        self.prepared = None
        self.trained = False
        self.approved = []

    def deactivate(self, path: Path, reason: str = "") -> None:
        """Leave one photo out of training and keep the reason shown on its tile."""
        key = photo_key(path)
        self.inactive.add(key)
        if reason:
            self.photo_notes[key] = reason
        self.prepared = None
        self.trained = False
        self.approved = []

    def set_source(self, source: str) -> None:
        """``photos``: train on the user's own photos. ``refs``: build the set from 1-10 references."""
        if source not in SOURCES:
            raise ValueError(f"unknown source {source!r}")
        if source != self.source:
            self.source = source
            self.approved = []
            self.prepared = None
            self.trained = False

    def training_photos(self) -> list[Path]:
        """What ``lora prepare`` gets: the approved set when building from references, else the photos."""
        return list(self.approved) if self.source == "refs" else self.active_photos()

    def set_names(self, name: str, stage_name: str) -> None:
        """Store the names. The trigger stays the rare token unless the user edited it."""
        self.name, self.stage_name = name.strip(), stage_name.strip()

    def set_verbal(self, verbal: bool) -> None:
        """Verbal consent needs no file, so ticking it detaches the release."""
        self.verbal = bool(verbal)
        if self.verbal:
            self.release = None

    def set_adult(self, adult: bool) -> None:
        """Unticking 18 or older also clears the adult-content label and the optional subject attribute."""
        self.adult = bool(adult)
        if not self.adult:
            self.adult_content = False
            self.attr = None

    @property
    def mode(self) -> str:
        """``expert`` or ``simple``: the engine's ``--mode``."""
        return "expert" if self.expert else "simple"

    def consent_method(self) -> str:
        """``verbal`` or ``file`` for a person, else empty."""
        if self.kind != "person":
            return ""
        return "verbal" if self.verbal else "file"

    def check_photos(self) -> Check:
        """Photo count and kind (step 1). References: 1 to 10 in both modes."""
        count = len(self.active_photos())
        if self.source == "refs":
            if not REF_MIN <= count <= REF_MAX:
                return Check(False, "forge.app.err.refs_count", {"low": REF_MIN, "high": REF_MAX})
            if self.kind not in KINDS:
                return Check(False, "forge.app.err.no_kind")
            return Check(True)
        need = need_more(count, self.mode)
        if need:
            return Check(False, "lora.err.too_few", {"need": need})
        if self.expert:
            if count > EXPERT_CAP:
                return Check(False, "lora.err.above_cap", {"cap": EXPERT_CAP})
        else:
            if count > SIMPLE_MAX:
                return Check(False, "lora.err.too_many", {"extra": count - SIMPLE_MAX, "high": SIMPLE_MAX})
        if self.kind not in KINDS:
            return Check(False, "forge.app.err.no_kind")
        if self.expert and not SIMPLE_MIN <= count <= SIMPLE_MAX:
            return Check(True, "lora.warn.count", {"count": count}, warning=True)
        return Check(True)

    def check_subject(self) -> Check:
        """Names and the consent boxes for the kind (step 2).

        A person needs Consent obtained plus either Verbal or an attached
        release, and 18 or older. Adult content is a label only.
        An optional subject attribute is a whole number from 18 to 99.
        """
        if not (self.name or self.stage_name):
            return Check(False, "lora.err.no_name")
        if not self.trigger.strip():
            return Check(False, "forge.app.err.no_trigger")
        if self.kind == "person" and self.attr is not None and accepted(self.attr) is None:
            return Check(False, "lora.err.subject_attr")
        if self.kind == "person" and self.attr is not None and not self.adult:
            return Check(False, "lora.err.subject_attr")
        if self.kind == "person" and not (self.consent and self.adult):
            return Check(False, "lora.err.no_consent")
        if self.kind == "person" and not (self.verbal or self.release is not None):
            return Check(False, "lora.err.no_release")
        if self.adult_content and not (self.kind == "person" and self.adult):
            return Check(False, "lora.err.adult_content")
        if self.kind in ("object", "style") and not self.rights:
            return Check(False, "lora.err.no_rights")
        if looks_like_words(self.trigger):
            return Check(True, "lora.warn.trigger_word", warning=True)
        return Check(True)

    def check_refset(self) -> Check:
        """The set built from references is approved ("This is it"). Own photos do not open this step."""
        if self.source != "refs":
            return Check(True)
        if not self.approved:
            return Check(False, "forge.app.err.refset_not_done")
        return Check(True)

    def check_prepare(self) -> Check:
        """The engine's preparation result (step 3). A safety block is final."""
        if self.prepared is None:
            return Check(False, "forge.app.err.not_prepared")
        if self.prepared.blocked:
            return Check(False, "lora.err.blocked")
        need = need_more(len(self.prepared.kept), self.mode)
        if need:
            return Check(False, "lora.err.too_few", {"need": need})
        return Check(True)

    def check_train(self) -> Check:
        """Training finished (step 4)."""
        return Check(True) if self.trained else Check(False, "forge.app.err.not_trained")

    def check(self, step: str) -> Check:
        """The validator for ``step``; every earlier step must pass too."""
        validators = {
            "photos": self.check_photos,
            "subject": self.check_subject,
            "refset": self.check_refset,
            "prepare": self.check_prepare,
            "train": self.check_train,
            "result": lambda: Check(True),
        }
        for name in STEPS:
            result = validators[name]()
            if name == step or not result.ok:
                return result
        raise ValueError(f"unknown step {step!r}")



def suggest_trigger(name: str = "") -> str:
    """A rare ``vxp`` token. ``name`` is not used."""
    del name
    return rare_trigger()
