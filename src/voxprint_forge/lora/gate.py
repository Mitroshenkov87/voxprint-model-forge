"""Input safety gate for training photos: the minors ban and the age check (always on, fail closed).

Ported from the forge age gate v3 and closeup_qc rules with the training cut-off
of 20 and no soft path: any doubt drops the photo.

* A visible face (or an unsure answer) sends the photo down the full path: the
  VLM judge and the age estimator must both read every person as 20 or older.
* Without a visible face the VLM judge alone decides, on the same age cues.
* Any age cue answered present or unsure, a low confidence, a missing or failing
  judge, or an incomplete answer drops the photo.
* A clear minor signal blocks the whole run.

There are no thresholds to configure and no override.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

RULE_VERSION = 1
AGE_MIN = 20.0
AGE_HARD_MIN = 18.0
MIN_CONFIDENCE = 0.5
INDICATORS = (
    "child_face",
    "child_body",
    "undeveloped_figure",
    "school_context",
    "child_props",
    "child_clothing",
    "minor_age_text",
    "infant_or_putto",
    "childlike_body",
    "lacks_adult_features",
)
MINOR_SIGNALS = ("child_face", "child_body", "undeveloped_figure", "minor_age_text", "infant_or_putto")
TRI = ("present", "absent", "unsure")
YES_NO = ("yes", "no", "unsure")


@dataclass(frozen=True)
class Verdict:
    """What the VLM judge answered for one photo."""

    face_visible: str
    person: str
    min_age: float | None
    confidence: float | None
    reason_code: str | None
    could_be_under_20: bool | None
    indicators: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AgeReading:
    """Per-face ages from the age estimator."""

    n_faces: int
    ages: tuple[float | None, ...] = ()


class Judge(Protocol):
    """A VLM backend that answers the age-cue questions."""

    name: str

    def judge(self, image: Any) -> Verdict:
        """Answer for one image."""
        ...


class AgeEstimator(Protocol):
    """An age backend that reads faces (MiVOLO v2 by default)."""

    name: str

    def estimate(self, image: Any) -> AgeReading:
        """Ages for one image."""
        ...


@dataclass(frozen=True)
class Decision:
    """Gate result for one photo. ``reasons`` empty means keep."""

    reasons: tuple[str, ...]
    minor: bool
    path: str

    @property
    def keep(self) -> bool:
        """True when the photo may be used."""
        return not self.reasons


def decide(verdict: Verdict | None, reading: AgeReading | None, *, estimator_ok: bool) -> Decision:
    """Apply the rules to one photo. ``estimator_ok`` is False when the estimator was missing or failed."""
    if verdict is None:
        return Decision(("judge_missing",), False, "none")
    reasons: list[str] = []
    minor = False
    if (
        verdict.face_visible not in YES_NO
        or verdict.person not in YES_NO
        or set(verdict.indicators) != set(INDICATORS)
        or any(value not in TRI for value in verdict.indicators.values())
    ):
        reasons.append("judge_incomplete")
    for key in INDICATORS:
        value = verdict.indicators.get(key)
        if value in ("present", "unsure"):
            reasons.append(f"{key}_{value}")
            minor = minor or (value == "present" and key in MINOR_SIGNALS)
    confidence = verdict.confidence
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or confidence < MIN_CONFIDENCE:
        reasons.append("low_confidence")
    if verdict.reason_code == "child_or_adolescent":
        reasons.append("child_or_adolescent")
        minor = True
    elif verdict.reason_code in ("youthful_adult", "could_be_under_20", "image_unclear"):
        reasons.append(verdict.reason_code)
    if verdict.could_be_under_20 is not False and verdict.person != "no":
        reasons.append("could_be_under_20")
    if verdict.person != "no":
        if verdict.min_age is None:
            reasons.append("age_missing")
        elif verdict.min_age < AGE_HARD_MIN:
            reasons.append("under_18")
            minor = True
        elif verdict.min_age < AGE_MIN:
            reasons.append("under_20")
    face = verdict.face_visible != "no" or (reading is not None and reading.n_faces > 0)
    path = "full" if face else ("body" if verdict.person != "no" else "no_person")
    if face:
        if not estimator_ok or reading is None:
            reasons.append("estimator_missing")
        else:
            if not reading.ages or reading.n_faces < 1:
                reasons.append("estimator_no_face")
            for age in reading.ages:
                if age is None:
                    reasons.append("estimator_age_missing")
                elif age < AGE_HARD_MIN:
                    reasons.append("estimator_under_18")
                    minor = True
                elif age < AGE_MIN:
                    reasons.append("estimator_under_20")
    return Decision(tuple(dict.fromkeys(reasons)), minor, path)


class Gate:
    """The judge plus the age estimator. A missing judge makes the gate unavailable."""

    def __init__(self, judge: Judge | None, estimator: AgeEstimator | None) -> None:
        self.judge = judge
        self.estimator = estimator

    @property
    def available(self) -> bool:
        """False blocks the run before any photo is checked."""
        return self.judge is not None and self.estimator is not None

    def check(self, image: Any) -> Decision:
        """Judge one photo. Exceptions drop it (fail closed)."""
        try:
            verdict = self.judge.judge(image) if self.judge is not None else None
        except Exception:  # noqa: BLE001 - any judge failure drops the photo
            return Decision(("judge_error",), False, "none")
        reading: AgeReading | None = None
        estimator_ok = self.estimator is not None
        if estimator_ok:
            try:
                reading = self.estimator.estimate(image)  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001 - any estimator failure drops a face photo
                estimator_ok = False
        return decide(verdict, reading, estimator_ok=estimator_ok)


def check_all(gate: Gate, images: Sequence[Any]) -> list[Decision]:
    """Decisions for several images, in order."""
    return [gate.check(image) for image in images]
