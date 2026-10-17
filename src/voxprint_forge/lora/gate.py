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

Rule v1 (strict) always decides. Rule v3 runs only as a shadow (decision 01:24): its would-be result is
recorded for measuring false refusals and never changes keep or drop.
Shadow v3: a face-age estimate of 18 or more but under 20 no
longer drops a face photo on its own. It passes only when the judge confirms an adult (clearly or youthful
adult, confident, no "could be under 20") and every age cue is absent. Any estimate under 18 still drops
and blocks. Each photo kept this way carries an ``audit`` record (id-free: estimates, verdict, cues).
Photos without a face are judged exactly as in v1.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

RULES = ("v1", "v3")
DEFAULT_RULE = "v1"
RULE_VERSION = 1
SHADOW_RULE = "v3"
ADULT_CODES = ("clearly_adult", "youthful_adult")
RELAXABLE = ("youthful_adult", "under_20", "estimator_under_20")
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
    trace: dict[str, Any] = field(default_factory=dict, compare=False)
    audit: dict[str, Any] | None = field(default=None, compare=False)
    shadow: dict[str, Any] | None = field(default=None, compare=False)

    @property
    def keep(self) -> bool:
        """True when the photo may be used."""
        return not self.reasons


def decide(
    verdict: Verdict | None, reading: AgeReading | None, *, estimator_ok: bool, rule: str = DEFAULT_RULE
) -> Decision:
    """Apply the rules to one photo. ``estimator_ok`` is False when the estimator was missing or failed."""
    if rule not in RULES:
        raise ValueError(f"unknown gate rule {rule!r}")
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
    reasons = list(dict.fromkeys(reasons))
    audit = None
    if rule == "v3" and path == "full" and reasons and not minor and set(reasons) <= set(RELAXABLE):
        adult = (
            verdict.reason_code in ADULT_CODES
            and verdict.could_be_under_20 is False
            and all(verdict.indicators.get(k) == "absent" for k in INDICATORS)
            and reading is not None
            and all(a is not None and a >= AGE_HARD_MIN for a in reading.ages)
            and verdict.min_age is not None
            and verdict.min_age >= AGE_HARD_MIN
        )
        if adult:
            ests = [verdict.min_age, *reading.ages]  # type: ignore[union-attr]
            audit = {
                "rule": "v3",
                "min_estimate": min(ests),
                "judge_age": verdict.min_age,
                "estimator_ages": list(reading.ages),  # type: ignore[union-attr]
                "verdict": _code(verdict.reason_code),
                "markers": {_code(k): _code(v) for k, v in verdict.indicators.items()},
                "relaxed": reasons,
            }
            reasons = []
    return Decision(tuple(reasons), minor, path, audit=audit)


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
        """Judge one photo. Exceptions drop it (fail closed); the trace says which stage failed and how."""
        trace: dict[str, Any] = {"judge": {"ok": False}, "estimator": {"ok": False}}
        try:
            verdict = self.judge.judge(image) if self.judge is not None else None
        except Exception as exc:  # noqa: BLE001 - any judge failure drops the photo
            trace["judge"] = {"ok": False, "stage": getattr(exc, "stage", "judge"), **error_info(exc)}
            trace["judge"].update(_stats(self.judge))
            trace["estimator"] = {"ok": False, "skipped": True}
            return Decision(("judge_error",), False, "none", trace)
        trace["judge"] = {"ok": verdict is not None, **_stats(self.judge)}
        if verdict is not None:
            trace["judge"]["verdict"] = verdict_summary(verdict)
        reading: AgeReading | None = None
        estimator_ok = self.estimator is not None
        if estimator_ok:
            try:
                reading = self.estimator.estimate(image)  # type: ignore[union-attr]
                trace["estimator"] = {"ok": True, "n_faces": reading.n_faces, "ages": list(reading.ages)}
            except Exception as exc:  # noqa: BLE001 - any estimator failure drops a face photo
                estimator_ok = False
                trace["estimator"] = {"ok": False, "stage": getattr(exc, "stage", "estimator"), **error_info(exc)}
        else:
            trace["estimator"] = {"ok": False, "missing": True}
        decision = decide(verdict, reading, estimator_ok=estimator_ok, rule=DEFAULT_RULE)
        shadow = shadow_record(verdict, reading, estimator_ok)
        return Decision(decision.reasons, decision.minor, decision.path, trace, shadow=shadow)


def shadow_record(verdict: Verdict | None, reading: AgeReading | None, estimator_ok: bool) -> dict[str, Any]:
    """What shadow v3 would have decided (codes and numbers only). Never used for keep/drop."""
    s = decide(verdict, reading, estimator_ok=estimator_ok, rule=SHADOW_RULE)
    return {"rule": SHADOW_RULE, "keep": s.keep, "reasons": list(s.reasons), "audit": s.audit}


def error_info(exc: BaseException) -> dict[str, str]:
    """Exception type and a short code, never the message text (it may hold paths or model output)."""
    info = {"error": type(exc).__name__}
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        info["code"] = code[:64]
    return info


def verdict_summary(verdict: Verdict) -> dict[str, Any]:
    """The judge's answer as codes and numbers only."""
    return {
        "face_visible": _code(verdict.face_visible),
        "person": _code(verdict.person),
        "min_age": verdict.min_age,
        "confidence": verdict.confidence if isinstance(verdict.confidence, (int, float)) else _code(verdict.confidence),
        "reason_code": _code(verdict.reason_code),
        "could_be_under_20": verdict.could_be_under_20
        if isinstance(verdict.could_be_under_20, bool)
        else _code(verdict.could_be_under_20),
        "indicators": {_code(k): _code(v) for k, v in verdict.indicators.items()},
    }


def _code(value: Any) -> Any:
    """Short codes pass; free text is cut so a trace never carries a description of the image."""
    if value is None or isinstance(value, bool):
        return value
    return str(value)[:24]


def _stats(backend: Any) -> dict[str, Any]:
    stats = getattr(backend, "last_stats", None)
    return dict(stats) if isinstance(stats, dict) else {}


def check_all(gate: Gate, images: Sequence[Any]) -> list[Decision]:
    """Decisions for several images, in order."""
    return [gate.check(image) for image in images]
