"""LoRA input gate. Settings are the module constants."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

RULES = ("v1", "v3", "v4")
PRODUCT_RULE = "v4"
INTERNAL_RULE = "v1"
DEFAULT_RULE = PRODUCT_RULE
RULE_VERSION = 4
AVG_MIN = 18.0
PHOTO_FLOOR = 16.0
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
BODY_MARKERS = (
    "child_body",
    "undeveloped_figure",
    "school_context",
    "minor_age_text",
    "infant_or_putto",
    "childlike_body",
    "lacks_adult_features",
)
CHILD_WORDS = (
    "child",
    "children",
    "kid",
    "kids",
    "minor",
    "minors",
    "teen",
    "teens",
    "teenager",
    "underage",
    "preteen",
    "schoolgirl",
    "schoolboy",
    "loli",
    "lolita",
    "shota",
    "toddler",
    "infant",
    "putto",
    "putti",
    "cupid",
    "cupids",
    "cherub",
    "cherubs",
    "amorino",
)
CHILD_STEMS = (
    "ребен",
    "ребён",
    "детск",
    "подрост",
    "школьн",
    "несовершеннолет",
    "малолет",
    "путти",
    "путто",
    "купидон",
)
CHILD_WORDS_RU = ("дети", "детей", "амур", "амуры", "амурчик")
TRI = ("present", "absent", "unsure")
YES_NO = ("yes", "no", "unsure")


@dataclass(frozen=True)
class Verdict:

    face_visible: str
    person: str
    min_age: float | None
    confidence: float | None
    reason_code: str | None
    could_be_under_20: bool | None
    indicators: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class AgeReading:

    n_faces: int
    ages: tuple[float | None, ...] = ()


class Judge(Protocol):

    name: str

    def judge(self, image: Any) -> Verdict:
        ...


class AgeEstimator(Protocol):

    name: str

    def estimate(self, image: Any) -> AgeReading:
        ...


@dataclass(frozen=True)
class Decision:

    reasons: tuple[str, ...]
    minor: bool
    path: str
    trace: dict[str, Any] = field(default_factory=dict, compare=False)
    audit: dict[str, Any] | None = field(default=None, compare=False)
    shadow: dict[str, Any] | None = field(default=None, compare=False)

    @property
    def keep(self) -> bool:
        return not self.reasons


def decide(
    verdict: Verdict | None,
    reading: AgeReading | None,
    *,
    estimator_ok: bool,
    rule: str = DEFAULT_RULE,
    attested: bool = True,
) -> Decision:
    if rule not in RULES:
        raise ValueError(f"unknown gate rule {rule!r}")
    if verdict is None:
        return Decision(("judge_missing",), False, "none")
    if rule == "v4":
        return _decide_v4(verdict, reading, estimator_ok=estimator_ok, attested=attested)
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


def _decide_v4(verdict: Verdict, reading: AgeReading | None, *, estimator_ok: bool, attested: bool) -> Decision:
    reasons: list[str] = []
    minor = False
    if (
        verdict.face_visible not in YES_NO
        or verdict.person not in YES_NO
        or set(verdict.indicators) != set(INDICATORS)
        or any(value not in TRI for value in verdict.indicators.values())
    ):
        reasons.append("judge_incomplete")
    face = verdict.face_visible == "yes" or (reading is not None and reading.n_faces > 0)
    path = "full" if face else ("body" if verdict.person != "no" else "no_person")
    if not attested and (face or verdict.person != "no"):
        reasons.append("not_attested")
    markers = INDICATORS if face else BODY_MARKERS
    for key in markers:
        if verdict.indicators.get(key) == "present":
            reasons.append("markers")
            minor = minor or key in MINOR_SIGNALS
    if verdict.reason_code == "child_or_adolescent":
        reasons.append("markers")
        minor = True
    estimate: float | None = None
    if face:
        judge_age = verdict.min_age
        if not estimator_ok or reading is None:
            reasons.append("estimator_missing")
        elif not reading.ages or reading.n_faces < 1:
            reasons.append("estimator_no_face")
        elif any(not _num(a) for a in reading.ages):
            reasons.append("estimator_age_missing")
        elif not _num(judge_age):
            reasons.append("age_missing")
        else:
            youngest = min(float(a) for a in reading.ages)  # type: ignore[arg-type]
            estimate = (float(judge_age) + youngest) / 2.0  # type: ignore[arg-type]
            if min(float(judge_age), youngest) < PHOTO_FLOOR:  # type: ignore[arg-type]
                reasons.append("single_under_16")
            if estimate < AVG_MIN:
                reasons.append("average_under_18")
                minor = True
    reasons = list(dict.fromkeys(reasons))
    return Decision(tuple(reasons), minor, path, trace={"estimate": estimate})


def _num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def child_keyword(*texts: str) -> bool:
    for text in texts:
        for token in re.split(r"[^a-zа-яё]+", str(text).lower()):
            if token in CHILD_WORDS or token in CHILD_WORDS_RU or any(token.startswith(stem) for stem in CHILD_STEMS):
                return True
    return False


class Gate:

    def __init__(
        self,
        judge: Judge | None,
        estimator: AgeEstimator | None,
        *,
        internal: bool = False,
        identify: Callable[[Any], str | None] | None = None,
    ) -> None:
        self.judge = judge
        self.estimator = estimator
        self.internal = bool(internal)
        self.identify = identify

    @property
    def rule(self) -> str:
        return INTERNAL_RULE if self.internal else PRODUCT_RULE

    def rule_for(self, attested: bool) -> str:
        return self.rule

    @property
    def available(self) -> bool:
        return self.judge is not None and self.estimator is not None

    def check(self, image: Any, *, attested: bool = False) -> Decision:
        trace: dict[str, Any] = {"judge": {"ok": False}, "estimator": {"ok": False}}
        try:
            verdict = self.judge.judge(image) if self.judge is not None else None
        except Exception as exc:  # noqa: BLE001
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
            except Exception as exc:  # noqa: BLE001
                estimator_ok = False
                trace["estimator"] = {"ok": False, "stage": getattr(exc, "stage", "estimator"), **error_info(exc)}
        else:
            trace["estimator"] = {"ok": False, "missing": True}
        decision = decide(verdict, reading, estimator_ok=estimator_ok, rule=self.rule_for(attested), attested=attested)
        if not self.internal:
            trace["estimate"] = decision.trace.get("estimate")
            return Decision(decision.reasons, decision.minor, decision.path, trace)
        shadow = shadow_record(verdict, reading, estimator_ok)
        return Decision(decision.reasons, decision.minor, decision.path, trace, shadow=shadow)


def shadow_record(verdict: Verdict | None, reading: AgeReading | None, estimator_ok: bool) -> dict[str, Any]:
    s = decide(verdict, reading, estimator_ok=estimator_ok, rule=SHADOW_RULE)
    return {"rule": SHADOW_RULE, "keep": s.keep, "reasons": list(s.reasons), "audit": s.audit}


def error_info(exc: BaseException) -> dict[str, str]:
    info = {"error": type(exc).__name__}
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        info["code"] = code[:64]
    return info


def verdict_summary(verdict: Verdict) -> dict[str, Any]:
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
    if value is None or isinstance(value, bool):
        return value
    return str(value)[:24]


def _stats(backend: Any) -> dict[str, Any]:
    stats = getattr(backend, "last_stats", None)
    return dict(stats) if isinstance(stats, dict) else {}


def check_all(gate: Gate, images: Sequence[Any]) -> list[Decision]:
    return [gate.check(image) for image in images]
