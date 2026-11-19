"""LoRA input gate. Settings are the module constants."""

from __future__ import annotations

import math
import re
import threading
import unicodedata
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Protocol

RULES = ("v1", "v3", "v4")
PRODUCT_RULE = "v4"
INTERNAL_RULE = "v1"
DEFAULT_RULE = PRODUCT_RULE
RULE_VERSION = 6
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
    "teenage",
    "teenaged",
    "teenagers",
    "tween",
    "tweens",
    "kiddo",
    "kiddie",
    "kiddies",
    "jailbait",
    "schoolchild",
    "schoolchildren",
    "schoolgirls",
    "schoolboys",
    "underaged",
    "pubescent",
    "lolis",
    "shotas",
    "toddlers",
    "infants",
    "newborn",
    "pedo",
)
CHILD_STEMS = (
    "ребен",
    "детск",
    "подрост",
    "школьн",
    "несовершеннолет",
    "малолет",
    "путти",
    "путто",
    "купидон",
    # RU declensions, diminutives and slang. "мальчишник", "детектив", "девушка",
    # "малышка" and "ребята" are adult words and must not match.
    "девочк",
    "девочек",
    "девчушк",
    "мальчик",
    "мальчишк",
    "мальчишек",
    "мальчуган",
    "мальчонк",
    "ребятишк",
    "ребятишек",
    "ребятен",
    "детишк",
    "детишек",
    "детвор",
    "младен",
    "младенч",
    "грудничк",
    "грудничок",
    "карапуз",
    "малютк",
    "дочурк",
    "сынишк",
    "пацанен",
    "школот",
    "школяр",
    "первоклашк",
    "первоклассн",
    "второклассн",
    "третьеклассн",
    "старшеклассн",
    "младшеклассн",
    "дошкол",
    "детсад",
    "тинейдж",
    "отроковиц",
    "лоли",
    "шотакон",
    # Transliterated RU.
    "rebenk",
    "rebenok",
    "rebyon",
    "rebjon",
    "devochk",
    "devochek",
    "malchik",
    "malchishk",
    "shkoln",
    "podrost",
    "nesovershennolet",
    "malolet",
    "detsk",
    "detishk",
    "tineydzh",
    "tinejdzh",
    "mladen",
    "lolicon",
    "lolikon",
    "shotacon",
    "preteen",
    "prepubescen",
    "kindergart",
    "preschool",
)
CHILD_WORDS_RU = (
    "дети",
    "детей",
    "детям",
    "детьми",
    "детях",
    "детки",
    "деток",
    "дитя",
    "дите",
    "дитяти",
    "отрок",
    "отрока",
    "шота",
    "амур",
    "амуры",
    "амурчик",
    "deti",
    "detey",
    "detei",
    "detej",
    "ditya",
)
_EN_NUMBERS = (
    "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen"
).split()
_RU_NUMBERS = (
    "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять", "одиннадцать",
    "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать",
)
# "-летний" compounds from three up; "двухлетний" also means a two-year span.
_RU_COMPOUND = (
    "трех", "четырех", "пяти", "шести", "семи", "восьми", "девяти", "десяти", "одиннадцати", "двенадцати",
    "тринадцати", "четырнадцати", "пятнадцати", "шестнадцати", "семнадцати",
)
_AGE_UNIT = r"(?:y\.?\s?o\b\.?|y/o|yrs?\b|years?[\s-]*old|year[\s-]*old|лет\b|летн\w*|летк\w*|годик\w*|года?\b)"
_AGE = re.compile(
    rf"(?<![\d.,])(\d{{1,2}})(?![\d.,])\s*-?\s*{_AGE_UNIT}"
    rf"|(?:\baged?|\bвозраст\w*)[\s:=-]*(\d{{1,2}})(?!\d)"
    rf"|\b({'|'.join(_EN_NUMBERS)})[\s-]*(?:years?[\s-]*old|yo\b)"
    rf"|\b({'|'.join(_RU_NUMBERS)})\s+(?:лет|года?)\b"
    rf"|\b({'|'.join(_RU_COMPOUND)})[\s-]*летн\w*"
)
# "5 лет назад", "за 10 лет", "10 лет опыта": a span of time, not an age.
_SPAN_AFTER = re.compile(r"\s*(?:назад|спустя|тому|опыт\w*|стаж\w*|работ\w*|ago)\b")
_SPAN_BEFORE = re.compile(r"(?:\bза|\bчерез|\bпоследн\w*|\bпосле|\bуже|\bоколо|\bболее|\bменее|\bв течение)\s*$")
_SCHOOL_UNIFORM = re.compile(r"school[\s-]*(?:uniform|girl\s*outfit)|\bseifuku\b|школьн\w*\s+форм\w*")
_UNDER_AGE = re.compile(r"\bunder[\s-]*(?:age|aged|18)\b|\bнесовершеннолет")
# Adult idioms that contain a child word: removed before token matching. Narrow on purpose:
# "kid gloves" (the leather), and "minor" as an adjective right before a non-person noun.
# "a minor", "minors", "minor girl", "minor child" still match.
_ADULT_IDIOMS = re.compile(
    r"\bkid[\s-]+gloves?\b"
    r"|\bminor[\s-]+(?:details?|scratch(?:es)?|keys?|changes?|flaws?|imperfections?|blemish(?:es)?|wrinkles?"
    r"|issues?|edits?|adjustments?|differences?|marks?|tweaks?|roles?|chords?|injur(?:y|ies)|damage|defects?"
    r"|scuffs?|stains?|cracks?|dents?|blur|noise|artifacts?|variations?|asymmetry|smudges?|spots?)\b"
)
_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"))
_LEET = str.maketrans("013457@$", "oieastas")
_CYR_TO_LAT = str.maketrans("аеорсухкмтні", "aeopcyxkmthi")
_LAT_TO_CYR = str.maketrans("aeopcyxkmthi", "аеорсухкмтні")
_WORDS = frozenset(w.replace("ё", "е") for w in (*CHILD_WORDS, *CHILD_WORDS_RU))
_STEMS = tuple(stem.replace("ё", "е") for stem in CHILD_STEMS)
TRI = ("present", "absent", "unsure")
YES_NO = ("yes", "no", "unsure")

# Minors-gate job policy (rc.12, approved 2026-11-01). Each checked photo gets one tier:
#   strong: refuses the job (unless CLEAR_SHARE of the set is clear and it has no child keyword);
#   ambiguous: dropped, counts toward the allowance;
#   margin: 18-20 adult margin, dropped, never refuses. Since rule 6 a face-path photo with no reasons
#   and both the judge and the youngest MiVOLO age at 18 or older is clear, not margin; margin stays for
#   a judge-only reading under 20 and for leftover reasons the other tiers do not cover;
#   unchecked: the photo could not be fully read, dropped, never refuses by itself
#   (a judge or estimator that is missing or errors, ``UNAVAILABLE``, still refuses the job: fail closed);
#   clear: the only tier that is trained on.
# On attested-adult person sets, :func:`apply_attested_keeps` then makes no-face photos and up to
# AGE_FLAG_KEEP_SHARE of the input photos' MiVOLO-only age flags clear (final rule, 2026-11-18).
TIERS = ("clear", "margin", "unchecked", "ambiguous", "strong")
MINOR_AGE = AVG_MIN  # below it a model says "minor"
YOUNG_AGE = PHOTO_FLOOR  # below it a model says "clearly young"
KEEP_AGE = AGE_MIN  # clear needs both models at or above it
CLEAR_PERCENT = 90  # this share of clear photos makes every other photo a false flag
ALLOWANCE_MIN = 1  # ambiguous photos allowed at ALLOWANCE_FROM photos or fewer
ALLOWANCE_MAX = 10  # ambiguous photos allowed at ALLOWANCE_TO photos or more
ALLOWANCE_FROM = 30
ALLOWANCE_TO = 60
UNCHECKED = (
    "judge_missing",
    "judge_error",
    "judge_incomplete",
    "not_attested",
    "estimator_missing",
    "estimator_no_face",
    "estimator_age_missing",
    "age_missing",
)
FACE_RULES = ("all", "primary")
PRIMARY_MIN_SCORE = 0.9
STRAY_MAX_AREA = 0.25
# No-face majority (set-level, after every photo is decided). A photo whose only reason is
# ``estimator_no_face`` is kept when at least this share of the set passed the age check with a face.
NO_FACE_MAJORITY = 0.80
NO_FACE_KEEP = "kept_no_face_majority"
# Attested-adult person sets (final rule, 2026-11-18). See :func:`apply_attested_keeps`.
# Rule 1: a photo where MiVOLO ran and found no face is kept, at any no-face ratio.
# Rule 2: MiVOLO-only under-age flags are false positives; the first
# floor(AGE_FLAG_KEEP_SHARE * input photos) of them, in file-name order, are kept, the rest dropped.
AGE_FLAG_KEEP_SHARE = 0.10
NO_FACE_KEEP_ATTESTED = "kept_no_face"
AGE_FLAG_KEEP = "kept_age_flag_false_positive"
# The reasons a MiVOLO under-age reading gives when the judge itself reads 20 or older.
AGE_FLAG_REASONS = ("single_under_16", "average_under_18")
# ``age_flag`` on a flagged photo's trace: kept as a false positive, or dropped past the limit.
AGE_FLAG_KEPT = "kept_false_positive"
AGE_FLAG_DROPPED = "dropped_over_limit"


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
class Face:

    box: tuple[float, float, float, float]
    score: float | None


@dataclass(frozen=True)
class AgeReading:

    n_faces: int
    ages: tuple[float | None, ...] = ()
    faces: tuple[Face, ...] = ()


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
    tier: str = field(default="unchecked", compare=False)

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
    tier = "clear" if not reasons else ("strong" if minor else "ambiguous")
    return Decision(tuple(reasons), minor, path, audit=audit, tier=tier)


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
    youngest: float | None = None
    judge_age = verdict.min_age
    select = select_faces(verdict, reading) if face else None
    if face:
        if not estimator_ok or reading is None:
            reasons.append("estimator_missing")
        elif not reading.ages or reading.n_faces < 1:
            reasons.append("estimator_no_face")
        elif any(not _num(a) for a in reading.ages):
            reasons.append("estimator_age_missing")
        elif not _num(judge_age):
            reasons.append("age_missing")
        else:
            used = reading.ages if select is None else select["ages_used"]
            youngest = min(float(a) for a in used)  # type: ignore[arg-type]
            estimate = (float(judge_age) + youngest) / 2.0  # type: ignore[arg-type]
            if min(float(judge_age), youngest) < PHOTO_FLOOR:  # type: ignore[arg-type]
                reasons.append("single_under_16")
            if estimate < AVG_MIN:
                reasons.append("average_under_18")
                minor = True
    reasons = list(dict.fromkeys(reasons))
    judge =float(judge_age) if _num(judge_age) else None  # type: ignore[arg-type]
    tier = tier_v4(verdict, reasons, minor, judge, youngest)
    return Decision(tuple(reasons), minor, path, trace={"estimate": estimate, "faces": select}, tier=tier)


def select_faces(verdict: Verdict, reading: AgeReading | None) -> dict[str, Any] | None:
    """Per-photo face record: ``rule`` (``FACE_RULES``), ``why``, ``faces``, ``primary``, ``stray``, ``ages_used``."""
    if reading is None:
        return None
    ages = reading.ages
    faces = reading.faces
    rows: list[dict[str, Any]] = []
    for index, face in enumerate(faces):
        box = tuple(face.box) if isinstance(face.box, (tuple, list)) else ()
        area = float(box[2]) * float(box[3]) if len(box) == 4 and all(_num(v) for v in box) else None
        rows.append(
            {
                "i": index,
                "box": [float(v) if _num(v) else None for v in box],
                "area": area,
                "score": float(face.score) if _num(face.score) else None,
                "age": ages[index] if index < len(ages) else None,
            }
        )
    record: dict[str, Any] = {
        "rule": "all",
        "why": "",
        "faces": rows,
        "primary": None,
        "stray": [],
        "ages_used": list(ages),
    }

    def keep_all(why: str) -> dict[str, Any]:
        record["why"] = why
        return record

    if reading.n_faces < 2 or len(ages) != reading.n_faces:
        return keep_all("n_faces")
    if len(faces) != len(ages):
        return keep_all("no_boxes")
    if any(row["area"] is None or row["area"] <= 0 or row["score"] is None for row in rows):
        return keep_all("bad_box")
    if any(not _num(a) for a in ages):
        return keep_all("age_missing")
    order = sorted(rows, key=lambda row: row["area"], reverse=True)
    top = order[0]
    record["primary"] = top["i"]
    if top["score"] < PRIMARY_MIN_SCORE:
        return keep_all("low_score")
    if any(row["area"] >= STRAY_MAX_AREA * top["area"] for row in order[1:]):
        return keep_all("similar_size")
    if float(top["age"]) < KEEP_AGE:
        return keep_all("primary_age")
    judge = verdict.min_age
    if (
        not _num(judge)
        or float(judge) < KEEP_AGE  # type: ignore[arg-type]
        or not _confident(verdict.confidence)
        or verdict.reason_code == "child_or_adolescent"
        or verdict.could_be_under_20 is not False
        or set(verdict.indicators) != set(INDICATORS)
        or any(value != "absent" for value in verdict.indicators.values())
    ):
        return keep_all("judge")
    record.update(
        rule="primary",
        why="stray",
        stray=[row["i"] for row in order[1:]],
        ages_used=[top["age"]],
    )
    return record


def face_pass_ratio(decisions: Sequence[Decision]) -> float:
    """Share of ``decisions`` that passed the age check with a detected face. Empty set: 0.

    A photo counts when it has no reasons, the ``full`` (face) path and the ``clear`` tier: a photo with no
    reasons in the margin or ambiguous tier is still dropped, so it did not pass.
    """
    if not decisions:
        return 0.0
    return sum(1 for d in decisions if d.keep and d.path == "full" and d.tier == "clear") / len(decisions)


def apply_no_face_majority(decisions: Sequence[Decision], *, threshold: float = NO_FACE_MAJORITY) -> list[Decision]:
    """Set-level pass: keep the photos where no face was found when the set's face pass ratio is at least ``threshold``.

    A photo is eligible only when ``estimator_no_face`` is its one reason and nothing says it is under
    age: not ``minor``, tier not ``strong``, judge age not under ``MINOR_AGE``. A kept photo has no
    reasons, tier ``clear`` and ``trace`` gets ``keep_reason`` and ``no_face_ratio``. Every other
    decision is returned as it is.
    """
    ratio = face_pass_ratio(decisions)
    if ratio < threshold:
        return list(decisions)
    out: list[Decision] = []
    for decision in decisions:
        if _no_face_only(decision):
            trace = {
                **decision.trace,
                "keep_reason": NO_FACE_KEEP,
                "no_face_ratio": ratio,
                "kept_reasons": list(decision.reasons),
            }
            decision = Decision((), False, decision.path, trace, decision.audit, decision.shadow, tier="clear")
        out.append(decision)
    return out


def age_flag_keep_limit(n_inputs: int, share: float = AGE_FLAG_KEEP_SHARE) -> int:
    """``floor(share * n_inputs)``, computed exactly (no float rounding): 180 -> 18, 9 -> 0, 0 -> 0."""
    if n_inputs <= 0:
        return 0
    return math.floor(Fraction(str(share)) * int(n_inputs))


def apply_attested_keeps(
    decisions: Sequence[Decision],
    names: Sequence[str],
    ids: Sequence[str] | None = None,
    *,
    attested: bool,
    n_inputs: int,
    share: float = AGE_FLAG_KEEP_SHARE,
) -> tuple[list[Decision], dict[str, Any]]:
    """Set-level pass for an attested-adult person set: the decisions after rules 1 and 2, and their audit record.

    ``names`` (and ``ids``, for a tie between equal names) are the photos' file names, parallel to
    ``decisions``. ``n_inputs`` is the number of input photos of the run. A set that is not attested,
    and every photo neither rule covers, is returned unchanged.

    Rule 1 (no face, :func:`no_face_keepable`): MiVOLO ran and found no face, the judge read the photo
    with confidence and does not say under 18, and nothing else was found. The photo is kept whatever
    the set's no-face ratio: tier ``clear``, no reasons, ``keep_reason`` :data:`NO_FACE_KEEP_ATTESTED`.

    Rule 2 (age flags, :func:`age_flagged`): the only reasons are MiVOLO under-age readings
    (:data:`AGE_FLAG_REASONS`) while the judge reads the photo with confidence at 20 or older and finds
    no marker. These are estimator false positives on an attested set. ``keep_limit`` is
    :func:`age_flag_keep_limit` of ``n_inputs`` (``floor(AGE_FLAG_KEEP_SHARE * n_inputs)``, so 18 of 180;
    small sets round down: 9 inputs give 0, so a single flagged photo is dropped). The flagged photos are
    sorted by file name (then id), never by batch or processing order, and the first ``keep_limit`` are
    kept: tier ``clear``, no reasons, ``keep_reason`` :data:`AGE_FLAG_KEEP`, ``age_flag``
    :data:`AGE_FLAG_KEPT`. The rest keep their tier and reasons (the normal safety drop) and get
    ``age_flag`` :data:`AGE_FLAG_DROPPED`. There is no refusal and nothing to confirm.

    Judge-side minor findings, markers, child keywords, checks that did not run and non-attested sets
    are never covered. A kept photo's trace also gets ``kept_reasons`` (its reasons before the keep).
    """
    limit = age_flag_keep_limit(n_inputs, share) if attested else 0
    audit: dict[str, Any] = {
        "rule": "attested_keeps",
        "attested": bool(attested),
        "age_flag_keep_share": share,
        "inputs": int(n_inputs),
        "age_flag_keep_limit": limit,
        "age_flagged": 0,
        "age_flag_kept": [],
        "age_flag_dropped": [],
        "no_face_kept": [],
    }
    out = list(decisions)
    if not attested:
        return out, audit
    keys = [(str(names[i]), str(ids[i]) if ids is not None else "") for i in range(len(out))]

    def entry(index: int) -> dict[str, Any]:
        row: dict[str, Any] = {"file": keys[index][0], "reasons": list(out[index].reasons)}
        if ids is not None:
            row["id"] = keys[index][1]
        return row

    def kept(decision: Decision, **extra: Any) -> Decision:
        trace = {**decision.trace, **extra, "kept_reasons": list(decision.reasons)}
        return Decision((), False, decision.path, trace, decision.audit, decision.shadow, tier="clear")

    for index, decision in enumerate(out):
        if no_face_keepable(decision):
            audit["no_face_kept"].append(entry(index))
            out[index] = kept(decision, keep_reason=NO_FACE_KEEP_ATTESTED)
    flagged = sorted((i for i, decision in enumerate(out) if age_flagged(decision)), key=lambda i: keys[i])
    audit["age_flagged"] = len(flagged)
    for rank, index in enumerate(flagged):
        decision = out[index]
        if rank < limit:
            audit["age_flag_kept"].append(entry(index))
            out[index] = kept(
                decision, keep_reason=AGE_FLAG_KEEP, age_flag=AGE_FLAG_KEPT, age_flag_limit=limit
            )
        else:
            audit["age_flag_dropped"].append(entry(index))
            trace = {**decision.trace, "age_flag": AGE_FLAG_DROPPED, "age_flag_limit": limit}
            out[index] = Decision(
                decision.reasons, decision.minor, decision.path, trace, decision.audit, decision.shadow,
                tier=decision.tier,
            )
    return out, audit


def no_face_keepable(decision: Decision) -> bool:
    """Rule 1: MiVOLO ran and found no face, and nothing else stops the photo.

    Needs: not kept yet; no reason but ``estimator_no_face``; not ``minor``, not ``strong`` or
    ``ambiguous``; an estimator trace that ran (``ok``) with no face; a judge verdict whose confidence is
    at least ``MIN_CONFIDENCE`` and whose age, when given, is not under ``MINOR_AGE``.
    """
    if decision.tier == "clear" or decision.minor or decision.tier in ("strong", "ambiguous"):
        return False
    if not set(decision.reasons) <= {"estimator_no_face"}:
        return False
    estimator = decision.trace.get("estimator")
    if not isinstance(estimator, dict) or estimator.get("ok") is not True:
        return False
    n_faces, ages = estimator.get("n_faces"), estimator.get("ages")
    if not (isinstance(ages, list) and (not ages or (_num(n_faces) and n_faces < 1))):
        return False
    verdict = _verdict(decision)
    if verdict is None or not _confident(verdict.get("confidence")):
        return False
    age = verdict.get("min_age")
    return not (_num(age) and float(age) < MINOR_AGE)


def age_flagged(decision: Decision) -> bool:
    """Rule 2: a MiVOLO-only under-age flag the judge does not share.

    Needs: not kept; the face path; reasons only from :data:`AGE_FLAG_REASONS`; a judge verdict with
    confidence of at least ``MIN_CONFIDENCE``, an age of ``KEEP_AGE`` (20) or older, reason code not
    ``child_or_adolescent`` and no indicator ``present``; MiVOLO ran and its youngest age is under ``MINOR_AGE``.
    """
    if decision.tier == "clear" or decision.path != "full" or not decision.reasons:
        return False
    if not set(decision.reasons) <= set(AGE_FLAG_REASONS):
        return False
    verdict = _verdict(decision)
    if verdict is None or not _confident(verdict.get("confidence")):
        return False
    age = verdict.get("min_age")
    if not _num(age) or float(age) < KEEP_AGE or verdict.get("reason_code") == "child_or_adolescent":
        return False
    indicators = verdict.get("indicators")
    if not isinstance(indicators, dict) or any(value == "present" for value in indicators.values()):
        return False
    estimator = decision.trace.get("estimator")
    ages = estimator.get("ages") if isinstance(estimator, dict) and estimator.get("ok") is True else None
    return isinstance(ages, list) and bool(ages) and any(_num(a) and float(a) < MINOR_AGE for a in ages)


def _verdict(decision: Decision) -> dict[str, Any] | None:
    """The judge verdict summary on ``decision.trace``, or ``None`` when the judge did not answer."""
    judge = decision.trace.get("judge")
    if not isinstance(judge, dict) or judge.get("ok") is not True:
        return None
    verdict = judge.get("verdict")
    return verdict if isinstance(verdict, dict) else None


def _no_face_only(decision: Decision) -> bool:
    if decision.reasons != ("estimator_no_face",) or decision.minor or decision.tier == "strong":
        return False
    judge = decision.trace.get("judge")
    verdict = judge.get("verdict") if isinstance(judge, dict) else None
    age = verdict.get("min_age") if isinstance(verdict, dict) else None
    return not (_num(age) and float(age) < MINOR_AGE)


def tier_v4(
    verdict: Verdict, reasons: Sequence[str], minor: bool, judge: float | None, youngest: float | None
) -> str:
    """The photo's tier under the job policy. ``youngest`` is the youngest MiVOLO age, set only on the face path."""
    marked = "markers" in reasons
    child = verdict.reason_code == "child_or_adolescent"
    if judge is not None and youngest is not None:
        low, high = min(judge, youngest), max(judge, youngest)
        mean = (judge + youngest) / 2.0
        says_minor = marked or judge < MINOR_AGE or youngest < MINOR_AGE
        # Strong: two models agree it is under 18.
        if (child and youngest < MINOR_AGE) or (youngest < YOUNG_AGE and judge < MINOR_AGE):
            return "strong"
        # Ambiguous: one model says minor and the other says 20 or older, or the average sits in the margin.
        if (
            (marked and youngest >= KEEP_AGE)
            or (youngest < MINOR_AGE and judge >= KEEP_AGE)
            or (low < YOUNG_AGE and high >= KEEP_AGE)
            or (says_minor and MINOR_AGE <= mean < KEEP_AGE)
        ):
            return "ambiguous"
        # Any other minor indication is not covered by the policy: fail closed, as v4 did.
        if minor or low < MINOR_AGE:
            return "strong"
    elif minor or (judge is not None and judge < MINOR_AGE):
        # Only the judge could read this photo and it says minor: fail closed.
        return "strong"
    elif judge is not None and judge < KEEP_AGE:
        return "margin"
    if marked:
        return "ambiguous"
    if any(reason in UNCHECKED for reason in reasons):
        return "unchecked"
    if not _confident(verdict.confidence):
        return "ambiguous"
    if reasons:
        return "margin"
    # Rule 6: both models read 18 or older and nothing else was found: the 18-20 margin is trained on.
    if judge is not None and youngest is not None and min(judge, youngest) < MINOR_AGE:
        return "margin"
    return "clear"


def _confident(value: Any) -> bool:
    return _num(value) and float(value) >= MIN_CONFIDENCE


def allowance(n: int) -> int:
    """Ambiguous photos a set of ``n`` checked photos may drop and still train (1 to 10, half rounds up)."""
    if n <= ALLOWANCE_FROM:
        return ALLOWANCE_MIN
    if n >= ALLOWANCE_TO:
        return ALLOWANCE_MAX
    span, width = ALLOWANCE_MAX - ALLOWANCE_MIN, ALLOWANCE_TO - ALLOWANCE_FROM
    return ALLOWANCE_MIN + ((n - ALLOWANCE_FROM) * span * 2 + width) // (width * 2)


@dataclass(frozen=True)
class JobDecision:

    refuse: bool
    why: str
    n: int
    counts: dict[str, int]
    allowance: int
    relaxed: int = 0
    exempt: int = 0


def decide_job(tiers: Sequence[str], *, keyword: bool, relaxed: int = 0, exempt: int = 0) -> JobDecision:
    """Whole-job verdict from the checked photos' tiers. Only clear photos are ever trained on.

    ``relaxed`` counts the clear photos that :func:`apply_attested_keeps` kept (no face, or an age flag
    kept as a false positive). They are trained on, but they do not count toward the ``CLEAR_PERCENT``
    share that lets a set with a strong photo train: that share is reached as it was before those rules.
    ``exempt`` counts the ambiguous photos it dropped past the age-flag limit: they are dropped and never
    refuse the job, so they do not count toward ``allowance(n)``.
    """
    n = len(tiers)
    counts = {tier: sum(1 for t in tiers if t == tier) for tier in TIERS}
    limit = allowance(n)
    relaxed = max(0, min(int(relaxed), counts["clear"]))
    exempt = max(0, min(int(exempt), counts["ambiguous"]))

    def job(refuse: bool, why: str) -> JobDecision:
        return JobDecision(refuse, why, n, counts, limit, relaxed, exempt)

    if keyword:
        return job(True, "child_keyword")
    if n and (counts["clear"] - relaxed) * 100 >= n * CLEAR_PERCENT:
        return job(False, "clear_share")
    if counts["strong"]:
        return job(True, "strong")
    if counts["ambiguous"] - exempt > limit:
        return job(True, "over_allowance")
    return job(False, "within_allowance")


# A check that did not run on a photo. Whatever the tiers say, the job is refused (fail closed).
UNAVAILABLE = ("judge_missing", "judge_error", "estimator_missing")


def _num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def child_keyword(*texts: str) -> bool:
    """True when any text names a child or a minor age, in EN, RU or transliterated RU.

    Text is NFKC-folded, lower-cased and ``ё`` becomes ``е``. Leetspeak (``t33n``)
    and mixed Latin/Cyrillic look-alikes (``сhild``) are read as letters. An age
    counts when it is under 18; ``18 лет`` and ``19yo`` pass. A few adult idioms
    (``kid gloves``, ``minor details``) do not count.
    """
    return child_keyword_match(*texts) is not None


def child_keyword_match(*texts: str) -> str | None:
    """The first word or phrase in ``texts`` that names a child or a minor age, or ``None``."""
    for text in texts:
        norm = normalize_text(text)
        for age, phrase in _age_matches(norm):
            if age < AGE_HARD_MIN:
                return phrase
        found = _UNDER_AGE.search(norm)
        if found:
            return found.group(0)
        found = _SCHOOL_UNIFORM.search(norm)
        if found and _ages(norm):
            return found.group(0)
        for token in re.split(r"[^0-9a-zа-яі]+", _ADULT_IDIOMS.sub(" ", norm)):
            if token and any(_child_token(variant) for variant in _variants(token)):
                return token
    return None


def normalize_text(text: object) -> str:
    return unicodedata.normalize("NFKC", str(text)).translate(_INVISIBLE).lower().replace("ё", "е")


def minor_age(text: str) -> bool:
    """True when ``text`` states an age under 18 (``16yo``, ``16 лет``, ``шестнадцатилетняя``)."""
    return any(age < AGE_HARD_MIN for age in _ages(normalize_text(text)))


def _ages(norm: str) -> list[int]:
    return [age for age, _ in _age_matches(norm)]


def _age_matches(norm: str) -> list[tuple[int, str]]:
    """Each stated age in ``norm`` with the text that stated it; spans of time are skipped."""
    found: list[tuple[int, str]] = []
    for match in _AGE.finditer(norm):
        if _SPAN_AFTER.match(norm, match.end()) or _SPAN_BEFORE.search(norm, 0, match.start()):
            continue
        digits = match.group(1) or match.group(2)
        if digits:
            found.append((int(digits), match.group(0)))
        elif match.group(3):
            found.append((_EN_NUMBERS.index(match.group(3)) + 1, match.group(0)))
        elif match.group(4):
            found.append((_RU_NUMBERS.index(match.group(4)) + 1, match.group(0)))
        elif match.group(5):
            found.append((_RU_COMPOUND.index(match.group(5)) + 3, match.group(0)))
    return found


def _variants(token: str) -> set[str]:
    out = {token, *re.findall(r"[a-zа-яі]+", token)}
    if re.search(r"[a-z]", token) and re.search(r"[0-9]", token):
        out.add(token.translate(_LEET))
    if re.search(r"[a-z]", token) and re.search(r"[а-яі]", token):
        out |= {token.translate(_CYR_TO_LAT), token.translate(_LAT_TO_CYR)}
    return out


def _child_token(token: str) -> bool:
    return token in _WORDS or any(token.startswith(stem) for stem in _STEMS)


# One age-estimator run on a photo: the reading, whether it is usable, and its trace entry.
_Estimate = tuple[AgeReading | None, bool, dict[str, Any]]


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
        """The decision for one photo: the judge, then the age estimator, then :func:`decide`."""
        try:
            verdict = self.judge.judge(image) if self.judge is not None else None
        except Exception as exc:  # noqa: BLE001
            return _judge_failed(exc, _stats(self.judge))
        return self._after_judge(image, verdict, _stats(self.judge), attested=attested)

    def check_batch(self, images: Sequence[Any], *, attested: bool = False) -> list[Decision]:
        """One decision per image, in input order, each the same as :meth:`check` would give.

        The judge reads the whole batch in one ``judge_batch`` call when it has one. Meanwhile the
        age estimator reads each image once, in order, on one background thread, with the same
        per-image error capture as :meth:`check`; :func:`decide` then runs per image. A judge
        without ``judge_batch``, a single image, or a subclass that overrides :meth:`check`
        takes :meth:`check` per image. A ``judge_batch`` that raises, or returns the wrong
        number of items or an item that is not a verdict, an exception or ``None``, is a batch
        failure: every image of the batch is then checked one by one with :meth:`check` (the
        estimator runs again). An exception item is that image's ``judge_error``, as a raising
        judge is in :meth:`check`: its estimator reading is discarded and the trace says skipped.
        """
        images = list(images)
        batch = getattr(self.judge, "judge_batch", None)
        if not callable(batch) or len(images) <= 1 or type(self).check is not Gate.check:
            return [self.check(image, attested=attested) for image in images]
        # The age estimator reads the batch on one background thread, in input order, while the
        # judge runs; nothing else touches the estimator until that thread is done.
        stop = threading.Event()
        with ThreadPoolExecutor(1, thread_name_prefix="gate-estimator") as pool:
            pending = pool.submit(self._estimate_each, images, stop) if self.estimator is not None else None
            try:
                results, failure = _judge_batch(batch, images)
            except BaseException:
                stop.set()
                raise
            if failure is not None:
                stop.set()
            # Re-raises anything that escaped the per-image capture (a BaseException) here.
            estimates = pending.result() if pending is not None else None
        if failure is not None:
            _report_batch(failure)
            return [self.check(image, attested=attested) for image in images]
        if estimates is not None and len(estimates) != len(images):
            raise RuntimeError(f"age estimator read {len(estimates)} of {len(images)} photos")
        stats = _batch_stats(self.judge, len(images))
        out: list[Decision] = []
        for index, (image, item, item_stats) in enumerate(zip(images, results, stats, strict=True)):
            if isinstance(item, Exception):
                # As in ``check``: a judge error skips the estimator; its reading is discarded.
                out.append(_judge_failed(item, item_stats))
            else:
                estimated = estimates[index] if estimates is not None else None
                out.append(self._after_judge(image, item, item_stats, attested=attested, estimated=estimated))
        return out

    def _estimate_each(self, images: Sequence[Any], stop: threading.Event) -> list[_Estimate]:
        """:meth:`_estimate` for each image in order; stops early once ``stop`` is set."""
        out: list[_Estimate] = []
        for image in images:
            if stop.is_set():
                break
            out.append(self._estimate(image))
        return out

    def _estimate(self, image: Any) -> _Estimate:
        """The age estimator on ``image``: ``(reading, estimator_ok, estimator trace)``. An error is captured."""
        if self.estimator is None:
            return None, False, {"ok": False, "missing": True}
        try:
            reading = self.estimator.estimate(image)
            return reading, True, {"ok": True, "n_faces": reading.n_faces, "ages": list(reading.ages)}
        except Exception as exc:  # noqa: BLE001
            return None, False, {"ok": False, "stage": getattr(exc, "stage", "estimator"), **error_info(exc)}

    def _after_judge(
        self,
        image: Any,
        verdict: Verdict | None,
        stats: dict[str, Any],
        *,
        attested: bool,
        estimated: _Estimate | None = None,
    ) -> Decision:
        """Everything after the judge answered for ``image``: the estimator, :func:`decide` and the trace.

        ``estimated`` is the :meth:`_estimate` result when it already ran (``check_batch``); else it runs here.
        """
        trace: dict[str, Any] = {"judge": {"ok": verdict is not None, **stats}, "estimator": {"ok": False}}
        if verdict is not None:
            trace["judge"]["verdict"] = verdict_summary(verdict)
        reading, estimator_ok, trace["estimator"] = estimated if estimated is not None else self._estimate(image)
        decision = decide(verdict, reading, estimator_ok=estimator_ok, rule=self.rule_for(attested), attested=attested)
        trace["faces"] = decision.trace.get("faces")
        if not self.internal:
            trace["estimate"] = decision.trace.get("estimate")
            return Decision(decision.reasons, decision.minor, decision.path, trace, tier=decision.tier)
        shadow = shadow_record(verdict, reading, estimator_ok)
        return Decision(decision.reasons, decision.minor, decision.path, trace, shadow=shadow, tier=decision.tier)


def _judge_failed(exc: Exception, stats: dict[str, Any]) -> Decision:
    """The ``judge_error`` decision for a photo whose judge raised ``exc``; the estimator is skipped."""
    trace: dict[str, Any] = {"judge": {"ok": False, "stage": getattr(exc, "stage", "judge"), **error_info(exc)}}
    trace["judge"].update(stats)
    trace["estimator"] = {"ok": False, "skipped": True}
    return Decision(("judge_error",), False, "none", trace)


def _judge_batch(batch: Callable[[list[Any]], Any], images: list[Any]) -> tuple[list[Any], str | None]:
    """``judge_batch(images)`` as a list, and why it cannot be used (``None`` when it can).

    A raised ``Exception``, another number of items than images, or an item that is not a verdict,
    an exception or ``None`` makes it unusable; a ``BaseException`` propagates.
    """
    try:
        results = list(batch(images))
    except Exception as exc:  # noqa: BLE001 - the caller checks every image again, one by one
        return [], f"failed ({type(exc).__name__})"
    if len(results) != len(images) or not all(
        item is None or isinstance(item, (Verdict, Exception)) for item in results
    ):
        return [], f"returned {len(results)} items for {len(images)} photos"
    return results, None


def _batch_stats(judge: Any, n: int) -> list[dict[str, Any]]:
    """The judge's per-item stats of its last ``judge_batch`` call, or empty dicts when it kept none."""
    stats = getattr(judge, "last_batch_stats", None)
    if not isinstance(stats, list) or len(stats) != n:
        return [{} for _ in range(n)]
    return [dict(item) if isinstance(item, dict) else {} for item in stats]


def _report_batch(what: str) -> None:
    import sys

    print(f"safety gate batch judge {what}: checking those photos one by one", file=sys.stderr)


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
