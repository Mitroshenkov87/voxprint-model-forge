"""Decide whether the own Qwen-Edit-2511 INT4 quant runs.

The default mode is ``on``: the own r256 quant always runs. ``auto`` is
opt-in. It reads the Phase 0 ``result.json`` and skips that quant only when
QuantFunc r128 finishes every hard case (c1 identity, c2 hands, c3 text)
with ``fits_16gb`` true. The same file, when present, is an eval and card
baseline and does not choose the winner. Wan 2.2 is not part of this decision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Never

from voxprint_forge.diffusion.int4.pins import HARD_CASES
from voxprint_forge.errors import StageError, UsageError

OwnEditMode = Literal["auto", "on", "off"]
OwnEditAction = Literal["quantize", "mirror"]


@dataclass(frozen=True)
class OwnEditDecision:
    """What the rental will do for the own edit quant."""

    mode: OwnEditMode
    action: OwnEditAction
    reason: str
    phase0_result: str
    fits_16gb: bool | None
    hard_cases: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        """JSON object stored on ``status.json`` and the run result."""
        return {
            "mode": self.mode,
            "action": self.action,
            "reason": self.reason,
            "phase0_result": self.phase0_result,
            "fits_16gb": self.fits_16gb,
            "hard_cases": list(self.hard_cases),
        }


def parse_own_edit(value: str) -> OwnEditMode:
    """Accept ``auto``, ``on``, or ``off``."""
    if value == "auto" or value == "on" or value == "off":
        return value
    raise UsageError("own-edit must be auto, on, or off", code="usage")


def _unexpected_mode(mode: Never) -> Never:
    raise UsageError(f"unhandled own-edit mode {mode}", code="usage")


def decide_own_edit(mode: OwnEditMode, phase0_result: Path | None) -> OwnEditDecision:
    """Choose quantize or mirror. ``on`` always quantizes. ``auto`` is opt-in."""
    text = "" if phase0_result is None else str(phase0_result)
    if mode == "on":
        return OwnEditDecision(
            "on",
            "quantize",
            "own-edit is on, so the own INT4 quant runs.",
            text,
            None,
            (),
        )
    if mode == "off":
        return OwnEditDecision(
            "off",
            "mirror",
            "own-edit is off, so QuantFunc r128 and r32 are mirrored and the own quant is skipped.",
            text,
            None,
            (),
        )
    if mode == "auto":
        return _from_phase0(text, phase0_result)
    return _unexpected_mode(mode)


def _from_phase0(text: str, phase0_result: Path | None) -> OwnEditDecision:
    if phase0_result is None or not phase0_result.is_file():
        return OwnEditDecision(
            "auto",
            "quantize",
            "Phase 0 result is missing, so the own quant runs.",
            text,
            None,
            (),
        )
    loaded = _load(phase0_result)
    passed, reason, fits, cases = quantfunc_r128_gate(loaded)
    action: OwnEditAction = "mirror" if passed else "quantize"
    return OwnEditDecision("auto", action, reason, text, fits, cases)


def _load(path: Path) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageError(f"Phase 0 result {path} is not JSON: {exc}", code="phase0") from None
    if not isinstance(loaded, dict):
        raise StageError(f"Phase 0 result {path} is not an object", code="phase0")
    return loaded


def quantfunc_r128_gate(result: dict[str, object]) -> tuple[bool, str, bool | None, tuple[str, ...]]:
    """Return whether qwen-r128 passed every hard case and fits in 16 GB."""
    arms = result.get("arms")
    if not isinstance(arms, list):
        return False, "Phase 0 result has no arms list, so the own quant runs.", None, ()
    arm = next((item for item in arms if isinstance(item, dict) and item.get("id") == "qwen-r128"), None)
    if not isinstance(arm, dict):
        return False, "Phase 0 result has no QuantFunc qwen-r128 arm, so the own quant runs.", None, ()
    cap = _cap16(arm)
    if cap is None:
        return False, "Phase 0 qwen-r128 has no cap16 result, so the own quant runs.", None, ()
    fits = cap.get("fits_16gb") is True
    found = _hard_case_map(cap.get("cases"))
    missing = [case_id for case_id in HARD_CASES if case_id not in found]
    if missing:
        listed = ", ".join(missing)
        return False, f"Phase 0 qwen-r128 is missing hard case {listed}, so the own quant runs.", fits, ()
    failed = [case_id for case_id in HARD_CASES if not _case_passed(found[case_id])]
    if failed:
        listed = ", ".join(failed)
        return (
            False,
            f"Phase 0 qwen-r128 hard case {listed} did not pass, so the own quant runs.",
            fits,
            tuple(HARD_CASES),
        )
    if cap.get("ok") is False or _error_text(cap.get("error")):
        return False, "Phase 0 qwen-r128 cap16 did not finish cleanly, so the own quant runs.", fits, tuple(HARD_CASES)
    if not fits:
        return False, "Phase 0 qwen-r128 fits_16gb is false, so the own quant runs.", False, tuple(HARD_CASES)
    return (
        True,
        "QuantFunc r128 passed every hard case and fits_16gb, so the own quant is skipped.",
        True,
        tuple(HARD_CASES),
    )


def _cap16(arm: dict[str, object]) -> dict[str, object] | None:
    caps = arm.get("caps")
    if not isinstance(caps, dict):
        return None
    cap = caps.get("cap16")
    if isinstance(cap, dict):
        return cap
    return None


def _hard_case_map(cases: object) -> dict[str, dict[str, object]]:
    found: dict[str, dict[str, object]] = {}
    if not isinstance(cases, list):
        return found
    for item in cases:
        if not isinstance(item, dict):
            continue
        case_id = str(item.get("id") or "")
        if case_id in HARD_CASES:
            found[case_id] = item
    return found


def _case_passed(case: dict[str, object]) -> bool:
    """A hard case passes when it finished and no recorded score says it failed."""
    if case.get("passed") is False or case.get("ok") is False:
        return False
    if _error_text(case.get("error")):
        return False
    scores = case.get("scores")
    if not isinstance(scores, dict):
        return True
    if scores.get("passed") is False:
        return False
    if case.get("id") == "c3" and scores.get("ocr_match") is False:
        return False
    return True


def _error_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


_BASELINE_ARMS = ("qwen-r128", "qwen-r32")
_SCORE_KEYS = ("ocr_match", "identity_cosine_proxy", "background_ssim_proxy")


def load_phase0_baseline(path: Path | None) -> dict[str, object] | None:
    """Comparison baseline from Phase 0, or None when that file is absent.

    A file that is not a JSON object raises ``StageError`` with code ``phase0``.
    The baseline does not choose the edit winner.
    """
    if path is None or not path.is_file():
        return None
    return phase0_baseline(_load(path), source=str(path))


def phase0_baseline(result: dict[str, object], *, source: str) -> dict[str, object]:
    """QuantFunc qwen-r128 and qwen-r32 cap16 rows for the eval and the card."""
    summary = result.get("fits_16gb")
    schema = result.get("schema")
    return {
        "source": source,
        "schema": schema if isinstance(schema, str) else "",
        "summary_fits_16gb": summary if isinstance(summary, bool) else None,
        "arms": _baseline_arms(result.get("arms")),
    }


def _baseline_arms(arms: object) -> list[dict[str, object]]:
    if not isinstance(arms, list):
        return []
    rows: list[dict[str, object]] = []
    for arm_id in _BASELINE_ARMS:
        arm = next((item for item in arms if isinstance(item, dict) and item.get("id") == arm_id), None)
        if isinstance(arm, dict):
            rows.append(_arm_baseline(arm))
    return rows


def _arm_baseline(arm: dict[str, object]) -> dict[str, object]:
    cap = _cap16(arm) or {}
    fits = cap.get("fits_16gb")
    ok = cap.get("ok")
    rank = arm.get("rank")
    weight = arm.get("weight_file")
    return {
        "id": str(arm.get("id") or ""),
        "rank": rank if isinstance(rank, int) and not isinstance(rank, bool) else None,
        "weight_file": weight if isinstance(weight, str) else "",
        "fits_16gb": fits if isinstance(fits, bool) else None,
        "ok": ok if isinstance(ok, bool) else None,
        "cases": _baseline_cases(cap.get("cases")),
    }


def _baseline_cases(cases: object) -> list[dict[str, object]]:
    found = _hard_case_map(cases)
    rows: list[dict[str, object]] = []
    for case_id in HARD_CASES:
        case = found.get(case_id)
        if case is None:
            continue
        seconds = case.get("seconds")
        row: dict[str, object] = {"id": case_id, "scores": _baseline_scores(case.get("scores"))}
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
            row["seconds"] = float(seconds)
        rows.append(row)
    return rows


def _baseline_scores(scores: object) -> dict[str, object]:
    if not isinstance(scores, dict):
        return {}
    kept: dict[str, object] = {}
    for key in _SCORE_KEYS:
        value = scores.get(key)
        if isinstance(value, bool) or (isinstance(value, (int, float)) and not isinstance(value, bool)):
            kept[key] = value
    return kept
