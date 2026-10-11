"""Winner rule for own INT4, QuantFunc r128/r256, and FP8.

When every arm has identity, hands, and text, the gate is the canvas hard-case
rule: identity within 0.05 of FP8, text exact, hands no worse than FP8, and a
16 GB fit (smi peak at most 15.0 GiB) preferred over a 24 GB fit (23.0 GiB).
Missing quality numbers fall through to the fastest arm that fits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from voxprint_forge.diffusion.int4.pins import FITS_16_MIB, FITS_24_MIB, IDENTITY_DELTA

Rule = Literal["quality", "speed"]


@dataclass(frozen=True)
class Arm:
    """One measured candidate."""

    arm_id: str
    peak_vram_smi_mib: float
    seconds_per_image: float
    identity: float | None
    hands: float | None
    text: bool | None
    edit_fidelity: float | None
    rank: int | None = None


@dataclass(frozen=True)
class Decision:
    """The recorded winner and the rule that chose it."""

    winner: str | None
    rule: Rule
    fits_16gb: dict[str, bool]
    fits_24gb: dict[str, bool]
    passed: dict[str, bool]


def fits_16(peak_mib: float) -> bool:
    """True when the smi peak stays inside the 16 GB cap."""
    return peak_mib <= FITS_16_MIB


def fits_24(peak_mib: float) -> bool:
    """True when the smi peak stays inside the 24 GB cap."""
    return peak_mib <= FITS_24_MIB


def _quality_pass(arm: Arm, reference: Arm) -> bool:
    if arm.identity is None or reference.identity is None or arm.hands is None or reference.hands is None:
        return False
    if arm.text is not True or reference.text is not True:
        return False
    if abs(arm.identity - reference.identity) > IDENTITY_DELTA:
        return False
    return arm.hands + 1e-9 >= reference.hands


def _best(arms: list[Arm], *, quality: dict[str, bool], need_16: bool) -> Arm | None:
    pool = [
        arm
        for arm in arms
        if quality[arm.arm_id] and (fits_16(arm.peak_vram_smi_mib) if need_16 else fits_24(arm.peak_vram_smi_mib))
    ]
    if not pool:
        return None
    reference = next(arm for arm in arms if arm.arm_id == "fp8")

    def key(arm: Arm) -> tuple[float, float]:
        delta = abs((arm.identity or 0.0) - (reference.identity or 0.0))
        return (delta, arm.seconds_per_image)

    return min(pool, key=key)


def _speed_best(arms: list[Arm], *, need_16: bool) -> Arm | None:
    pool = []
    for arm in arms:
        fits = fits_16(arm.peak_vram_smi_mib) if need_16 else fits_24(arm.peak_vram_smi_mib)
        if fits:
            pool.append(arm)
    if not pool:
        return None
    return min(pool, key=lambda arm: arm.seconds_per_image)


def decide_winner(arms: list[Arm]) -> Decision:
    """Pick the winner. FP8 is the quality reference and can also win."""
    by_id = {arm.arm_id: arm for arm in arms}
    if "fp8" not in by_id:
        raise ValueError("fp8 arm is required")
    fit16 = {arm.arm_id: fits_16(arm.peak_vram_smi_mib) for arm in arms}
    fit24 = {arm.arm_id: fits_24(arm.peak_vram_smi_mib) for arm in arms}
    complete = all(arm.identity is not None and arm.hands is not None and arm.text is not None for arm in arms)
    reference = by_id["fp8"]
    if complete:
        passed = {arm.arm_id: _quality_pass(arm, reference) for arm in arms}
        chosen = _best(arms, quality=passed, need_16=True) or _best(arms, quality=passed, need_16=False)
        return Decision(
            winner=None if chosen is None else chosen.arm_id,
            rule="quality",
            fits_16gb=fit16,
            fits_24gb=fit24,
            passed=passed,
        )
    passed = {arm.arm_id: False for arm in arms}
    chosen = _speed_best(arms, need_16=True) or _speed_best(arms, need_16=False)
    return Decision(
        winner=None if chosen is None else chosen.arm_id,
        rule="speed",
        fits_16gb=fit16,
        fits_24gb=fit24,
        passed=passed,
    )


def dry_run_arms() -> list[Arm]:
    """Synthetic CPU-smoke scores. They are not a quality measurement.

    own-r256 passes the gate and fits 16 GB. QuantFunc r256 is closer to FP8
    on identity but misses the 15.0 GiB cap. r128 misses the identity delta.
    """
    return [
        Arm("fp8", 21 * 1024, 42.0, 0.92, 0.80, True, 0.90, None),
        Arm("quantfunc-r128", 14.2 * 1024, 9.0, 0.86, 0.70, True, 0.70, 128),
        Arm("quantfunc-r256", 15.6 * 1024, 11.0, 0.90, 0.80, True, 0.84, 256),
        Arm("own-r256", 14.8 * 1024, 10.0, 0.91, 0.82, True, 0.86, 256),
    ]
