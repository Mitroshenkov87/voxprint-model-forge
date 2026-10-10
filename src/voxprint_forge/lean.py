"""Stop a check once a confidence interval decides the gate.

GPU evals run a few hard cases. After a minimum count, a 95% interval that lies
entirely inside the gate, or entirely outside it, ends that component. A clear
failure of one metric ends it even when another metric is still wide.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

# Two-sided 95% Student-t critical values for df = 1..30. Larger df uses 1.96.
_T95: tuple[float, ...] = (
    0.0,
    12.706,
    4.303,
    3.182,
    2.776,
    2.571,
    2.447,
    2.365,
    2.306,
    2.262,
    2.228,
    2.201,
    2.179,
    2.160,
    2.145,
    2.131,
    2.120,
    2.110,
    2.101,
    2.093,
    2.086,
    2.080,
    2.074,
    2.069,
    2.064,
    2.060,
    2.056,
    2.052,
    2.048,
    2.045,
    2.042,
)

REFUSAL_MINIMUM = 4
REFUSAL_LINE = 0.5
REVIEW_LIMIT = 4


def t_critical(df: int) -> float:
    """Student-t 95% critical value for ``df`` degrees of freedom."""
    if df < 1:
        raise ValueError("df must be at least 1")
    if df < len(_T95):
        return _T95[df]
    return 1.96


def mean_interval(values: Sequence[float]) -> tuple[float, float] | None:
    """95% Student-t interval for the mean of LPIPS or a CLIP score.

    Two samples are required. The sample variance uses ``n - 1``. A constant
    series has a zero-width interval at that value, so a mean that is already
    on one side of the gate is decided.
    """
    count = len(values)
    if count < 2:
        return None
    mean = sum(values) / count
    variance = sum((item - mean) ** 2 for item in values) / (count - 1)
    half = t_critical(count - 1) * math.sqrt(variance / count)
    return mean - half, mean + half


def proportion_interval(hits: int, trials: int) -> tuple[float, float] | None:
    """Wilson score 95% interval for a refusal rate.

    ``hits`` is the number of refusals. The interval stays inside ``[0, 1]``.
    Four refusals out of four sit entirely above one half. Zero refusals out of
    four sit entirely below one half. Two out of four still covers one half, so
    that series is not decided.
    """
    if trials < 1 or hits < 0 or hits > trials:
        return None
    z = 1.96
    phat = hits / trials
    z2 = z * z
    denom = 1.0 + z2 / trials
    centre = (phat + z2 / (2 * trials)) / denom
    margin = z * math.sqrt((phat * (1.0 - phat) / trials) + z2 / (4 * trials * trials)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def separates(low: float, high: float, line: float, *, higher_is_worse: bool) -> bool:
    """True when the interval is entirely inside the gate or entirely outside it."""
    if higher_is_worse:
        return high <= line or low > line
    return low >= line or high < line


def metrics_decided(
    series: Sequence[tuple[Sequence[float], float, bool]],
    *,
    minimum: int,
) -> bool:
    """True when the samples already decide every gate, or one gate has failed.

    Each item is ``(values, line, higher_is_worse)``. Fewer than ``minimum``
    samples never decides. A clear failure returns immediately. A pass waits
    until every series is on one side of its line.
    """
    if not series:
        return False
    count = len(series[0][0])
    if count < minimum or any(len(values) != count for values, _line, _worse in series):
        return False
    pending = False
    for values, line, higher_is_worse in series:
        interval = mean_interval(values)
        if interval is None:
            return False
        low, high = interval
        if not separates(low, high, line, higher_is_worse=higher_is_worse):
            pending = True
            continue
        failed = low > line if higher_is_worse else high < line
        if failed:
            return True
    return not pending


def wave_size(done: int, cap: int, *, decided: bool, minimum: int) -> int:
    """How many new cases to run. Zero when the cap is met or the gate is decided."""
    if done >= cap or (done >= minimum and decided):
        return 0
    if done == 0:
        return min(minimum, cap)
    return 1


def review_document(rows: Sequence[tuple[str, str, int, str]]) -> str:
    """Labels for a full-resolution review pack.

    Each row is filename, prompt, seed, and role (``reference`` or ``candidate``).
    """
    lines = ["# Review pack", ""]
    for filename, prompt, seed, role in rows:
        lines.append(f"- {role} seed {seed}: `{filename}`")
        lines.append(f"  {prompt}")
    lines.append("")
    return "\n".join(lines)
