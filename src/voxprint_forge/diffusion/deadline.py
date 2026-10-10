"""Skip a part when its estimate is longer than the time left in the rental."""

from __future__ import annotations

import os
from datetime import datetime, timezone

from voxprint_forge.errors import UsageError

DEFERRED = "deferred to next rental"


def resolve_deadline(flag: str | None) -> datetime | None:
    """`--deadline` wins. Otherwise read FORGE_DEADLINE. Empty means no deadline."""
    text = (flag or "").strip() or os.environ.get("FORGE_DEADLINE", "").strip()
    if not text:
        return None
    raw = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise UsageError(f"deadline {text!r} is not ISO-8601", code="deadline") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def remaining_min(deadline: datetime, now: datetime) -> float:
    return (deadline - now).total_seconds() / 60.0


def defer_part(estimate_min: int, deadline: datetime | None, now: datetime) -> bool:
    """True when the estimate exceeds the minutes left. Equal time still runs."""
    if deadline is None:
        return False
    return estimate_min > remaining_min(deadline, now)
