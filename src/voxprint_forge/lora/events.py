"""Plain-words progress events shared by the CLI and the GUI (plan §6)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from voxprint_forge.lora.i18n import text

STATES = ("running", "paused", "done", "failed", "blocked")
WARMUP_UNITS = 2


@dataclass(frozen=True)
class Event:
    """One progress event: what runs now, why, the percent and the time left."""

    step: str
    message_key: str
    params: dict[str, object] = field(default_factory=dict)
    reason_key: str | None = None
    percent: float | None = None
    eta_seconds: int | None = None
    state: str = "running"

    def to_dict(self) -> dict[str, object]:
        """The event as a JSON-ready dict."""
        return asdict(self)


def sentence(event: Event, lang: str | None = None) -> str:
    """The human sentence for ``event``: what, why, and the time left."""
    parts = [text(event.message_key, lang, **event.params)]
    if event.reason_key:
        parts.append(text(event.reason_key, lang))
    if event.state == "running" and event.percent is not None:
        if event.eta_seconds is None:
            parts.append(text("lora.progress.eta_unknown", lang))
        else:
            parts.append(text("lora.progress.eta", lang, minutes=max(1, round(event.eta_seconds / 60))))
    return " — ".join(parts)


def json_line(event: Event) -> str:
    """One compact JSON line for ``--json``."""
    return json.dumps({"event": event.to_dict()}, ensure_ascii=False, sort_keys=True)


class Progress:
    """Emit events with an ETA measured from step times after a short warm-up."""

    def __init__(self, sink: Callable[[Event], None] | None = None, *, clock: Callable[[], float] = time.monotonic):
        self.sink = sink
        self.clock = clock
        self.events: list[Event] = []
        self._phase: str | None = None
        self._mark: tuple[float, int] | None = None

    def emit(
        self,
        step: str,
        message_key: str,
        params: dict[str, object] | None = None,
        reason_key: str | None = None,
        *,
        state: str = "running",
        done: int | None = None,
        total: int | None = None,
        offset: int = 0,
    ) -> Event:
        """Send one event. ``done``/``total`` give the percent and ETA; ``offset`` is units done before a resume."""
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}")
        percent = eta = None
        if done is not None and total:
            percent = round(100.0 * done / total, 1)
            eta = self._eta(step, done - offset, total - offset)
        event = Event(step, message_key, dict(params or {}), reason_key, percent, eta, state)
        self.events.append(event)
        if self.sink is not None:
            self.sink(event)
        return event

    def _eta(self, phase: str, done: int, total: int) -> int | None:
        now = self.clock()
        if phase != self._phase:
            self._phase, self._mark = phase, None
        if done < WARMUP_UNITS:
            return None
        if self._mark is None:
            self._mark = (now, done)
            return None
        start, units = self._mark
        if done <= units:
            return None
        per_unit = (now - start) / (done - units)
        return int(round(per_unit * max(0, total - done)))
