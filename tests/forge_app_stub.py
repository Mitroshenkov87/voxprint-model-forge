"""Test helpers for the Model Forge app: procedural photos and a stub engine runner.

Photos are generated gradients (no people, no faces). The stub runner replays
scripted ``lora`` events and envelopes instead of starting the CLI.
"""

from __future__ import annotations

from pathlib import Path

from voxprint_forge.app.engine import EngineEvent, Outcome

ALL_COMMANDS = frozenset({"prepare", "train", "extend", "export"})


def make_photos(folder: Path, count: int) -> list[Path]:
    """Write ``count`` small gradient PNGs and return their paths."""
    from PySide6 import QtGui

    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for index in range(count):
        image = QtGui.QImage(96, 96, QtGui.QImage.Format.Format_RGB32)
        for y in range(96):
            for x in range(96):
                image.setPixel(x, y, QtGui.qRgb((x * 2 + index * 20) % 256, (y * 2) % 256, (index * 37) % 256))
        path = folder / f"shape-{index:02d}.png"
        image.save(str(path))
        paths.append(path)
    return paths


def write_prepared(project: Path, photos: list[Path], drops: list[tuple[Path, str]]) -> None:
    """Write ``captions.jsonl`` and ``drops.jsonl`` the way ``lora prepare`` does."""
    import json

    project.mkdir(parents=True, exist_ok=True)
    rows = [{"id": f"p{i}", "file": str(photo), "long": "a gradient test pattern", "short": f"a gradient {i}",
             "round": 0} for i, photo in enumerate(photos)]
    (project / "captions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    lines = [json.dumps({"id": f"d{i}", "name": path.name, "reason": reason}) for i, (path, reason) in
             enumerate(drops)]
    (project / "drops.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")


class StubRunner:
    """Answers each ``lora`` command from ``script``: events, then an outcome."""

    def __init__(self, script: dict[str, tuple[list[EngineEvent], Outcome]] | None = None) -> None:
        self.script = script or {}
        self.calls: list[list[str]] = []
        self.effects: dict[str, object] = {}
        self.busy = False
        self.interrupted = 0
        self.hold = False
        self._pending: tuple[object, object, Outcome] | None = None

    def start(self, args, on_event, on_done) -> None:
        self.calls.append(list(args))
        events, outcome = self.script.get(args[2], ([], Outcome(True, {})))
        effect = self.effects.get(args[2])
        if effect is not None:
            effect(Path(args[args.index("--project") + 1]))
        self.busy = True
        for event in events:
            on_event(event)
        if self.hold:
            self._pending = (on_event, on_done, outcome)
            return
        self.busy = False
        on_done(outcome)

    def release(self, outcome: Outcome | None = None) -> None:
        """Finish a held job."""
        _on_event, on_done, scripted = self._pending  # type: ignore[misc]
        self._pending = None
        self.busy = False
        on_done(outcome or scripted)  # type: ignore[operator]

    def interrupt(self) -> None:
        self.interrupted += 1
