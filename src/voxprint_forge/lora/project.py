"""Training project folder (plan §2).

Fixed layout, documented in docs/LORA.md ("Project folder")::

    project.json           subject, settings, train and likeness records
    prepared/              cropped training photos (one PNG per photo id)
    captions.jsonl         one prepared photo per line: id, file, captions
    drops.jsonl            photos set aside, with a reason key
    gate_trace.jsonl       per checked photo, the safety gate's codes and numbers
    gate_run.json          state of the last safety-check run (running, complete, interrupted)
    checkpoints/           intermediate LoRA versions and training state
    previews/              test pictures drawn during training
    review/                review-round state and samples
    eval/                  likeness score rows and reports
    export/                finished .voxlora packages and .safetensors exports
    release/               an attached consent release file
    refset/                reference-set rounds built for training
    train.json             latest training progress (step, loss, ETA)
    train.log              one progress line per report
    vram.csv               VRAM samples on the same cadence
    PAUSE                  while it exists, training saves a version and stops

``prepared/``, ``checkpoints/``, ``previews/``, ``review/``, ``eval/``,
``export/``, ``release/`` and ``refset/`` are created on demand. All code
builds these paths through the :class:`Project` properties, so the on-disk
names stay stable for projects created by older versions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from voxprint_forge.lora.errors import LoraUsageError

SCHEMA = "voxprint-lora-project/1"


@dataclass(frozen=True)
class Item:
    """One prepared photo and its captions. ``round`` is 0 for the first set, then +1 per extend."""

    id: str
    file: str
    long: str
    short: str
    round: int = 0
    unclothed: bool | None = None


PAUSE_FILE = "PAUSE"


class Project:
    """Paths and files of one project folder."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    @property
    def file(self) -> Path:
        """``project.json``."""
        return self.root / "project.json"

    @property
    def prepared(self) -> Path:
        """Cropped training images."""
        return self.root / "prepared"

    @property
    def captions(self) -> Path:
        """``captions.jsonl``, one :class:`Item` per line."""
        return self.root / "captions.jsonl"

    @property
    def drops(self) -> Path:
        """``drops.jsonl``: photos set aside, with a reason key."""
        return self.root / "drops.jsonl"

    @property
    def gate_trace(self) -> Path:
        """``gate_trace.jsonl``: per checked photo, the safety gate's codes and numbers (no image data)."""
        return self.root / "gate_trace.jsonl"

    @property
    def gate_run(self) -> Path:
        """``gate_run.json``: the prepare safety-check run (``running``, ``interrupted`` or ``complete``)."""
        return self.root / "gate_run.json"

    @property
    def pause(self) -> Path:
        """``PAUSE``: while this file exists, training saves a version and stops.

        The GUI creates it to pause and deletes it before ``--resume``. The
        engine only reads it. This works the same on Windows, where a GUI
        cannot send Ctrl+C to the engine.
        """
        return self.root / PAUSE_FILE

    @property
    def checkpoints(self) -> Path:
        """Intermediate LoRA versions and training state."""
        return self.root / "checkpoints"

    @property
    def previews(self) -> Path:
        """Test pictures."""
        return self.root / "previews"

    @property
    def review(self) -> Path:
        """Review-round state and sample pictures."""
        return self.root / "review"

    @property
    def eval(self) -> Path:
        """Likeness score rows and reports."""
        return self.root / "eval"

    @property
    def export(self) -> Path:
        """Finished ``.voxlora`` packages and ``.safetensors`` exports."""
        return self.root / "export"

    @property
    def release(self) -> Path:
        """An attached consent release file."""
        return self.root / "release"

    @property
    def refset(self) -> Path:
        """Reference-set rounds built for training."""
        return self.root / "refset"

    @property
    def train_state(self) -> Path:
        """``train.json``: the latest training progress (step, loss, ETA)."""
        return self.root / "train.json"

    @property
    def train_log(self) -> Path:
        """``train.log``: one progress line per report."""
        return self.root / "train.log"

    @property
    def vram_log(self) -> Path:
        """``vram.csv``: VRAM samples on the progress cadence."""
        return self.root / "vram.csv"

    def resolve(self, stored: str | Path) -> Path:
        """A path stored in a project file (``captions.jsonl``, review state) back to an absolute path.

        Stored paths are relative to the project root; an absolute path is used as is.
        """
        path = Path(stored)
        return path if path.is_absolute() else self.root / path

    def read(self) -> dict[str, object]:
        """``project.json``; a missing file means the photos were never prepared."""
        if not self.file.is_file():
            raise LoraUsageError("lora.err.not_prepared")
        data = json.loads(self.file.read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise LoraUsageError("lora.err.not_prepared")
        return data

    def write(self, data: dict[str, object]) -> None:
        """Write ``project.json`` with the schema tag."""
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"schema": SCHEMA, **{key: value for key, value in data.items() if key != "schema"}}
        _atomic(self.file, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def items(self) -> list[Item]:
        """Prepared items whose image exists."""
        if not self.captions.is_file():
            return []
        found = []
        for line in self.captions.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                flag = row.get("unclothed", None)
                item = Item(
                    row["id"],
                    row["file"],
                    row.get("long", ""),
                    row.get("short", ""),
                    int(row.get("round", 0)),
                    flag if isinstance(flag, bool) else None,
                )
                if self.resolve(item.file).is_file():
                    found.append(item)
        return found

    def write_items(self, items: list[Item]) -> None:
        """Replace ``captions.jsonl``."""
        lines = [json.dumps(_item_row(i), ensure_ascii=False, sort_keys=True) for i in items]
        self.root.mkdir(parents=True, exist_ok=True)
        _atomic(self.captions, "".join(line + "\n" for line in lines))


def _item_row(item: Item) -> dict[str, object]:
    row: dict[str, object] = {
        "id": item.id, "file": item.file, "long": item.long, "short": item.short, "round": item.round,
    }
    if isinstance(item.unclothed, bool):
        row["unclothed"] = item.unclothed
    return row


def package_stem(data: dict[str, object]) -> str:
    """``.voxlora`` file stem: the stage name, or the name when the stage name is empty."""
    subject = data.get("subject")
    fields = subject if isinstance(subject, dict) else {}
    raw = str(fields.get("stage_name") or "").strip() or str(fields.get("name") or "").strip() or "lora"
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', " ", raw).strip().rstrip(".")
    return cleaned or "lora"


def _atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8", newline="\n")
    tmp.replace(path)
