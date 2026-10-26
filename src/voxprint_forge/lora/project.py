"""Training project folder (plan §2): photos/, prepared/, captions.jsonl, drops.jsonl, checkpoints/, previews/."""

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
                if (self.root / item.file).is_file():
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
