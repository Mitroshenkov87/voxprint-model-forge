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

A project that stores a ``bases`` list (written by ``lora prepare``) keeps
one dataset and puts each base's versions under ``checkpoints/<base>/`` and
``export/<base>/``. A project with only ``train.base``, or with neither, is
a legacy project: those two folders stay directly under the root, and the
base list is ``[train.base]`` or ``["z-image"]``. ``lora retrain`` appends a
base. On a legacy project it writes ``bases`` and moves root checkpoint and
export files under the base they already belong to.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from voxprint_forge.lora import DEFAULT_BASE, legacy_base
from voxprint_forge.lora.errors import LoraUsageError

# ``lora prepare --base`` names. ``both`` is Z-Image, then moss.
_PREPARE_CHOICES = {
    "moss": ("moss-image-base",),
    "z-image": ("z-image",),
    "both": ("z-image", "moss-image-base"),
}
# ``lora retrain --base`` spellings. The reverse of the single-base prepare names.
_CLI_BASE = {ids[0]: name for name, ids in _PREPARE_CHOICES.items() if len(ids) == 1}

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


def bases_for_choice(choice: str) -> list[str]:
    """MDL ids for a ``--base`` value.

    ``moss``, ``z-image`` and ``both`` are the prepare names. Any other
    string is an MDL id (``lora train --base tiny-test``) and is kept as given.
    """
    text = choice.strip()
    found = _PREPARE_CHOICES.get(text.lower())
    if found is not None:
        return list(found)
    if text.lower() == "moss-image-base":
        return ["moss-image-base"]
    return [text]


def train_bases(choice: str | None, recorded: Sequence[str], *, explicit: bool) -> list[str]:
    """Bases one ``lora train`` runs.

    An omitted ``--base`` uses the project's first base, so a legacy project
    stays on Z-Image. ``both`` trains every recorded base when the project
    lists more than one, and otherwise Z-Image then moss.
    """
    known = [item for item in recorded if item]
    if not explicit:
        return [known[0]] if known else [legacy_base()]
    text = (choice or "").strip()
    if text.lower() == "both":
        if len(known) > 1:
            return known
        return ["z-image", "moss-image-base"]
    return bases_for_choice(text) if text else ([known[0]] if known else [DEFAULT_BASE])


def choice_for_base(mdl_id: str) -> str:
    """``--base`` spelling for an MDL id. ``moss-image-base`` is ``moss``."""
    return _CLI_BASE.get(mdl_id, mdl_id)


def base_mismatch_params(found: str, target: str) -> dict[str, str]:
    """Parameters for ``lora.err.base_mismatch``, including the retrain command's ``--base``."""
    return {"base": found, "target": target, "cli": choice_for_base(target)}


def add_base(project: Project, base_id: str) -> list[str]:
    """Append ``base_id`` to the project's bases and return the list.

    The prepared dataset is left as it is. A legacy project (no ``bases``
    key) records its current base first. Files sitting directly in
    ``checkpoints/`` and ``export/`` move under that base, so they stay
    reachable once the project uses a folder per base.
    """
    cleaned = base_id.strip()
    if not cleaned:
        raise LoraUsageError("lora.err.base_missing")
    data = project.read()
    current = project.recorded_bases()
    legacy = project.legacy_layout()
    updated = list(current)
    if cleaned not in updated:
        updated.append(cleaned)
    if legacy:
        _relocate_legacy_outputs(project.root, current[0])
    if not legacy and data.get("bases") == updated:
        return updated
    project.write({**data, "bases": updated})
    return updated


class Project:
    """Paths and files of one project folder."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._active_base: str | None = None

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

    def recorded_bases(self) -> list[str]:
        """Target bases. A missing ``bases`` list is a legacy project.

        ``train.base`` when that was recorded, otherwise Z-Image. The dataset
        itself has no base.
        """
        try:
            data = self.read()
        except LoraUsageError:
            return [legacy_base()]
        cleaned = _id_list(data.get("bases"))
        if cleaned is not None and "bases" in data:
            return cleaned or [legacy_base()]
        train = data.get("train")
        if isinstance(train, dict):
            base_id = str(train.get("base") or "").strip()
            if base_id:
                return [base_id]
        return [legacy_base()]

    def legacy_layout(self) -> bool:
        """True when ``checkpoints/`` and ``export/`` stay at the project root.

        New projects store ``bases``. A project with only ``train.base``, or
        with neither, keeps the folders older versions wrote.
        """
        try:
            data = self.read()
        except LoraUsageError:
            return True
        return _id_list(data.get("bases")) is None or "bases" not in data

    def activate_base(self, base: str) -> None:
        """Point ``checkpoints`` and ``export`` at ``base`` for this process.

        Legacy projects ignore it: both folders stay at the root.
        """
        self._active_base = str(base).strip() or None

    def _output_base(self) -> str:
        """The base whose output folders are in use. Legacy projects do not call this."""
        if self._active_base:
            return self._active_base
        try:
            data = self.read()
        except LoraUsageError:
            return self.recorded_bases()[0]
        train = data.get("train")
        if isinstance(train, dict):
            base_id = str(train.get("base") or "").strip()
            if base_id:
                return base_id
        return self.recorded_bases()[0]

    @property
    def checkpoints(self) -> Path:
        """Intermediate LoRA versions and training state.

        Legacy projects: ``checkpoints/``. Projects with ``bases``:
        ``checkpoints/<base>/``.
        """
        root = self.root / "checkpoints"
        if self.legacy_layout():
            return root
        return root / self._output_base()

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
        """Finished ``.voxlora`` packages and ``.safetensors`` exports.

        Legacy projects: ``export/``. Projects with ``bases``: ``export/<base>/``.
        """
        root = self.root / "export"
        if self.legacy_layout():
            return root
        return root / self._output_base()

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


def _id_list(value: object) -> list[str] | None:
    """A stored list of MDL ids, or None when ``value`` is not a list."""
    if not isinstance(value, list):
        return None
    return [str(item).strip() for item in value if str(item).strip()]


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


def _relocate_legacy_outputs(root: Path, base_id: str) -> None:
    """Move root-level ``checkpoints/`` and ``export/`` entries under ``base_id``."""
    for name in ("checkpoints", "export"):
        folder = root / name
        if not folder.is_dir():
            continue
        entries = [path for path in folder.iterdir() if path.name != base_id]
        if not entries:
            continue
        dest = folder / base_id
        dest.mkdir(parents=True, exist_ok=True)
        for path in entries:
            target = dest / path.name
            if not target.exists():
                path.rename(target)
    _retarget_review_checkpoint(root, base_id)


def _retarget_review_checkpoint(root: Path, base_id: str) -> None:
    """Point an open review at the checkpoint after it moved under ``base_id``."""
    path = root / "review" / "state.json"
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if not isinstance(data, dict):
        return
    raw = data.get("checkpoint")
    if not isinstance(raw, str):
        return
    updated = _nested_checkpoint(raw, base_id)
    if updated == raw:
        return
    data["checkpoint"] = updated
    _atomic(path, json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _nested_checkpoint(raw: str, base_id: str) -> str:
    """``checkpoints/<file>`` becomes ``checkpoints/<base_id>/<file>``."""
    text = raw.replace("\\", "/")
    prefix = "checkpoints/"
    nested = f"checkpoints/{base_id}/"
    if text.startswith(nested) or not text.startswith(prefix):
        return raw
    rest = text[len(prefix):]
    if not rest:
        return raw
    return nested + rest
