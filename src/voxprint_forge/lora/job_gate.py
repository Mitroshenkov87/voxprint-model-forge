"""One job-level refusal before training (product rule v4, minors).

``prepare`` runs the image gate per photo. This module turns any minor signal
into a refusal of the whole job, and checks the text around the photos: captions,
the trigger, the project and subject names, sample prompts and settings. A check
that cannot run refuses the job (fail closed). ``pin`` stores the sha256 of every
approved file; ``preflight_refuse`` checks them again before the trainer starts,
so a file changed, added or removed after the gate is refused.

The refusal message is neutral. The source is recorded in ``drops.jsonl`` and
``refusals.jsonl`` for audit, never shown.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

from voxprint_forge.lora.errors import LoraBlocked
from voxprint_forge.lora.gate import AGE_HARD_MIN, RULE_VERSION, child_keyword
from voxprint_forge.lora.project import Item, Project

MANIFEST = "gate.json"
AUDIT = "refusals.jsonl"
SCHEMA = "voxprint-lora-gate/1"
BLOCKED = "lora.err.blocked"
UNAVAILABLE = "lora.err.check_unavailable"
CHANGED = "lora.err.dataset_changed"


@dataclass(frozen=True)
class Clearance:
    """What the trainer was cleared to read: the manifest digest and the rule."""

    digest: str
    rule: int


def refuse(project: Project, source: str, key: str = BLOCKED) -> NoReturn:
    """Mark the job not approved, log the source, and raise :class:`LoraBlocked`."""
    try:
        _record(project, source, key)
    finally:
        raise LoraBlocked(key)


def check_texts(project: Project, texts: Mapping[str, Iterable[object]]) -> None:
    """Refuse when any text names a child or an age under 18. Keys are the audit source."""
    for source, values in texts.items():
        for value in values:
            if child_keyword(str(value)):
                refuse(project, source)


def subject_texts(project: Project, data: Mapping[str, object]) -> dict[str, list[object]]:
    """Trigger, project folder name and subject names, by audit source."""
    subject = data.get("subject") if isinstance(data.get("subject"), dict) else {}
    return {
        "trigger": [subject.get("trigger") or ""],  # type: ignore[union-attr]
        "project_name": [
            project.root.name,
            subject.get("name") or "",  # type: ignore[union-attr]
            subject.get("stage_name") or "",  # type: ignore[union-attr]
            subject.get("provenance_note") or "",  # type: ignore[union-attr]
        ],
    }


def caption_texts(items: Sequence[Item]) -> list[str]:
    return [text for item in items for text in (item.long, item.short)]


def check_subject(project: Project, data: Mapping[str, object]) -> None:
    """A person must be confirmed adult, a declared age must be 18+, and no name may name a minor."""
    subject = data.get("subject") if isinstance(data.get("subject"), dict) else {}
    if subject.get("kind", data.get("kind")) == "person" and subject.get("adult_confirmed") is not True:  # type: ignore[union-attr]
        refuse(project, "setting")
    age = subject.get("attr")  # type: ignore[union-attr]
    if isinstance(age, (int, float)) and not isinstance(age, bool) and age < AGE_HARD_MIN:
        refuse(project, "setting")
    check_texts(project, subject_texts(project, data))


def pin(project: Project) -> Clearance:
    """Store the sha256 of every prepared file and ``captions.jsonl`` after the gate passed."""
    files = _hashes(project)
    payload = {"schema": SCHEMA, "rule": RULE_VERSION, "files": files}
    path = project.root / MANIFEST
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)
    return Clearance(_digest(files), RULE_VERSION)


def unpin(project: Project) -> None:
    """Drop the approval before a new prepare round: until it passes, training is refused."""
    (project.root / MANIFEST).unlink(missing_ok=True)
    if project.file.is_file():
        try:
            data = project.read()
        except Exception:  # noqa: BLE001
            data = {}
        project.write({**data, "gate_passed": False})


def repin_captions(project: Project) -> Clearance | None:
    """After a caption edit: images must still match, edited captions are checked, then pinned again.

    A project that was never pinned stays unpinned; ``preflight_refuse`` refuses it.
    """
    if not (project.root / MANIFEST).is_file():
        return None
    try:
        verify(project, skip=("captions.jsonl",))
        check_texts(project, {"caption": caption_texts(project.items())})
        return pin(project)
    except LoraBlocked:
        raise
    except Exception:  # noqa: BLE001
        refuse(project, "check_error", UNAVAILABLE)


def verify(project: Project, *, skip: Sequence[str] = ()) -> Clearance:
    """Refuse when a pinned file changed, or a file was added or removed."""
    path = project.root / MANIFEST
    if not path.is_file():
        refuse(project, "hash_missing", CHANGED)
    stored = json.loads(path.read_text(encoding="utf-8"))
    pinned = stored.get("files") if stored.get("schema") == SCHEMA else None
    if not isinstance(pinned, dict) or stored.get("rule") != RULE_VERSION:
        refuse(project, "hash_missing", CHANGED)
    now = _hashes(project)
    if {k: v for k, v in now.items() if k not in skip} != {k: v for k, v in pinned.items() if k not in skip}:
        refuse(project, "hash_mismatch", CHANGED)
    return Clearance(_digest(now), RULE_VERSION)


def preflight_refuse(
    project: Project,
    *,
    prompts: Sequence[str] = (),
    settings: Mapping[str, object] | None = None,
) -> Clearance:
    """The one refusal step before training. Returns the clearance or raises :class:`LoraBlocked`.

    Order: the gate passed, files match their pins, the subject and names, captions,
    sample prompts (given and stored for review), then settings. Any error while
    checking refuses the job.
    """
    try:
        data = project.read()
        if data.get("gate_passed") is not True:
            refuse(project, "not_approved")
        clearance = verify(project)
        check_subject(project, data)
        check_texts(project, {"caption": caption_texts(project.items())})
        check_texts(project, {"sample_prompt": [*prompts, *_stored_prompts(project)]})
        check_texts(project, {"setting": _setting_values({**(settings or {}), **_project_settings(data)})})
        return clearance
    except LoraBlocked:
        raise
    except Exception:  # noqa: BLE001
        refuse(project, "check_error", UNAVAILABLE)


def check_run(project: Project, prompts: Sequence[str], settings: Mapping[str, object]) -> None:
    """The prompts and settings the trainer will actually use, once they are final."""
    check_texts(project, {"sample_prompt": list(prompts), "setting": _setting_values(settings)})


def _stored_prompts(project: Project) -> list[str]:
    from voxprint_forge.lora.review import load_state

    prompts = load_state(project).get("prompts") or []
    return [str(p) for p in prompts] if isinstance(prompts, list) else [str(prompts)]


def _project_settings(data: Mapping[str, object]) -> dict[str, object]:
    return {k: v for k, v in data.items() if k not in ("subject", "schema")}


def _setting_values(settings: Mapping[str, object]) -> list[str]:
    out: list[str] = []
    for key, value in settings.items():
        if isinstance(value, Mapping):
            out += _setting_values(value)
        elif isinstance(value, (list, tuple)):
            out += [str(v) for v in value]
        elif isinstance(value, str):
            out.append(f"{key} {value}")
    return out


def _hashes(project: Project) -> dict[str, str]:
    files: dict[str, str] = {}
    if project.prepared.is_dir():
        for path in sorted(project.prepared.rglob("*")):
            if path.is_file():
                files[path.relative_to(project.root).as_posix()] = _sha256(path)
    if project.captions.is_file():
        files["captions.jsonl"] = _sha256(project.captions)
    return files


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(files: Mapping[str, str]) -> str:
    return hashlib.sha256(json.dumps(dict(files), sort_keys=True).encode()).hexdigest()


def _record(project: Project, source: str, key: str) -> None:
    project.root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    row = {"id": "job", "name": "", "reason": "lora.drop.safety", "scope": "job"}
    with project.drops.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (project.root / AUDIT).open("a", encoding="utf-8") as handle:
        audit = {"time": stamp, "source": source, "key": key, "rule": RULE_VERSION}
        handle.write(json.dumps(audit, ensure_ascii=False) + "\n")
    (project.root / MANIFEST).unlink(missing_ok=True)
    try:
        data = project.read()
    except Exception:  # noqa: BLE001
        data = {}
    project.write({**data, "gate_passed": False})
