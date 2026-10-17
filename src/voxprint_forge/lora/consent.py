"""Subject and consent per kind (plan §1.6, §9.2). Only typed values are kept; nothing is collected."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from voxprint_forge.lora.errors import LoraUsageError

KINDS = ("person", "object", "style")


@dataclass(frozen=True)
class Subject:
    """What the photos show and what the user confirmed."""

    kind: str
    name: str = ""
    stage_name: str = ""
    trigger: str = ""
    consent_obtained: bool = False
    adult_confirmed: bool = False
    rights_confirmed: bool = False
    provenance_note: str = ""
    release_attached: bool = False
    release_sha256: str = ""

    def to_dict(self) -> dict[str, object]:
        """Plain dict for ``project.json``."""
        return asdict(self)


def default_trigger(name: str, stage_name: str) -> str:
    """The stage name if given, else the name: lower case, letters and digits joined by ``_``."""
    source = stage_name or name
    return re.sub(r"[^\w]+", "_", source.strip().lower(), flags=re.UNICODE).strip("_")


def check_subject(
    kind: str,
    *,
    name: str = "",
    stage_name: str = "",
    trigger: str = "",
    consent_obtained: bool = False,
    subject_adult: bool = False,
    rights_confirmed: bool = False,
    provenance_note: str = "",
    release: Path | None = None,
) -> Subject:
    """Validate the inputs for ``kind`` and return the :class:`Subject`. The minors gate runs regardless."""
    if kind not in KINDS:
        raise LoraUsageError("lora.err.no_name")
    name, stage_name = name.strip(), stage_name.strip()
    if not name and not stage_name:
        raise LoraUsageError("lora.err.no_name")
    if kind == "person" and not (consent_obtained and subject_adult):
        raise LoraUsageError("lora.err.no_consent")
    if kind in ("object", "style") and not rights_confirmed:
        raise LoraUsageError("lora.err.no_rights")
    digest = ""
    if release is not None:
        digest = hashlib.sha256(Path(release).read_bytes()).hexdigest()
    return Subject(
        kind=kind,
        name=name,
        stage_name=stage_name,
        trigger=trigger.strip() or default_trigger(name, stage_name),
        consent_obtained=bool(consent_obtained) if kind == "person" else False,
        adult_confirmed=bool(subject_adult) if kind == "person" else False,
        rights_confirmed=bool(rights_confirmed) if kind != "person" else False,
        provenance_note=provenance_note,
        release_attached=release is not None,
        release_sha256=digest,
    )
