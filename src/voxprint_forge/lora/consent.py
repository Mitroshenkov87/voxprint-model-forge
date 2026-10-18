"""Subject and consent per kind."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from voxprint_forge.lora.errors import LoraUsageError

KINDS = ("person", "object", "style")
CONSENT_METHODS = ("verbal", "file")


@dataclass(frozen=True)
class Subject:

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
    consent_method: str = ""
    adult_content: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_trigger(name: str, stage_name: str) -> str:
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
    consent_method: str | None = None,
    adult_content: bool = False,
) -> Subject:
    if kind not in KINDS:
        raise LoraUsageError("lora.err.no_name")
    name, stage_name = name.strip(), stage_name.strip()
    if not name and not stage_name:
        raise LoraUsageError("lora.err.no_name")
    if adult_content and not (kind == "person" and subject_adult):
        raise LoraUsageError("lora.err.adult_content")
    if kind == "person" and not (consent_obtained and subject_adult):
        raise LoraUsageError("lora.err.no_consent")
    method = ""
    if kind == "person":
        method = consent_method or ("file" if release is not None else "verbal")
        if method not in CONSENT_METHODS:
            raise LoraUsageError("lora.err.consent_method")
        if method == "verbal" and release is not None:
            raise LoraUsageError("lora.err.consent_method")
        if method == "file" and release is None:
            raise LoraUsageError("lora.err.no_release")
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
        consent_method=method,
        adult_content=bool(adult_content),
    )
