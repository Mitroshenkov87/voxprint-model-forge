"""The model release file (``release.txt``): parsing shared by the GUI and ``forge-train`` (no Qt).

``release.txt`` sits directly inside a photo folder and records, in plain English
``Key: value`` lines, who the model is and what they agreed to. A complete file
(``Adult (18+): yes``, ``Consent to training: yes`` and ``LoRA training allowed: yes``)
answers the consent prompts by itself; a ``no`` or a missing line only means the usual
prompts are asked. The file never adds a restriction and never skips the per-photo
age check — that gate always runs on prepare.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RELEASE_FILE = "release.txt"
_EXPLANATION_MARKER = "--- How to fill this in ---"
_YES_NO_KEYS = {
    "adult (18+)": "adult",
    "no minors in photos": "no_minors",
    "consent to training": "consent_training",
    "consent to generation": "consent_generation",
    "lora training allowed": "lora_allowed",
    "adult content in training": "adult_content",
    "adult content in generation": "adult_generation",
}

# The approved template (Aleksandr, 2026-11-16 00:40 MSK), byte for byte.
TEMPLATE = (
    'Model name: "Tatiana Model"\n'
    'Adult (18+): yes\n'
    'Consent to training: yes\n'
    'Consent to generation: yes\n'
    'LoRA training allowed: yes\n'
    'Adult content in training: yes\n'
    'Adult content in generation: yes\n'
    'No minors in photos: yes\n'
    'Date: 2026-11-15\n'
    '\n'
    '--- How to fill this in ---\n'
    'Put this file, named release.txt, in the same folder as the photos.\n'
    'Forge reads it, attaches it to the project with its checksum, and asks no\n'
    'consent questions.\n'
    '\n'
    'Model name\n'
    "  The person's name, in double quotes. It becomes the default LoRA name;\n"
    '  spaces turn into underscores, so "Tatiana Model" gives\n'
    '  Tatiana_Model.safetensors.\n'
    '\n'
    'Adult (18+)\n'
    '  yes = the model is an adult. Required for training a person LoRA.\n'
    '\n'
    'Consent to training\n'
    '  yes = the model agreed that her photos are used to train a LoRA.\n'
    '\n'
    'Consent to generation\n'
    '  yes = the model agreed that pictures of her are generated with the LoRA.\n'
    '  no  = the LoRA is trained but marked "no generation".\n'
    '\n'
    'LoRA training allowed\n'
    '  yes = Forge may train right away.\n'
    '  no  = Forge asks you in the window before training.\n'
    '\n'
    'Adult content in training\n'
    '  yes = nude and adult photos are used in training (adult preset).\n'
    '  no  = only clothed photos are used.\n'
    '\n'
    'Adult content in generation\n'
    '  yes = the LoRA may be used for adult (18+) generation. This is written\n'
    '        into the LoRA so Canvas knows.\n'
    '  no  = the LoRA is marked for clothed generation only.\n'
    '\n'
    'No minors in photos\n'
    '  yes = you confirm that no one under 18 appears in any photo.\n'
    '  Forge also tells the description model that the subject is an adult.\n'
    '  The automatic age check on every photo always runs anyway.\n'
    '\n'
    'Date\n'
    '  The date of the permission, as YYYY-MM-DD.\n'
    '\n'
    'Example: an adult model who agreed to everything, adult content included,\n'
    'sets every line to "yes", as above.\n'
)


@dataclass(frozen=True)
class ReleaseInfo:
    """What a ``release.txt`` declares: its hash, the model name, the yes/no answers and the date.

    ``bool | None`` fields are ``True``/``False`` for an explicit ``yes``/``no`` and
    ``None`` when the line is missing or says anything else.
    """

    path: Path
    sha256: str
    model_name: str = ""
    adult: bool | None = None
    no_minors: bool | None = None
    consent_training: bool | None = None
    consent_generation: bool | None = None
    lora_allowed: bool | None = None
    adult_content: bool | None = None  # "Adult content in training"
    adult_generation: bool | None = None  # "Adult content in generation"
    date: str = ""

    @property
    def complete_consent(self) -> bool:
        """Whether the file alone answers the consent prompts: 18+, training consent and LoRA allowed."""
        return self.adult is True and self.consent_training is True and self.lora_allowed is True


def slug(name: str) -> str:
    """The LoRA/project slug, the same rule ``forge-train`` uses: lowercase, every char outside [a-z0-9] becomes '-'."""
    return re.sub(r"[^a-z0-9]", "-", name.lower())


def _yes_no(value: str) -> bool | None:
    """``yes``/``no`` case-insensitively; anything else is not set."""
    lowered = value.lower()
    if lowered == "yes":
        return True
    if lowered == "no":
        return False
    return None


def _model_name(value: str) -> str:
    """The name whole, with one pair of surrounding double quotes stripped when present."""
    if len(value) >= 2 and value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    return value


def parse_release(path: Path) -> ReleaseInfo:
    """Parse ``release.txt``: one ``Key: value`` per line, case-insensitive keys, trimmed values.

    A ``--- How to fill this in ---`` line ends the data (everything after it is ignored),
    blank lines are skipped, unknown keys are ignored. The sha256 is over the raw file bytes.
    """
    target = Path(path)
    raw = target.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8", errors="replace")
    values: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line == _EXPLANATION_MARKER:
            break
        key, sep, value = line.partition(":")
        if not sep:
            continue
        value = value.strip()
        field = _YES_NO_KEYS.get(key.strip().lower())
        if field is not None:
            values[field] = _yes_no(value)
        elif key.strip().lower() == "model name":
            values["model_name"] = _model_name(value)
        elif key.strip().lower() == "date":
            values["date"] = value
    return ReleaseInfo(path=target, sha256=digest, **values)


def find_release(folder: Path) -> Path | None:
    """``release.txt`` directly inside ``folder`` (case variants like ``Release.txt`` accepted), else None.

    The match must be a regular readable file; nested files and folders named ``release.txt`` do not count.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return None
    try:
        candidates = [
            entry
            for entry in folder.iterdir()
            if entry.is_file() and entry.name.lower() == RELEASE_FILE and os.access(entry, os.R_OK)
        ]
    except OSError:
        return None
    if not candidates:
        return None
    candidates.sort(key=lambda entry: entry.name != RELEASE_FILE)
    return candidates[0]


def _flag(value: bool | None) -> str:
    return "1" if value else "0"


def _print_shell(info: ReleaseInfo) -> None:
    """Shell-safe assignments for ``eval`` in ``forge-train`` (see ``--shell``)."""
    assignments = [
        f"RELEASE_SHA256={shlex.quote(info.sha256)}",
        f"RELEASE_NAME={shlex.quote(info.model_name)}",
        f"RELEASE_ADULT={_flag(info.adult)}",
        f"RELEASE_NO_MINORS={_flag(info.no_minors)}",
        f"RELEASE_CONSENT={'1' if info.complete_consent else '0'}",
        f"RELEASE_ADULT_CONTENT={_flag(info.adult_content)}",
        f"RELEASE_ADULT_GENERATION={_flag(info.adult_generation)}",
    ]
    print("\n".join(assignments))


def _print_json(info: ReleaseInfo) -> None:
    payload = {
        "path": str(info.path),
        "sha256": info.sha256,
        "model_name": info.model_name,
        "adult": info.adult,
        "no_minors": info.no_minors,
        "consent_training": info.consent_training,
        "consent_generation": info.consent_generation,
        "lora_allowed": info.lora_allowed,
        "adult_content": info.adult_content,
        "adult_generation": info.adult_generation,
        "date": info.date,
        "complete_consent": info.complete_consent,
    }
    print(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> int:
    """``python -m voxprint_forge.lora.release_file --shell FILE | --json FILE | --template``."""
    parser = argparse.ArgumentParser(prog="python -m voxprint_forge.lora.release_file")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--shell", metavar="FILE", help="print shell-safe assignments for eval")
    group.add_argument("--json", metavar="FILE", dest="json_file", help="print the parsed fields as JSON")
    group.add_argument("--template", action="store_true", help="print the release.txt template")
    ns = parser.parse_args(list(sys.argv[1:] if argv is None else argv))
    if ns.template:
        print(TEMPLATE, end="")
        return 0
    info = parse_release(ns.shell if ns.shell is not None else ns.json_file)
    if ns.shell is not None:
        _print_shell(info)
    else:
        _print_json(info)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
