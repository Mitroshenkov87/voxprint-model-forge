"""Prompt ids and the guard that keeps prompt text out of public artefacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from voxprint_forge.compare.codes import CompareSetup

PRIVATE_ENV = "FORGE_PRIVATE_PROMPTS"
_ID_LEN = 12


def prompt_id(text: str) -> str:
    """``sha256(prompt)[:12]``. The same text always yields the same id."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:_ID_LEN]


def private_root_from_env() -> Path | None:
    """Directory from ``FORGE_PRIVATE_PROMPTS``.

    Unset means the private set is skipped. A set value that is not a directory
    is a setup error. This variable is a local path, the same idea as
    ``FORGE_PRIVATE_CALIB``, and it is never a token.
    """
    raw = os.environ.get(PRIVATE_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_dir():
        raise CompareSetup(f"{PRIVATE_ENV} is not a directory: {path}", code="compare_private")
    return path


def leak_ids(text: str, secrets: Sequence[str]) -> list[str]:
    """Prompt ids whose text appears in ``text``. The text itself is not returned."""
    found: list[str] = []
    for secret in secrets:
        if secret and secret in text:
            found.append(prompt_id(secret))
    return found


def assert_text_clean(text: str, secrets: Sequence[str], *, where: str) -> None:
    """Raise a setup error when any secret string appears in ``text``."""
    found = leak_ids(text, secrets)
    if found:
        raise CompareSetup(f"prompt text leaked into {where} for ids {', '.join(found)}", code="compare_leak")


def assert_mapping_clean(payload: Mapping[str, object], secrets: Sequence[str], *, where: str) -> None:
    """Raise when the JSON form of ``payload`` contains a secret string."""
    assert_text_clean(json.dumps(payload, ensure_ascii=False, sort_keys=True), secrets, where=where)


def secret_strings(prompts: Iterable[str], negatives: Iterable[str] = ()) -> tuple[str, ...]:
    """Prompt and negative strings long enough to be worth scanning for."""
    found: list[str] = []
    for item in (*prompts, *negatives):
        text = item.strip()
        if len(text) >= 8 and text not in found:
            found.append(text)
    return tuple(found)


def assert_tree_clean(root: Path, secrets: Sequence[str]) -> list[str]:
    """Return leak ids if any file name or file bytes under ``root`` contain a secret.

    An empty list means the tree is clean. Callers that want a failure raise
    from this list. The returned ids are hashes, not the secret text.
    """
    hits: list[str] = []
    if not root.exists():
        return hits
    encoded = [(item, item.encode("utf-8")) for item in secrets if item]
    for path in sorted(root.rglob("*")):
        name_hits = leak_ids(path.name, secrets)
        hits.extend(name_hits)
        if not path.is_file():
            continue
        data = path.read_bytes()
        for secret, raw in encoded:
            if raw in data:
                hits.append(prompt_id(secret))
    return hits
