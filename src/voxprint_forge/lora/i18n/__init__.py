"""English and Russian strings for the LoRA trainer (``lora.*`` keys)."""

from __future__ import annotations

import json
import locale
import os
from functools import cache
from pathlib import Path

LANGS = ("en", "ru")
_DIR = Path(__file__).resolve().parent


@cache
def catalog(lang: str) -> dict[str, str]:
    """All strings for ``lang`` (``en`` or ``ru``)."""
    if lang not in LANGS:
        raise ValueError(f"unknown language {lang!r}")
    return json.loads((_DIR / f"{lang}.json").read_text(encoding="utf-8"))


def current_lang() -> str:
    """``ru`` when ``VOXPRINT_LANG`` or the system locale is Russian, else ``en``."""
    raw = os.environ.get("VOXPRINT_LANG", "") or (locale.getlocale()[0] or "") or os.environ.get("LANG", "")
    return "ru" if raw.lower().startswith("ru") else "en"


def text(key: str, lang: str | None = None, **params: object) -> str:
    """The string for ``key`` with ``{param}`` placeholders filled. Unknown keys return the key."""
    template = catalog(lang or current_lang()).get(key, key)
    try:
        return template.format(**params)
    except (KeyError, IndexError, ValueError):
        return template
