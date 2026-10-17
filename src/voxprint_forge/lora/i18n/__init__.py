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
def catalog(lang: str) -> dict[str, str | dict[str, str]]:
    """All strings for ``lang`` (``en`` or ``ru``)."""
    if lang not in LANGS:
        raise ValueError(f"unknown language {lang!r}")
    return json.loads((_DIR / f"{lang}.json").read_text(encoding="utf-8"))


def current_lang() -> str:
    """``ru`` when ``VOXPRINT_LANG`` or the system locale is Russian, else ``en``."""
    raw = os.environ.get("VOXPRINT_LANG", "") or (locale.getlocale()[0] or "") or os.environ.get("LANG", "")
    return "ru" if raw.lower().startswith("ru") else "en"


def text(key: str, lang: str | None = None, **params: object) -> str:
    """The string for ``key`` with ``{param}`` placeholders filled. Unknown keys return the key.

    The range placeholders (``{rec_low}`` …) come from :mod:`voxprint_forge.lora.limits`; an entry with
    plural forms picks one by ``count`` through :func:`voxprint_forge.plural.pick`.
    """
    from voxprint_forge.lora.limits import text_params
    from voxprint_forge.plural import fill, pick

    code = lang or current_lang()
    template = catalog(code).get(key, key)
    if isinstance(template, dict):
        template = pick(template, params, code)
    return fill(str(template), {**text_params(), **params})
