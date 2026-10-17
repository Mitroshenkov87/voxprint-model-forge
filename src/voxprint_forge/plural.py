"""Plural forms for count strings, shared by the GUI and the CLI.

A catalog entry whose value is an object holds one form per plural category:
Russian ``one`` / ``few`` / ``many``, English ``one`` / ``other``. The form is picked
by the parameter named in ``"by"`` (default ``count``) with :func:`ru_plural`
(Russian) or the English rule.
"""

from __future__ import annotations

from collections.abc import Mapping

RU_CATEGORIES = ("one", "few", "many")
EN_CATEGORIES = ("one", "other")


def ru_plural(count: int, one: str, few: str, many: str) -> str:
    """The Russian form for ``count``: 1, 21, 101 -> ``one``; 2-4, 22-24 -> ``few``; 0, 5-20, 25-30 -> ``many``."""
    number = abs(int(count))
    mod10, mod100 = number % 10, number % 100
    if mod10 == 1 and mod100 != 11:
        return one
    if 2 <= mod10 <= 4 and not 12 <= mod100 <= 14:
        return few
    return many


def category(count: int, lang: str) -> str:
    """The plural category name of ``count`` in ``lang`` (``ru`` via :func:`ru_plural`, else English)."""
    if lang == "ru":
        return ru_plural(count, "one", "few", "many")
    return "one" if abs(int(count)) == 1 else "other"


def pick(forms: Mapping[str, str], params: Mapping[str, object], lang: str) -> str:
    """The form of ``forms`` for its counting parameter; without an integer, the general form (``many`` / ``other``)."""
    count = params.get(forms.get("by", "count"))
    if isinstance(count, bool) or not isinstance(count, int):
        name = "many" if lang == "ru" else "other"
    else:
        name = category(count, lang)
    for key in (name, "other", "many"):
        if key in forms:
            return forms[key]
    return next(value for key, value in forms.items() if key != "by")


def fill(template: str, params: Mapping[str, object]) -> str:
    """``template.format(**params)``; a broken placeholder returns the template unchanged."""
    try:
        return template.format(**params)
    except (KeyError, IndexError, ValueError):
        return template
