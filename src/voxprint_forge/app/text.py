"""``tr`` for Model Forge: the shell's catalogs plus the shared range placeholders and plural forms.

Every GUI string goes through here, so ``{rec_low}``–``{rec_high}`` and the Expert range always
come from :mod:`voxprint_forge.lora.limits`, and count strings pick their form with
:func:`voxprint_forge.plural.pick` (the same helper the CLI uses).
"""

from __future__ import annotations

from voxprint_common.i18n import get_language, load_catalog
from voxprint_common.i18n import tr as shell_tr

from voxprint_forge.lora.limits import text_params
from voxprint_forge.plural import EN_CATEGORIES, RU_CATEGORIES, fill, pick


def _forms(key: str, lang: str) -> tuple[str, dict[str, str]]:
    """The plural forms of ``key`` and their language; empty when the entry is a plain string."""
    for code in dict.fromkeys((lang, "en")):
        catalog = load_catalog(code)
        if key in catalog:
            return code, {}
        names = (*(RU_CATEGORIES if code == "ru" else EN_CATEGORIES), "by")
        found = {name: catalog[f"{key}.{name}"] for name in names if f"{key}.{name}" in catalog}
        if found:
            return code, found
    return lang, {}


def tr(key: str, **params: object) -> str:
    """Translate ``key`` in the active language; ``count`` picks the plural form."""
    values = {**text_params(), **params}
    code, forms = _forms(key, get_language())
    if forms:
        return fill(pick(forms, params, code), values)
    return shell_tr(key, **values)
