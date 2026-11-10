"""LoRA trainer engine: prepare photos, train, extend and export a LoRA.

The GUI and the CLI (``voxprint-forge-cli lora ...``) both drive this
package. Every step reports plain-words progress events (see
:mod:`voxprint_forge.lora.events`) with English strings.
"""

from collections.abc import Mapping

# Default MDL id for new projects. Lives here so the CLI parser can name it without loading adapters.
DEFAULT_BASE = "z-image"
# Bases a build may also train on, besides the default (what the app offers as optional).
OPTIONAL_BASES = ("moss-image-base", "z-image-turbo")


def legacy_base() -> str:
    """MDL id for projects and LoRAs saved before the base was recorded.

    Z-Image was the only base then, so a record without base information
    resolves to it.
    """
    return "z-image"


def resolve_base(recorded: object = None) -> str:
    """MDL id from a stored value, or :func:`legacy_base` when none was stored.

    A string is an MDL id. A mapping is a sidecar ``base`` block and uses
    ``mdl_id``. Blank and missing values are projects and LoRAs from before
    the base was recorded, which trained on Z-Image.
    """
    if isinstance(recorded, Mapping):
        recorded = recorded.get("mdl_id")
    if isinstance(recorded, str):
        text = recorded.strip()
        if text:
            return text
    return legacy_base()
