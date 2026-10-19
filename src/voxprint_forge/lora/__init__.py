"""LoRA trainer engine: prepare photos, train, extend and export a LoRA.

The GUI and the CLI (``voxprint-forge-cli lora ...``) both drive this
package. Every step reports plain-words progress events (see
:mod:`voxprint_forge.lora.events`) with English and Russian strings.
"""

# Default MDL id. Lives here so the CLI parser can name it without loading adapters.
DEFAULT_BASE = "z-image"
