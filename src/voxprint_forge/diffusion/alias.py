"""Map Qwen text-encoder keys that differ by a `model.` prefix."""

from __future__ import annotations


def alias_key(key: str, current: dict[str, object]) -> str | None:
    """Accept `model.layers...` and `layers...` as the same tensor."""
    if key in current:
        return key
    if key.startswith("model.") and key[len("model.") :] in current:
        return key[len("model.") :]
    prefixed = "model." + key
    if prefixed in current:
        return prefixed
    return None
