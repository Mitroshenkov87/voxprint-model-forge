"""Strip secrets from anything that might be logged or returned."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

SECRET_ENV = (
    "HF_TOKEN",
    "KAGGLE_API_TOKEN",
    "GH_TOKEN",
    "FORGE_DRIVE_TOKEN",
    "FORGE_DRIVE_SERVICE_ACCOUNT",
)


def active_secrets(extra: Sequence[str] = ()) -> list[str]:
    """Secret values currently set in the environment, plus ``extra``."""
    found = [os.environ[name] for name in SECRET_ENV if os.environ.get(name)]
    found.extend(item for item in extra if item)
    return [item for item in found if len(item) >= 4]


def scrub(text: str, secrets: Sequence[str] | None = None) -> str:
    """Replace known secret values in ``text`` with a redaction mark."""
    cleaned = text
    for secret in secrets if secrets is not None else active_secrets():
        if secret:
            cleaned = cleaned.replace(secret, "***")
    return cleaned


def log(message: str) -> None:
    """Print a scrubbed line to stderr."""
    print(scrub(message), file=sys.stderr)
