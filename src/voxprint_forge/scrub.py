"""Strip secrets from anything that might be logged or returned."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

SECRET_ENV = ("HF_TOKEN", "KAGGLE_API_TOKEN", "GH_TOKEN")


def active_secrets(extra: Sequence[str] = ()) -> list[str]:
    found = [os.environ[name] for name in SECRET_ENV if os.environ.get(name)]
    found.extend(item for item in extra if item)
    return [item for item in found if len(item) >= 4]


def scrub(text: str, secrets: Sequence[str] | None = None) -> str:
    cleaned = text
    for secret in secrets if secrets is not None else active_secrets():
        if secret:
            cleaned = cleaned.replace(secret, "***")
    return cleaned


def log(message: str) -> None:
    print(scrub(message), file=sys.stderr)
