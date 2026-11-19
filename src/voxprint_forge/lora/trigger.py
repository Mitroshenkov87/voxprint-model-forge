"""Rare trigger tokens, and a warning when a typed trigger is an ordinary word.

The token is ``vxp`` plus consonants. It is never built from the project name,
so a prompt cannot ask the base model to draw that name as text.
"""

from __future__ import annotations

import gzip
import random
import re
from functools import cache
from pathlib import Path

PREFIX = "vxp"
CONSONANTS = "bcdfghjklmnpqrstvwxz"
BODY = 6
_RARE = re.compile(rf"{PREFIX}[{CONSONANTS}]{{{BODY}}}\Z")
_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
_MIN_WORD = 3


def rare_trigger(rng: random.Random | None = None) -> str:
    """``vxp`` plus a short consonant string. A dictionary hit is drawn again."""
    picker = rng or random.SystemRandom()
    token = ""
    for _ in range(32):
        body = "".join(picker.choice(CONSONANTS) for _ in range(BODY))
        token = PREFIX + body
        if not looks_like_words(token):
            return token
    return token


def is_rare_trigger(token: str) -> bool:
    """True when ``token`` is the auto-generated form, not a name slug."""
    return _RARE.fullmatch(token.strip()) is not None


def looks_like_words(trigger: str) -> bool:
    """True when a letter-run in ``trigger`` is an ordinary dictionary word.

    ``pg_synthetic_a`` warns because ``synthetic`` is a word. A ``vxp`` token
    does not. Pieces shorter than three letters (``pg``, ``a``) are ignored.
    """
    words = _dictionary()
    return any(len(part) >= _MIN_WORD and part.lower() in words for part in _TOKEN.findall(trigger))


def prepend_trigger(trigger: str, text: str) -> str:
    """Put ``trigger`` at the front of a training caption, once."""
    token = trigger.strip()
    body = text.strip()
    if not token:
        return body
    if not body:
        return token
    if _starts_with_trigger(body, token):
        return body
    return f"{token}, {body}"


def strip_trigger(trigger: str, text: str) -> str:
    """Remove a leading trigger so a subject phrase can be read from the caption."""
    token = trigger.strip()
    body = text.strip()
    if token and _starts_with_trigger(body, token):
        return body[len(token):].lstrip(" ,:;-")
    return body


def token_slug(text: str) -> str:
    """A file-id slug of text that is already a trigger or a name."""
    return re.sub(r"[^\w]+", "_", text.strip().lower(), flags=re.UNICODE).strip("_")


def _starts_with_trigger(body: str, token: str) -> bool:
    if not body.lower().startswith(token.lower()):
        return False
    rest = body[len(token):]
    return rest == "" or rest[0] in " ,:;-"


@cache
def _dictionary() -> frozenset[str]:
    path = Path(__file__).resolve().parent / "words.txt.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return frozenset(line.strip() for line in handle if line.strip())
