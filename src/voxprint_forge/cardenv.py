"""Jinja environment for model cards. Interpolated fields are escaped."""

from __future__ import annotations

from jinja2 import Environment, StrictUndefined


def card_environment(*, keep_trailing_newline: bool = False) -> Environment:
    """Return an environment that escapes every interpolated card field.

    ``autoescape=True`` is literal so every string and ``.jinja`` template
    escapes. A template emits a value raw only with ``|safe``, and only for
    a paragraph Forge wrote itself.
    """
    return Environment(
        undefined=StrictUndefined,
        autoescape=True,
        keep_trailing_newline=keep_trailing_newline,
    )
