"""Jinja environment for model cards. Interpolated fields are escaped."""

from __future__ import annotations

from jinja2 import Environment, StrictUndefined, select_autoescape


def card_environment(*, keep_trailing_newline: bool = False) -> Environment:
    """Return an environment that escapes every interpolated card field.

    Cards are built with ``from_string``, which has no filename, so
    extension-based autoescape would leave them raw. ``default_for_string``
    and ``default`` both stay on. A template emits a value raw only with
    ``|safe``, and only for a paragraph Forge wrote itself.
    """
    return Environment(
        undefined=StrictUndefined,
        autoescape=select_autoescape(default_for_string=True, default=True),
        keep_trailing_newline=keep_trailing_newline,
    )
