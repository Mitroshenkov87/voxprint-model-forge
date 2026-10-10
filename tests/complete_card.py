"""Minimal card text that passes the public-upload gate."""

from __future__ import annotations

APACHE_LINE = (
    "Apache-2.0, same as the base model. See https://www.apache.org/licenses/LICENSE-2.0"
)


def complete_card(line: str = APACHE_LINE) -> str:
    """Return a short card with a licence line, credits, and a modification notice."""
    return (
        "---\nlicense: apache-2.0\n---\n\n"
        "## License\n\n"
        f"{line}\n\n"
        "## Credits\n\n"
        "- Example\n\n"
        "## Modification notice\n\n"
        "Modified from the base model named above.\n"
    )
