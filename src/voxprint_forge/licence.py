"""Licence gate. The allow-list is ``voxprint_common.licence``.

Excluded repository ids stay in this module. A public upload still refuses a
non-commercial tag and an untagged card before any other licence id is considered.
"""

from __future__ import annotations

from voxprint_common.licence import LicenceError as CommonLicenceError
from voxprint_common.licence import check, is_noncommercial, is_unlicensed, normalize

from voxprint_forge.errors import LicenceError

# Never fetched, even if a card is mis-tagged.
EXCLUDED_REPOS = frozenset(
    {
        "yandex/yandexgpt-5-lite-8b-instruct",
        "yandex/yandexgpt-5-lite-8b-pretrain",
        "yandex/yandexgpt-5-lite-8b-instruct-gguf",
        "smkrv/t-lite-it-2.1-abl-gguf",
    }
)


def normalise_repo(repo: str) -> str:
    """Strip a Hub URL down to ``owner/name``."""
    return repo.strip().lower()


def assert_repo_allowed(repo: str) -> None:
    """Raise when this repo id is not an allowed weight source."""
    name = normalise_repo(repo)
    if not name:
        return
    if name in EXCLUDED_REPOS or "yandexgpt-5-lite" in name:
        raise LicenceError(
            f"{repo} is excluded (custom non-permissive terms, or an untagged third-party abliteration)",
            code="excluded_repo",
        )
    if name.startswith("smkrv/"):
        raise LicenceError(f"{repo} is excluded", code="excluded_repo")


def decide_licence(card_licence: str | None, *, repo: str = "") -> str:
    """Return the shared allow-list id, or raise :class:`LicenceError`.

    The id is the normalized SPDX form (``apache-2.0``, not ``Apache-2.0``).
    Aliases such as ``apache-2`` and ``cc-by`` follow ``voxprint_common.licence``.
    """
    if repo:
        assert_repo_allowed(repo)
    shown = repo or "model"
    if card_licence is None or is_unlicensed(card_licence):
        raise LicenceError(
            f"{shown} has no licence tag; refusing untagged weights",
            code="untagged",
        )
    if is_noncommercial(card_licence):
        tag = normalize(card_licence)
        raise LicenceError(
            f"{shown} licence tag {tag!r} is non-commercial; refusing",
            code="non_commercial",
        )
    try:
        check(card_licence, release=True)
    except CommonLicenceError as exc:
        tag = normalize(card_licence)
        raise LicenceError(
            f"{shown} licence tag {tag!r} is not permissive",
            code="non_permissive",
        ) from exc
    return normalize(card_licence)
