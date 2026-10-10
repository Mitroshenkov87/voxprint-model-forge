"""Permissive-licence gate. A model must be tagged and the tag must be permissive."""

from __future__ import annotations

from voxprint_forge.errors import LicenceError

PERMISSIVE = frozenset(
    {
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
        "bsd-3-clause-clear",
        "isc",
        "unlicense",
        "0bsd",
        "zlib",
    }
)

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
    return repo.strip().lower()


def assert_repo_allowed(repo: str) -> None:
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
    """Return the SPDX id, or raise LicenceError."""
    if repo:
        assert_repo_allowed(repo)
    if card_licence is None or str(card_licence).strip() == "":
        raise LicenceError(
            f"{repo or 'model'} has no licence tag; refusing untagged weights",
            code="untagged",
        )
    tag = str(card_licence).strip().lower()
    if tag not in PERMISSIVE:
        raise LicenceError(
            f"{repo or 'model'} licence tag {tag!r} is not permissive",
            code="non_permissive",
        )
    return tag
