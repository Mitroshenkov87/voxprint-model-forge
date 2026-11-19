"""Record the pack-time build and rc into the wheel and the installer payload.

The catalogue ``BUILD.json`` keeps ``offset`` and a null ``build`` for a dev tree.
``voxprint_common.buildinfo`` then prefers ``VOXPRINT_BUILD``, then
``GITHUB_RUN_NUMBER`` plus ``offset``, and only then a recorded ``build``. On a
laptop those variables are unset, so an installed copy that still has a null
``build`` prints ``dev``.

This writes the resolved number into the package (so the wheel carries it) and
into the staged app directory (so the installed ``BUILD.json`` carries it).
``build_info.json`` is the frozen file the shared loader merges over the catalogue.

A tag build takes ``rc`` from the tag: ``v0.1.0-rc.12`` stamps ``rc=12`` and
``v0.1.0`` stamps a final build. The tag comes from ``--tag``, or from
``GITHUB_REF_NAME`` when GitHub reports a tag ref. A tag whose version is not
the ``BUILD.json`` version fails the stamp. Without a tag, ``rc`` is copied from
the catalogue as before.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

try:
    from voxprint_common.buildinfo import _VERSION_RC
except ImportError:  # CI stamps before anything is installed; same pattern as voxprint_common 0.5.2.
    _VERSION_RC = re.compile(r"^(\d+\.\d+\.\d+)(?:-rc(?:\.(\d+))?)?$", re.IGNORECASE)

__all__ = ["frozen_document", "main", "rc_from_tag", "release_tag", "resolve_build", "write_pack"]


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def resolve_build(catalog: Mapping[str, object], env: Mapping[str, str]) -> int | None:
    """Build number the shared loader would print for this catalogue and environment.

    ``VOXPRINT_BUILD`` wins. Otherwise the number is ``GITHUB_RUN_NUMBER`` plus
    ``offset``. With neither, a recorded ``build`` is kept, and a missing one
    stays null.
    """
    override = str(env.get("VOXPRINT_BUILD", "")).strip()
    if override.isdigit():
        return int(override)
    run = str(env.get("GITHUB_RUN_NUMBER", "")).strip()
    if run.isdigit():
        offset = _optional_int(catalog.get("offset")) or 0
        return int(run) + offset
    return _optional_int(catalog.get("build"))


def release_tag(env: Mapping[str, str]) -> str | None:
    """The tag GitHub is building, or None for a branch, pull request or local build."""
    ref = str(env.get("GITHUB_REF", "")).strip()
    if str(env.get("GITHUB_REF_TYPE", "")).strip() == "tag" or ref.startswith("refs/tags/"):
        name = str(env.get("GITHUB_REF_NAME", "")).strip() or ref.removeprefix("refs/tags/")
        return name or None
    return None


def rc_from_tag(tag: str, version: str) -> int | None:
    """RC number of ``v<version>-rc.N``, or None for the final tag ``v<version>``.

    Raises ValueError when the tag is not a release tag, has no RC number, or
    names a version other than ``version`` (the ``BUILD.json`` version).
    """
    text = tag.strip().removeprefix("refs/tags/")
    match = _VERSION_RC.fullmatch(text[1:]) if text[:1] in ("v", "V") else None
    if match is None:
        raise ValueError(f"tag {tag!r} is not v<version> or v<version>-rc.N")
    if match.group(1) != version:
        raise ValueError(f"tag {tag!r} is version {match.group(1)}, but BUILD.json is version {version}")
    if "-rc" in text.lower() and match.group(2) is None:
        raise ValueError(f"tag {tag!r} has no RC number; use v{version}-rc.N")
    return int(match.group(2)) if match.group(2) is not None else None


def frozen_document(catalog: Mapping[str, object], env: Mapping[str, str], tag: str | None = None) -> dict[str, object]:
    """Five-key identity: catalogue fields, with ``build`` and ``rc`` resolved.

    ``tag`` (or the tag in ``env``) sets ``rc``; see :func:`rc_from_tag`.
    """
    version = catalog.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("BUILD.json has no version")
    app = catalog.get("app")
    codename = catalog.get("codename")
    tag = tag if tag is not None else release_tag(env)
    rc = rc_from_tag(tag, version.strip()) if tag else _optional_int(catalog.get("rc"))
    return {
        "app": app.strip() if isinstance(app, str) and app.strip() else "model-forge",
        "version": version.strip(),
        "rc": rc,
        "build": resolve_build(catalog, env),
        "codename": codename.strip() if isinstance(codename, str) else "",
    }


def _dump(document: Mapping[str, object]) -> str:
    return json.dumps(document, indent=2) + "\n"


def write_pack(
    catalog_path: Path,
    package_dir: Path,
    stage_dir: Path,
    env: Mapping[str, str] | None = None,
    tag: str | None = None,
) -> dict[str, object]:
    """Write the stamped identity into the package directory and the stage directory.

    The package copy is what ``pip wheel`` puts in the installed distribution.
    The stage copy is ``{app}\\BUILD.json`` and ``{app}\\build_info.json``.
    """
    source = os.environ if env is None else env
    catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
    if not isinstance(catalog, dict):
        raise ValueError(f"{catalog_path} must contain a JSON object")
    frozen = frozen_document(catalog, source, tag)
    payload = dict(catalog)
    for key in ("app", "version", "rc", "build", "codename"):
        payload[key] = frozen[key]
    package_dir.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "BUILD.json").write_text(_dump(payload), encoding="utf-8")
    (package_dir / "build_info.json").write_text(_dump(frozen), encoding="utf-8")
    (stage_dir / "BUILD.json").write_text(_dump(payload), encoding="utf-8")
    (stage_dir / "build_info.json").write_text(_dump(frozen), encoding="utf-8")
    return frozen


def main(argv: list[str] | None = None) -> int:
    """Stamp the wheel package and the installer stage. Exit 2 when ``--require-build`` has no number."""
    parser = argparse.ArgumentParser(description="Record the pack-time build and rc.")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--require-build", action="store_true")
    parser.add_argument("--tag", help="release tag v<version> or v<version>-rc.N (default: GITHUB_REF_NAME on a tag)")
    args = parser.parse_args(argv)
    try:
        frozen = write_pack(args.catalog, args.package_dir, args.stage_dir, tag=args.tag or None)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"build stamp failed: {exc}", file=sys.stderr)
        return 2
    if args.require_build and frozen["build"] is None:
        print("no build number: set GITHUB_RUN_NUMBER or VOXPRINT_BUILD", file=sys.stderr)
        return 2
    print(f"stamped build={frozen['build']} rc={frozen['rc']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
