"""Fail when a vendored wheel disagrees with its pin or the declared version.

    python tools/check_vendor_pins.py

The pin file is ``vendor/pins.txt`` (sha256sum format, LF). Every
``vendor/*.whl`` must match a row there and its ``.sha256`` sidecar. Each
``==`` pin for that distribution in ``pyproject.toml`` and ``requirements*.txt``
must name the same version.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SKIP = {".git", ".venv", "venv", "site-packages", "__pycache__", "node_modules", ".pytest_cache", ".ruff_cache"}
_NAME = re.compile(r"[-_.]+")
_SPEC = re.compile(r"([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[^\]]+\])?(.*)")


def _normalize(name: str) -> str:
    return _NAME.sub("-", name).lower()


def _parse_pins(text: str, label: str) -> tuple[list[tuple[str, str]], list[str]]:
    rows: list[tuple[str, str]] = []
    found: list[str] = []
    if "\r" in text:
        found.append(f"{label}: CR")
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        digest = parts[0] if parts else ""
        if len(parts) != 2 or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            found.append(f"{label}:{number}")
            continue
        rows.append((parts[1], digest))
    return rows, found


def _wheel_identity(filename: str) -> tuple[str, str] | None:
    if not filename.endswith(".whl"):
        return None
    pieces = filename[: -len(".whl")].split("-")
    if len(pieces) not in (5, 6):
        return None
    return _normalize(pieces[0]), pieces[1]


def _requirement_name(text: str, wanted: dict[str, str]) -> tuple[str, str] | None:
    token = text.split(";", 1)[0].strip().split()[0] if text.strip() else ""
    match = _SPEC.fullmatch(token)
    if match is None:
        return None
    name = _normalize(match.group(1))
    if name not in wanted:
        return None
    return name, token


def _dependency_strings(document: object) -> list[str]:
    if not isinstance(document, dict):
        return []
    project = document.get("project")
    if not isinstance(project, dict):
        return []
    found: list[str] = []
    dependencies = project.get("dependencies", [])
    if isinstance(dependencies, list):
        found.extend(item for item in dependencies if isinstance(item, str))
    optional = project.get("optional-dependencies", {})
    if isinstance(optional, dict):
        for items in optional.values():
            if isinstance(items, list):
                found.extend(item for item in items if isinstance(item, str))
    return found


def _requirement_files(root: Path) -> list[Path]:
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in _SKIP]
        for name in filenames:
            if name.startswith("requirements") and name.endswith(".txt"):
                found.append(Path(dirpath) / name)
    return found


def problems(root: Path) -> list[str]:
    """Return problem lines. Empty when every vendored wheel matches its pins."""
    found: list[str] = []
    vendor = root / "vendor"
    pin_path = vendor / "pins.txt"
    if not pin_path.is_file():
        return ["vendor/pins.txt missing"]
    rows, parsed = _parse_pins(pin_path.read_text(encoding="utf-8"), "vendor/pins.txt")
    found.extend(parsed)
    pinned = {name: digest for name, digest in rows}
    wheels = sorted(vendor.glob("*.whl"))
    wheel_names = {path.name for path in wheels}
    for name in sorted(pinned):
        if name not in wheel_names:
            found.append(f"pin {name}: no wheel")
    versions: dict[str, set[str]] = {}
    for path in wheels:
        identity = _wheel_identity(path.name)
        if identity is None:
            found.append(f"{path.name}: cannot parse wheel name")
            continue
        dist, version = identity
        versions.setdefault(dist, set()).add(version)
        digest = pinned.get(path.name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest is None:
            found.append(f"{path.name}: no pin")
        elif digest != actual:
            found.append(f"{path.name}: pin {digest} != sha256 {actual}")
        sidecar = path.parent / f"{path.name}.sha256"
        label = f"vendor/{sidecar.name}"
        if not sidecar.is_file():
            found.append(f"{label}: missing")
            continue
        side_rows, side_parsed = _parse_pins(sidecar.read_text(encoding="utf-8"), label)
        found.extend(side_parsed)
        side = [item for item_name, item in side_rows if item_name == path.name]
        recorded = digest if digest is not None else actual
        if len(side) != 1 or side[0] != recorded:
            found.append(f"{label}: sidecar != pin {recorded}")
    wanted: dict[str, str] = {}
    for dist, values in sorted(versions.items()):
        if len(values) != 1:
            found.append(f"{dist}: vendor versions {', '.join(sorted(values))}")
            continue
        wanted[dist] = next(iter(values))
    pyproject = root / "pyproject.toml"
    seen: set[str] = set()
    if not pyproject.is_file():
        found.append("pyproject.toml missing")
    else:
        try:
            document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            found.append(f"pyproject.toml: {exc}")
            document = {}
        for spec in _dependency_strings(document):
            mentioned = _requirement_name(spec, wanted)
            if mentioned is None:
                continue
            dist, token = mentioned
            if _rest_is_exact(token, wanted[dist]):
                seen.add(dist)
            else:
                found.append(f"pyproject.toml: {token} != {dist}=={wanted[dist]}")
    for path in _requirement_files(root):
        source = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            found.append(f"{source}: not utf-8")
            continue
        for line in text.splitlines():
            stripped = line.split("#", 1)[0].strip()
            if not stripped or stripped.startswith("-"):
                continue
            mentioned = _requirement_name(stripped, wanted)
            if mentioned is None:
                continue
            dist, token = mentioned
            if not _rest_is_exact(token, wanted[dist]):
                found.append(f"{source}: {token} != {dist}=={wanted[dist]}")
    for dist, version in sorted(wanted.items()):
        if dist not in seen:
            found.append(f"pyproject.toml: {dist} is not pinned to {version}")
    return found


def _rest_is_exact(token: str, version: str) -> bool:
    match = _SPEC.fullmatch(token.split(";", 1)[0].strip().split()[0])
    if match is None:
        return False
    return match.group(2).strip() == f"=={version}"


def main() -> int:
    """Print problem lines and return 1 when any are found."""
    found = problems(ROOT)
    for line in found:
        print(line, file=sys.stderr)
    if not found:
        print("vendor pins ok")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
