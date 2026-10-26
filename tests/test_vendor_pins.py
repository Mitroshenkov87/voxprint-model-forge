"""Vendored wheels must match vendor/pins.txt and the declared version."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import check_vendor_pins  # noqa: E402


def _wheel(version: str = "1.2.3", payload: bytes = b"wheel-bytes") -> tuple[str, str]:
    name = f"demo_pkg-{version}-py3-none-any.whl"
    return name, hashlib.sha256(payload).hexdigest()


def _tree(
    root: Path,
    *,
    payload: bytes = b"wheel-bytes",
    version: str = "1.2.3",
    pin: str | None = None,
    py_version: str = "1.2.3",
    extra_pin: str | None = None,
    sidecar: str | None = None,
    omit_pin: bool = False,
    requirements: str | None = None,
) -> None:
    vendor = root / "vendor"
    vendor.mkdir(parents=True)
    name, digest = _wheel(version, payload)
    (vendor / name).write_bytes(payload)
    recorded = pin if pin is not None else digest
    lines = [] if omit_pin else [f"{recorded}  {name}"]
    if extra_pin is not None:
        lines.append(extra_pin)
    (vendor / "pins.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    side = sidecar if sidecar is not None else recorded
    (vendor / f"{name}.sha256").write_text(f"{side}  {name}\n", encoding="utf-8", newline="\n")
    (root / "pyproject.toml").write_text(
        "\n".join(
            [
                "[project]",
                'name = "demo"',
                'version = "0"',
                "dependencies = [",
                f'  "demo-pkg=={py_version}",',
                "]",
                "[project.optional-dependencies]",
                "ui = [",
                f'  "demo-pkg[ui]=={py_version}",',
                "]",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    if requirements is not None:
        (root / "requirements.txt").write_text(requirements, encoding="utf-8", newline="\n")


def test_repository_pins_match() -> None:
    assert check_vendor_pins.problems(ROOT) == []


def test_wrong_wheel_hash_fails(tmp_path: Path) -> None:
    _tree(tmp_path, pin="ab" * 32, sidecar="ab" * 32)
    found = check_vendor_pins.problems(tmp_path)
    assert any("!= sha256" in item for item in found)


def test_pyproject_version_must_match_the_wheel(tmp_path: Path) -> None:
    _tree(tmp_path, py_version="9.9.9")
    found = check_vendor_pins.problems(tmp_path)
    assert any("pyproject.toml: demo-pkg==9.9.9 != demo-pkg==1.2.3" in item for item in found)


def test_missing_and_extra_pins_fail(tmp_path: Path) -> None:
    _tree(tmp_path, omit_pin=True, extra_pin="cd" * 32 + "  other-1.0.0-py3-none-any.whl")
    found = check_vendor_pins.problems(tmp_path)
    assert any(item.endswith(": no pin") for item in found)
    assert any("no wheel" in item for item in found)


def test_requirements_version_must_match_the_wheel(tmp_path: Path) -> None:
    _tree(tmp_path, requirements="demo-pkg>=1.2.3\n")
    found = check_vendor_pins.problems(tmp_path)
    assert any(item.startswith("requirements.txt: demo-pkg>=1.2.3") for item in found)


def test_venv_requirements_are_ignored(tmp_path: Path) -> None:
    _tree(tmp_path)
    folder = tmp_path / ".venv"
    folder.mkdir()
    (folder / "requirements.txt").write_text("demo-pkg==9.9.9\n", encoding="utf-8", newline="\n")
    assert check_vendor_pins.problems(tmp_path) == []


def test_cli_is_quiet_on_success(capsys: pytest.CaptureFixture[str]) -> None:
    assert check_vendor_pins.main() == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "vendor pins ok"
    assert captured.err == ""
