"""The installer records build and rc into the wheel, not only into the CI environment."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from voxprint_common.buildinfo import load

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("stamp_build", ROOT / "installer" / "stamp_build.py")
assert _SPEC is not None and _SPEC.loader is not None
stamp_build = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stamp_build)

WORKFLOW = (ROOT / ".github" / "workflows" / "installer.yml").read_text(encoding="utf-8")
SMOKE = (ROOT / "scripts" / "windows_smoke.ps1").read_text(encoding="utf-8")
PACKAGE_DATA = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

CATALOG = {
    "app": "model-forge",
    "version": "0.1.0",
    "rc": None,
    "build": None,
    "codename": "Charash",
    "offset": 20,
}


def _write_catalog(path: Path, catalog: dict[str, object] | None = None) -> None:
    path.write_text(json.dumps(catalog if catalog is not None else CATALOG, indent=2) + "\n", encoding="utf-8")


def test_resolve_build_matches_the_shared_loader() -> None:
    """VOXPRINT_BUILD wins, otherwise the run number plus offset, otherwise the recorded build."""
    assert stamp_build.resolve_build(CATALOG, {"VOXPRINT_BUILD": "9", "GITHUB_RUN_NUMBER": "3"}) == 9
    assert stamp_build.resolve_build(CATALOG, {"GITHUB_RUN_NUMBER": "3"}) == 23
    assert stamp_build.resolve_build(CATALOG, {}) is None
    assert stamp_build.resolve_build({**CATALOG, "build": 5}, {}) == 5


def test_null_catalogue_without_a_frozen_file_is_dev(tmp_path: Path) -> None:
    """A packaged BUILD.json with a null build prints dev when the CI variables are absent."""
    package = tmp_path / "pkg"
    package.mkdir()
    _write_catalog(package / "BUILD.json", {**CATALOG, "offset": 0})
    info = load(package_dir=package, env={})
    assert info.build is None
    assert info.rc is None
    assert info.label() == '0.1.0 dev "Charash"'


def test_pack_stamp_is_what_an_installed_import_reads(tmp_path: Path) -> None:
    """The wheel's build_info.json supplies build and rc even if a parent catalogue is null."""
    repo = tmp_path / "repo"
    package = repo / "pkg"
    stage = tmp_path / "stage"
    catalog_path = tmp_path / "BUILD.json"
    recorded = {**CATALOG, "rc": 4}
    _write_catalog(catalog_path, recorded)
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'parent'\n", encoding="utf-8")
    _write_catalog(
        repo / "BUILD.json",
        {
            "app": "model-forge",
            "version": "0.0.0",
            "rc": None,
            "build": None,
            "codename": "Other",
            "offset": 0,
        },
    )
    frozen = stamp_build.write_pack(catalog_path, package, stage, {"GITHUB_RUN_NUMBER": "3"})
    assert frozen == {
        "app": "model-forge",
        "version": "0.1.0",
        "rc": 4,
        "build": 23,
        "codename": "Charash",
    }
    info = load(package_dir=package, env={})
    assert info.build == 23
    assert info.rc == 4
    assert info.label() == '0.1.0 RC 4 build 23 "Charash"'
    payload = json.loads((stage / "BUILD.json").read_text(encoding="utf-8"))
    assert payload["build"] == 23
    assert payload["rc"] == 4
    assert payload["offset"] == 20
    wheel = json.loads((package / "build_info.json").read_text(encoding="utf-8"))
    assert wheel["build"] == 23
    assert wheel["rc"] == 4
    assert "offset" not in wheel


def test_null_rc_is_kept_in_the_stamped_files(tmp_path: Path) -> None:
    """A final build still carries the rc key, with JSON null."""
    catalog_path = tmp_path / "BUILD.json"
    _write_catalog(catalog_path)
    frozen = stamp_build.write_pack(catalog_path, tmp_path / "pkg", tmp_path / "stage", {"GITHUB_RUN_NUMBER": "110"})
    assert frozen["rc"] is None
    assert frozen["build"] == 130
    text = (tmp_path / "pkg" / "BUILD.json").read_text(encoding="utf-8")
    assert '"build": 130' in text
    assert '"rc": null' in text


def test_require_build_fails_closed_without_a_number(tmp_path: Path, monkeypatch) -> None:
    """CI passes --require-build so a pack with no run number does not ship 'dev'."""
    monkeypatch.delenv("GITHUB_RUN_NUMBER", raising=False)
    monkeypatch.delenv("VOXPRINT_BUILD", raising=False)
    catalog_path = tmp_path / "BUILD.json"
    _write_catalog(catalog_path, {**CATALOG, "offset": 0})
    argv = [
        "--catalog",
        str(catalog_path),
        "--package-dir",
        str(tmp_path / "pkg"),
        "--stage-dir",
        str(tmp_path / "stage"),
        "--require-build",
    ]
    assert stamp_build.main(argv) == 2
    monkeypatch.setenv("VOXPRINT_BUILD", "9")
    assert stamp_build.main(argv) == 0
    stamped = json.loads((tmp_path / "pkg" / "build_info.json").read_text(encoding="utf-8"))
    assert stamped["build"] == 9


def test_workflow_stamps_the_wheel_before_it_is_built() -> None:
    """The stamp runs before pip wheel, and the raw catalogue is not copied over it."""
    assert "build_info.json" in PACKAGE_DATA
    stage = WORKFLOW.split("- name: Stage the app", 1)[1].split("- name: PowerShell 7", 1)[0]
    assert stage.index("stamp_build.py") < stage.index("pip wheel")
    assert "--require-build" in stage
    assert stage.index("pip wheel") < stage.index("wheels-incoming")
    copied = stage.split("foreach ($name in @(", 1)[1].split(")", 1)[0]
    assert "BUILD.json" not in copied
    version = SMOKE.split('Invoke-Step "version"', 1)[1].split('Invoke-Step "upgrade-busy"', 1)[0]
    assert "build_info.json" in version
    assert "-InstalledOnly" in version
    assert "GITHUB_RUN_NUMBER" in SMOKE
    assert "checkout is on PATH" in SMOKE
