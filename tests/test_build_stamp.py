"""The installer records build and rc into the wheel, not only into the CI environment."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
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


def _stamp_argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--catalog",
        str(tmp_path / "BUILD.json"),
        "--package-dir",
        str(tmp_path / "pkg"),
        "--stage-dir",
        str(tmp_path / "stage"),
        "--require-build",
        *extra,
    ]


def _no_tag_env(monkeypatch) -> None:
    for name in ("GITHUB_REF", "GITHUB_REF_NAME", "GITHUB_REF_TYPE"):
        monkeypatch.delenv(name, raising=False)


def test_rc_tag_stamps_rc_and_the_label_shows_it(tmp_path: Path, monkeypatch) -> None:
    """A v0.1.0-rc.12 tag build stamps rc=12 over the catalogue's null rc, and the label says RC 12."""
    _no_tag_env(monkeypatch)
    monkeypatch.setenv("VOXPRINT_BUILD", "201")
    _write_catalog(tmp_path / "BUILD.json", {**CATALOG, "offset": 0})
    assert stamp_build.main(_stamp_argv(tmp_path, "--tag", "v0.1.0-rc.12")) == 0
    for folder in ("pkg", "stage"):
        stamped = json.loads((tmp_path / folder / "build_info.json").read_text(encoding="utf-8"))
        assert stamped["rc"] == 12 and stamped["build"] == 201
    info = load(package_dir=tmp_path / "pkg", env={})
    assert info.rc == 12
    assert info.label() == '0.1.0 RC 12 build 201 "Charash"'


def test_rc_comes_from_github_ref_name_on_a_tag_build(tmp_path: Path) -> None:
    """Without --tag, GITHUB_REF_NAME is read only when GitHub reports a tag ref."""
    catalog_path = tmp_path / "BUILD.json"
    _write_catalog(catalog_path)
    env = {"GITHUB_RUN_NUMBER": "3", "GITHUB_REF_TYPE": "tag", "GITHUB_REF_NAME": "v0.1.0-rc.7"}
    assert stamp_build.write_pack(catalog_path, tmp_path / "pkg", tmp_path / "stage", env)["rc"] == 7
    branch = {"GITHUB_RUN_NUMBER": "3", "GITHUB_REF_TYPE": "branch", "GITHUB_REF_NAME": "v0.1.0-rc.7"}
    assert stamp_build.write_pack(catalog_path, tmp_path / "pkg", tmp_path / "stage", branch)["rc"] is None


def test_final_tag_stamps_a_final_build() -> None:
    """v0.1.0 is the final release: rc stays null and the release check keeps '0.1.0 build N'."""
    assert stamp_build.frozen_document(CATALOG, {"GITHUB_RUN_NUMBER": "3"}, "v0.1.0")["rc"] is None


@pytest.mark.parametrize(
    ("tag", "words"),
    [
        ("v0.2.0-rc.1", "BUILD.json is version 0.1.0"),
        ("v0.1.1", "BUILD.json is version 0.1.0"),
        ("v0.1.0-rc", "no RC number"),
        ("release-12", "is not v<version>"),
    ],
)
def test_tag_that_does_not_match_build_json_fails(tmp_path: Path, monkeypatch, capsys, tag, words) -> None:
    """A tag naming another version (or no RC number) fails the stamp with the reason, and writes nothing."""
    _no_tag_env(monkeypatch)
    monkeypatch.setenv("VOXPRINT_BUILD", "201")
    _write_catalog(tmp_path / "BUILD.json")
    assert stamp_build.main(_stamp_argv(tmp_path, "--tag", tag)) == 2
    err = capsys.readouterr().err
    assert err.startswith("build stamp failed:") and words in err
    assert not (tmp_path / "pkg").exists()


def test_non_tag_build_keeps_the_catalogue_rc(tmp_path: Path, monkeypatch) -> None:
    """Branch, pull request and laptop builds are unchanged: rc is copied from BUILD.json."""
    _no_tag_env(monkeypatch)
    monkeypatch.setenv("VOXPRINT_BUILD", "201")
    _write_catalog(tmp_path / "BUILD.json", {**CATALOG, "rc": 4})
    assert stamp_build.main(_stamp_argv(tmp_path)) == 0
    assert json.loads((tmp_path / "pkg" / "build_info.json").read_text(encoding="utf-8"))["rc"] == 4
    _write_catalog(tmp_path / "BUILD.json")
    assert stamp_build.main(_stamp_argv(tmp_path)) == 0
    assert json.loads((tmp_path / "pkg" / "build_info.json").read_text(encoding="utf-8"))["rc"] is None
    assert load(package_dir=tmp_path / "pkg", env={}).label() == '0.1.0 build 201 "Charash"'


def test_tag_builds_pass_the_tag_and_release_expects_rc() -> None:
    """installer.yml passes the tag on tag builds; release.yml expects 'RC N' for rc tags only."""
    stage = WORKFLOW.split("- name: Stage the app", 1)[1].split("- name: PowerShell 7", 1)[0]
    assert '$env:GITHUB_REF_TYPE -eq "tag"' in stage and '"--tag", $env:GITHUB_REF_NAME' in stage
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    identity = release.split("- name: Print the release identity", 1)[1]
    assert identity.index("stamp_build.py") < identity.index("version_line")
    assert '--tag "$GITHUB_REF_NAME"' in identity
    assert 'expected="0.1.0 RC ${BASH_REMATCH[1]} build ${build} \\"Charash\\""' in identity
    assert 'expected="0.1.0 build ${build} \\"Charash\\""' in identity


def test_require_build_fails_closed_without_a_number(tmp_path: Path, monkeypatch) -> None:
    """CI passes --require-build so a pack with no run number does not ship 'dev'."""
    _no_tag_env(monkeypatch)
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
