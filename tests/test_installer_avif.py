"""The Windows installer checks that Pillow can encode AVIF on Python 3.15.2."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_app_venv_is_python_315_and_checks_pillow_avif() -> None:
    install = (ROOT / "installer" / "windows" / "install.ps1").read_text(encoding="utf-8")
    assert '"--python", "3.15.2"' in install
    assert '"--python", "3.12"' in install
    assert "requirements-avif.txt" not in install
    assert "pillow_avif" not in install
    assert "features.check('avif')" in install
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'pillow-avif-plugin==1.6.0; python_version < \'3.15\'' in pyproject
    assert "pillow>=13" in pyproject


def test_installer_workflow_does_not_bundle_the_cp314_plugin() -> None:
    workflow = (ROOT / ".github" / "workflows" / "installer.yml").read_text(encoding="utf-8")
    assert 'python-version: "3.15.2"' in workflow
    assert 'python-version: "3.14"' not in workflow
    assert "requirements-avif.txt" not in workflow
    assert "pillow_avif_plugin-1.6.0-cp314-cp314-win_amd64.whl" not in workflow
    assert "cp314t" not in workflow
    assert "0.13.7" in workflow


def test_gui_launcher_prints_version_and_help_on_the_parent_console() -> None:
    source = (ROOT / "installer" / "windows" / "launcher.cs").read_text(encoding="utf-8")
    assert "AttachConsole" in source
    assert "BeginOutputReadLine" in source
    assert "BeginErrorReadLine" in source
    assert "python.exe" in source
    assert "--version" in source and "--help" in source
    assert "pythonw.exe" in source
